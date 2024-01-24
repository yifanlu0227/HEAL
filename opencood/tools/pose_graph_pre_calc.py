# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import importlib
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
import copy
from collections import OrderedDict
import json

import numpy as np
from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, default="/root/OpenCOODv2/opencood/hypes_yaml/v2xset/lidar_only/coalign/precalc.yaml",
                        help='data generation yaml file needed ')
    parser.add_argument("--model_dir", type=str, default="")
    opt = parser.parse_args()
    return opt

SAVE_BOXES= True

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    pos_std_list = [0]#, 0.2, 0.4, 0.6, 0.8, 1.0]
    rot_std_list = [0]#, 0.2, 0.4, 0.6, 0.8, 1.0]
    pos_mean_list = [0]#, 0, 0, 0, 0]
    rot_mean_list = [0]#, 0, 0, 0, 0]

    for (pos_mean, pos_std, rot_mean, rot_std) in zip(pos_mean_list, pos_std_list, rot_mean_list, rot_std_list):
        # setting noise
        noise_setting = OrderedDict()
        noise_args = {'pos_std': pos_std,
                      'rot_std': rot_std,
                      'pos_mean': pos_mean,
                      'rot_mean': rot_mean}

        noise_setting['add_noise'] = True
        noise_setting['args'] = noise_args

        # build dataset for each noise setting
        print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
        hypes.update({"noise_setting": noise_setting})

        print('Dataset Building')
        opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
        opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)
        hypes_ = copy.deepcopy(hypes)
        hypes_['validate_dir'] = hypes_['test_dir']
        opencood_test_dataset = build_dataset(hypes_, visualize=False, train=False)

        train_loader = DataLoader(opencood_train_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)
        
        test_loader = DataLoader(opencood_test_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ########################################################################

        hypes = yaml_utils.load_voxel_params(hypes)

        stage1_model_name = hypes['box_align_pre_calc']['stage1_model'] # point_pillar_disconet_teacher
        stage1_model_config = hypes['box_align_pre_calc']['stage1_model_config']
        stage1_checkpoint_path = hypes['box_align_pre_calc']['stage1_model_path']

        # import the model
        model_filename = "opencood.models." + stage1_model_name
        model_lib = importlib.import_module(model_filename)
        stage1_model_class = None
        target_model_name = stage1_model_name.replace('_', '')

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                stage1_model_class = cls
        
        stage1_model = stage1_model_class(stage1_model_config)
        stage1_model.load_state_dict(torch.load(stage1_checkpoint_path), strict=False)

        # import the postprocessor
        stage1_postprocessor_name = hypes['box_align_pre_calc']['stage1_postprocessor_name']
        stage1_postprocessor_config = hypes['box_align_pre_calc']['stage1_postprocessor_config']
        postprocessor_lib = importlib.import_module('opencood.data_utils.post_processor')
        stage1_postprocessor_class = None
        target_postprocessor_name = stage1_postprocessor_name.replace('_', '')

        for name, cls in postprocessor_lib.__dict__.items():
            if name.lower() == target_postprocessor_name:
                stage1_postprocessor_class = cls
        
        stage1_postprocessor = stage1_postprocessor_class(stage1_postprocessor_config, train=False)
        
        for p in stage1_model.parameters():
            p.requires_grad_(False)

        if torch.cuda.is_available():
                stage1_model.to(device)

        stage1_model.eval()
        stage1_anchor_box = torch.from_numpy(stage1_postprocessor.generate_anchor_box())


        for split in ['train', 'val', 'test']:

            stage1_boxes_dict = dict()

            stage1_boxes_save_dir = f"{hypes['box_align_pre_calc']['output_save_path']}/{split}"
            if not os.path.exists(stage1_boxes_save_dir):
                os.makedirs(stage1_boxes_save_dir)
            stage1_boxes_save_path = os.path.join(stage1_boxes_save_dir, "stage1_boxes.json")
            
            for i, batch_data in enumerate(eval(f"{split}_loader")):
                if batch_data is None:
                    continue

                batch_data = train_utils.to_device(batch_data, device)
                print(i, batch_data['ego']['sample_idx'], batch_data['ego']['cav_id_list'])
                output_stage1 = stage1_model(batch_data['ego'])
                pred_corner3d_list, pred_box3d_list, uncertainty_list = \
                stage1_postprocessor.post_process_stage1(output_stage1, stage1_anchor_box)
                record_len = batch_data['ego']['record_len']
                lidar_pose = batch_data['ego']['lidar_pose']
                lidar_pose_clean = batch_data['ego']['lidar_pose_clean']

                if pred_corner3d_list is None:
                    continue

                """
                    Save the corners, uncertainty, lidar_pose_clean
                """

                if SAVE_BOXES:
                    if pred_corner3d_list is None:
                        stage1_boxes_dict[batch_data['ego']['sample_idx']] = None
                        continue
                    sample_idx = batch_data['ego']['sample_idx']
                    pred_corner3d_np_list = [x.cpu().numpy().tolist() for x in pred_corner3d_list]
                    uncertainty_np_list = [x.cpu().numpy().tolist() for x in uncertainty_list]
                    lidar_pose_clean_np = lidar_pose_clean.cpu().numpy().tolist()
                    stage1_boxes_dict[sample_idx] = OrderedDict()
                    
                    stage1_boxes_dict[sample_idx]['pred_corner3d_np_list'] = pred_corner3d_np_list
                    stage1_boxes_dict[sample_idx]['uncertainty_np_list'] = uncertainty_np_list
                    stage1_boxes_dict[sample_idx]['lidar_pose_clean_np'] = lidar_pose_clean_np
                    stage1_boxes_dict[sample_idx]['cav_id_list'] = batch_data['ego']['cav_id_list']


            if SAVE_BOXES:
                with open(stage1_boxes_save_path, "w") as f:
                    json.dump(stage1_boxes_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
