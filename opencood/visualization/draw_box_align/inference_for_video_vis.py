# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from typing import OrderedDict

import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--use_laplace', action='store_true',
                        help="whether to use laplace to simulate noise. Otherwise Gaussian")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, late_w_ba, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=1,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty']

    hypes = yaml_utils.load_yaml(None, opt)
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    left_hand = True if "OPV2V" in hypes['test_dir'] else False
    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']


    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # add noise to pose.
    pos_std_list = [0.6, 0.4]
    rot_std_list = [0.6, 0.4]
    pos_mean_list = [0, 0]
    rot_mean_list = [0, 0] 

    vis_idx = list(range(200,300))

    for laplace_noise in [False]:
        opt.use_laplace = laplace_noise
        AP50 = []
        AP70 = []
        for (pos_mean, pos_std, rot_mean, rot_std) in zip(pos_mean_list, pos_std_list, rot_mean_list, rot_std_list):
            noise_setting = OrderedDict()
            noise_args = {'pos_std': pos_std,
                            'rot_std': rot_std,
                            'pos_mean': pos_mean,
                            'rot_mean': rot_mean}

            noise_setting['add_noise'] = True
            noise_setting['args'] = noise_args

            suffix = "_video_vis"
            if opt.use_laplace:
                noise_setting['args']['laplace'] = True
                suffix = "_laplace_video_vis"

            # build dataset for each noise setting
            print('Dataset Building')
            print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
            hypes.update({"noise_setting": noise_setting})
            opencood_dataset = build_dataset(hypes, visualize=True, train=False)
            opencood_dataset_vis = Subset(opencood_dataset, vis_idx)
            data_loader = DataLoader(opencood_dataset_vis,
                                    batch_size=1,
                                    num_workers=4,
                                    collate_fn=opencood_dataset.collate_batch_test,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
            
            # Create the dictionary for evaluation
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                        0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.7: {'tp': [], 'fp': [], 'gt': 0}}
            
            noise_level = f"{pos_std}_{rot_std}_{pos_mean}_{rot_mean}" + suffix


            for i, batch_data in enumerate(data_loader):
                print(f"{noise_level}_{i}")
                np.random.seed(303)
                if batch_data is None:
                    continue
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    uncertainty_tensor = None
                    if opt.fusion_method == 'late':
                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_late_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'early':
                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_early_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'intermediate':
                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_intermediate_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'no':
                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_no_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'no_w_uncertainty':
                        pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
                            inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    else:
                        raise NotImplementedError('Only no, early, late and intermediate'
                                                'fusion is supported.')
                    
                    
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.3)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.5)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.7)
                    if opt.save_npy:
                        npy_save_path = os.path.join(opt.model_dir, 'npy' + suffix)
                        if not os.path.exists(npy_save_path):
                            os.makedirs(npy_save_path)
                        inference_utils.save_prediction_gt(pred_box_tensor,
                                                        gt_box_tensor,
                                                        batch_data['ego'][
                                                            'origin_lidar'][0],
                                                        i,
                                                        npy_save_path)

                    if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None) and (opt.use_laplace is False):
                        vis_save_path_root = os.path.join(opt.model_dir, f'vis_{noise_level}')
                        if not os.path.exists(vis_save_path_root):
                            os.makedirs(vis_save_path_root)

                        vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                        simple_vis.visualize(pred_box_tensor,
                                            gt_box_tensor,
                                            batch_data['ego'][
                                                'origin_lidar'][0],
                                            hypes['postprocess']['gt_range'],
                                            vis_save_path,
                                            method='3d',
                                            left_hand=left_hand,
                                            uncertainty=uncertainty_tensor)
                        
                        vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                        simple_vis.visualize(pred_box_tensor,
                                            gt_box_tensor,
                                            batch_data['ego'][
                                                'origin_lidar'][0],
                                            hypes['postprocess']['gt_range'],
                                            vis_save_path,
                                            method='bev',
                                            left_hand=left_hand,
                                            uncertainty=uncertainty_tensor)
                torch.cuda.empty_cache()

        #     _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
        #                                 opt.model_dir, noise_level)

        #     AP50.append(ap50)
        #     AP70.append(ap70)

        # dump_dict = {'ap50': AP50, 'ap70': AP70}
        # yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, f'AP0507{suffix}.yaml'))


if __name__ == '__main__':
    main()
