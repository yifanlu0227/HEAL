"""
-*- coding: utf-8 -*-
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

Incrementally increase heterogeneous agents in order.

Actual collaborator m1 -> m1+m2 -> m1+m2+m3 -> m1+m2+m3+m4

Ego is always m1

commrange is 180 (large enough)

For Intermediate Fusion, we will switch to IntermediateHeterinferFusionDataset
"""

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')



def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="204.8,102.4",
                        help="detection range is [-204.8, +204.8, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--use_cav', type=str, default="[1,2,3,4]",
                        help="evaluate with real collaborator number")
    parser.add_argument('--lidar_degrade', action='store_true',
                        help="whether to degrade lidar. {m1:32, m3:16} and {m1:16, m3:16}")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        hypes = update_dict(hypes, {
            "mapping_dict": {
                "m1": "m1",
                "m2": "m2",
                "m3": "m3",
                "m4": "m4"
            }
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)

    if opt.fusion_method == 'intermediate':
        hypes['fusion']['core_method'] += 'infer' 
    hypes['comm_range'] = 180
    hypes['heter']['assignment_path'] = hypes['heter']['assignment_path'].replace(".json", "_in_order.json")
    hypes = update_dict(hypes, {
            "ego_modality": 'm1'
        })
    
    if opt.lidar_degrade:
        lidar_dict1 = {
            "m1": 32,
            "m3": 16
        }
        lidar_dict2 = {
            "m1": 16,
            "m3": 16
        }
        opt.use_cav = "[4]"
        use_cav_and_lidar_config_pair = [(4, lidar_dict1), (4, lidar_dict2)]
    else:
        lidar_dict0 = {
            'm3': 32
        }
        use_cav_and_lidar_config_pair = [(x, lidar_dict0) for x in eval(opt.use_cav)]

    for (use_cav, lidar_config) in use_cav_and_lidar_config_pair:
        hypes['use_cav'] = use_cav
        if lidar_config is not None:
            hypes['heter']['lidar_channels_dict'] = lidar_config
            print(hypes['heter']['lidar_channels_dict'])

        # build dataset for each noise setting
        print('Dataset Building')
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        # opencood_dataset_subset = Subset(opencood_dataset, range(1220,1260))
        # data_loader = DataLoader(opencood_dataset_subset,
        data_loader = DataLoader(opencood_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)
        
        # Create the dictionary for evaluation
        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

        
        infer_info = opt.fusion_method + opt.note + f"_use_cav{use_cav}"
        if opt.lidar_degrade:
            infer_info += f"_m1_{lidar_config['m1']}_m3_{lidar_config['m3']}"


        for i, batch_data in enumerate(data_loader):
            # we can just save the batch_data to file. For perfomance test.
            # intermediate dataset
            # command: python opencood/tools/inference_heter_in_order.py --model_dir opencood/logs_HEAL/Pyramid_collaboration_m1_base/final_infer --range 102.4,102.4
            # import pickle
            # print(batch_data['ego']['agent_modality_list'])
            # collabortor = "".join(batch_data['ego']['agent_modality_list'])
            # save_dir = f"opencood/logs_HEAL/FLOPs_calc/{collabortor}_online"
            # with open(os.path.join(save_dir, 'input.pkl'), 'wb') as f:
            #     pickle.dump(batch_data, f)
            # break

            # late fusion dataset
            # command: python opencood/tools/inference_heter_in_order.py --model_dir opencood/logs_HEAL/HEAL_late_final --fusion_method late --use_cav [4] --range 102.4,102.4
            # import pickle
            # for i_, (cav_id, cav_content) in enumerate(batch_data.items()):
            #     modality_name = cav_content['modality_name']
            #     print(cav_id, modality_name)
            #     save_dir = f"opencood/logs_HEAL/FLOPs_calc/{modality_name}_single"
            #     with open(os.path.join(save_dir, 'input.pkl'), 'wb') as f:
            #         pickle.dump({'ego':cav_content}, f)
            #     if i_ >= 4:
            #         break
            # raise

            print(f"{infer_info}_{i}")
            if batch_data is None:
                continue
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == 'late':
                    infer_result = inference_late_fusion_heter_in_order(batch_data,
                                                            model,
                                                            opencood_dataset,
                                                            use_cav)
                elif opt.fusion_method == 'intermediate':
                    infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'no':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'single':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset,
                                                                    single_gt=True)
                else:
                    raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                            'fusion is supported.')

                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']
                
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

                if not opt.no_score:
                    infer_result.update({'score_tensor': pred_score})

                if getattr(opencood_dataset, "heterogeneous", False):
                    cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                    infer_result.update({"cav_box_np": cav_box_np, \
                                        "agent_modality_list": agent_modality_list})

                if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                    vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                    if not os.path.exists(vis_save_path_root):
                        os.makedirs(vis_save_path_root)

                    vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                    simple_vis.visualize(infer_result,
                                        batch_data['ego'][
                                            'origin_lidar'][0],
                                        hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand)
            torch.cuda.empty_cache()

        _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                    opt.model_dir, infer_info)




def inference_late_fusion_heter_in_order(batch_data, model, dataset, use_cav):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    # ['ego', "650", "659", ...]  keys in batch_data is in order
    for i_, (cav_id, cav_content) in enumerate(batch_data.items()):
        if i_ >= use_cav:
            break
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict


if __name__ == '__main__':
    main()
