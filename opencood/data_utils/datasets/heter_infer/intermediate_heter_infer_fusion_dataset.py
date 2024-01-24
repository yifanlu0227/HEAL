'''
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
    
Successively adding agent, evaluation for HEAL

We add 'use_cav' to control the number of cars that really participate in collaboration.
We use 'max_cav' to genenrate all gt boxes in the scene.
''' 

import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import merge_features_to_dict, compute_iou, convert_format
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json
from opencood.utils.heter_utils import Adaptor
from opencood.data_utils.datasets.intermediate_heter_fusion_dataset import getIntermediateheterFusionDataset

def getIntermediateheterinferFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    Intermediate_heter_Fusion_Dataset = getIntermediateheterFusionDataset(cls)

    class Intermediate_heter_Infer_Fusion_Dataset(Intermediate_heter_Fusion_Dataset):
        """
        We add 'use_cav' to control the number of cars that really participate in collaboration.
        We use 'max_cav' to genenrate all gt boxes in the scene.
        """
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            self.use_cav = params['use_cav']
            # e.g. 2, and self.max_cav = 5
            # we will use 5 cav to generate gt boxes, but only use 2 cav to do the fusion.
            self.sensor_type_dict = {
                'm1': 'lidar',
                'm2': 'camera',
                'm3': 'lidar',
                'm4': 'camera',
            }
            print("\n\nCaution: make sure your sensor type is consistent with:")
            print(self.sensor_type_dict)
            print("Otherwise, modify it in opencood/data_utils/datasets/heter_infer/intermediate_heter_infer_fusion_dataset.py\n\n")


        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
                
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            
            input_list_m1 = [] # can contain lidar or camera
            input_list_m2 = []
            input_list_m3 = []
            input_list_m4 = []

            agent_modality_list = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            exclude_agent = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            projected_lidar_clean_list = [] # disconet

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    exclude_agent.append(cav_id)
                    continue
                
                ########### Modified Here! 2023.9.19 Yifan Lu ######################
                ## even not match, should be included to get full GT boxes

                # if self.adaptor.unmatched_modality(selected_cav_base['modality_name']):
                #     exclude_agent.append(cav_id)
                #     continue
                ####################################################################

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)   

            if len(cav_id_list) == 0:
                return None

            for cav_id in exclude_agent:
                base_data_dict.pop(cav_id)

            ########## Updated by Yifan Lu 2022.1.26 ############
            # box align to correct pose.
            # stage1_content contains all agent. Even out of comm range.
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content['cav_id_list'] # include those out of range
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent) for cur_agent in cur_agent_id_list] # indexing current agent in `all_agent_id_list`

                    pred_corners_list = [np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]

                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                        cur_agnet_pose, 
                                                                        uncertainty_list=uncertainty_list, 
                                                                        **self.box_align_args)
                        cur_agnet_pose[:,[0,1,4]] = refined_pose 

                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agnet_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agnet_pose[i].tolist()



            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]

            
            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                modality_name = selected_cav_base['modality_name']
                sensor_type = self.sensor_type_dict[selected_cav_base['modality_name']]

                # dynamic object center generator! for heterogeneous input
                if not self.visualize:
                    self.generate_object_center = eval(f"self.generate_object_center_{sensor_type}")
                # need discussion. In test phase, use lidar label.
                else: 
                    self.generate_object_center = self.generate_object_center_lidar

                object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                        ego_cav_base['params']['lidar_pose_clean'])
                
                object_stack.append(object_bbx_center[object_bbx_mask == 1])
                object_id_stack += object_ids

                ##### Modified Here! 2023.8.16 Yifan Lu #####
                if _i >= self.use_cav:
                    continue
                #############################################

                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_cav_base)

                if sensor_type == "lidar":
                    eval(f"input_list_{modality_name}").append(selected_cav_processed[f"processed_features_{modality_name}"])
                elif sensor_type == "camera":
                    eval(f"input_list_{modality_name}").append(selected_cav_processed[f"image_inputs_{modality_name}"])
                else:
                    raise
                
                agent_modality_list.append(modality_name)

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(
                        selected_cav_processed['projected_lidar'])
                
                if self.supervise_single or self.heterogeneous:
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

            cav_num = len(agent_modality_list)

            # generate single view GT label
            if self.supervise_single or self.heterogeneous:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                    })

            if self.kd_flag:
                stack_lidar_np = np.vstack(projected_lidar_stack)
                stack_lidar_np = mask_points_by_range(stack_lidar_np,
                                            self.params['preprocess'][
                                                'cav_lidar_range'])
                stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                processed_data_dict['ego'].update({'teacher_processed_lidar':
                stack_feature_processed})

            
            # exculude all repetitve objects, DAIR-V2X
            if self.params['fusion']['dataset'] == 'dairv2x':
                if len(object_stack) == 1:
                    object_stack = object_stack[0]
                else:
                    ego_boxes_np = object_stack[0]
                    cav_boxes_np = object_stack[1]
                    order = self.params['postprocess']['order']
                    ego_corners_np = box_utils.boxes_to_corners_3d(ego_boxes_np, order)
                    cav_corners_np = box_utils.boxes_to_corners_3d(cav_boxes_np, order)
                    ego_polygon_list = list(convert_format(ego_corners_np))
                    cav_polygon_list = list(convert_format(cav_corners_np))
                    iou_thresh = 0.05 


                    gt_boxes_from_cav = []
                    for i in range(len(cav_polygon_list)):
                        cav_polygon = cav_polygon_list[i]
                        ious = compute_iou(cav_polygon, ego_polygon_list)
                        if (ious > iou_thresh).any():
                            continue
                        gt_boxes_from_cav.append(cav_boxes_np[i])
                    
                    if len(gt_boxes_from_cav):
                        object_stack_from_cav = np.stack(gt_boxes_from_cav)
                        object_stack = np.vstack([ego_boxes_np, object_stack_from_cav])
                    else:
                        object_stack = ego_boxes_np

                unique_indices = np.arange(object_stack.shape[0])
                object_id_stack = np.arange(object_stack.shape[0])
            else:
                # exclude all repetitive objects, OPV2V-H
                unique_indices = \
                    [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            
            for modality_name in self.modality_name_list:
                if self.sensor_type_dict[modality_name] == "lidar":
                    merged_feature_dict = merge_features_to_dict(eval(f"input_list_{modality_name}")) 
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_feature_dict}) # maybe None
                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(eval(f"input_list_{modality_name}"), merge='stack')
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_image_inputs_dict}) # maybe None

            processed_data_dict['ego'].update({'agent_modality_list': agent_modality_list})

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=self.anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses})


            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})


            processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})

            return processed_data_dict


    return Intermediate_heter_Infer_Fusion_Dataset


