# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import random
import math
from collections import OrderedDict
import cv2
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
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.heter_utils import Adaptor

def getLateheterFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class LateheterFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.heterogeneous = True
            self.modality_assignment = None if ('assignment_path' not in params['heter'] or params['heter']['assignment_path'] is None) \
                                            else read_json(params['heter']['assignment_path'])
            self.ego_modality = params['heter']['ego_modality'] # "m1" or "m1&m2" or "m3"

            self.modality_name_list = list(params['heter']['modality_setting'].keys())
            self.sensor_type_dict = OrderedDict()
            
            lidar_channels_dict = params['heter'].get('lidar_channels_dict', OrderedDict())
            mapping_dict = params['heter']['mapping_dict']

            self.adaptor = Adaptor(self.ego_modality, 
                                   self.modality_name_list,
                                   self.modality_assignment,
                                   lidar_channels_dict,
                                   mapping_dict,
                                   None,
                                   train)

            for modality_name, modal_setting in params['heter']['modality_setting'].items():
                self.sensor_type_dict[modality_name] = modal_setting['sensor_type']
                if modal_setting['sensor_type'] == 'lidar':
                    setattr(self, f"pre_processor_{modality_name}", build_preprocessor(modal_setting['preprocess'], train))

                elif modal_setting['sensor_type'] == 'camera':
                    setattr(self, f"data_aug_conf_{modality_name}", modal_setting['data_aug_conf'])

                else:
                    raise("Not support this type of sensor")

            self.reinitialize()

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            if self.train:
                reformat_data_dict = self.get_item_train(base_data_dict)
            else:
                reformat_data_dict = self.get_item_test(base_data_dict, idx)
            return reformat_data_dict

        def get_item_train(self, base_data_dict):
            processed_data_dict = OrderedDict()
            base_data_dict = add_noise_data_dict(
                base_data_dict, self.params["noise_setting"]
            )
            # during training, we return a random cav's data
            # only one vehicle is in processed_data_dict
            if not self.visualize:
                options = []
                for cav_id, cav_content in base_data_dict.items():
                    if cav_content['modality_name'] in self.ego_modality:
                        options.append(cav_id)
                selected_cav_base = base_data_dict[random.choice(options)]
            else:
                selected_cav_id, selected_cav_base = list(base_data_dict.items())[0]
            
            selected_cav_processed = self.get_item_single_car(selected_cav_base)
            processed_data_dict.update({"ego": selected_cav_processed})

            return processed_data_dict


        def get_item_test(self, base_data_dict, idx):
            """
                processed_data_dict.keys() = ['ego', "650", "659", ...]
            """
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            ego_id = -1
            ego_lidar_pose = []
            cav_id_list = []
            lidar_pose_list = []

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                    break

            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)
                if distance > self.params['comm_range']:
                    continue

                if self.adaptor.unmatched_modality(selected_cav_base['modality_name']):
                    continue

                cav_id_list.append(cav_id)
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])

            cav_id_list_newname = []
            for cav_id in cav_id_list:
                selected_cav_base = base_data_dict[cav_id]
                # find the transformation matrix from current cav to ego.
                cav_lidar_pose = selected_cav_base['params']['lidar_pose']
                transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
                cav_lidar_pose_clean = selected_cav_base['params']['lidar_pose_clean']
                transformation_matrix_clean = x1_to_x2(cav_lidar_pose_clean, ego_lidar_pose_clean)

                # In test phase, we all use lidar label for fair comparison. (need discussion)
                self.label_type = 'lidar' # DAIRV2X
                self.generate_object_center = self.generate_object_center_lidar # OPV2V, V2XSET

                selected_cav_processed = \
                    self.get_item_single_car(selected_cav_base)
                selected_cav_processed.update({'transformation_matrix': transformation_matrix,
                                            'transformation_matrix_clean': transformation_matrix_clean})
                update_cav = "ego" if cav_id == ego_id else cav_id
                processed_data_dict.update({update_cav: selected_cav_processed})
                cav_id_list_newname.append(update_cav)
            

            return processed_data_dict


        def get_item_single_car(self, selected_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            modality_name = selected_cav_base['modality_name']
            sensor_type = self.sensor_type_dict[modality_name]

            # label
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center_single(
                [selected_cav_base], selected_cav_base["params"]["lidar_pose_clean"]
            )

            # lidar
            if sensor_type == "lidar" or self.visualize:
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                lidar_np = mask_points_by_range(lidar_np,
                                                self.params['preprocess'][
                                                    'cav_lidar_range'])
                # remove points that hit ego vehicle
                lidar_np = mask_ego_points(lidar_np)

                # data augmentation, seems very important for single agent training, because lack of data diversity.
                # only work for lidar modality in training.
                if not self.visualize:
                    lidar_np, object_bbx_center, object_bbx_mask = \
                    self.augment(lidar_np, object_bbx_center, object_bbx_mask)
                if sensor_type == "lidar":
                    processed_lidar = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_np)
                    selected_cav_processed.update({f'processed_features_{modality_name}': processed_lidar})
                

            if self.visualize:
                selected_cav_processed.update({'origin_lidar': lidar_np})

            # camera
            if sensor_type == "camera":
                # adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py
                camera_data_list = selected_cav_base["camera_data"]

                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                extrinsics = [] # cam_to_lidar
                post_rots = []
                post_trans = []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        eval(f"self.data_aug_conf_{modality_name}"), self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)

                selected_cav_processed.update(
                    {
                    f"image_inputs_{modality_name}": 
                        {
                            "imgs": torch.stack(imgs), # [N, 3or4, H, W]
                            "intrins": torch.stack(intrins),
                            "extrinsics": torch.stack(extrinsics),
                            "rots": torch.stack(rots),
                            "trans": torch.stack(trans),
                            "post_rots": torch.stack(post_rots),
                            "post_trans": torch.stack(post_trans),
                        }
                    }
                )

            
            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center,
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    "modality_name": modality_name
                }
            )

            # generate targets label
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
            )
            selected_cav_processed.update({"label_dict": label_dict})

            return selected_cav_processed


        def collate_batch_train(self, batch):
            """
            Customized collate function for pytorch dataloader during training
            for early and late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # during training, we only care about ego.
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            label_dict_list = []
            origin_lidar = []
            inputs_list_m1 = [] 
            inputs_list_m2 = []
            inputs_list_m3 = []
            inputs_list_m4 = []
            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                label_dict_list.append(ego_dict['label_dict'])
                
                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask})

            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'anchor_box': torch.from_numpy(self.anchor_box),
                                    'label_dict': label_torch_dict})
            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})


            
                
            for modality_name in self.modality_name_list:
                sensor_type = self.sensor_type_dict[modality_name]
                for i in range(len(batch)):
                    ego_dict = batch[i]['ego']
                    if f'processed_features_{modality_name}' in ego_dict:
                        eval(f"inputs_list_{modality_name}").append(ego_dict[f'processed_features_{modality_name}']) 
                    elif f'image_inputs_{modality_name}' in ego_dict:
                        eval(f"inputs_list_{modality_name}").append(ego_dict[f'image_inputs_{modality_name}']) 

                if self.sensor_type_dict[modality_name] == "lidar":
                    processed_lidar_torch_dict = eval(f"self.pre_processor_{modality_name}").collate_batch(eval(f"inputs_list_{modality_name}"))
                    output_dict['ego'].update({f'inputs_{modality_name}': processed_lidar_torch_dict})
                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"), merge='stack')
                    output_dict['ego'].update({f'inputs_{modality_name}': merged_image_inputs_dict})

            return output_dict

        def collate_batch_test(self, batch):
            """
            Customized collate function for pytorch dataloader during testing
            for late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # currently, we only support batch size of 1 during testing
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            batch = batch[0]

            output_dict = {}

            # for late fusion, we also need to stack the lidar for better
            # visualization
            if self.visualize:
                projected_lidar_list = []
                origin_lidar = []

            for cav_id, cav_content in batch.items():
                modality_name = cav_content['modality_name']
                sensor_type = self.sensor_type_dict[modality_name]

                output_dict.update({cav_id: {}})
                # shape: (1, max_num, 7)
                object_bbx_center = \
                    torch.from_numpy(np.array([cav_content['object_bbx_center']]))
                object_bbx_mask = \
                    torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
                object_ids = cav_content['object_ids']

                # the anchor box is the same for all bounding boxes usually, thus
                # we don't need the batch dimension.
                output_dict[cav_id].update(
                    {"anchor_box": self.anchor_box_torch}
                )

                transformation_matrix = cav_content['transformation_matrix']
                if self.visualize:
                    origin_lidar = [cav_content['origin_lidar']]
                    if (self.params.get('only_vis_ego', True) is False) or (cav_id=='ego'):
                        projected_lidar = copy.deepcopy(cav_content['origin_lidar'])
                        projected_lidar[:, :3] = \
                            box_utils.project_points_by_matrix_torch(
                                projected_lidar[:, :3],
                                transformation_matrix)
                        projected_lidar_list.append(projected_lidar)

                if sensor_type == "lidar":
                    # processed lidar dictionary
                    processed_lidar_torch_dict = \
                        eval(f"self.pre_processor_{modality_name}").collate_batch([cav_content[f'processed_features_{modality_name}']])
                    output_dict[cav_id].update({f'inputs_{modality_name}': processed_lidar_torch_dict})

                if sensor_type == 'camera':
                    imgs_batch = [cav_content[f"image_inputs_{modality_name}"]["imgs"]]
                    rots_batch = [cav_content[f"image_inputs_{modality_name}"]["rots"]]
                    trans_batch = [cav_content[f"image_inputs_{modality_name}"]["trans"]]
                    intrins_batch = [cav_content[f"image_inputs_{modality_name}"]["intrins"]]
                    extrinsics_batch = [cav_content[f"image_inputs_{modality_name}"]["extrinsics"]]
                    post_trans_batch = [cav_content[f"image_inputs_{modality_name}"]["post_trans"]]
                    post_rots_batch = [cav_content[f"image_inputs_{modality_name}"]["post_rots"]]

                    output_dict[cav_id].update({
                        f"inputs_{modality_name}":
                            {
                                "imgs": torch.stack(imgs_batch),
                                "rots": torch.stack(rots_batch),
                                "trans": torch.stack(trans_batch),
                                "intrins": torch.stack(intrins_batch),
                                "extrinsics": torch.stack(extrinsics_batch),
                                "post_trans": torch.stack(post_trans_batch),
                                "post_rots": torch.stack(post_rots_batch),
                            }
                        }
                    )


                # label dictionary
                label_torch_dict = \
                    self.post_processor.collate_batch([cav_content['label_dict']])
                    
                # for centerpoint
                label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                         'object_bbx_mask': object_bbx_mask})

                # save the transformation matrix (4, 4) to ego vehicle
                transformation_matrix_torch = \
                    torch.from_numpy(
                        np.array(cav_content['transformation_matrix'])).float()
                
                transformation_matrix_clean_torch = \
                    torch.from_numpy(
                        np.array(cav_content['transformation_matrix_clean'])).float()

                output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                            'object_bbx_mask': object_bbx_mask,
                                            'label_dict': label_torch_dict,
                                            'object_ids': object_ids,
                                            'transformation_matrix': transformation_matrix_torch,
                                            'transformation_matrix_clean': transformation_matrix_clean_torch,
                                            'modality_name': modality_name})

                if self.visualize:
                    origin_lidar = \
                        np.array(
                            downsample_lidar_minimum(pcd_np_list=origin_lidar))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[cav_id].update({'origin_lidar': origin_lidar})

            if self.visualize:
                projected_lidar_stack = [torch.from_numpy(
                    np.vstack(projected_lidar_list))]
                output_dict['ego'].update({'origin_lidar': projected_lidar_stack})
                # output_dict['ego'].update({'projected_lidar_list': projected_lidar_list})

            return output_dict


        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = self.post_processor.post_process(
                data_dict, output_dict
            )
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor

        def post_process_no_fusion(self, data_dict, output_dict_ego):
            data_dict_ego = OrderedDict()
            data_dict_ego["ego"] = data_dict["ego"]
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            pred_box_tensor, pred_score = self.post_processor.post_process(
                data_dict_ego, output_dict_ego
            )
            return pred_box_tensor, pred_score, gt_box_tensor

        def post_process_no_fusion_uncertainty(self, data_dict, output_dict_ego):
            data_dict_ego = OrderedDict()
            data_dict_ego['ego'] = data_dict['ego']
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            pred_box_tensor, pred_score, uncertainty = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
            return pred_box_tensor, pred_score, gt_box_tensor, uncertainty

    return LateheterFusionDataset