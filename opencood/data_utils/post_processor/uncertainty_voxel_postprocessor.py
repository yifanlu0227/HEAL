# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
from torch.nn.functional import sigmoid
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.data_utils.post_processor.voxel_postprocessor \
    import VoxelPostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils
from opencood.utils.common_utils import limit_period


class UncertaintyVoxelPostprocessor(VoxelPostprocessor):
    def __init__(self, anchor_params, train):
        super(UncertaintyVoxelPostprocessor, self).__init__(anchor_params, train)
    
    def post_process_stage1(self, stage1_output_dict, anchor_box):
        """
        This function is used to calculate the detections in advance 
        and save them(after return) for CoAlign box alignment.
        """
        cls_preds = stage1_output_dict['cls_preds']
        reg_preds = stage1_output_dict['reg_preds']
        unc_preds = stage1_output_dict['unc_preds']

        # the final bounding box list
        uncertainty_dim = unc_preds.shape[1] // cls_preds.shape[1]
        cls_preds = F.sigmoid(cls_preds.permute(0, 2, 3, 1).contiguous())  # [N, H, W, anchor_num]
        unc_preds = unc_preds.permute(0,2,3,1).contiguous() #[N, H, W, anchor_num * 2]

        # convert regression map back to bounding box
        batch_box3d = self.delta_to_boxes3d(reg_preds, anchor_box)  # (N, W*L*2, 7)
        mask = torch.gt(cls_preds, self.params['target_args']['score_threshold'])
        batch_num_box_count = [int(m.sum()) for m in mask]
        mask = mask.view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
        mask_sm = mask.unsqueeze(2).repeat(1, 1, uncertainty_dim)
        

        boxes3d = torch.masked_select(batch_box3d.view(-1, 7), mask_reg[0]).view(-1, 7) 
        uncertainty = torch.masked_select(unc_preds.view(-1,uncertainty_dim), mask_sm[0]).view(-1,uncertainty_dim) # [N*H*W*#anchor_num, 2] -> [num_select, 2]
        scores = torch.masked_select(cls_preds.view(-1), mask[0])
        if 'dir_preds' in stage1_output_dict and len(boxes3d) != 0:
            dir_preds = stage1_output_dict['dir_preds']
            dir_offset = self.params['dir_args']['dir_offset']
            num_bins = self.params['dir_args']['num_bins']

            dir_cls_preds = dir_preds.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
            dir_cls_preds = dir_cls_preds[mask]
            # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
            
            period = (2 * np.pi / num_bins) # pi
            dir_rot = limit_period(
                boxes3d[..., 6] - dir_offset, 0, period
            ) # 限制在0到pi之间
            boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(boxes3d.dtype) # 转化0.25pi到2.5pi
            boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

        # convert output to bounding box
        if len(boxes3d) != 0:
            # save origianl format box. [N, 7]
            pred_box3d_original = boxes3d.detach()
            # (N, 8, 3)
            boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order'])
            # (N, 8, 3)
            pred_corners_tensor = boxes3d_corner  # box_utils.project_box3d(boxes3d_corner, transformation_matrix)
            # convert 3d bbx to 2d, (N,4)
            projected_boxes2d = box_utils.corner_to_standup_box_torch(pred_corners_tensor)
            # (N, 5)
            pred_box2d_score_tensor = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)
            scores = pred_box2d_score_tensor[:, -1]

        else:
             return None, None, None

        # divide boxes to each cav

        cur_idx = 0
        batch_pred_corners3d = [] # [[N1, 8, 3], [N2, 8, 3], ...]
        batch_pred_boxes3d = [] # [[N1, 7], [N2, 7], ...]
        batch_uncertainty = [] # [[N1, 2], [N2, 2], ...]
        batch_scores = []
        for n in batch_num_box_count:
            cur_corners = pred_corners_tensor[cur_idx: cur_idx+n]
            cur_boxes = pred_box3d_original[cur_idx: cur_idx+n]
            cur_scores = scores[cur_idx:cur_idx+n]
            cur_uncertainty = uncertainty[cur_idx: cur_idx+n]
            # nms
            keep_index = box_utils.nms_rotated(cur_corners,
                                               cur_scores,
                                               self.params['nms_thresh']
                                               )
            batch_pred_corners3d.append(cur_corners[keep_index])
            batch_pred_boxes3d.append(cur_boxes[keep_index])
            batch_scores.append(cur_scores[keep_index])
            batch_uncertainty.append(cur_uncertainty[keep_index])
            cur_idx += n

        return batch_pred_corners3d, batch_pred_boxes3d, batch_uncertainty


    def post_process(self, data_dict, output_dict, return_uncertainty=False):
        """
        For fusion_method: no_w_uncertainty
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        uncertainty_list = []
        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                continue
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix'] # no clean

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            uncertainty_dim = output_dict[cav_id]['unc_preds'].shape[1] // output_dict[cav_id]['cls_preds'].shape[1]
            prob = output_dict[cav_id]['cls_preds']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]['reg_preds']

            # uncertainty map
            unc_preds = output_dict[cav_id]['unc_preds'].permute(0, 2, 3, 1).contiguous()
            unc_preds = unc_preds.view(unc_preds.shape[0], -1, uncertainty_dim) # [N, H*W*#anchor_num, 2]

            # convert regression map back to bounding box
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box) # (N, H*W*#anchor_num, 7)
            mask = \
                torch.gt(prob, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
            mask_sm = mask.unsqueeze(2).repeat(1, 1, uncertainty_dim)
            
            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])
            uncertainty = torch.masked_select(unc_preds[0], mask_sm[0]).view(-1, uncertainty_dim)


            # adding dir classifier
            if 'dir_preds' in output_dict[cav_id].keys() and len(boxes3d) != 0:
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']


                dir_preds  = output_dict[cav_id]['dir_preds'] # [N, H, W, 4]
                dir_cls_preds = dir_preds.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
                period = (2 * np.pi / num_bins) # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]


            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)
                uncertainty_list.append(uncertainty)


        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            if return_uncertainty:
                return None, None, None
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        uncertainty_list = torch.vstack(uncertainty_list)
        uncertainty = uncertainty_list
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        uncertainty = uncertainty[keep_index]

        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]
        uncertainty = uncertainty[keep_index]

        # filter out the prediction out of the range.
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor, self.params['gt_range'])
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]
        uncertainty = uncertainty[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]
        
        if return_uncertainty:
            return pred_box3d_tensor, scores, uncertainty

        return pred_box3d_tensor, scores

