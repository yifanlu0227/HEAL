"""
3D Anchor Generator for Voxel
"""
import numpy as np
import torch

from opencood.data_utils.post_processor.voxel_postprocessor \
    import VoxelPostprocessor
from opencood.utils import box_utils
from opencood.utils import common_utils
from opencood.utils.common_utils import limit_period
from icecream import ic

class FpvrcnnPostprocessor(VoxelPostprocessor):
    def __init__(self, anchor_params, train):
        super(FpvrcnnPostprocessor, self).__init__(anchor_params, train)
        # redetect box in stage2
        self.redet = True if 'redet' in anchor_params and anchor_params['redet'] else False
        print("Postprocessor Stage2 ReDetect: ", self.redet)

    def post_process(self, data_dict, output_dict, stage1=False):
        if stage1:
            return self.post_process_stage1(data_dict, output_dict)
        elif not self.redet: # stage2 refinement
            return self.post_process_stage2(data_dict)
        else: # stage2 redetect
            return self.post_process_stage2_redet(data_dict, output_dict)

    def post_process_stage1(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        No NMS


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_corners_list = []
        pred_box3d_list = []
        score_list = []

        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict

            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # prediction result
            preds_dict = output_dict[cav_id]['stage1_out']

            # preds
            prob = preds_dict['cls_preds']
            prob = torch.sigmoid(prob.permute(0, 2, 3, 1).contiguous())
            reg = preds_dict['reg_preds']  # .permute(0, 2, 3, 1).contiguous()
            dir = preds_dict['dir_preds'].permute(0, 2, 3, 1).contiguous().reshape(1, -1, 2)

            batch_box3d = self.delta_to_boxes3d(reg, anchor_box) # hwl
            mask = torch.gt(prob, self.params['target_args']['score_threshold'])
            batch_num_box_count = [int(m.sum()) for m in mask]
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            boxes3d = torch.masked_select(batch_box3d.view(-1, 7), mask_reg[0]).view(-1, 7) # hwl. right
            scores = torch.masked_select(prob.view(-1), mask[0])

            dir_labels = torch.max(dir, dim=-1)[1]
            dir_labels = dir_labels[mask]

            if scores.shape[0] != 0:
                if 'iou_preds' in preds_dict:
                    iou = torch.sigmoid(preds_dict['iou_preds'].permute(0, 2, 3, 1).contiguous()).reshape(1, -1)
                    iou = torch.clamp(iou, min=0.0, max=1.0)
                    iou = (iou + 1) * 0.5
                    scores = scores * torch.pow(iou.masked_select(mask), 4)

                # correct_direction
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']

                dir = preds_dict['dir_preds'].permute(0, 2, 3, 1).contiguous().reshape(1, -1, 2)
                dir_cls_preds = dir[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
                period = (2 * np.pi / num_bins) # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]


                # filter invalid boxes
                keep_idx = torch.logical_and((boxes3d[:, 3:6] > 1).all(dim=1), (boxes3d[:, 3:6] < 10).all(dim=1))
                idx_start = 0
                count = []
                for i, n in enumerate(batch_num_box_count):
                    count.append(int(keep_idx[idx_start:idx_start+n].sum()))
                batch_num_box_count = count
                boxes3d = boxes3d[keep_idx] # hwl
                scores = scores[keep_idx]

                # if the number of boxes is too huge, this would consume a lot of memory in the second stage
                # therefore, randomly select some boxes if the box number is too big at the beginning of the training

                # if len(boxes3d) > 300:
                #     keep_idx = torch.multinomial(scores, 300)
                #     idx_start = 0
                #     count = []
                #     for i, n in enumerate(batch_num_box_count):
                #         count.append(int(torch.logical_and(keep_idx>=idx_start, keep_idx<idx_start + n).sum()))
                #     batch_num_box_count = count
                #     boxes3d = boxes3d[keep_idx] 
                #     scores = scores[keep_idx]

                pred_corners_list.append(box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order']))
                pred_box3d_list.append(boxes3d)
                score_list.append(scores)


        if len(pred_box3d_list) == 0:
            return None, None

        # predicted 3d bbx
        pred_corners_tensor = torch.vstack(pred_corners_list)
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        scores = torch.hstack(score_list)

        cur_idx = 0
        batch_pred_boxes3d = []
        batch_scores = []
        for n in batch_num_box_count:
            cur_corners = pred_corners_tensor[cur_idx:cur_idx+n]
            cur_scores = scores[cur_idx:cur_idx+n]
            # nms
            keep_index = box_utils.nms_rotated(cur_corners,
                                               cur_scores,
                                               self.params['nms_thresh']
                                               )
            cur_boxes = pred_box3d_tensor[cur_idx:cur_idx+n] # keep hwl, no need to transform
            batch_pred_boxes3d.append(cur_boxes[keep_index])
            batch_scores.append(cur_scores[keep_index])
            cur_idx += n

        return batch_pred_boxes3d, batch_scores

    def post_process_stage2(self, data_dict):
        from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import nms_gpu
        if 'stage2_out' not in data_dict['ego'].keys():
            return None, None
        output_dict = data_dict['ego']['stage2_out']
        label_dict = data_dict['ego']['rcnn_label_dict']
        rcnn_cls = output_dict['rcnn_cls'].sigmoid().view(-1)
        
        # use stage2 score
        if 'rcnn_iou' in output_dict:
            rcnn_iou = output_dict['rcnn_iou'].view(-1)
            rcnn_iou = rcnn_iou / 2 + 0.5 # renormalize
            rcnn_score = rcnn_cls * rcnn_iou**4
        else:
            rcnn_score = rcnn_cls
        
        # use stage1 score 
        # rcnn_score = label_dict['rois_scores_stage1']


        rcnn_reg = output_dict['rcnn_reg'].view(-1, 7)
        rois_anchor = label_dict['rois_anchor'] # lwh order
        rois = label_dict['rois'] # lwh order 
        roi_center = rois[:, 0:3]
        roi_ry = rois[:, 6] % (2 * np.pi)
        boxes_local = box_utils.box_decode(rcnn_reg, rois_anchor)

        # boxes_local = rcnn_reg + rois_anchor
        detections = common_utils.rotate_points_along_z(
            points=boxes_local.view(-1, 1, boxes_local.shape[-1]), angle=roi_ry.view(-1)
        ).view(-1, boxes_local.shape[-1])
        detections[:, :3] = detections[:, :3] + roi_center
        detections[:, 6] = detections[:, 6] + roi_ry
        mask = rcnn_score >= 0

        detections = detections[mask]
        scores = rcnn_score[mask]
        # gt_boxes = label_dict['gt_of_rois_src'][mask]
        mask = nms_gpu(detections, scores, thresh=0.01)[0]
        boxes3d = detections[mask] # keep hwl

        projected_boxes3d = None
        if len(boxes3d) != 0:
            # (N, 8, 3)
            boxes3d_corner = \
                box_utils.boxes_to_corners_3d(boxes3d,
                                              order="lwh") # in stage 2, box encoding is dxdydz order
            # (N, 8, 3)
            projected_boxes3d = \
                box_utils.project_box3d(boxes3d_corner,
                                        data_dict['ego']['transformation_matrix'])

        ## Added by Yifan Lu, filter box outside of GT range
        if projected_boxes3d is None:
            return None, None
        scores = scores[mask]
        cav_range = self.params['gt_range']
        mask = box_utils.get_mask_for_boxes_within_range_torch(projected_boxes3d, cav_range)
        projected_boxes3d = projected_boxes3d[mask]
        scores = scores[mask]


        return projected_boxes3d, scores

    # def post_process_stage2(self, data_dict):
    #     """
    #     it's a pseduo stage2 process, but only output the stage1 rpn result.
    #     """
    #     from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import nms_gpu
    #     if 'stage2_out' not in data_dict['ego'].keys():
    #         return None, None
    #     output_dict = data_dict['ego']['stage2_out']
    #     label_dict = data_dict['ego']['rcnn_label_dict']
    #     rcnn_score = label_dict['rois_scores_stage1']
    #     rois = label_dict['rois'][:,[0,1,2,5,4,3,6]]

    #     boxes3d_corner = \
    #         box_utils.boxes_to_corners_3d(rois,
    #                                         order=self.params['order'])
    #     mask = box_utils.get_mask_for_boxes_within_range_torch(boxes3d_corner, self.params['gt_range'])
    #     boxes3d_corner = boxes3d_corner[mask]
    #     rcnn_score = rcnn_score[mask]

    #     return boxes3d_corner, rcnn_score.flatten()


    def post_process_stage2_redet(self, data_dict, output_dict):
        return super().post_process(data_dict, output_dict)