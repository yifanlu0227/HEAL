# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.point_pillar_loss import sigmoid_focal_loss

class PointPillarPyramidLoss(PointPillarDepthLoss):
    def __init__(self, args):
        super().__init__(args)
        self.pyramid = args['pyramid']

        # relative downsampled GT cls map from fused labels.
        self.relative_downsample = self.pyramid['relative_downsample']
        self.pyramid_weight = self.pyramid['weight']
        self.num_levels = len(self.relative_downsample)
    
    def forward(self, output_dict, target_dict, suffix=""):
        if output_dict['pyramid'] == 'collab': # intermediate fusion, pyramid collab.
            return self.forward_collab(output_dict, target_dict, suffix)

        elif output_dict['pyramid'] == 'single': # late fusion, pyramid single 
            return self.forward_single(output_dict, target_dict, suffix)
        raise
        
    def forward_single(self, output_dict, target_dict, suffix):
        """
        for heter_pyramid_single
        """
        batch_size = target_dict['pos_equal_one'].shape[0]
        total_loss = super().forward(output_dict, target_dict, suffix)

        occ_single_list = output_dict['occ_single_list']    
        occ_loss = self.calc_occ_loss(occ_single_list, target_dict['pos_equal_one'], target_dict['neg_equal_one'], batch_size)
        total_loss += occ_loss
        self.loss_dict.update({
            'pyramid_loss': occ_loss.item(),
            'total_loss': total_loss.item()
        })
        return total_loss

    def forward_collab(self, output_dict, target_dict, suffix):
        """
        for heter_pyramid_collab
        """
        if suffix == "": 
            return super().forward(output_dict, target_dict)
        assert suffix == "_single"
        batch_size = target_dict['pos_equal_one'].shape[0]

        positives = target_dict['pos_equal_one']
        negatives = target_dict['neg_equal_one']

        occ_single_list = output_dict['occ_single_list']
        occ_loss = self.calc_occ_loss(occ_single_list, positives, negatives, batch_size)
        total_loss = occ_loss
        self.loss_dict = {
            'pyramid_loss': occ_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss


    def calc_occ_loss(self, occ_single_list, positives, negatives, batch_size):
        total_occ_loss = 0
        occ_positives = torch.logical_or(positives[...,0], positives[...,1]).unsqueeze(-1).float() # N, H, W
        occ_negatives = torch.logical_and(negatives[...,0], negatives[...,1]).unsqueeze(-1).float() # N, H, W

        for i, occ_preds_single in enumerate(occ_single_list):
            """
            occ_preds_single: N, 1, H, W

            occ_positives: N, H, W, 1
            occ_negatives: N, H, W, 1

            """

            positives_level = F.max_pool2d(occ_positives.permute(0,3,1,2), kernel_size=self.relative_downsample[i]).permute(0,2,3,1)
            negatives_level = 1 - F.max_pool2d((1 - occ_negatives).permute(0,3,1,2), kernel_size=self.relative_downsample[i]).permute(0,2,3,1)

            occ_labls = positives_level.view(batch_size, -1, 1)
            positives_level = occ_labls
            negatives_level = negatives_level.view(batch_size, -1, 1)

            pos_normalizer = positives_level.sum(1, keepdim=True).float()

            occ_preds = occ_preds_single.permute(0, 2, 3, 1).contiguous() \
                        .view(batch_size, -1,  1)
            occ_weights = positives_level * self.pos_cls_weight + negatives_level * 1.0
            occ_weights /= torch.clamp(pos_normalizer, min=1.0)
            occ_loss = sigmoid_focal_loss(occ_preds, occ_labls, weights=occ_weights, **self.cls)
            occ_loss = occ_loss.sum() / batch_size
            occ_loss *= self.pyramid_weight[i]

            total_occ_loss += occ_loss


        return total_occ_loss
    



    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)
        pyramid_loss = self.loss_dict.get('pyramid_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || Pyramid Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss, pyramid_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss' + suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Pyramid_loss' + suffix, pyramid_loss,
                epoch*batch_len + batch_id)

