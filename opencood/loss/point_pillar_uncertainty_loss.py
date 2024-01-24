# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.loss.point_pillar_loss import PointPillarLoss, \
    one_hot_f, softmax_cross_entropy_with_logits, weighted_smooth_l1_loss, sigmoid_focal_loss
import d3d.mathh as mathh
from opencood.utils.common_utils import limit_period
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from functools import partial

class PointPillarUncertaintyLoss(PointPillarLoss):
    def __init__(self, args):
        super(PointPillarUncertaintyLoss, self).__init__(args)
        self.uncertainty = args['uncertainty']
        self.uncertainty_dim = args['uncertainty']['dim'] # 2 means x, y; 3 means x, y, yaw; 7 means x y z dh dw dl yaw
        self.unc_loss_func = KLLoss(args['uncertainty'])


    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        elif 'batch_size' in output_dict:
            batch_size = output_dict['batch_size']
        else:
            batch_size = target_dict['pos_equal_one'].shape[0]

        cls_labls = target_dict['pos_equal_one'].view(batch_size, -1,  1)
        positives = cls_labls > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1,  1) > 0

        pos_normalizer = positives.sum(1, keepdim=True).float()

        # rename variable
        if f'psm{suffix}' in output_dict:
            output_dict[f'cls_preds{suffix}'] = output_dict[f'psm{suffix}']
        if f'rm{suffix}' in output_dict:
            output_dict[f'reg_preds{suffix}'] = output_dict[f'rm{suffix}']
        if f'dm{suffix}' in output_dict:
            output_dict[f'dir_preds{suffix}'] = output_dict[f'dm{suffix}']
        if f'sm{suffix}' in output_dict:
            output_dict[f'unc_preds{suffix}'] = output_dict[f'sm{suffix}']

        total_loss = 0

        # cls loss
        cls_preds = output_dict[f'cls_preds{suffix}'].permute(0, 2, 3, 1).contiguous() \
                    .view(batch_size, -1,  1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
        cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

        # reg loss
        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        reg_targets = target_dict['targets'].view(batch_size, -1, 7)
        reg_preds_w_angle, reg_targets_w_angle = self.add_sin_difference_and_angle(reg_preds, reg_targets) # note the difference 
        reg_loss = weighted_smooth_l1_loss(reg_preds_w_angle[...,:7], reg_targets_w_angle[...,:7], weights=reg_weights, sigma=self.reg['sigma'])
        reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size

        # uncertainty loss
        ######## kl #########
        unc_preds = output_dict[f'unc_preds{suffix}'].permute(0, 2, 3, 1).contiguous() # [N, H, W, #anchor_num * 3]
        unc_preds = unc_preds.view(unc_preds.size(0), -1, self.uncertainty_dim)

        unc_loss = self.unc_loss_func(reg_preds_w_angle,
                                     reg_targets_w_angle,
                                     unc_preds,
                                     reg_weights)

        unc_loss = unc_loss.sum() / unc_preds.shape[0]
        unc_loss *= self.uncertainty['weight']


        ######## direction ##########
        if self.dir:
            dir_targets = self.get_direction_target(target_dict['targets'].view(batch_size, -1, 7))
            dir_logits = output_dict[f"dir_preds{suffix}"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # [N, H*W*#anchor, 2]

            dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.anchor_num), dir_targets.view(-1, self.anchor_num)) 
            dir_loss = dir_loss.flatten() * reg_weights.flatten() 
            dir_loss = dir_loss.sum() * self.dir['weight'] / batch_size
            total_loss += dir_loss
            self.loss_dict.update({'dir_loss': dir_loss.item()})


        ######## IoU ###########
        if self.iou:
            iou_preds = output_dict["iou_preds{suffix}"].permute(0, 2, 3, 1).contiguous()
            pos_pred_mask = reg_weights.squeeze(dim=-1) > 0 # (4, 70400)
            iou_pos_preds = iou_preds.view(batch_size, -1)[pos_pred_mask]
            boxes3d_pred = VoxelPostprocessor.delta_to_boxes3d(output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().detach(),
                                                            output_dict['anchor_box'])[pos_pred_mask]
            boxes3d_tgt = VoxelPostprocessor.delta_to_boxes3d(target_dict['targets'],
                                                            output_dict['anchor_box'])[pos_pred_mask]
            iou_weights = reg_weights[pos_pred_mask].view(-1)
            iou_pos_targets = self.iou_loss_func(boxes3d_pred.float()[:, [0, 1, 2, 5, 4, 3, 6]], # hwl -> dx dy dz
                                                    boxes3d_tgt.float()[:, [0, 1, 2, 5, 4, 3, 6]]).detach().squeeze()
            iou_pos_targets = 2 * iou_pos_targets.view(-1) - 1
            iou_loss = weighted_smooth_l1_loss(iou_pos_preds, iou_pos_targets, weights=iou_weights, sigma=self.iou['sigma'])

            iou_loss = iou_loss.sum() * self.iou['weight'] / batch_size
            total_loss += iou_loss
            self.loss_dict.update({'iou_loss': iou_loss.item()})

        total_loss += reg_loss + cls_loss + unc_loss

        self.loss_dict.update({'total_loss': total_loss.item(),
                               'reg_loss': reg_loss.item(),
                               'cls_loss': cls_loss.item(),
                               'unc_loss': unc_loss.item()})

        return total_loss


    def logging(self, epoch, batch_id, batch_len, writer = None):
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
        unc_loss = self.loss_dict.get('unc_loss', 0)


        print("[epoch %d][%d/%d] || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Unc Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, unc_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss', cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss', dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss', iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Unc_loss', unc_loss,
                            epoch*batch_len + batch_id)

    @staticmethod
    def add_sin_difference_and_angle(boxes1, boxes2, dim=6):
        """
        This is different with base PointPillarLoss's add_sin_difference function.
        We retain the angle, and put it at last dimension

            add_sin_difference returns [B, H*W, 7]
            -> 
            add_sin_difference_and_angle returns [B, H*W, 8]

        """
        assert dim != -1

        # sin(theta1 - theta2) = sin(theta1)*cos(theta2) - cos(theta1)*sin(theta2) 
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])

        rad_tg_encoding = torch.cos(boxes1[..., dim: dim + 1]) * \
                          torch.sin(boxes2[..., dim: dim + 1])
        
        boxes1_w_angle = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim:]], dim=-1) # originally, boxes1[..., dim + 1:]], dim=-1)
        boxes2_w_angle = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim:]], dim=-1) # originally, boxes1[..., dim + 1:]], dim=-1)

        return boxes1_w_angle, boxes2_w_angle


class KLLoss(nn.Module):
    def __init__(self, args):
        super(KLLoss, self).__init__()

        self.angle_weight = args['angle_weight']
        self.uncertainty_dim = args['dim']
        if args['xy_loss_type'] == "l2":
            self.xy_loss = self.kl_loss_l2
        elif args['xy_loss_type'] == "l1":
            self.xy_loss = self.kl_loss_l1
        else:
            raise "not implemented"

        if args['angle_loss_type'] == "l2":
            self.angle_loss = self.kl_loss_l2
        elif args['angle_loss_type'] == "von-mise":
            lambda_V = args['lambda_V']
            s0 = args['s0']
            limit_period = args['limit_period']
            self.angle_loss = partial(self.kl_loss_angular, lambda_V=lambda_V, s0=s0, limit_period=limit_period)
        else:
            raise "not implemented"

    @staticmethod
    def kl_loss_l2(diff, s):
        """
        Args:
            diff: [B, 2]
            s:    [B, 2]
        Returns:
            loss: [B, 2]
        """
        loss = 0.5*(torch.exp(-s) * (diff**2) + s)
        return loss
    
    @staticmethod
    def kl_loss_l1(diff, s):
        """
        Args:
            diff: [B, 2]
            s:    [B, 2]
        Returns:
            loss: [B, 2]
        """
        loss = 0.5*torch.exp(-s) * torch.abs(diff) + s
        return loss
    
    @staticmethod
    def kl_loss_angular(diff, s, lambda_V=1, s0=1, limit_period=False):
        """
        Args:
            diff: [B, 1]
            s:    [B, 1]
            if limit_period, 
                diff + 180 ~ diff. 
        Returns:
            loss: [B, 1]
        """
        exp_minus_s = torch.exp(-s)
        if limit_period:
            cos_abs = torch.abs(torch.cos(diff))
            loss = loss = torch.log(mathh.i0e_cuda(exp_minus_s)*torch.exp(exp_minus_s)) - exp_minus_s * cos_abs.detach() + lambda_V * F.elu(s-s0)
        else:
            loss = torch.log(mathh.i0e_cuda(exp_minus_s)*torch.exp(exp_minus_s)) - exp_minus_s * torch.cos(diff) + lambda_V * F.elu(s-s0)

        return loss

    def forward(self, input: torch.Tensor,
                      target: torch.Tensor, 
                      sm: torch.Tensor, 
                      weights: torch.Tensor = None):
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets
        
        if self.uncertainty_dim == 3: # x,y,yaw
            xy_diff = input[...,:2] - target[...,:2]
            loss1 = self.xy_loss(xy_diff, sm[...,:2])
            theta_diff = input[...,7:8] - target[...,7:8]
            loss2 = self.angle_weight * self.angle_loss(theta_diff, sm[...,2:3])
            loss = torch.cat((loss1, loss2), dim=-1)
            
        elif self.uncertainty_dim == 7: # all regression target
            other_diff = input[...,:6] - target[...,:6]
            theta_diff = input[...,7:8] - target[...,7:8]
            diff = torch.cat((other_diff, theta_diff), dim=-1)
            loss = self.xy_loss(diff, sm)

        elif self.uncertainty_dim == 2: # x,y
            xy_diff = input[...,:2] - target[...,:2]
            loss = self.xy_loss(xy_diff, sm[...,:2])
        else:
            raise "not implemented"

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]

            loss = loss * weights
        
        return loss