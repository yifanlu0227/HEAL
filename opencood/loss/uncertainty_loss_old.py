# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import d3d.mathh as mathh
from opencood.utils.common_utils import limit_period
from functools import partial

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss




class KLLoss(nn.Module):
    def __init__(self, args):
        super(KLLoss, self).__init__()

        self.angle_weight = args['angle_weight']
        self.uncertainty_dim = args['uncertainty_dim']
        if args['xy_loss_type'] == "l2":
            self.xy_loss = self.kl_loss_l2
        elif args['xy_loss_type'] == "l1":
            self.xy_loss = self.kl_loss_l1
        else:
            raise "not implemented"

        if args['angle_loss_type'] == "l2":
            self.angle_loss = self.kl_loss_l2
        elif args['angle_loss_type'] == "von":
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
        
        

        if self.uncertainty_dim == 3:
            xy_diff = input[...,:2] - target[...,:2]
            loss1 = self.xy_loss(xy_diff, sm[...,:2])
            
            theta_diff = input[...,7:8] - target[...,7:8]

            loss2 = self.angle_weight * self.angle_loss(theta_diff, sm[...,2:3])

            loss = torch.cat((loss1, loss2), dim=-1)
            
        elif self.uncertainty_dim == 7:
            ## is this right?
            other_diff = input[...,:6] - target[...,:6]
            theta_diff = input[...,7:8] - target[...,7:8]

            diff = torch.cat((other_diff, theta_diff), dim=-1)
            loss = self.xy_loss(diff, sm)

        elif self.uncertainty_dim == 2:
            xy_diff = input[...,:2] - target[...,:2]
            loss = self.xy_loss(xy_diff, sm[...,:2])
        else:
            raise "not implemented"

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)
        
        return loss



class PointPillarUncertaintyLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarUncertaintyLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.kl_weight = args['kl_weight']
        self.reg_coe = args['reg']
        self.uncertainty_dim = args['kl_args']['uncertainty_dim']

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_weight = args['dir_args']['dir_weight']
            self.dir_offset = args['dir_args']['args']['dir_offset']
            self.num_bins = args['dir_args']['args']['num_bins']
            anchor_yaw = np.deg2rad(np.array(args['dir_args']['anchor_yaw']))  # for direction classification
            self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1,-1,1)  # [1,2,1]
            self.anchor_num = self.anchor_yaw_map.shape[1]

        else:
            self.use_dir =False


        self.kl_loss_func = KLLoss(args['kl_args'])

        self.loss_dict = {}

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        rm = output_dict['rm']  # [B, 14, 50, 176]
        psm = output_dict['psm'] # [B, 2, 50, 176]
        sm = output_dict['sm']  # log of sigma^2 / scale [B, 6, 50 176]
        targets = target_dict['targets']

        cls_preds = psm.permute(0, 2, 3, 1).contiguous() # N, C, H, W -> N, H, W, C

        box_cls_labels = target_dict['pos_equal_one']  # [B, 50, 176, 2] 
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous() # -> [B, 50*176*2], two types of anchor

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float() # all 1
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float() # positive number per sample
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        ########## regression ##########
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)

        box_preds_sin, reg_targets_sin = self.add_sin_difference_dim(rm,
                                                                 targets)
        loc_loss_src =\
            self.reg_loss_func(box_preds_sin[...,:7],
                               reg_targets_sin[...,:7],
                               weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe


        ######## direction ##########
        if self.use_dir:
            dir_targets = self.get_direction_target(targets)
            N =  output_dict["dm"].shape[0]
            dir_logits = output_dict["dm"].permute(0, 2, 3, 1).contiguous().view(N, -1, 2) # [N, H*W*#anchor, 2]


            dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.anchor_num), dir_targets.view(-1, self.anchor_num)) 

            dir_loss = dir_loss.view(dir_logits.shape[:2]) * reg_weights # [N, H*W*anchor_num]

            dir_loss = dir_loss.sum() * self.dir_weight / N

        ######## kl #########
        sm = sm.permute(0, 2, 3, 1).contiguous() # [N, H, W, #anchor_num * 3]
        sm = sm.view(sm.size(0), -1, self.uncertainty_dim)

        kl_loss_src = \
            self.kl_loss_func(box_preds_sin,
                              reg_targets_sin,
                              sm,
                              reg_weights)

        kl_loss = kl_loss_src.sum() / sm.shape[0]
        kl_loss *= self.kl_weight

        # total_loss = reg_loss + conf_loss + kl_loss
        total_loss = reg_loss + conf_loss

        self.loss_dict.update({'total_loss': total_loss,
                               'reg_loss': reg_loss,
                               'conf_loss': conf_loss,
                               'kl_loss': kl_loss})
        
        if self.use_dir:
            # total_loss += dir_loss
            self.loss_dict.update({'dir_loss': dir_loss})


        return total_loss

    def get_direction_target(self, reg_targets):
        """
        Args:
            reg_targets:  [N, H * W * #anchor_num, 7]
                The last term is (theta_gt - theta_a)
        
        Returns:
            dir_targets:
                theta_gt: [N, H * W * #anchor_num, NUM_BIN] 
                NUM_BIN = 2
        """
        # (1, 2, 1)
        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num//self.anchor_num, 1).to(reg_targets.device) # [1, H * W * #anchor_num, 1]
        rot_gt = reg_targets[..., -1] + anchor_map[..., -1] # [N, H*W*anchornum]
        offset_rot = limit_period(rot_gt - self.dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / self.num_bins)).long()  # [N, H*W*anchornum]
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=self.num_bins - 1)
        # one_hot:
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_cls_targets = one_hot_f(dir_cls_targets, self.num_bins)
        return dir_cls_targets



    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference_dim(boxes1, boxes2, dim=6):
        """
        This is different with other loss function.
        Here we especially retain the angel

            Add sin difference ?
            Replace sin difference !

        Returns:
            [B, H*W, 7] -> [B, H*W, 8]
        """
        assert dim != -1

        # sin(theta1 - theta2) = sin(theta1)*cos(theta2) - cos(theta1)*sin(theta2) 

        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])

        rad_tg_encoding = torch.cos(boxes1[..., dim: dim + 1]) * \
                          torch.sin(boxes2[..., dim: dim + 1])

        # boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
        #                     boxes1[..., dim + 1:]], dim=-1)
        # boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
        #                     boxes2[..., dim + 1:]], dim=-1)
        
        boxes1_encoded = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim:]], dim=-1)
        boxes2_encoded = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim:]], dim=-1)

        return boxes1_encoded, boxes2_encoded


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
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']
        kl_loss = self.loss_dict['kl_loss']
        

        print_msg = ("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                    " || Loc Loss: %.4f || KL Loss: %.4f" % (
                        epoch, batch_id + 1, batch_len,
                        total_loss.item(), conf_loss.item(), reg_loss.item(),  kl_loss.item()))
        
        if self.use_dir:
            dir_loss = self.loss_dict['dir_loss']
            print_msg += " || Dir Loss: %.4f" % dir_loss.item()

        print(print_msg)

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss', conf_loss.item(),
                            epoch*batch_len + batch_id)
            writer.add_scalar('kl_loss', kl_loss.item(),
                            epoch*batch_len + batch_id)
            if self.use_dir:
                writer.add_scalar('dir_loss', dir_loss.item(),
                            epoch*batch_len + batch_id)

def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), depth, dtype=dtype, device=tensor.device) # [4, 70400, 2]
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)                        # [4, 70400, 2]
    return tensor_onehot

def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss
