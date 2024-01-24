# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.point_pillar_loss import PointPillarLoss

class PointPillarDepthLoss(PointPillarLoss):
    def __init__(self, args):
        super().__init__(args)
        self.depth = args['depth']


        self.depth_weight = self.depth['weight']
        self.smooth_target = True if 'smooth_target' in self.depth and self.depth['smooth_target'] else False
        self.use_fg_mask = True if 'use_fg_mask' in self.depth and self.depth['use_fg_mask'] else False
        self.fg_weight = 3.25
        self.bg_weight = 0.25
        if self.smooth_target:
            self.depth_loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="none", smooth_target=True)
        else:
            self.depth_loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")


    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """

        total_loss = super().forward(output_dict, target_dict, suffix)
        all_depth_loss = 0
        depth_items_list = [x for x in output_dict.keys() if x.startswith(f"depth_items{suffix}")]
        ######## Depth Supervision ########
        for depth_item_name in depth_items_list:
            depth_item = output_dict[depth_item_name]

            # depth logdit: [N, D, H, W]
            # depth gt indices: [N, H, W]
            # fg_mask: [N, H, W]
            depth_logit, depth_gt_indices = depth_item[0], depth_item[1]
            depth_loss = self.depth_loss_func(depth_logit, depth_gt_indices) 
            if self.use_fg_mask:
                fg_mask = depth_item[-1]
                weight_mask = (fg_mask > 0) * self.fg_weight + (fg_mask == 0) * self.bg_weight
                depth_loss *= weight_mask

            depth_loss = depth_loss.mean() * self.depth_weight 
            all_depth_loss += depth_loss

        total_loss += all_depth_loss
        self.loss_dict.update({'depth_loss': all_depth_loss}) # no update the total loss in dict
        
        return total_loss


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


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss))

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


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha, gamma = 2.0, reduction= 'none', smooth_target = False , eps = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth_target = smooth_target
        self.eps = eps
        if self.smooth_target:
            self.smooth_kernel = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
            self.smooth_kernel.weight = torch.nn.Parameter(torch.tensor([[[0.2, 0.9, 0.2]]]), requires_grad=False)
            self.smooth_kernel = self.smooth_kernel.to(torch.device("cuda"))

    def forward(self, input, target):
        n = input.shape[0]
        out_size = (n,) + input.shape[2:]

        # compute softmax over the classes axis
        input_soft = input.softmax(1)
        log_input_soft = input.log_softmax(1)

        # create the labels one hot tensor
        D = input.shape[1]
        if self.smooth_target:
            target_one_hot = F.one_hot(target, num_classes=D).to(input).view(-1, D) # [N*H*W, D]
            target_one_hot = self.smooth_kernel(target_one_hot.float().unsqueeze(1)).squeeze(1) # [N*H*W, D]
            target_one_hot = target_one_hot.view(*target.shape, D).permute(0, 3, 1, 2)
        else:
            target_one_hot = F.one_hot(target, num_classes=D).to(input).permute(0, 3, 1, 2)
        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, self.gamma)

        focal = -self.alpha * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss