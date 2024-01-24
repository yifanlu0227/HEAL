# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


"""
Gaussian Loss 
"""
class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negtive samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg

def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    device = pred.device
    pos_weights = gaussian_target.eq(1)
    pos_weights = pos_weights.to(device)
    neg_weights = (1 - gaussian_target).pow(gamma)
    neg_weights = neg_weights.to(device)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


def clip_sigmoid(x, eps=1e-4):
    """Sigmoid function for input feature.

    Args:
        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float): Lower bound of the range to be clamped to. Defaults
            to 1e-4.

    Returns:
        torch.Tensor: Feature map after sigmoid.
    """
    y = torch.clamp(torch.sigmoid(x), min=eps, max=1 - eps)
    # y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

def _gather_feat(feat, ind, mask=None):
    # feat : [bs, wxh, c]
    dim  = feat.size(2)  
    # ind : [bs, index, c]
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)  # 按照dim=1获取ind
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()   ## # from [bs c h w] to [bs, h, w, c] 
    feat = feat.view(feat.size(0), -1, feat.size(3))  # to [bs, wxh, c]
    feat = _gather_feat(feat, ind)
    return feat



class RegLoss(nn.Module):
    '''Regression loss for an output tensor
        Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''
    def __init__(self):
        super(RegLoss, self).__init__()
    
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float().unsqueeze(2) 

        loss = F.l1_loss(pred*mask, target*mask, reduction='none')
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
        return loss



class FastFocalLoss(nn.Module):
    '''
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    '''
    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, out, target, ind, mask, cat):
        '''
        Arguments:
        out, target: B x C x H x W
        ind, mask: B x M
        cat (category id for peaks): B x M
        '''
        mask = mask.float()
        gt = torch.pow(1 - target, 4)
        neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
        neg_loss = neg_loss.sum()

        pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
                mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return - neg_loss
        return - (pos_loss + neg_loss) / num_pos

class CenterPointLoss(nn.Module):
    def __init__(self, args):
        super(CenterPointLoss, self).__init__()

        self.cls_weight = args['cls_weight']
        self.loc_weight = args['loc_weight']
        self.code_weights = args['code_weights']
        self.target_cfg = args['target_assigner_config']
        self.lidar_range = self.target_cfg['cav_lidar_range']
        self.voxel_size = self.target_cfg['voxel_size']

        self.loss_cls = GaussianFocalLoss(reduction='mean')
        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        # Predictions 
        box_preds = output_dict['bbox_preds{}'.format(suffix)].permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        cls_preds = clip_sigmoid(output_dict['cls_preds{}'.format(suffix)])
        
        # GTs
        bbox_center = target_dict['object_bbx_center{}'.format(suffix)].cpu().numpy()
        bbox_mask = target_dict['object_bbx_mask{}'.format(suffix)].cpu().numpy()
        batch_size = bbox_mask.shape[0]

        max_gt = int(max(bbox_mask.sum(axis=1)))
        gt_boxes3d = np.zeros((batch_size, max_gt, bbox_center[0].shape[-1]), dtype=np.float32)  # [B, max_anchor_num, 7]
        for k in range(batch_size):
            gt_boxes3d[k, :int(bbox_mask[k].sum()), :] = bbox_center[k, :int(bbox_mask[k].sum()), :]
        gt_boxes3d = torch.from_numpy(gt_boxes3d).to(box_preds.device)

        targets_dict = self.assign_targets(
            gt_boxes=gt_boxes3d   #    [B, max_anchor_num, 7 + C ]      heatmap [2,1,h,w]  anno_boxes [2,100,8] inds [2, 100]
        )

        cls_gt =  targets_dict['heatmaps']
        box_gt = (targets_dict['anno_boxes'], targets_dict['inds'], targets_dict['masks'])

        cls_loss = self.get_cls_layer_loss(cls_preds, cls_gt)
        box_loss = self.get_box_reg_layer_loss(box_preds, box_gt)
        rpn_loss = cls_loss + box_loss

        self.loss_dict.update({ 'total_loss': rpn_loss.item(),
                                'reg_loss': box_loss.item(),
                                'cls_loss': cls_loss.item()})

        return rpn_loss
  
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
        
        print("[epoch %d][%d/%d]%s, || Loss: %.4f || Conf Loss: %.4f"
                    " || Loc Loss: %.4f" % (
                        epoch, batch_id + 1, batch_len, suffix,
                        total_loss, cls_loss, reg_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss', cls_loss,
                            epoch*batch_len + batch_id)
                        

    def get_cls_layer_loss(self, pred_heatmaps, gt_heatmaps):
        num_pos = gt_heatmaps.eq(1).float().sum().item()

        cls_loss = self.loss_cls(
            pred_heatmaps,
            gt_heatmaps,
            avg_factor=max(num_pos, 1))

        cls_loss = cls_loss * self.cls_weight
        return cls_loss


    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        device = feat.device  
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)   # 把 ind 和 dim 拼接在一起
        feat = feat.gather(1, ind.to(device))
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat


    def get_box_reg_layer_loss(self, bbox_preds, bbox_gt):
        target_box, inds, masks = bbox_gt
        pred = bbox_preds
        ind = inds
        num = masks.float().sum()
        pred = pred.view(pred.size(0), -1, pred.size(3))     # [n, h*w, 8 ]
        pred = self._gather_feat(pred, ind)
        mask = masks.unsqueeze(2).expand_as(target_box).float()  ## 把 mask 的维度进行扩展
        isnotnan = (~torch.isnan(target_box)).float()
        mask *= isnotnan

        code_weights = self.code_weights
        bbox_weights = mask * mask.new_tensor(code_weights)
        
        loc_loss = l1_loss(
            pred, target_box, bbox_weights, avg_factor=(num + 1e-4))

        loc_loss = loc_loss * self.loc_weight
        return loc_loss


    def assign_targets(self, gt_boxes):
        """Generate targets.

        Args:
            gt_boxes: ( M, 7+c) box + cls   ## 这个地方函数和centerpoint-kitti 那个不太一样，这里是分开进行计算的 

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        if gt_boxes.shape[-1] == 8:
            gt_bboxes_3d, gt_labels_3d = gt_boxes[..., :-1], gt_boxes[..., -1]    # gt_box [2,14,8] batch_size * bbox_num * 8
            heatmaps, anno_boxes, inds, masks = self.get_targets_single(gt_bboxes_3d, gt_labels_3d)
        elif gt_boxes.shape[-1] == 7:
            gt_bboxes_3d = gt_boxes
            heatmaps, anno_boxes, inds, masks = self.get_targets_single(gt_bboxes_3d)

        # transpose heatmaps, because the dimension of tensors in each task is
        # different, we have to use numpy instead of torch to do the transpose.
        # heatmaps = np.array(heatmaps).transpose(1, 0).tolist()
        # heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # # heatmaps = torch.from_numpy(np.array(heatmaps))
        # # transpose anno_boxes
        # anno_boxes = np.array(anno_boxes).transpose(1, 0).tolist()
        # anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # # transpose inds
        # inds = np.array(inds).transpose(1, 0).tolist()
        # inds = [torch.stack(inds_) for inds_ in inds]
        # # transpose inds
        # masks = np.array(masks).transpose(1, 0).tolist()
        # masks = [torch.stack(masks_) for masks_ in masks]

        all_targets_dict = {
            'heatmaps': heatmaps,
            'anno_boxes': anno_boxes,
            'inds': inds,
            'masks': masks
        }
        
        return all_targets_dict


    def get_targets_single(self, gt_bbox_3d, gt_labels_3d=None):
        
        batch_size = gt_bbox_3d.shape[0]
        device = gt_bbox_3d.device
        max_objs = self.target_cfg['max_objs']
        pc_range = self.lidar_range
        voxel_size = self.voxel_size

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        feature_map_size = grid_size[:2] // self.target_cfg['out_size_factor']

        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for batch in range(batch_size):
            task_boxes = gt_bbox_3d[batch, :, :]
            if not gt_labels_3d is None:
                task_classes = gt_labels_3d[batch, :]

            heatmap = gt_bbox_3d.new_zeros(    # 辅助gt_bboxes_3d的属性
                (1, feature_map_size[1],feature_map_size[0])) 

            anno_box = gt_bbox_3d.new_zeros((max_objs, 8), 
                                            dtype = torch.float32)
            
            ind = gt_bbox_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bbox_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes.shape[0], max_objs)
        
            for k in range(num_objs):
                # 计算x的heatmap坐标
                coor_x = (task_boxes[k][0] - pc_range[0]) / voxel_size[0] / self.target_cfg['out_size_factor']
                coor_y = (task_boxes[k][1] - pc_range[1]) / voxel_size[1] / self.target_cfg['out_size_factor']
                coor_z = (task_boxes[k][2] - pc_range[2]) / voxel_size[2] / self.target_cfg['out_size_factor']
                h = task_boxes[k][3] / voxel_size[0] / self.target_cfg['out_size_factor']
                w = task_boxes[k][4] / voxel_size[1] / self.target_cfg['out_size_factor']
                l = task_boxes[k][5] / voxel_size[2] / self.target_cfg['out_size_factor']
                rot = task_boxes[k][6]

                if h > 0 and w > 0:
                    radius = gaussian_radius(
                        (h, w),
                        min_overlap=self.target_cfg['gaussian_overlap'])
                    radius = max(self.target_cfg['min_radius'], int(radius))

                    center = torch.tensor([coor_x, coor_y],
                                        dtype=torch.float32,
                                        device=device)
                    center_int = center.to(torch.int32)   ## bbox 的中心在heatmap 中的位置

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0].item()
                            and 0 <= center_int[1] < feature_map_size[1].item()):
                        continue

                    draw_gaussian(heatmap[0], center_int, radius) 
                    
                    x, y = center_int[0], center_int[1]
                    assert (center_int[1] * feature_map_size[0] + center_int[0] <
                                        feature_map_size[0] * feature_map_size[1])
                    ind[k] = y * feature_map_size[0] + x
                    mask[k] = 1
                    # box_dim = task_boxes[k][3:6]
                    # box_dim = box_dim.log()
                    box_dim = torch.cat([h.unsqueeze(0), w.unsqueeze(0), l.unsqueeze(0)], dim=0)
                    anno_box[k] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        coor_z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                    ])   # [x,y,z, w, h, l, sin(heading), cos(heading)]

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            inds.append(ind)
            masks.append(mask)
            # import cv2; cv2.imwrite('test_{}.png'.format(batch), heatmap.cpu().numpy()[0]*255)
        heatmaps = torch.stack(heatmaps)
        anno_boxes = torch.stack(anno_boxes)
        inds = torch.stack(inds)
        masks = torch.stack(masks)
        return heatmaps, anno_boxes, inds, masks  # [B, H, W]


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


 
def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap



def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)



import functools

import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    
    if weight is not None:
        device = loss.device
        weight = weight.to(device)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@weighted_loss
def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    device = pred.device
    pos_weights = gaussian_target.eq(1)
    pos_weights = pos_weights.to(device)
    neg_weights = (1 - gaussian_target).pow(gamma)
    neg_weights = neg_weights.to(device)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss

@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    device = pred.device
    target = target.to(device)
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss