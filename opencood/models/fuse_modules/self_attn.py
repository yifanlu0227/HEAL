# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

DEBUG=False

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len, pairwise_t_matrix):
        """
        pairwise_t_matrix : [N,N,2,3]
        """
        split_x = self.regroup(x, record_len)
        batch_size = len(record_len)
        C, H, W = split_x[0].shape[1:]  # C, W, H before
        out = []
        for b, xx in enumerate(split_x):
          N = xx.shape[0]
          t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
          i = 0 # ego
          xx = warp_affine_simple(xx, t_matrix[i, :, :, :], (H, W))

          cav_num = xx.shape[0]
          xx = xx.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
          h = self.att(xx, xx, xx)
          h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...].unsqueeze(0)  # C, W, H before
          out.append(h)
        return torch.cat(out, dim=0)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x



    def forward_debug(self, x, origin_x, record_len, pairwise_t_matrix):
        split_x = self.regroup(x, record_len)
        split_origin_x = self.regroup(origin_x, record_len)
        batch_size = len(record_len)
        C, H, W = split_x[0].shape[1:]  # C, W, H before
        H_origin, W_origin = split_origin_x[0].shape[2:]
        out = []
        from matplotlib import pyplot as plt
        for b, xx in enumerate(split_x):
          N = xx.shape[0]
          t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
          i = 0
          xx = warp_affine_simple(xx, t_matrix[i, :, :, :], (H, W))
          origin_xx = warp_affine_simple(split_origin_x[b], t_matrix[i, :, :, :], (H_origin, W_origin))

          for idx in range(N):
            plt.imshow(torch.max(xx[idx],0)[0].detach().cpu().numpy())
            plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/feature_{b}_{idx}")
            plt.clf()
            plt.imshow(torch.max(origin_xx[idx],0)[0].detach().cpu().numpy())
            plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/origin_feature_{b}_{idx}")
            plt.clf()
        raise