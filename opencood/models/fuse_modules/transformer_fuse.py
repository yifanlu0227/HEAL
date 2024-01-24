"""
Implementation of transformer encoder fusion.
"""

import torch
import torch.nn as nn

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine_simple, get_rotated_roi
import torch.nn.functional as F
from icecream import ic
from matplotlib import pyplot as plt

# class MultiheadAttBlock(nn.Module):
#     def __init__(self, channels, n_head=8, dropout=0):
#         super(MultiheadAttBlock, self).__init__()
#         self.attn = nn.MultiheadAttention(channels, n_head, dropout)

#     def forward(self, q, k, v):
#         """
#         order (seq, batch, feature)
#         Args:
#             q: (1, H*W, C)
#             k: (N, H*W, C)
#             v: (N, H*W, C)
#         Returns:
#             outputs: ()
#         """
#         context, weight = self.attn(q,k,v) # (1, H*W, C)

#         return context


class EncodeLayer(nn.Module):
    def __init__(self, channels, n_head=8, dropout=0):
        super(EncodeLayer, self).__init__()
        self.attn = nn.MultiheadAttention(channels, n_head, dropout)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, q, k, v):
        """
        order (seq, batch, feature)
        Args:
            q: (1, H*W, C)
            k: (N, H*W, C)
            v: (N, H*W, C)
        Returns:
            outputs: ()
        """
        residual = q
        context, weight = self.attn(q,k,v) # (1, H*W, C)
        context = self.dropout1(context)
        output1 = self.norm1(residual + context)

        # feed forward net
        residual = output1 # (1, H*W, C)
        context = self.linear2(self.relu(self.linear1(output1)))
        context = self.dropout2(context)
        output2 = self.norm2(residual + context)

        return output2





class TransformerFusion(nn.Module):
    def __init__(self, args):
        super(TransformerFusion, self).__init__()
        
        self.channels = args['in_channels']
        self.n_head = args['n_head']
        self.dropout = args['dropout_rate']

        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

        self.encode_layer = EncodeLayer(self.channels, self.n_head, self.dropout)
    
    def add_pe_map(self, x, normalized=True):
        # scale = 2 * math.pi
        temperature = 10000
        num_pos_feats = x.shape[-3] // 2  # positional encoding dimension. C = 2d

        mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool, device=x.device)  #[H, W]
        not_mask = ~mask
        y_embed = not_mask.cumsum(0, dtype=torch.float32)  # [H, W]
        x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [H, W]

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)  # [0,1,2,...,d]
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # 10000^(2k/d), k is [0,0,1,1,...,d/2,d/2]

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # [C, H, W]

        if len(x.shape) == 4:
            x = x + pos[None,:,:,:]
        elif len(x.shape) == 5:
            x = x + pos[None,None,:,:,:]
        return x

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        split_x = self.regroup(x, record_len)

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2


        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L,1,H,W)).to(x)
                roi_mask[b,i] = warp_affine_simple(one_tensor, pairwise_t_matrix[b][i, :, :, :],(H, W))

        batch_node_features = split_x
        # iteratively update the features for num_iteration times

        out = []
        # iterate each batch
        for b in range(B):

            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            updated_node_features = []

            # update each node i
            i = 0 # ego
            # (N,1,H,W)
            mask = roi_mask[b, i, :N, ...]

            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))

            neighbor_feature_flat = neighbor_feature.view(N,C,H*W)  # (N, C, H*W)
            neighbor_feature_flat_pe = self.add_pe_map(neighbor_feature).view(N,C,H*W)  # (N, C, H*W)
            
            query = neighbor_feature_flat_pe[0:1,...].permute(0,2,1)  # (1, H*W, C)
            key = neighbor_feature_flat_pe.permute(0,2,1)  # (N, H*W, C)
            value = neighbor_feature_flat.permute(0,2,1)



            fusion_result = self.encode_layer(query, key, value)  # (1, H*W, C)
            fusion_result = fusion_result.permute(0,2,1).reshape(1, C, H, W)[0]

            out.append(fusion_result)
            
        out = torch.stack(out)
        
        return out








