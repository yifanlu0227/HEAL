# -*- coding: utf-8 -*-
# Author: Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Implementation of When2com Fusion
"""

import torch
import torch.nn as nn
import numpy as np

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class When2comFusion(nn.Module):
    def __init__(self, args):
        super(When2comFusion, self).__init__()

        self.discrete_ratio = args['voxel_size'][0]  
        self.downsample_rate = args['downsample_rate']  

        self.in_channels = args['in_channels']
        self.feat_H = args['H']
        self.feat_W = args['W']
        self.query_size = args['query_size']
        self.key_size = args['key_size']
        self.mode = args['mode']
        self.agent_num = 2

        self.query_key_net = policy_net4(self.in_channels)
        self.key_net = km_generator(out_size=self.key_size, input_feat_h=self.feat_H//4, input_feat_w=self.feat_W//4)
        self.query_net = km_generator(out_size=self.query_size, input_feat_h=self.feat_H//4, input_feat_w=self.feat_W//4)
        self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size)

    def activated_select(self, val_mat, prob_action, thres=0.2):
        coef_act = torch.mul(prob_action, (prob_action > thres).float())
        attn_shape = coef_act.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_act_exp = coef_act.view(bats, key_num, query_num, 1, 1, 1)

        output = coef_act_exp * val_mat  # (batch,4,channel,size,size)
        feat_act = output.sum(1)  # (batch,1,channel,size,size)

        # compute connect
        count_coef = coef_act.clone()
        ind = np.diag_indices(self.agent_num)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (
            self.agent_num * count_coef.shape[0]
        )
        return feat_act, coef_act, num_connect

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, pairwise_t_matrix, weight=None):
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
        
        weight: torch.Tensor
            Weight of aggregating coming message
            shape: (B, L, L)
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        split_x = self.regroup(x, record_len)
        batch_node_features = split_x
        updated_node_features = []
        for b in range(B):

            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            # (N,1,H,W)
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            query_key_maps = self.query_key_net(neighbor_feature)
            keys = self.key_net(query_key_maps)
            query = self.query_net(query_key_maps[0].unsqueeze(0))

            query = query.unsqueeze(0)
            keys = keys.unsqueeze(0)
            neighbor_feature = neighbor_feature.unsqueeze(1).unsqueeze(0)

            feat_fuse, prob_action = self.attention_net(query, keys, neighbor_feature, sparse=True)

            if self.mode == "activated":
                feat_fuse, connect_mat, num_connect = self.activated_select(neighbor_feature, prob_action)

            updated_node_features.append(feat_fuse.squeeze(0))

        out = torch.cat(updated_node_features, dim=0)
        
        return out

class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))
        
        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.range(start=1, end=number_of_logits, device=input.device).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

class km_generator(nn.Module):
    def __init__(self, out_size=128, input_feat_h=25, input_feat_w=63):
        super(km_generator, self).__init__()
        # self.n_feat = int(256 * (input_feat_h//4 + 1) * (input_feat_w//4 + 1))
        self.n_feat = int(256 * input_feat_h * input_feat_w)
        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256), #            
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), #             
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)) #            

    def forward(self, feat_map):
        outputs = self.fc(feat_map.view(-1, self.n_feat))
        return outputs

class km_generator_v2(nn.Module):
    def __init__(self, out_size=128):
        super(km_generator_v2, self).__init__()
        # N, C = 256, H, W
        self.conv1 = conv2DBatchNormRelu(256, 128, k_size=3, stride=2, padding=1)
        self.avgp = nn.AdaptiveAvgPool2d((5, 7))
        self.n_feat = int(128*5*7)
        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256), #            
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), #             
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)) #    

    def forward(self, feat_map):
        feat_map = self.avgp(self.conv1(feat_map))
        outputs = self.fc(feat_map.view(-1, self.n_feat))
        return outputs

class policy_net4(nn.Module):
    def __init__(self, in_channel):
        super(policy_net4, self).__init__()
        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(in_channel, 512, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = conv2DBatchNormRelu(256, 256, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(256, 256, k_size=3, stride=2, padding=1)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs

class MIMOGeneralDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, warp_flag=True, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        self.warp_flag = warp_flag
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, qu, k, v, sparse=True):
        # qu (b, q_agents, query_size)
        # k (b, k_agents, key_size)
        # v (b, k_agents, q_agents, c, h, w)
        query = self.linear(qu)  # (b, q_agents, key_size)

        # normalization
        # query_norm = query.norm(p=2,dim=2).unsqueeze(2).expand_as(query)
        # query = query.div(query_norm + 1e-9)

        # k_norm = k.norm(p=2,dim=2).unsqueeze(2).expand_as(k)
        # k = k.div(k_norm + 1e-9)
        # generate the
        attn_orig = torch.bmm(k, query.transpose(2, 1))  # (b, k_agents, q_agents)  column: differnt keys and the same query

        # scaling [not sure]
        # scaling = torch.sqrt(torch.tensor(k.shape[2],dtype=torch.float32)).cuda()
        # attn_orig = attn_orig/ scaling # (b,5,5)  column: differnt keys and the same query

        attn_orig_softmax = self.softmax(attn_orig)  # (b, k_agents, q_agents)
        # attn_orig_softmax = self.sparsemax(attn_orig)

        attn_shape = attn_orig_softmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        attn_orig_softmax_exp = attn_orig_softmax.view(bats, key_num, query_num, 1, 1, 1)

        if self.warp_flag:
            v_exp = v
        else:
            v_exp = torch.unsqueeze(v, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = attn_orig_softmax_exp * v_exp  # (b, k_agents, q_agents, c, h, w)
        output_sum = output.sum(1)  # (b, q_agents, c, h, w)

        return output_sum, attn_orig_softmax


class AdditiveAttentin(nn.Module):
    def __init__(self, c_k, c_q):
        super().__init__()
        # self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)
        self.sparsemax = Sparsemax(dim=1)
        self.linear_feat = nn.Linear(c_k, 128)
        self.linear_context = nn.Linear(c_q, 128)
        self.linear_out = nn.Linear(128, 1)

    def forward(self, q, k, v, sparse=True):
        temp1 = self.linear_feat(k)  # [b, N, 128]
        temp2 = self.linear_context(q)  # [b, 1, 128]
        attn_orig = torch.bmm(temp1, temp2.transpose(2, 1))
        if sparse:
            attn_orig = self.sparsemax(attn_orig)  # [b, N, 1]
        else:
            attn_orig = self.softmax(attn_orig)  # [b, N, 1]
        attn = attn_orig.unsqueeze(-1).unsqueeze(-1) # [b, N, 1, 1, 1]
        output = attn * v # [b, N, C, H, W]
        output = output.sum(1)  # (b, C, H, W)
        return output, attn