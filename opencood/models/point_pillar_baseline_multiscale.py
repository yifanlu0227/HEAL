# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch.nn as nn
from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm

class PointPillarBaselineMultiscale(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarBaselineMultiscale, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", True) # default true
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
 
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']

        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)

        # multiscale fusion
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, affine_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list) 

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        return output_dict
