# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone


class CenterPointIntermediate(nn.Module):
    def __init__(self, args):
        super(CenterPointIntermediate, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)
        # self.out_size_factor = args['']
        self.voxel_size = args['voxel_size']
        self.out_size_factor = args['out_size_factor']
        self.cav_lidar_range  = args['lidar_range']
        self.conv_cls = nn.Conv2d(
            128*3, 1,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            128*3, 8,               ## xyz,hwl,sin(r),cos(r)
            kernel_size=1
        )  # in_channels  out_channels
        self.init_weight()

    def init_weight(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi) )
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)



    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        lidar_pose = data_dict['lidar_pose']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        

        cls = self.conv_cls(spatial_features_2d)   ## [2, 1, h, w]
        bbox = self.conv_box(spatial_features_2d)    ## [2, 8, h, w]
    

        # 把bbox 的第二维度变成7 
        _, bbox_temp = self.generate_predicted_boxes(cls, bbox)

        # print(bbox.equal(bbox_temp))
        output_dict = {'cls_preds': cls,
                       'reg_preds': bbox_temp,
                       'cls': cls,
                       'bbox_preds':bbox}    # 计算loss的时候使用 'bbox', 在生成output的时候 'rm'
                       
        return output_dict


    def generate_predicted_boxes(self, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        
        batch, H, W, code_size = box_preds.size()   ## code_size 表示的是预测的尺寸
        
        # batch_reg = box_preds[:, 0:2, :, :]  # x,y,z
        # batch_hei = box_preds[:, 2:3, :, :]
        # batch_dim = torch.exp(box_preds[:, 3:6, :, :])
        # # batch_dim = box_preds[:, 3:6, :, :]   # w h l 
        # batch_rots = box_preds[:, 6:7, :, :]
        # batch_rotc = box_preds[:, 7:8, :, :]
        # rot = torch.atan2(batch_rots, batch_rotc)
        
        
        box_preds = box_preds.reshape(batch, H*W, code_size)

        batch_reg = box_preds[..., 0:2]
        batch_hei = box_preds[..., 2:3]

        batch_dim = torch.exp(box_preds[..., 3:6])

        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]

        rot = torch.atan2(batch_rots, batch_rotc)

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)

        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.cav_lidar_range[0]   ## 基于feature_map 的size求解真实的坐标
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.cav_lidar_range[1]


        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
        # batch_box_preds = batch_box_preds.reshape(batch, H, W, batch_box_preds.shape[-1])
        # batch_box_preds = batch_box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_box_preds_temp = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=1)
        # box_preds = box_preds.permute(0, 3, 1, 2).contiguous()

        # batch_cls_preds = cls_preds.view(batch, H*W, -1)
        return cls_preds, batch_box_preds
