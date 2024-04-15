# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import torchvision
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import importlib


class HeterModelLate(nn.Module):
    def __init__(self, args):
        super(HeterModelLate, self).__init__()
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            # depth supervision for camera
            if model_setting['encoder_args'].get("depth_supervision", False) :
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            # setup backbone (very light-weight)
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))

            # setup layers (actual backbone)
            setattr(self, f"layers_{modality_name}", ResNetBEVBackbone(model_setting['layers_args']))
            setattr(self, f"layers_num_{modality_name}", len(model_setting['layers_args']['num_upsample_filter']))

            # setup shrink head
            setattr(self, f"shrink_conv_{modality_name}",  DownsampleConv(model_setting['shrink_header']))

            # setup detection head
            in_head = model_setting['head_args']['in_head']
            setattr(self, f'cls_head_{modality_name}', nn.Conv2d(in_head, args['anchor_number'], kernel_size=1))
            setattr(self, f'reg_head_{modality_name}', nn.Conv2d(in_head, args['anchor_number'] * 7, kernel_size=1))
            setattr(self, f'dir_head_{modality_name}', nn.Conv2d(in_head, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))


    def forward(self, data_dict):
        output_dict = {}
        modality_name = [x for x in list(data_dict.keys()) if x.startswith("inputs_")]
        assert len(modality_name) == 1
        modality_name = modality_name[0].lstrip('inputs_')

        feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
        feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']

        if self.sensor_type_dict[modality_name] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = feature.shape
            feature = torchvision.transforms.CenterCrop(
                    (int(H*eval(f"self.crop_ratio_H_{modality_name}")), int(W*eval(f"self.crop_ratio_W_{modality_name}")))
                )(feature)

            if eval(f"self.depth_supervision_{modality_name}"):
                output_dict.update({
                    f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                })

        # multiscale fusion. 
        # Here we do not use layer0 of the "self.layers_{modality_name}"
        # We assume feature from the "self.backbone_{modality_name}" is the first-scale feature
        feature_list = [feature]

        for i in range(1, eval(f"self.layers_num_{modality_name}")):
            feature = eval(f"self.layers_{modality_name}").get_layer_i_feature(feature, layer_i=i)
            feature_list.append(feature)

        feature = eval(f"self.layers_{modality_name}").decode_multiscale_feature(feature_list)
        
        feature = eval(f"self.shrink_conv_{modality_name}")(feature)

        cls_preds = eval(f"self.cls_head_{modality_name}")(feature)
        reg_preds = eval(f"self.reg_head_{modality_name}")(feature)
        dir_preds = eval(f"self.dir_head_{modality_name}")(feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})

        return output_dict
