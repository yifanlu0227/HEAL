# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.feature_alignnet_modules import SCAligner, Res1x1Aligner, \
    Res3x3Aligner, Res3x3Aligner, CBAM, ConvNeXt, FANet, SDTAAgliner


class AlignNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args['core_method']
        
        if model_name == "scaligner":
            self.channel_align = SCAligner(args['args'])
        elif model_name == "resnet1x1":
            self.channel_align = Res1x1Aligner(args['args'])
        elif model_name == "resnet3x3":
            self.channel_align = Res3x3Aligner(args['args'])
        elif model_name == "sdta":
            self.channel_align = SDTAAgliner(args['args'])
        elif model_name == "cbam":
            self.channel_align = CBAM(args['args'])
        elif model_name == "convnext":
            self.channel_align = ConvNeXt(args['args'])
        elif model_name == "fanet":
            self.channel_align = FANet(args['args'])
        elif model_name == 'identity':
            self.channel_align = nn.Identity()

        self.spatial_align_flag = args.get("spatial_align", False)
        if self.spatial_align_flag:
            raise NotImplementedError

    def forward(self, x):
        return self.channel_align(x)
