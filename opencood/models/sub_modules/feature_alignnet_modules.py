# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from opencood.models.sub_modules.cbam import BasicBlock
import math

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class ConvEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=1, deformable=False):
        super().__init__()
        if not deformable:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        else:
            from mmcv.ops import DeformConv2dPack as dconv2d
            self.dwconv = dconv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class SDTAEncoder(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
                 use_pos_emb=False, num_heads=4, qkv_bias=True, attn_drop=0., drop=0., num_conv=2, deformable=False):
        super().__init__()
        width = dim
        convs = []
        if not deformable:
            for i in range(num_conv):
                convs.append(nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=width))
                # convs.append(nn.BatchNorm2d(dim))
                convs.append(nn.ReLU())
        else:
            from mmcv.ops import DeformConv2dPack as dconv2d
            for i in range(num_conv):
                convs.append(dconv2d(dim, dim, kernel_size=1, padding=0, groups=width))
                # convs.append(nn.BatchNorm2d(dim))
                convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)


        self.norm_xca = LayerNorm(dim, eps=1e-6)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()  # TODO: MobileViT is using 'swish'
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.convs(x)

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x



class SDTA(nn.Module):
    def __init__(self, args, deform):
        super().__init__()
        in_ch = args['in_ch']
        self.model = nn.ModuleList()

        for i in range(args['layer_num']):
            self.model.append(ConvEncoder(dim=in_ch, deformable=deform))
            self.model.append(SDTAEncoder(dim=in_ch, deformable=deform))
            
    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x


class Resnet3x3(nn.Module):
    def __init__(self, args, deform=False):
        super().__init__()
        in_ch = args['in_ch']
        layernum = args['layer_num']
        model_list = nn.ModuleList()
        for _ in range(layernum):
            model_list.append(ResidualBlock(in_ch, in_ch, kernel_size=3, deform=deform))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)


class Resnet1x1(nn.Module):
    def __init__(self, args, deform=False):
        super().__init__()
        in_ch = args['in_ch']
        layernum = args['layer_num']
        model_list = nn.ModuleList()
        for _ in range(layernum):
            model_list.append(ResidualBlock(in_ch, in_ch, kernel_size=1, deform=deform))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)


"""
Feature-Align Network with Knowledge Distillation for Efficient Denoising
"""
class ARNetBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(indim, indim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(indim, indim, kernel_size=3, padding=1, groups=8),
            nn.ReLU(),
            nn.Conv2d(indim, outdim, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)

class FALayer(nn.Module):
    def __init__(self, indim, outdim, imgdim):
        super().__init__()
        self.conv1 = nn.Conv2d(imgdim, imgdim, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(imgdim, outdim, 1)
        self.conv3 = nn.Conv2d(imgdim, outdim, 1)
        self.arblock = ARNetBlock(indim, outdim)
    
    def forward(self, feature, img):
        feature = self.arblock(feature)
        inter = self.relu(self.conv1(img))
        gamma = self.conv2(inter)
        beta = self.conv3(inter)

        return feature * gamma + beta

class FANet(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        self.falayer1 = FALayer(dim, dim, dim)
        self.falayer2 = FALayer(dim, dim*2, dim)
        self.falayer3 = FALayer(dim*2, dim*4, dim)
        self.falayer4 = FALayer(dim*4, dim*2, dim)
        self.falayer5 = FALayer(dim*2, dim, dim)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample2d = nn.Upsample(scale_factor=2, mode='bilinear')

        self.skip_conv1 = nn.Conv2d(dim*2, dim*2, 1)
        self.skip_conv2 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        x_detach = x.detach()
        # fake image input
        img0 = x_detach 
        img1 = self.maxpool(img0)
        img2 = self.maxpool(img1)

        feature0 = self.falayer1(x, img0) # H,W, dim
        feature1 = self.falayer2(self.maxpool(feature0), img1) # H/2, W/2, dim*2
        feature2 = self.falayer3(self.maxpool(feature1), img2) # H/4, W/4, dim*4

        feature3 = self.falayer4(self.upsample2d(feature2), img1) + self.skip_conv1(feature1)
        feature4 = self.falayer5(self.upsample2d(feature3), img0) + self.skip_conv2(feature0)

        return feature4



"""
CBAM: Convolutional Block Attention Module
"""
class CBAM(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        num_of_blocks = args['num_of_blocks']
        model_list = nn.ModuleList()
        for _ in range(num_of_blocks):
            model_list.append(BasicBlock(dim, dim))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)


"""
ConvNeXt 
"""
class ConvNeXtBlock(nn.Module):
    r""" 
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7, deform=False):
        super().__init__()
        self.deform = deform
        if self.deform:
            from mmcv.ops import DeformConv2dPack as dconv2d
            self.dfconv = dconv2d(dim, dim, kernel_size=3, padding=1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        if self.deform:
            x = self.dfconv(x)
            x = self.act(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        kernel_size = args.get("kernel_size", 7)
        num_of_blocks = args['num_of_blocks']
        deform = args.get('deform', False)
        model_list = nn.ModuleList()
        for _ in range(num_of_blocks):
            model_list.append(ConvNeXtBlock(dim, kernel_size=kernel_size, deform=deform))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)




"""
Resnet1x1 Aligner
"""
class ResidualBlock(nn.Module):  
    def __init__(self, in_channels, out_channels, use_1x1conv=False, kernel_size=3, deform=False):
        super(ResidualBlock, self).__init__()
        if kernel_size == 3:
            padding = 1
            stride = 1
        elif kernel_size == 1:
            padding = 0
            stride = 1
        else:
            raise("Not Supported")

        if not deform:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        else:
            from mmcv.ops import DeformConv2dPack as dconv2d
            self.conv1 = dconv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
            self.conv2 = dconv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        # 1x1conv来升维
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class Res1x1Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        num_of_blocks = args['num_of_blocks']
        deform = args.get('deform', False)
        model_list = nn.ModuleList()
        for _ in range(num_of_blocks):
            model_list.append(ResidualBlock(dim, dim, kernel_size=1, deform=deform))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)

class Res3x3Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        num_of_blocks = args['num_of_blocks']
        deform = args.get('deform', False)
        model_list = nn.ModuleList()
        for _ in range(num_of_blocks):
            model_list.append(ResidualBlock(dim, dim, kernel_size=3, deform=deform))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)


class SDTAAgliner(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_ch = args['dim']
        self.model = nn.ModuleList()

        for i in range(args['num_of_blocks']):
            self.model.append(ConvEncoder(dim=in_ch, deformable=False))
            self.model.append(SDTAEncoder(dim=in_ch, deformable=False))
            
    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x

"""
Laynorm + MLP
"""
class ResMLP(nn.Module):
    def __init__(self, num_of_layers=2, dim=64):
        super().__init__()
        model_list = [nn.LayerNorm(dim)]
        for i in range(num_of_layers):
            model_list.append(nn.Linear(dim, dim))
            model_list.append(nn.GELU())
        self.model = nn.Sequential(*model_list)
    
    def forward(self, x):
        return x + self.model(x)

class SCAligner(nn.Module):
    """
    Structure:

    Input:
        FeatureMap (NCHW) 
    Model:
        Permute -> (NHWC)
        ------------------------ x M
        LayerNorm -> (NHWC)
        MLP(GELU) x n + skip_conn-> (NHWC) 
        ------------------------
        Permute -> (NCHW)

    if Camera, additionally

    Input:
        FeatureMap (NCHW)
        Coming FeatureMap Mean (NCHW)
    Model:
        cat -> (N 2C HW)
        conv2d -> (N2HW)
        warp FeatureMap (NCHW)

    """
    def __init__(self, args):   
        super().__init__()
        num_of_blocks = args['num_of_blocks']
        num_of_layers = args['num_of_layers']
        dim = args['dim']
        model_list = []
        for _ in range(num_of_blocks):
            model_list.append(ResMLP(num_of_layers, dim))
        self.backbone = nn.Sequential(*model_list)


    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.backbone(x)
        x = x.permute(0,3,1,2)
        return x
