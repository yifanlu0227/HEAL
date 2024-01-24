# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
from collections import OrderedDict

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
def unfix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def has_trainable_params(module: torch.nn.Module) -> bool:
    any_require_grad = any(p.requires_grad for p in module.parameters())
    any_bn_in_train_mode = any(m.training for m in module.modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)))
    return any_require_grad or any_bn_in_train_mode

def has_untrainable_params(module: torch.nn.Module) -> bool:
    any_not_require_grad = any((not p.requires_grad) for p in module.parameters())
    any_bn_in_eval_mode = any((not m.training) for m in module.modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)))
    return any_not_require_grad or any_bn_in_eval_mode

def check_trainable_module(model):
    appeared_module_list = []
    has_trainable_list = []
    has_untrainable_list = []
    for name, module in model.named_modules():
        if any([name.startswith(appeared_module_name) for appeared_module_name in appeared_module_list]) or name=='': # the whole model has name ''
            continue
        appeared_module_list.append(name)

        if has_trainable_params(module):
            has_trainable_list.append(name)
        if has_untrainable_params(module):
            has_untrainable_list.append(name)

    print("=========Those modules have trainable component=========")
    print(*has_trainable_list,sep='\n',end='\n\n')
    print("=========Those modules have untrainable component=========")
    print(*has_untrainable_list,sep='\n',end='\n\n')


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=0.1)
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=0.1)
        # if hasattr(m, 'bias'):
        #     nn.init.constant_(m.bias, 0)

    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.xavier_normal_(m.weight, gain=0.05)
    #     nn.init.constant_(m.bias, 0)

def rename_model_dict_keys(model_dict_path, rename_dict):
    """
    Args:
        model_dict_path : str
            path to the model checkpoints
        rename_dict : dict
            key: old name
            value: new name

    Usage:
        Case 1: remove model parameters
            rename_dict = {"camera_encoder.*": "",
                            "camera_backbone.*": "",
                            "shrink_camera.*": "",
                            "cls_head_camera.*": "",
                            "reg_head_camera.*": "",
                            "dir_head_camera.*": "",}
            if the value is "", then the key will be removed from the model dict
        
        Case 2: rename model parameters' keys
            rename_dict = {"camencode.*": "camera_encoder.camencode.*",
                           "bevencode.*": "camera_encoder.bevencode.*",
                           "head.cls_head.*": "cls_head_camera.*",
                           "head.reg_head.*": "reg_head_camera.*",
                           "head.dir_head.*": "dir_head_camera.*",
                           "shrink_conv.*": "shrink_camera.*"}
            if the value is not "", then the key will be renamed to the value. * is supported to match multiple keys
    
    """
    pretrained_dict = torch.load(model_dict_path)
    torch.save(pretrained_dict, model_dict_path.replace('.pth', '_before_rename.pth'))
    # 1. filter out unnecessary keys
    for oldname, newname in rename_dict.items():
        if oldname.endswith("*"):
            _oldnames = list(pretrained_dict.keys())
            _oldnames = [x for x in _oldnames if x.startswith(oldname[:-1])]
            for _oldname in _oldnames:
                if newname != "":
                    _newname = _oldname.replace(oldname[:-1], newname[:-1])
                    pretrained_dict[_newname] = pretrained_dict[_oldname]
                pretrained_dict.pop(_oldname)
        else:
            if newname != "":
                pretrained_dict[newname] = pretrained_dict[oldname]
            pretrained_dict.pop(oldname)
    torch.save(pretrained_dict, model_dict_path)


def compose_model(model1, keyname1, model2, keyname2, output_model):
    pretrained_dict1 = torch.load(model1)
    pretrained_dict2 = torch.load(model2)

    new_dict = OrderedDict()
    for keyname in keyname1:
        if keyname.endswith("*"):
            _oldnames = list(pretrained_dict1.keys())
            _oldnames = [x for x in _oldnames if x.startswith(keyname[:-1])]
            for _oldname in _oldnames:
                new_dict[_oldname] = pretrained_dict1[_oldname]

    for keyname in keyname2:
        if keyname.endswith("*"):
            _oldnames = list(pretrained_dict2.keys())
            _oldnames = [x for x in _oldnames if x.startswith(keyname[:-1])]
            for _oldname in _oldnames:
                new_dict[_oldname] = pretrained_dict2[_oldname]

    torch.save(new_dict, output_model)



if __name__ == "__main__":
    # exemplar usage 2: rename model parameters' keys!
    dict_path = "/GPFS/rhome/yifanlu/workspace/OpenCOODv2/opencood/logs/v2xset_heter_late_fusion/net_epoch_bestval_at28.pth"
    rename_dict = {"camencode.*": "camera_encoder.camencode.*",
                   "bevencode.*": "camera_encoder.bevencode.*",
                   "head.cls_head.*": "cls_head_camera.*",
                   "head.reg_head.*": "reg_head_camera.*",
                   "head.dir_head.*": "dir_head_camera.*",
                   "shrink_conv.*": "shrink_camera.*"}
    rename_model_dict_keys(dict_path, rename_dict)

