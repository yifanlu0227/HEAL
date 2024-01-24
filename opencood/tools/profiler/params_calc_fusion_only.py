from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
import argparse
import pickle
from opencood.utils.common_utils import update_dict
import importlib
import time
import sys
import torch
import numpy as np
from icecream import ic 

def train_parser():
    parser = argparse.ArgumentParser(description="Profiler.")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed. config.yaml in the experimnet log.')
    parser.add_argument('--model_dir', default='',
                        help='Place holder')
    parser.add_argument('--input_data', type=str, required=True,
                        help='input data pickle file.')
    opt = parser.parse_args()
    return opt

def inference_throughput_naive(sub_model, sub_input):
    print("\nstart inference throughput performance test")
    run_num = 50
    print('warm up ...')
    with torch.no_grad(): # warm up
        for i in range(run_num):
            if len(sub_input)==5:
                output=sub_model(sub_input[0], sub_input[1], sub_input[2], sub_input[3], sub_input[4])
            elif len(sub_input)==3:
                output=sub_model(sub_input[0], sub_input[1], sub_input[2])
            else:
                raise ValueError(f"Invalid number of input tensors, len(sub_input)={len(sub_input)}.")
    print('warm up done.')
    run_num = 300
    with torch.no_grad():
        start_time = time.time()
        for i in range(run_num):
            if len(sub_input)==5:
                output=sub_model(sub_input[0], sub_input[1], sub_input[2], sub_input[3], sub_input[4])
            elif len(sub_input)==3:
                output=sub_model(sub_input[0], sub_input[1], sub_input[2])
            else:
                raise ValueError(f"Invalid number of input tensors, len(sub_input)={len(sub_input)}.")
        end_time = time.time()
        infer_thro = run_num / (end_time - start_time)
        print("inference throughput (naive): ", infer_thro)

    return infer_thro

def inference_throughput_cuda_event(sub_model, sub_input):
    print("\nstart inference throughput performance test")
    run_num = 50
    print('warm up ...')
    with torch.no_grad(): # warm up
        for i in range(run_num):
            if len(sub_input)==5:
                output=sub_model(sub_input[0], sub_input[1], sub_input[2], sub_input[3], sub_input[4])
            elif len(sub_input)==3:
                output=sub_model(sub_input[0], sub_input[1], sub_input[2])
            else:
                raise ValueError(f"Invalid number of input tensors, len(sub_input)={len(sub_input)}.")
    print('warm up done.')

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    run_num = 300
    timings = np.zeros((run_num,))

    print('start testing ...')
    with torch.no_grad():
        for i in range(run_num):
            starter.record()
            if len(sub_input)==5:
                output=sub_model(sub_input[0], sub_input[1], sub_input[2], sub_input[3], sub_input[4])
            elif len(sub_input)==3:
                output=sub_model(sub_input[0], sub_input[1], sub_input[2])
            else:
                raise ValueError(f"Invalid number of input tensors, len(sub_input)={len(sub_input)}.")
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[i] = curr_time / 1000

    infer_thro = run_num / timings.sum()
    print("inference throughput (cuda event): ", infer_thro)

    return infer_thro

from typing import Dict, Iterable, Callable
class InputFeatureExtractor(torch.nn.Module):
    """
    Retrieve the input the forward pass of the given layer(s).

    Usage:
        features_ex = InputFeatureExtractor(model, layers=feature_layer)
        sub_input = features_ex(input_data['ego'])
    """
    def __init__(self, model: torch.nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_inputs_hook(layer_id))
        self.hooks = [layer.register_forward_hook(self.save_inputs_hook(layer_id)) for layer_id in layers]

    def __del__(self):
        for hook in self.hooks:
            hook.remove()

    def save_inputs_hook(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self._features[layer_id] = input
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            _ = self.model(x)
        return self._features
    
class SubModel(torch.nn.Module):
    def __init__(self, model):
        super(SubModel, self).__init__()
        self.fusion_net = model.fusion_net
        self.cls_head = model.cls_head
        self.dir_head = model.dir_head
        self.reg_head = model.reg_head
    def forward(self, heter_feature_2d, record_len, affine_matrix):
        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)
        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)
        output_dict={}
        output_dict.update({'cls_preds': cls_preds,'reg_preds': reg_preds,'dir_preds': dir_preds})
        return output_dict
    
class PyramidModel(torch.nn.Module):
    def __init__(self, model):
        super(PyramidModel, self).__init__()
        self.pyramid_backbone = model.pyramid_backbone
        self.shrink_flag = model.shrink_flag
        if self.shrink_flag:
            self.shrink_conv = model.shrink_conv
        self.cls_head = model.cls_head
        self.dir_head = model.dir_head
        self.reg_head = model.reg_head
    def forward(self, heter_feature_2d, record_len, affine_matrix, agent_modality_list, cam_crop_info):
        fused_feature, _ = self.pyramid_backbone(heter_feature_2d, record_len, affine_matrix, agent_modality_list, cam_crop_info)
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)
        output_dict={}
        output_dict.update({'cls_preds': cls_preds,'reg_preds': reg_preds,'dir_preds': dir_preds})
        return output_dict
    
class HmvitModel(torch.nn.Module):
    def __init__(self, model):
        super(HmvitModel, self).__init__()
        self.fusion_net = model.fusion_net
        self.decoder = model.decoder
    def forward(self, x, pairwise_t_matrix, mode, record_len, mask):
        x = self.fusion_net(x, pairwise_t_matrix, mode, record_len, mask).squeeze(1)
        psm, rm, dm = self.decoder(x.unsqueeze(1), mode, use_upsample=False)
        output_dict={}
        output_dict.update({'cls_preds': psm,'reg_preds': rm,'dir_preds': dm})
        return output_dict

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    new_cav_range = [-102.4, -102.4, -3, 102.4, 102.4, 1]

    # replace all appearance
    hypes = update_dict(hypes, {
        "cav_lidar_range": new_cav_range,
        "lidar_range": new_cav_range,
        "gt_range": new_cav_range
    })
    # but infact no need to reload anchor if not decode box
    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    for name, func in yaml_utils_lib.__dict__.items():
        if name == hypes["yaml_parser"]:
            parser_func = func
    hypes = parser_func(hypes)

    model = train_utils.create_model(hypes)

    with open(opt.input_data, 'rb') as f:
        input_data = pickle.load(f)
    model = model.cuda()
    model.eval()
    input_data = train_utils.to_device(input_data, 'cuda')

    ###################### set sub model and data ###############################        
    feature_layer = ["fusion_net"]
    
    if "hmvit" in opt.hypes_yaml:
        sub_model = HmvitModel(model)
    elif "Pyramid" in opt.hypes_yaml:
        sub_model = PyramidModel(model)
        feature_layer = ['pyramid_backbone']
    else:
        sub_model = SubModel(model)

    features_ex = InputFeatureExtractor(model, layers=feature_layer)
    sub_input = features_ex(input_data['ego'])
    # sub_input = sub_input['fusion_net']
    sub_input = sub_input[feature_layer[0]]
    #############################################################################

    infer_throu_naive = inference_throughput_naive(sub_model, sub_input)
    infer_throu_cuda_event = inference_throughput_cuda_event(sub_model, sub_input)
    
    original_stdout = sys.stdout
    output_file = opt.hypes_yaml.replace('config.yaml', 'config_infer_perf_fusion_only.txt')
    with open(output_file, 'w') as f:
        sys.stdout = f  # 重定向标准输出到文件
        localtime = time.asctime(time.localtime(time.time()))
        print ("Test Time: ", localtime)
        print("Range: ", new_cav_range)
        print("Config: ", opt.hypes_yaml)
        print("Input Data: ", opt.input_data)
        print("Device: ", torch.cuda.get_device_name(0))
        print()
        print("inference throughput (by time.time):",  infer_throu_naive, ' sample/sec.')
        print()
        print("inference throughput (by cuda.Event):",  infer_throu_cuda_event, ' sample/sec.')
        torch.cuda.empty_cache()
        print()
       
        mem_before_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        mem_before_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        
        print("Mem reserved (before):", mem_before_reserved, " MB")
        print("Mem allocated (before):", mem_before_allocated, " MB")
        with torch.no_grad():
            if len(sub_input)==5:
                sub_model(sub_input[0], sub_input[1], sub_input[2], sub_input[3], sub_input[4])
            elif len(sub_input)==3:
                sub_model(sub_input[0], sub_input[1], sub_input[2])
            else:
                raise ValueError(f"Invalid number of input tensors, len(sub_input)={len(sub_input)}.")

        mem_after_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        mem_after_allocated = torch.cuda.memory_allocated() / (1024 ** 2)

        print("Mem reserved (after):", mem_after_reserved, " MB")
        print("Mem allocated (after):", mem_after_allocated, " MB")

    # 恢复原始的标准输出
    sys.stdout = original_stdout

    print("Mem reserved (before):", mem_before_reserved, " MB")
    print("Mem allocated (before):", mem_before_allocated, " MB")
    print("Mem reserved (after):", mem_after_reserved, " MB")
    print("Mem allocated (after):", mem_after_allocated, " MB")

if __name__ == "__main__":
    main()

 