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
import os
import subprocess
import numpy as np
from opencood.tools.profiler.params_calc import inference_throughput_naive, inference_throughput_cuda_event
from torch.profiler import profile, record_function, ProfilerActivity
import time


def train_parser():
    parser = argparse.ArgumentParser(description="Profiler.")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed. config.yaml in the experimnet log.')
    parser.add_argument('--model_dir', default='',
                        help='Place holder')
    parser.add_argument("--half", action='store_true',
                        help='if communication range set to half.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    new_cav_range = [-102.4, -102.4, -3, 102.4, 102.4, 1]
    if opt.half:
        new_cav_range = [-102.4, -51.2, -3, 102.4, 51.2, 1]

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
    print('Loading Model from checkpoint') 
    resume_epoch, model = train_utils.load_saved_model(opt.hypes_yaml.rstrip('config.yaml'), model)
    print(f"resume from {resume_epoch} epoch.")

    
    if opt.half:
        output_file = opt.hypes_yaml.replace('config.yaml', 'config_infer_perf_multi_half_0917.txt')
    else:
        output_file = opt.hypes_yaml.replace('config.yaml', 'config_infer_perf_multi_0917.txt')

    infer_throu_naive_list = []
    infer_throu_cuda_event_list = []
    flop_analysis_list = []
    mem_before_reserved_list = []
    mem_after_reserved_list = []
    mem_before_allocated_list = []
    mem_after_allocated_list = []

    for use_cav in [1,2,3,4,5]:
        if opt.half:
            input_data_file = f"opencood/logs_HEAL/FLOPs_calc/MoreAgents_m1/input_half_{use_cav}.pkl"
        else:
            input_data_file = f"opencood/logs_HEAL/FLOPs_calc/MoreAgents_m1/input_{use_cav}.pkl"
        with open(input_data_file, 'rb') as f:
            input_data = pickle.load(f)
        model = model.cuda()
        model.eval()
        input_data = train_utils.to_device(input_data, 'cuda')

        flops = FlopCountAnalysis(model, input_data['ego'])
        infer_throu_naive = inference_throughput_naive(model, input_data['ego'])
        infer_throu_cuda_event = inference_throughput_cuda_event(model, input_data['ego'])
        # 重定向标准输出到文件
        
        print("Use cav: ", use_cav)
        print("Range: ", new_cav_range)
        print("Config: ", opt.hypes_yaml)
        print("Input Data: ", input_data_file)

        print("inference throughput (by time.time):",  infer_throu_naive, ' sample/sec.')
        print("inference throughput (by cuda.Event):",  infer_throu_cuda_event, ' sample/sec.')
        print("Device: ", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
        print()

        mem_before_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        mem_before_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        
        print("Mem reserved (before):", mem_before_reserved, " MB")
        print("Mem allocated (before):", mem_before_allocated, " MB")
        with torch.no_grad():
            model(input_data['ego'])
            mem_after_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            mem_after_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            # torch profiler

        print("Mem reserved (after):", mem_after_reserved, " MB")
        print("Mem allocated (after):", mem_after_allocated, " MB")
        
        # pytorch profiler
        # with torch.no_grad():
        #     with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, use_cuda=True) as prof:
        #         with record_function("model_inference"):
        #             model(input_data['ego'])
        #     print("GPU time sorted operators:")
        #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        #     print("CPU time sorted operators:")
        #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


        infer_throu_naive_list.append(infer_throu_naive)
        infer_throu_cuda_event_list.append(infer_throu_cuda_event)
        flop_analysis_list.append(flop_count_table(flops))
        mem_before_reserved_list.append(mem_before_reserved)
        mem_after_reserved_list.append(mem_after_reserved)
        mem_before_allocated_list.append(mem_before_allocated)
        mem_after_allocated_list.append(mem_after_allocated)
        print("###########################################################")


    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        localtime = time.asctime(time.localtime(time.time()))
        print("Test Time: ", localtime)
        print("Use cav: ", use_cav, end='\n\n')
        print("Device: ", torch.cuda.get_device_name(0), end='\n\n')
        print("Range: ", new_cav_range, end='\n\n')
        print("Config: ", opt.hypes_yaml, end='\n\n')
        print("infer_throu_naive: ", *infer_throu_naive_list, sep='\n', end='\n\n')
        print("infer_throu_cuda_event: ", *infer_throu_cuda_event_list, sep='\n', end='\n\n')
        print("flop_analysis: ", *flop_analysis_list, sep='\n', end='\n\n')

        # 恢复原始的标准输出
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()

 