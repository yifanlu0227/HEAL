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
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import time

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

def inference_throughput_naive(model, data):
    print("start inference throughput performance test")
    run_num = 50
    print('warm up ...\n')
    with torch.no_grad(): # warm up
        for i in range(run_num):
            output = model(data)
    print('warm up done.')
    run_num = 200
    with torch.no_grad():
        start_time = time.time()
        for i in range(run_num):
            output = model(data)
        end_time = time.time()
        infer_thro = run_num / (end_time - start_time)
        print("inference throughput (naive): ", infer_thro)

    return infer_thro

def inference_throughput_cuda_event(model, data):
    print("start inference throughput performance test")
    run_num = 50
    print('warm up ...\n')
    with torch.no_grad(): # warm up
        for i in range(run_num):
            output = model(data)
    print('warm up done.')

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    run_num = 200
    timings = np.zeros((run_num,))

    print('start testing ...\n')
    with torch.no_grad():
        for i in range(run_num):
            starter.record()
            output = model(data)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[i] = curr_time / 1000

    infer_thro = run_num / timings.sum()
    print("inference throughput (cuda event): ", infer_thro)

    return infer_thro

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
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

    with open(opt.input_data, 'rb') as f:
        input_data = pickle.load(f)
    model = model.cuda()
    model.eval()
    input_data = train_utils.to_device(input_data, 'cuda')
    if "DAIR" in opt.hypes_yaml:
        input_data['ego']['object_ids'] = torch.IntTensor(input_data['ego']['object_ids'])
        input_data['ego']['sample_idx'] = torch.IntTensor(input_data['ego']['sample_idx'])
        input_data['ego']['cav_id_list'] = torch.IntTensor(input_data['ego']['cav_id_list'])


    flops = FlopCountAnalysis(model, input_data['ego'])
    print(flop_count_table(flops))
    # infer_throu_naive = inference_throughput_naive(model, input_data['ego'])
    infer_throu_cuda_event = inference_throughput_cuda_event(model, input_data['ego'])
    
    original_stdout = sys.stdout

    # 打开一个文件用于写入
    output_file = opt.hypes_yaml.replace('config.yaml', 'config_infer_perf.txt')
    with open(output_file, 'w') as f:
        sys.stdout = f  # 重定向标准输出到文件
        localtime = time.asctime(time.localtime(time.time()))
        print ("Test Time: ", localtime)
        print("Range: ", new_cav_range)
        print("Config: ", opt.hypes_yaml)
        print()
        print("Input Data: ", opt.input_data)
        print(flop_count_table(flops))
        print()
        print("Device: ", torch.cuda.get_device_name(0))
        # print()
        # print("inference throughput (by time.time):",  infer_throu_naive, ' sample/sec.')
        print()
        print("inference throughput (by cuda.Event):",  infer_throu_cuda_event, ' sample/sec.')
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

        print("Mem reserved (after):", mem_after_reserved, " MB")
        print("Mem allocated (after):", mem_after_allocated, " MB")

        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, use_cuda=True) as prof:
                with record_function("model_inference"):
                    model(input_data['ego'])
            print("GPU time sorted operators:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print("CPU time sorted operators:")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # 恢复原始的标准输出
    sys.stdout = original_stdout

    print("Mem reserved (before):", mem_before_reserved, " MB")
    print("Mem allocated (before):", mem_before_allocated, " MB")
    print("Mem reserved (after):", mem_after_reserved, " MB")
    print("Mem allocated (after):", mem_after_allocated, " MB")

if __name__ == "__main__":
    main()

 