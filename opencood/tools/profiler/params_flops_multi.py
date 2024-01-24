from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
import argparse
import pickle
from opencood.utils.common_utils import update_dict
import importlib
import sys
import torch
import time

def train_parser():
    parser = argparse.ArgumentParser(description="Profiler.")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed. config.yaml in the experimnet log.')
    parser.add_argument("--half", action='store_true',
                        help='if communication range set to half.')
    parser.add_argument('--model_dir', default='',
                        help='Place holder')
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
    
    if opt.half:
        output_file = opt.hypes_yaml.replace('config.yaml', 'config_flops_multi_half.txt')
    else:
        output_file = opt.hypes_yaml.replace('config.yaml', 'config_flops_multi.txt')

    flop_analysis_list = []

    for use_cav in [1,2,3,4,5]:
        if opt.half:
            input_data_file = f"opencood/logs_HEAL/FLOPS_calc/MoreAgents_m1/input_half_{use_cav}.pkl"
        else:
            input_data_file = f"opencood/logs_HEAL/FLOPS_calc/MoreAgents_m1/input_{use_cav}.pkl"
        
        with open(input_data_file, 'rb') as f:
            input_data = pickle.load(f)
        model.eval()

        flops = FlopCountAnalysis(model, input_data['ego'])
        
        print("Use cav: ", use_cav)
        print("Range: ", new_cav_range)
        print("Config: ", opt.hypes_yaml)
        print("Input Data: ", input_data_file)
        print("Device: ", torch.cuda.get_device_name(0))

        flop_analysis_list.append(flop_count_table(flops))
        
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
        print("flop_analysis: ", *flop_analysis_list, sep='\n', end='\n\n')
        # 恢复原始的标准输出
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()

 