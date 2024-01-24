"""
Training throughput calculation
"""


import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.common_utils import update_dict
import importlib
from icecream import ic

import time


def test_performance(model, train_dataloader, criterion=None, optimizer=None, supervise_single_flag=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 峰值显存占用
    peak_memory_reserved = 0
    peak_memory_allocated = 0
    
    # 训练样本吞吐量
    model.train()
    try: 
        model.model_train_init()
    except:
        print("No model_train_init function")
    start_time = time.time()
    
    for i, batch_data in enumerate(train_dataloader):
        if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
            continue
        model.zero_grad()
        optimizer.zero_grad()
        batch_data = train_utils.to_device(batch_data, device)
        ouput_dict = model(batch_data['ego'])
        
        final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

        if supervise_single_flag:
            final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")

        # back-propagation
        final_loss.backward()
        optimizer.step()
            
        current_memory_reserved =  torch.cuda.memory_reserved() / (1024 ** 2)
        peak_memory_reserved = max(peak_memory_reserved, current_memory_reserved)

        current_memory_allocated =  torch.cuda.memory_reserved() / (1024 ** 2)
        peak_memory_allocated = max(peak_memory_allocated, current_memory_allocated)
    
    elapsed_time = time.time() - start_time
    train_throughput = len(train_dataloader) / elapsed_time
    print(f"Training throughput: {train_throughput:.2f} samples/s")
    print(f"Peak GPU memory reserved usage during training: {current_memory_reserved:.2f} MB")
    print(f"Peak GPU memory allocated usage during training: {peak_memory_allocated:.2f} MB")
    return train_throughput, peak_memory_reserved, peak_memory_allocated



def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Place holder. NOT USED.')
    opt = parser.parse_args()
    return opt

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
    # reload anchor
    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    for name, func in yaml_utils_lib.__dict__.items():
        if name == hypes["yaml_parser"]:
            parser_func = func
    hypes = parser_func(hypes)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=1,
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)

    if torch.cuda.is_available():
        model.to(device)

    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single

    train_throughput, peak_memory_re, peak_memory_al = test_performance(model, train_loader, criterion, optimizer, supervise_single_flag)

    output_file = opt.hypes_yaml.replace('config.yaml', 'config_train_perf.txt')

    with open(output_file, 'w') as f:
        localtime = time.asctime(time.localtime(time.time()))
        print("Test Time: ", localtime)
        f.write(f"range: {new_cav_range}")
        f.write(f"\nDevice: {torch.cuda.get_device_name(0)} \n")
        f.write(f"training throughput: {train_throughput} samples/sec. \n")
        f.write(f"training peak_memory (reserved): {peak_memory_re} MB. \n")
        f.write(f"training peak_memory (allocated): {peak_memory_al} MB. \n")

if __name__ == "__main__":
    main()