# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import json
import copy
import numpy as np
from opencood.models.sub_modules.box_align_v2 import vis_pose_graph, box_alignment_relative_sample_np
from opencood.utils.pose_utils import generate_noise
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib import rcParams
sns.set(rc={'figure.figsize':(11.7,8.27)})


DEBUG = True

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data

# def plot_cdf(trans_error_list, rot_error_list, filenames, save_path, std):
#     legends = copy.deepcopy(filenames)
#     legends.reverse() # sns's legend order is different from plt?

#     for error_type in ['trans', 'rot']:
#         error_list = eval(f"{error_type}_error_list")
#         fig = plt.figure(figsize=[6.4, 5.8])
#         xtitle = " error (m)" if error_type=='trans' else " error (°)"
#         xtitle = error_type + xtitle

#         sns.ecdfplot(error_list)
#         plt.xlim(0, 0.8)
#         plt.xlabel(xtitle)
#         plt.legend(labels=legends, prop = {'size':12})
#         plt.tight_layout()
#         plt_filename = f"{std}_".replace(".","") + f"{error_type}_" +  "cdf.png"
#         plt_filename = os.path.join(save_path, plt_filename)
#         plt.savefig(plt_filename, dpi=300)
#         plt.clf()


def vis_data(trans_error_list, rot_error_list, filenames, save_path, std):
    # vis

    legends = copy.deepcopy(filenames)
    legends.reverse() # sns's legend order is different from plt?
    plt.style.use('ggplot')
    for error_type in ['trans', 'rot']:
        error_list = eval(f"{error_type}_error_list")
        
        # pdf
        sns.displot(error_list, kind='kde', bw_adjust=0.1, legend=False, aspect=1.2, linewidth=2)
        xtitle = " error (m)" if error_type=='trans' else " error (°)"
        xtitle = error_type + xtitle
        plt.xlabel(xtitle)
        plt.legend(labels=legends, prop = {'size':16})
        plt.tight_layout()
        plt_filename = f"{std}_".replace(".","") + f"{error_type}_" + "all.png"
        plt_filename = os.path.join(save_path, plt_filename)
        plt.savefig(plt_filename, dpi=300)
        plt.clf()

        # pdf
        # limit in [0,1]
        error_list_np = [np.array(item) for item in error_list]
        for error_term in error_list_np:
            error_term[error_term>1] = 1+np.random.randn(np.sum(error_term>1))*0.05

        sns.displot(error_list_np, kind='kde', bw_adjust=0.2, legend=False, aspect=1.2, linewidth=2)
        plt.xlim(0, 0.6)
        xtitle = " error (m)" if error_type=='trans' else " error (°)"
        xtitle = error_type + xtitle
        plt.xlabel(xtitle)
        plt.legend(labels=legends, prop = {'size':16})
        plt.tight_layout()
        plt_filename = f"{std}_".replace(".","") + f"{error_type}_" +  "all_lim.png"
        plt_filename = os.path.join(save_path, plt_filename)
        plt.savefig(plt_filename, dpi=300)
        plt.clf()

        sns.displot(error_list_np, kind='kde', bw_adjust=0.2, legend=False, aspect=1.2, linewidth=3)
        plt.xlim(0, 0.6)
        xtitle = " error (m)" if error_type=='trans' else " error (°)"
        xtitle = error_type + xtitle
        plt.xlabel(xtitle)
        plt.legend(labels=legends, prop = {'size':16})
        plt.tight_layout()
        plt_filename = f"{std}_".replace(".","") + f"{error_type}_" +  "all_lim3.png"
        plt_filename = os.path.join(save_path, plt_filename)
        plt.savefig(plt_filename, dpi=300)
        plt.clf()

        # cdf
        sns.ecdfplot(error_list)
        plt.xlim(0, 0.2 + std*2)
        plt.xlabel(xtitle)
        plt.legend(labels=legends, prop = {'size':14})
        plt.tight_layout()
        plt_filename = f"{std}_".replace(".","") + f"{error_type}_" +  "cdf.png"
        plt_filename = os.path.join(save_path, plt_filename)
        plt.savefig(plt_filename, dpi=300)
        plt.clf()




def calc_data(trans_error_list, rot_error_list, filenames, save_path, std):
    for error_type in ['trans', 'rot']:
        error_list = eval(f"{error_type}_error_list")
        txtname = os.path.join(save_path, f"{std}_{error_type}_stat.txt")
        with open(txtname ,'w') as f:
            for i in range(len(filenames)):
                data = np.array(error_list[i])
                f.write(f"=========== {filenames[i]} =========== \n")
                f.write(f"mean: {np.mean(data):.4f}\n")
                f.write(f"max: {np.max(data):.4f}\n")
                quantile_list = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.98, 0.99]
                for q in quantile_list:
                    f.write(f"quantile {q:.2f}: {np.quantile(data, q):.4f}\n")
                f.write("\n\n")
            

def evaluate_pose_graph(data_dict, save_path, std=0.2):

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    filenames = ['correction w/ uncertainty', 'correction w/o uncertainty', 'w/o correction']
    # filenames = ['correction w/ c.d.', 'w/o correction']

    test_term_num = len(filenames)

    trans_error_list = [[] for i in range(test_term_num)]
    
    rot_error_list = [[] for i in range(test_term_num)]


    np.random.seed(100)

    for sample_idx, content in tqdm(data_dict.items()): 
        print(sample_idx)
        if content is None:
            continue
        pred_corners_list = content['pred_corner3d_np_list']
        pred_corners_list = [np.array(corners, dtype=np.float64) for corners in pred_corners_list]
        uncertainty_list = content['uncertainty_np_list']
        uncertainty_list = [np.array(uncertainty, dtype=np.float64) for uncertainty in uncertainty_list]
        lidar_pose_clean_np = np.array(content['lidar_pose_clean_np'], dtype=np.float64)
        lidar_pose_clean_dof3 = lidar_pose_clean_np[:,[0,1,4]]
        cav_id_list = content['cav_id_list']
        N = lidar_pose_clean_np.shape[0]

        noisy_lidar_pose = copy.deepcopy(lidar_pose_clean_np)
        noisy_lidar_pose[1:,[0,1,4]] += np.random.normal(0, std, size=(N-1,3))
        noisy_lidar_pose_dof3 = noisy_lidar_pose[:,[0,1,4]]


        pose_after = [
                      box_alignment_relative_sample_np(pred_corners_list, # l2_cd_1.5
                                                        noisy_lidar_pose, 
                                                        uncertainty_list=uncertainty_list, 
                                                        landmark_SE2=True,
                                                        adaptive_landmark=False,
                                                        normalize_uncertainty=False,
                                                        abandon_hard_cases=True,
                                                        drop_hard_boxes=True,
                                                        use_uncertainty=True),

                      box_alignment_relative_sample_np(pred_corners_list, # l2_1.5
                                                        noisy_lidar_pose, 
                                                        uncertainty_list=uncertainty_list, 
                                                        landmark_SE2=True,
                                                        adaptive_landmark=False,
                                                        normalize_uncertainty=False,
                                                        abandon_hard_cases=True,
                                                        drop_hard_boxes=True,
                                                        use_uncertainty=False),

                      noisy_lidar_pose_dof3,
                    ]

        diffs = [np.abs(lidar_pose_clean_dof3 - pose) for pose in pose_after]
        diffs = np.stack(diffs)
        diffs[:,1:,2] = np.minimum(diffs[:,1:,2], 360 - diffs[:,1:,2]) # do not include ego
        
        for i, diff in enumerate(diffs):
            pos_diff = diff[1:,:2] # do not include ego
            angle_diff = diff[1:,2] # do not include ego
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            trans_error_list[i].extend(pos_diff.flatten().tolist())
            rot_error_list[i].extend(angle_diff.flatten().tolist())

        DEBUG = False
        if DEBUG:
            if (diffs[0] > 1).any():
                pose_graph_save_dir = os.path.join(save_path, f"pg_vis/{sample_idx}")
                vis_pose_graph(pose_after,
                            pred_corners_list, 
                            save_dir_path=pose_graph_save_dir)
                np.savetxt(os.path.join(pose_graph_save_dir,"pose_info.txt"), diffs.reshape(-1,3), fmt="%.4f")

        


    vis_data(trans_error_list, rot_error_list, filenames, save_path, std)

    calc_data(trans_error_list, rot_error_list, filenames, save_path, std)



evaluate_json = "/remote-home/share/yifanlu/OpenCOODv2/opencood/logs/stage1_boxes.json"
data_dict = read_json(evaluate_json)
# data_dict = {"1":data_dict["1"], "2": data_dict["2"]}

# output_path = "/GPFS/rhome/yifanlu/OpenCOOD/vis_result/A_opv2v_BA_clip_flatten/dist_02"
# evaluate_pose_graph(data_dict, output_path, std=0.2)


output_path = "/remote-home/share/yifanlu/OpenCOODv2/vis_result/opv2v_0404"
evaluate_pose_graph(data_dict, output_path, std=0.4)

# output_path = "/GPFS/rhome/yifanlu/OpenCOOD/vis_result/box_align_dist_08"
# evaluate_pose_graph(data_dict, output_path, std=0.8)