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

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def evaluate_pose_graph(data_dict, save_path, std=0.2):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.random.seed(100)
    interval=30
    cnt = 0
    for sample_idx, content in tqdm(data_dict.items()): 
        print(sample_idx)
        cnt += 1
        if cnt % interval != 0:
            continue
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


        pose_after = [noisy_lidar_pose_dof3]

        for i in range(1,60):
            pose_after += [box_alignment_relative_sample_np(pred_corners_list, 
                                                            noisy_lidar_pose, 
                                                            uncertainty_list=uncertainty_list, 
                                                            landmark_SE2=True,
                                                            adaptive_landmark=False,
                                                            normalize_uncertainty=False,
                                                            abandon_hard_cases=True,
                                                            drop_hard_boxes=True,
                                                            use_uncertainty=True,
                                                            thres=1.5,
                                                            max_iterations = i)]
            if(np.sum(np.abs(pose_after[-1] - pose_after[-2]))) < 1e-2:
                break


        diffs = [np.abs(lidar_pose_clean_dof3 - pose) for pose in pose_after]
        diffs = np.stack(diffs)
        diffs[:,1:,2] = np.minimum(diffs[:,1:,2], 360 - diffs[:,1:,2]) # do not include ego
       

        # pose_graph_save_dir = os.path.join(save_path, f"pg_vis/{sample_idx}")
        # vis_pose_graph(pose_after,
        #             pred_corners_list, 
        #             save_dir_path=pose_graph_save_dir,
        #             vis_agent=True)
        # np.savetxt(os.path.join(pose_graph_save_dir,"pose_info.txt"), diffs.reshape(-1,3), fmt="%.4f")



evaluate_json = "/GPFS/rhome/yifanlu/OpenCOOD/stage1_boxes/opv2v/test/stage1_boxes.json"
data_dict = read_json(evaluate_json)

output_path = "/GPFS/rhome/yifanlu/OpenCOOD/opencood/visualization/draw_box_align/0606_test"
evaluate_pose_graph(data_dict, output_path, std=0.6)

# output_path = "/GPFS/rhome/yifanlu/OpenCOOD/vis_result/box_align_dist_08"
# evaluate_pose_graph(data_dict, output_path, std=0.8)