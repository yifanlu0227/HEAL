# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import open3d as o3d
import numpy as np
import cv2
from matplotlib import pyplot as plt
from opencood.utils.subsampling_utils import get_random_subset
from multiprocessing import Process

vis = False

def mask_points_by_range(points, limit_range, return_mask=False):
    if len(limit_range) == 6:
        mask =  (points[:, 0] > limit_range[0]) & \
                (points[:, 0] < limit_range[3]) & \
                (points[:, 1] > limit_range[1]) & \
                (points[:, 1] < limit_range[4]) & \
                (points[:, 2] > limit_range[2]) & \
                (points[:, 2] < limit_range[5])
    elif len(limit_range) == 4:
        mask =  (points[:, 0] > limit_range[0]) & \
                (points[:, 0] < limit_range[2]) & \
                (points[:, 1] > limit_range[1]) & \
                (points[:, 1] < limit_range[3]) 

    points_mask = points[mask]
    
    if return_mask:
        return points_mask, mask
    else:
        return points_mask

def project_bev(pcd_np, lidar_range, voxel_size):
    """ project pcd to bev
    Args:
        pcd_np: np.ndarray, (N, 3)

        lidar_range: list
            range for bev, [x_min, y_min, z_min, x_max, y_max, z_max]

    Return
        bev: np.array, (H, W), 
            H = (y_max - y_min) / voxel_size
            W = (x_max - x_min) / voxel_size

        pcd_np_with_idx: np.ndarray, (N_, 4)
            last index show it belongs to which grid
    """
    [x_min, y_min, z_min, x_max, y_max, z_max] = lidar_range

    pcd_crop_np, mask = mask_points_by_range(pcd_np, lidar_range, return_mask=True)

    pcd_np_with_idx = np.zeros((pcd_np.shape[0], 4))
    pcd_np_with_idx[:,:3] = pcd_np

    H = round((y_max - y_min) / voxel_size)
    W = round((x_max - x_min) / voxel_size)
    # print(f"BEV map with shape ({H}, {W}).")

    bev = np.zeros((H, W), dtype=np.uint8)
    for i, (x,y,z) in enumerate(pcd_np):
        y_idx = int((y - y_min) / voxel_size)
        x_idx = int((x - x_min) / voxel_size)
        if mask[i]:
            bev[y_idx, x_idx] = 255
        pcd_np_with_idx[i][3] = y_idx * W + x_idx

    if vis:
        plt.imshow(bev)
        plt.show()

    return bev, pcd_np_with_idx

def line_detection(bev_img):
    """
    Should we really need detect line?
    Is edge enough to use?
    """
    edges = cv2.Canny(bev_img, 100, 200)
    if vis:
        plt.imshow(edges)
        plt.show()

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 25  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    line_image = np.copy(bev_img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255),1)

    if vis:
        plt.imshow(line_image)
        plt.show()

    return line_image


def get_point_in_voxels(pcd_np, rows, cols, lidar_range, voxel_size, pcd_with_idx):
    """ use indice in image to filter point cloud, then sample within it.
    Args:
        pcd_np: [N, 3]
        rows: [M,] non zero index -> row
        cols: [M,] non zero index -> col
        pcd_with_idx: [N, 4]
    Returns:
        points_select: [N_, 3]
    """
    [x_min, y_min, z_min, x_max, y_max, z_max] = lidar_range
    H = round((y_max - y_min) / voxel_size)
    W = round((x_max - x_min) / voxel_size)

    M = rows.shape[0]
    points_select = np.zeros((0,4))

    for i in range(M):
        # voxel_range = [x_min + voxel_size * cols[i],
        #                 y_min + voxel_size * rows[i], 
        #                 x_min + voxel_size * (cols[i] + 1),
        #                 y_min + voxel_size * (rows[i] + 1)]
        # points_in_voxel = mask_points_by_range(pcd_np, voxel_range)

        # if not points_in_voxel.any():
        #     continue    

        points_in_voxel = pcd_with_idx[pcd_with_idx[:,3]==(rows[i]*W + cols[i])]
        if not points_in_voxel.any():
            continue
        points_select = np.concatenate((points_select, points_in_voxel), axis=0)

    points_select = points_select[:,:3]
    
    return points_select


def get_keypoints(pcd_all_np, pcd_select_np, n_samples, mode = 'farthest'):
    if pcd_select_np.shape[0] >= n_samples:
        keypoints = get_random_subset(pcd_select_np, n_samples, mode)
    else:
        keypoints = get_random_subset(pcd_all_np, n_samples - pcd_select_np.shape[0], mode)
        keypoints = np.concatenate((keypoints, pcd_select_np), axis=0)

    return keypoints

def bev_sample(pcd_np, lidar_range, n_samples, mode, voxel_size=0.2, all_samples=False):
    """
    Args:
        pcd_np: 
            [N, 3] or [N, 4]
        lidar_range: 
            list len = 4 or len = 6, please use this to remove ground
        all_samples: 
            if True, not use n_samples to subsampling
    Returns:
        keypoints: np.ndarray
            [n_samples, 3]
    """

    pcd_np = pcd_np[:,:3]
    print(1)
    bev_img, pcd_with_idx = project_bev(pcd_np, lidar_range, voxel_size)
    print(2)
    lines = line_detection(bev_img)
    rows, cols = np.nonzero(lines)
    print(3)
    points_select = get_point_in_voxels(pcd_np, rows, cols, lidar_range, voxel_size, pcd_with_idx)
    print(4)

    if all_samples:
        keypoints = points_select
    else:
        keypoints = get_keypoints(pcd_np, points_select, n_samples, mode)

    print(keypoints.shape)

    return keypoints

def seq_generate():
    dirs = ["/GPFS/rhome/yifanlu/workspace/dataset/OPV2V/train",
            "/GPFS/rhome/yifanlu/workspace/dataset/OPV2V/validate"
            "/GPFS/rhome/yifanlu/workspace/dataset/OPV2V/test"]


    kp_store_path = '/GPFS/rhome/yifanlu/workspace/OpenCOOD/keypoints_file/bev_keypoints'
    lidar_range = [-140, -80, -1.5, 140, 80, 1]
    n_samples = 1500

    import os
    import opencood.utils.pcd_utils as pcd_utils

    for root_dir in dirs:
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        scenario_folders_name = sorted([x
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                               if os.path.isdir(
                    os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml')])
                timestamps = []

                # extract timestamp
                for file in yaml_files:
                    res = file.split('/')[-1]

                    timestamp = res.replace('.yaml', '')
                    timestamps.append(timestamp)


                for timestamp in timestamps:

                    yaml_file = os.path.join(cav_path,
                                                timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                                timestamp + '.pcd')

                    # when init the dataset, it read over all pcd files.
                    # it maybe slow, but no need to perform keypoint sampling for each time.\
                    kp_path = f"{kp_store_path}/{scenario_folders_name[i]}/{cav_id}/{timestamp}.npy"
                    kp_dir = kp_path.rsplit('/',1)[0] # before filename

                    if not os.path.exists(kp_dir):
                        os.makedirs(kp_dir)

                    if not os.path.exists(kp_path):
                        pcd_np = pcd_utils.pcd_to_np(lidar_file)
                        kp_file = bev_sample(pcd_np,
                                    lidar_range,
                                    n_samples,
                                    mode='uniform',
                                    all_samples=True)

                        np.save(kp_path, kp_file)


def parallel_generate(scenario_folder, scenario_folder_name):

    kp_store_path = '/GPFS/rhome/yifanlu/workspace/OpenCOOD/keypoints_file/bev_keypoints'
    lidar_range = [-140, -80, -1.5, 140, 80, 1]

    cav_list = sorted([x for x in os.listdir(scenario_folder)
                        if os.path.isdir(
            os.path.join(scenario_folder, x))])
    assert len(cav_list) > 0
    print(cav_list)

    # loop over all CAV data
    for (j, cav_id) in enumerate(cav_list):
        print(cav_id)
        # save all yaml files to the dictionary
        cav_path = os.path.join(scenario_folder, cav_id)

        yaml_files = \
            sorted([os.path.join(cav_path, x)
                    for x in os.listdir(cav_path) if
                    x.endswith('.yaml')])
        timestamps = []

        # extract timestamp
        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)


        for timestamp in timestamps:

            yaml_file = os.path.join(cav_path,
                                        timestamp + '.yaml')
            lidar_file = os.path.join(cav_path,
                                        timestamp + '.pcd')

            # when init the dataset, it read over all pcd files.
            # it maybe slow, but no need to perform keypoint sampling for each time.\
            target = [250,500,750,1000,1250,1500,2000,2500]
            kp_paths = [f"{kp_store_path}/{scenario_folder_name}/{cav_id}/{timestamp}.npy"]
            kp_paths += [f"{kp_store_path}_{n_samples}/{scenario_folder_name}/{cav_id}/{timestamp}.npy" for n_samples in target]
            flag = True
            for kp_path in kp_paths:
                if not os.path.exists(kp_path):
                    flag = False
            if flag:
                continue


            pcd_np = pcd_utils.pcd_to_np(lidar_file)[:,:3]

            all_keypoint = bev_sample(pcd_np,
                                lidar_range,
                                np.inf,
                                mode='uniform',
                                all_samples=True)

            kp_path = f"{kp_store_path}/{scenario_folder_name}/{cav_id}/{timestamp}.npy"
            kp_dir = kp_path.rsplit('/',1)[0] # before filename
            if not os.path.exists(kp_dir):
                os.makedirs(kp_dir)

            if not os.path.exists(kp_path):
                np.save(kp_path, all_keypoint)
                print(f"saving to {kp_path}")


            for n_samples in target:
                kp_path = f"{kp_store_path}_{n_samples}/{scenario_folder_name}/{cav_id}/{timestamp}.npy"
                kp_dir = kp_path.rsplit('/',1)[0] # before filename

                if not os.path.exists(kp_dir):
                    os.makedirs(kp_dir)
                
                select_keypoint = get_keypoints(pcd_np, all_keypoint, n_samples)

                if not os.path.exists(kp_path):
                    np.save(kp_path, select_keypoint)
                    print(f"saving to {kp_path}")




if __name__=="__main__":
    dirs = ["/GPFS/rhome/yifanlu/workspace/dataset/OPV2V/train",
            "/GPFS/rhome/yifanlu/workspace/dataset/OPV2V/validate",
            "/GPFS/rhome/yifanlu/workspace/dataset/OPV2V/test"]

    import os
    import opencood.utils.pcd_utils as pcd_utils

    scenario_folders = []
    scenario_folders_name = []

    for root_dir in dirs:
        scenario_folders += sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        scenario_folders_name += sorted([x
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
    
    

    scenario_folders = ['/GPFS/rhome/yifanlu/workspace/OpenCOOD/dataset_link/validate/2021_08_21_17_30_41']
    scenario_folders_name = ['2021_08_21_17_30_41']
    num = len(scenario_folders)

    for i in range(num):
        p = Process(target=parallel_generate, args=(scenario_folders[i],scenario_folders_name[i]))
        p.start()