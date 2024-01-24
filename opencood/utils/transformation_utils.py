# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transformation utils
"""

from re import X
import numpy as np
import torch
from icecream import ic
from pyquaternion import Quaternion
from opencood.utils.common_utils import check_numpy_to_torch

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

def get_pairwise_transformation(base_data_dict, max_cav, proj_first):
    """
    Get pair-wise transformation matrix accross different agents.

    Parameters
    ----------
    base_data_dict : dict
        Key : cav id, item: transformation matrix to ego, lidar points.

    max_cav : int
        The maximum number of cav, default 5

    Return
    ------
    pairwise_t_matrix : np.array
        The pairwise transformation matrix across each cav.
        shape: (L, L, 4, 4), L is the max cav number in a scene
        pairwise_t_matrix[i, j] is Tji, i_to_j
    """
    pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

    if proj_first:
        # if lidar projected to ego first, then the pairwise matrix
        # becomes identity
        # no need to warp again in fusion time.

        # pairwise_t_matrix[:, :] = np.identity(4)
        return pairwise_t_matrix
    else:
        t_list = []

        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            lidar_pose = cav_content['params']['lidar_pose']
            t_list.append(x_to_world(lidar_pose))  # Twx

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i != j:
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                    pairwise_t_matrix[i, j] = t_matrix

    return pairwise_t_matrix

def normalize_pairwise_tfm(pairwise_t_matrix, H, W, discrete_ratio, downsample_rate=1):
    """
    normalize the pairwise transformation matrix to affine matrix need by torch.nn.functional.affine_grid()
    Args:
        pairwise_t_matrix: torch.tensor
            [B, L, L, 4, 4], B batchsize, L max_cav
        H: num.
            Feature map height
        W: num.
            Feature map width
        discrete_ratio * downsample_rate: num.
            One pixel on the feature map corresponds to the actual physical distance

    Returns:
        affine_matrix: torch.tensor
            [B, L, L, 2, 3]
    """

    affine_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
    affine_matrix[...,0,1] = affine_matrix[...,0,1] * H / W
    affine_matrix[...,1,0] = affine_matrix[...,1,0] * W / H
    affine_matrix[...,0,2] = affine_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
    affine_matrix[...,1,2] = affine_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2

    return affine_matrix

def pose_to_tfm(pose):
    """ Transform batch of pose to tfm
    Args:
        pose: torch.Tensor or np.ndarray
            [N, 3], x, y, yaw, in degree
            [N, 6], x, y, z, roll, yaw, pitch, in degree

            roll and pitch follows carla coordinate
    Returns:
        tfm: torch.Tensor
            [N, 4, 4] 
    """

    pose_tensor, is_np = check_numpy_to_torch(pose)
    pose = pose_tensor


    if pose.shape[1] == 3:
        N = pose.shape[0]
        x = pose[:,0]
        y = pose[:,1]
        yaw = pose[:,2]

        tfm = torch.eye(4, device=pose.device).view(1,4,4).repeat(N,1,1)
        tfm[:,0,0] = torch.cos(torch.deg2rad(yaw))
        tfm[:,0,1] = - torch.sin(torch.deg2rad(yaw))
        tfm[:,1,0] = torch.sin(torch.deg2rad(yaw))
        tfm[:,1,1] = torch.cos(torch.deg2rad(yaw))
        tfm[:,0,3] = x
        tfm[:,1,3] = y

    elif pose.shape[1] == 6:
        N = pose.shape[0]
        x = pose[:,0]
        y = pose[:,1]
        z = pose[:,2]
        roll = pose[:,3]
        yaw = pose[:,4]
        pitch = pose[:,5]

        c_y = torch.cos(torch.deg2rad(yaw))
        s_y = torch.sin(torch.deg2rad(yaw))
        c_r = torch.cos(torch.deg2rad(roll))
        s_r = torch.sin(torch.deg2rad(roll))
        c_p = torch.cos(torch.deg2rad(pitch))
        s_p = torch.sin(torch.deg2rad(pitch))

        tfm = torch.eye(4, device=pose.device).view(1,4,4).repeat(N,1,1)

        # translation matrix
        tfm[:, 0, 3] = x
        tfm[:, 1, 3] = y
        tfm[:, 2, 3] = z

        # rotation matrix
        tfm[:, 0, 0] = c_p * c_y
        tfm[:, 0, 1] = c_y * s_p * s_r - s_y * c_r
        tfm[:, 0, 2] = -c_y * s_p * c_r - s_y * s_r
        tfm[:, 1, 0] = s_y * c_p
        tfm[:, 1, 1] = s_y * s_p * s_r + c_y * c_r
        tfm[:, 1, 2] = -s_y * s_p * c_r + c_y * s_r
        tfm[:, 2, 0] = s_p
        tfm[:, 2, 1] = -c_p * s_r
        tfm[:, 2, 2] = c_p * c_r

    if is_np:
        tfm = tfm.numpy()

    return tfm




def tfm_to_pose(tfm: np.ndarray):
    """
    turn transformation matrix to [x, y, z, roll, yaw, pitch]
    we use radians format.
    tfm is pose in transformation format, and XYZ order, i.e. roll-pitch-yaw
    """
    # There forumlas are designed from x_to_world, but equal to the one below.
    yaw = np.degrees(np.arctan2(tfm[1,0], tfm[0,0])) # clockwise in carla
    roll = np.degrees(np.arctan2(-tfm[2,1], tfm[2,2])) # but counter-clockwise in carla
    pitch = np.degrees(np.arctan2(tfm[2,0], ((tfm[2,1]**2 + tfm[2,2]**2) ** 0.5)) ) # but counter-clockwise in carla


    # These formulas are designed for consistent axis orientation
    # yaw = np.degrees(np.arctan2(tfm[1,0], tfm[0,0])) # clockwise in carla
    # roll = np.degrees(np.arctan2(tfm[2,1], tfm[2,2])) # but counter-clockwise in carla
    # pitch = np.degrees(np.arctan2(-tfm[2,0], ((tfm[2,1]**2 + tfm[2,2]**2) ** 0.5)) ) # but counter-clockwise in carla

    # roll = - roll
    # pitch = - pitch

    x, y, z = tfm[:3,3]
    return([x, y, z, roll, yaw, pitch])

def tfm_to_xycs_torch(tfm: torch.Tensor):
    """
        similar to tfm_to_pose_torch,
        return x/y/cos(yaw)/sin(yaw)
    """
    x = tfm[:,0,3]
    y = tfm[:,1,3]
    
    cos = tfm[:,0,0]
    sin = tfm[:,1,0]

    pose = torch.stack([x,y,cos,sin]).T # (N, 4)

    return pose

def xycs_to_tfm_torch(xycs: torch.Tensor):
    """
        Args: xycs
            [N, 4]
    """
    N = xycs.shape[0]
    tfm = torch.eye(4, device=xycs.device).view(1,4,4).repeat(N,1,1)

    x, y, cos, sin = xycs[:,0], xycs[:,1], xycs[:,2], xycs[:,3]

    tfm[:,0,0] = cos
    tfm[:,0,1] = - sin
    tfm[:,1,0] = sin
    tfm[:,1,1] = cos
    tfm[:,0,3] = x
    tfm[:,1,3] = y

    return tfm

def tfm_to_pose_torch(tfm: torch.Tensor, dof: int):
    """
    turn transformation matrix to [x, y, z, roll, yaw, pitch]
    we use degree format.
    tfm is pose in transformation format, and XYZ order, i.e. roll-pitch-yaw

    Args:
        tfm: [N, 4, 4]
        dof: 3 or 6
    Returns:
        6dof pose: [N, 6]
    """

    # There forumlas are designed from x_to_world, but equal to the one below.
    yaw = torch.rad2deg(torch.atan2(tfm[:,1,0], tfm[:,0,0])) # clockwise in carla
    roll = torch.rad2deg(torch.atan2(-tfm[:,2,1], tfm[:,2,2])) # but counter-clockwise in carla
    pitch = torch.rad2deg(torch.atan2(tfm[:,2,0], (tfm[:,2,1]**2 + tfm[:,2,2]**2) ** 0.5)) # but counter-clockwise in carla

    # These formulas are designed for consistent axis orientation
    # yaw = torch.rad2deg(torch.atan2(tfm[:,1,0], tfm[:,0,0])) # clockwise in carla
    # roll = torch.rad2deg(torch.atan2(tfm[:,2,1], tfm[:,2,2])) # but counter-clockwise in carla
    # pitch = torch.rad2deg(torch.atan2(-tfm[:,2,0], (tfm[:,2,1]**2 + tfm[:,2,2]**2) ** 0.5)) # but counter-clockwise in carla

    # roll = - roll
    # pitch = - pitch

    x = tfm[:,0,3]
    y = tfm[:,1,3]
    z = tfm[:,2,3]
    
    if dof == 6:
        pose = torch.stack([x,y,z,roll,yaw,pitch]).T # (N, 6)
    elif dof == 3:
        pose = torch.stack([x,y,yaw]).T
    else:
        raise("Only support returning 3dof/6dof pose.")

    return pose


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system
    Also is the pose in world coordinate: T_world_x

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch], degree

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)

    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2. T_x2_x1

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

        yaw, pitch, roll in degree

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1) # wP = x1_to_world * 1P, so x1_to_world is Tw1
    x2_to_world = x_to_world(x2) # Tw2
    world_to_x2 = np.linalg.inv(x2_to_world) # T2w

    transformation_matrix = np.dot(world_to_x2, x1_to_world) # T2w * Tw1 = T21
    return transformation_matrix


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous


def get_pairwise_transformation_torch(lidar_poses, max_cav, record_len, dof):
    """
    Get pair-wise transformation matrix accross different agents.
    Designed for batch data

    Parameters
    ----------
    lidar_poses : tensor, [N, 3] or [N, 6]
        3 or 6 dof pose of lidar.

    max_cav : int
        The maximum number of cav, default 5

    record: list
        shape (B)

    dof: int, 3 or 6

    Return
    ------
    pairwise_t_matrix : np.array
        The pairwise transformation matrix across each cav.
        shape: (B, L, L, 4, 4), L is the max cav number in a scene
        pairwise_t_matrix[i, j] is Tji, i_to_j
    """
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    B = len(record_len)
    lidar_poses_list = regroup(lidar_poses, record_len)

    pairwise_t_matrix = torch.eye(4, device=lidar_poses.device).view(1,1,1,4,4).repeat(B, max_cav, max_cav, 1, 1) # (B, L, L, 4, 4)
    # save all transformation matrix in a list in order first.
    for b in range(B):
        lidar_poses = lidar_poses_list[b]  # [N_cav, 3] or [N_cav, 6]. 
        t_list = pose_to_tfm(lidar_poses)  # Twx, [N_cav, 4, 4]

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i != j:
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    t_matrix = torch.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                    pairwise_t_matrix[b][i, j] = t_matrix

    return pairwise_t_matrix


def get_relative_transformation(lidar_poses):
    """
    Args:
        lidar_pose:  np.ndarray
            [N, dof], lidar pose in world coordinate
            N is the agent number, dof is 3/6.

            [x, y, z, roll, yaw, pitch], degree
        
    Returns:
        relative transformation, in ego's coordinate
    """
    N = lidar_poses.shape[0]
    dof = lidar_poses.shape[1]

    if dof == 3:
        full_lidar_poses = np.zeros((N, 6))
        full_lidar_poses[:,[0,1,4]] = lidar_poses
        lidar_poses = full_lidar_poses

    relative_t_matrix = np.eye(4).reshape(1,4,4).repeat(N, axis=0)  # [N, 4, 4]
    for i in range(1, N):
        relative_t_matrix[i] = x1_to_x2(lidar_poses[i], lidar_poses[0])
    
    return relative_t_matrix



def muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C):
    rotationA2B = np.array(rotationA2B).reshape(3, 3)
    rotationB2C = np.array(rotationB2C).reshape(3, 3)
    rotation = np.dot(rotationB2C, rotationA2B)
    translationA2B = np.array(translationA2B).reshape(3, 1)
    translationB2C = np.array(translationB2C).reshape(3, 1)
    translation = np.dot(rotationB2C, translationA2B) + translationB2C

    return rotation, translation


def veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file):
    matrix = np.empty([4,4])
    rotationA2B = lidar_to_novatel_json_file["transform"]["rotation"]
    translationA2B = lidar_to_novatel_json_file["transform"]["translation"]
    rotationB2C = novatel_to_world_json_file["rotation"]
    translationB2C = novatel_to_world_json_file["translation"]
    rotation,translation = muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C)
    matrix[0:3, 0:3] = rotation
    matrix[:, 3][0:3] = np.array(translation)[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1
    
    return matrix

def inf_side_rot_and_trans_to_trasnformation_matrix(json_file,system_error_offset):
    matrix = np.empty([4,4])
    matrix[0:3, 0:3] = json_file["rotation"]
    translation = np.array(json_file["translation"])
    translation[0][0] = translation[0][0] + system_error_offset["delta_x"]
    translation[1][0] = translation[1][0] + system_error_offset["delta_y"]  #为啥有[1][0]??? --> translation是(3,1)的
    matrix[:, 3][0:3] = translation[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1

    return matrix

def rot_and_trans_to_trasnformation_matrix(json_file):
    matrix = np.empty([4,4])
    matrix[0:3, 0:3] = json_file["rotation"]
    matrix[:, 3][0:3] = np.array(json_file["translation"])[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1

    return matrix


def test():
    random_pose = np.random.randn(6)
    tfm = x_to_world(random_pose)
    pose_result = tfm_to_pose(tfm)
    tfm2 = x_to_world(pose_result)

    print(random_pose)
    print(pose_result)
    print()
    print(tfm)
    print(tfm2)

if __name__ == "__main__":
    test()