# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Common utilities
"""

import numpy as np
import torch
from shapely.geometry import Polygon
import json
import pickle
from collections import OrderedDict

def update_dict(d1,d2):
    '''
    credit: https://github.com/yutu-75/update_dict/blob/main/update_dict/update_dict.py

    :param d1: Default nested dictionary,默认嵌套字典;
    :param d2: Updated dictionary 需要更新的字典;
    :return d1:
    Return a dict merged from default and custom
    # >>> recursive_update('a', 'b')
    Traceback (most recent call last):
        ...
    TypeError: Params of update_dict should be dicts
    # >>> update_dict({'a':{"b":{"c":{"d"}}},"e":{"e1":{"e5":'qwq'}},"e5": {},"ss":"1111"},
    {"e5":'www',"ss":"ssss",'c':{},'ss1':'ss'})
    {'a': {'b': {'c': {}}}, 'e': {'e1': {'e5': 'www'}}, 'e5': 'www', 'ss': 'ssss'
    # >>> update_dict({'a':{"b":{"c":{"d":'c'}}},"e":{"e1":{"e5":'qwq'}},"e5": {},"ss":"1111"},{"d":'www'})
    {'a': {'b': {'c': {'d': 'www'}}}, 'e': {'e1': {'e5': 'qwq'}}, 'e5': {}, 'ss': '1111'}
    # >>> update_dict({'a': {'c': 1, 'd': {}}, 'b': 4}, {'a': 2})
    {'a': 2, 'b': 4}
    '''

    if not isinstance(d1, dict) or not isinstance(d2, dict):
        raise TypeError('Params of update_dict should be dicts')
    for i in d1:
        if d2.get(i, None) is not None:
            d1[i] = d2[i]
        if isinstance(d1[i], dict):
            update_dict(d1[i],d2)
    return d1


def merge_features_to_dict(processed_feature_list, merge=None):
    """
    Merge the preprocessed features from different cavs to the same
    dictionary.

    Parameters
    ----------
    processed_feature_list : list
        A list of dictionary containing all processed features from
        different cavs.
    merge : "stack" or "cat". used for images

    Returns
    -------
    merged_feature_dict: dict
        key: feature names, value: list of features.
    """

    if len(processed_feature_list) == 0:
        return None

    merged_feature_dict = OrderedDict()

    for i in range(len(processed_feature_list)):
        for feature_name, feature in processed_feature_list[i].items():
            if feature_name not in merged_feature_dict:
                merged_feature_dict[feature_name] = []
            if isinstance(feature, list):
                merged_feature_dict[feature_name] += feature
            else:
                merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
    
    # stack them
    # it usually happens when merging cavs images -> v.shape = [N, Ncam, C, H, W]
    # cat them
    # it usually happens when merging batches cav images -> v is a list [(N1+N2+...Nn, Ncam, C, H, W))]
    if merge=='stack': 
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.stack(features, dim=0)
    elif merge=='cat':
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.cat(features, dim=0)

    return merged_feature_dict

def load_pkl_files(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data

def limit_period(val, offset=0.5, period=2*np.pi):
    """
    continous part: 
    [0 - period * offset, period - period * offset)
    """
    # 首先，numpy格式数据转换为torch格式
    val, is_numpy = check_numpy_to_torch(val)
    # 将方位角限制在[-pi, pi]
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def check_torch_to_numpy(x):
    if isinstance(x, torch.tensor):
        return x.cpu().numpy(), True
    return x, False


def check_contain_nan(x):
    if isinstance(x, dict):
        return any(check_contain_nan(v) for k, v in x.items())
    if isinstance(x, list):
        return any(check_contain_nan(itm) for itm in x)
    if isinstance(x, int) or isinstance(x, float):
        return False
    if isinstance(x, np.ndarray):
        return np.any(np.isnan(x))
    return torch.any(x.isnan()).detach().cpu().item()


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), radians, angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def rotate_points_along_z_2d(points, angle):
    """
    Rorate the points along z-axis.
    Parameters
    ----------
    points : torch.Tensor / np.ndarray
        (N, 2).
    angle : torch.Tensor / np.ndarray
        (N,)

    Returns
    -------
    points_rot : torch.Tensor / np.ndarray
        Rorated points with shape (N, 2)

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    # (N, 2, 2)
    rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2,
                                                                    2).float()
    points_rot = torch.einsum("ik, ikj->ij", points.float(), rot_matrix)
    return points_rot.numpy() if is_numpy else points_rot


def remove_ego_from_objects(objects, ego_id):
    """
    Avoid adding ego vehicle to the object dictionary.

    Parameters
    ----------
    objects : dict
        The dictionary contained all objects.

    ego_id : int
        Ego id.
    """
    if ego_id in objects:
        del objects[ego_id]


def retrieve_ego_id(base_data_dict):
    """
    Retrieve the ego vehicle id from sample(origin format).

    Parameters
    ----------
    base_data_dict : dict
        Data sample in origin format.

    Returns
    -------
    ego_id : str
        The id of ego vehicle.
    """
    ego_id = None

    for cav_id, cav_content in base_data_dict.items():
        if cav_content['ego']:
            ego_id = cav_id
            break
    return ego_id


def compute_iou(box, boxes):
    """
    Compute iou between box and boxes list
    Parameters
    ----------
    box : shapely.geometry.Polygon
        Bounding box Polygon.

    boxes : list
        List of shapely.geometry.Polygon.

    Returns
    -------
    iou : np.ndarray
        Array of iou between box and boxes.

    """
    # Calculate intersection areas
    if np.any(np.array([box.union(b).area for b in boxes])==0):
        print('debug')
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)


def convert_format(boxes_array):
    """
    Convert boxes array to shapely.geometry.Polygon format.
    Parameters
    ----------
    boxes_array : np.ndarray
        (N, 4, 2) or (N, 8, 3).

    Returns
    -------
        list of converted shapely.geometry.Polygon object.

    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in
                boxes_array]
    return np.array(polygons)


def torch_tensor_to_numpy(torch_tensor):
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    A numpy array.
    """
    return torch_tensor.numpy() if not torch_tensor.is_cuda else \
        torch_tensor.cpu().detach().numpy()


def get_voxel_centers(voxel_coords,
                      downsample_times,
                      voxel_size,
                      point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device) # 初始化结果 (8, 21, 800, 704)
    ndim = indices.shape[-1] # 获取坐标维度 4
    flattened_indices = indices.view(-1, ndim) # 将坐标展平 (204916, 4)
    # 以下两步是经典操作
    slices = [flattened_indices[:, i] for i in range(ndim)] # 分成4个list
    ret[slices] = point_inds # 将voxel的索引写入对应位置
    return ret

def generate_voxel2pinds(sparse_tensor):
    """
    计算有效voxel在原始空间shape中的索引
    """
    device = sparse_tensor.indices.device # 获取device
    batch_size = sparse_tensor.batch_size # 获取batch_size
    spatial_shape = sparse_tensor.spatial_shape # 获取空间形状 (21, 800, 704)
    indices = sparse_tensor.indices.long() # 获取索引
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32) # 生成索引 (204916,)
    output_shape = [batch_size] + list(spatial_shape) # 计算输出形状 (8, 21, 800, 704)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor
