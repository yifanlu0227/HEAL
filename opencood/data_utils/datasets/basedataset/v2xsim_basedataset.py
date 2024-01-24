# Author: Yangheng Zhao <zhaoyangheng-sjtu@sjtu.edu.cn>

import os
import pickle
from collections import OrderedDict
from typing import Dict
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor

class V2XSIMBaseDataset(Dataset):
    """
        First version.
        Load V2X-sim 2.0 using yifan lu's pickle file. 
        Only support LiDAR data. (even in hetero)
    """

    def __init__(self,
                 params: Dict,
                 visualize: bool = False,
                 train: bool = True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        if 'data_augment' in params: # late and early
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else: # intermediate
            self.data_augmentor = None

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir

        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        assert self.label_type in ['lidar', 'camera']

        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center

        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False
        
        with open(self.root_dir, 'rb') as f:
            dataset_info = pickle.load(f)
        self.dataset_info_pkl = dataset_info

        # TODO param: one as ego or all as ego?
        self.ego_mode = 'one'  # "all"

        self.reinitialize()

    def reinitialize(self):
        self.scene_database = OrderedDict()
        if self.ego_mode == 'one':
            self.len_record = len(self.dataset_info_pkl)
        else:
            raise NotImplementedError(self.ego_mode)

        for i, scene_info in enumerate(self.dataset_info_pkl):
            self.scene_database.update({i: OrderedDict()})
            cav_num = scene_info['agent_num']
            assert cav_num > 0

            if self.train:
                cav_ids = 1 + np.random.permutation(cav_num)
            else:
                cav_ids = list(range(1, cav_num + 1))
            

            for j, cav_id in enumerate(cav_ids):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break

                self.scene_database[i][cav_id] = OrderedDict()

                self.scene_database[i][cav_id]['ego'] = j==0

                self.scene_database[i][cav_id]['lidar'] = scene_info[f'lidar_path_{cav_id}']
                # need to delete this line is running in /GPFS
                self.scene_database[i][cav_id]['lidar'] = \
                    self.scene_database[i][cav_id]['lidar'].replace("/GPFS/rhome/yifanlu/workspace/dataset/v2xsim2-complete", "dataset/V2X-Sim-2.0")

                self.scene_database[i][cav_id]['params'] = OrderedDict()
                self.scene_database[i][cav_id][
                    'params']['lidar_pose'] = tfm_to_pose(
                        scene_info[f"lidar_pose_{cav_id}"]
                    )  # [x, y, z, roll, pitch, yaw]
                self.scene_database[i][cav_id]['params'][
                    'vehicles'] = scene_info[f'labels_{cav_id}'][
                        'gt_boxes_global']
                self.scene_database[i][cav_id]['params'][
                    'object_ids'] = scene_info[f'labels_{cav_id}'][
                        'gt_object_ids'].tolist()

    def __len__(self) -> int:
        return self.len_record

    @abstractmethod
    def __getitem__(self, index):
        pass

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """

        data = OrderedDict()
        # {
        #     'cav_id0':{
        #         'ego': bool,
        #         'params': {
        #           'lidar_pose': [x, y, z, roll, pitch, yaw],
        #           'vehicles':{
        #                   'id': {'angle', 'center', 'extent', 'location'},
        #                   ...
        #               }
        #           },# 包含agent位置信息和object信息
        #         'camera_data':,
        #         'depth_data':,
        #         'lidar_np':,
        #         ...
        #     }
        #     'cav_id1': ,
        #     ...
        # }
        scene = self.scene_database[idx]
        for cav_id, cav_content in scene.items():
            data[f'{cav_id}'] = OrderedDict()
            data[f'{cav_id}']['ego'] = cav_content['ego']

            data[f'{cav_id}']['params'] = cav_content['params']

            # load the corresponding data into the dictionary
            nbr_dims = 4  # x,y,z,intensity
            scan = np.fromfile(cav_content['lidar'], dtype='float32')
            points = scan.reshape((-1, 5))[:, :nbr_dims]
            data[f'{cav_id}']['lidar_np'] = points
            
            data[f'{cav_id}']['modality_name'] = 'm1' # Currently, there is only lidar's api.

        return data

    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_v2x(
            cav_contents, reference_lidar_pose)

    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        raise NotImplementedError()

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask