"""
 write a simple dataset to load data from v2xsim
 to draw different pictures.

 it is almost the same with basedataset, but have no params.
 it will retrieve global/local informations.

 it suppose root folder contains only one scene

 no ego, or everyone is ego. box information are saved for all 
"""


import time
from torch.utils.data import Dataset
import os
from collections import OrderedDict
import opencood.utils.pcd_utils as pcd_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, tfm_to_pose
from opencood.utils.pose_utils import generate_noise
import pickle
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self,
                 root_dir="/GPFS/rhome/yifanlu/workspace/OpenCOOD/opencood/visualization/draw_fancy/datasetv2x/v2xsim_infos_vis[31].pkl"):
        


        with open(root_dir, 'rb') as f:
            dataset_infos = pickle.load(f)  # dataset_infos is a list 

        self.max_cav = 5

        self.keyframe_database = OrderedDict()
        self.len_record = len(dataset_infos)
        agent_start = eval(min([i[-1] for i in dataset_infos[0].keys() if i.startswith("lidar_pose")]))
        self.agent_start = agent_start

        # loop over all keyframe.
        # data_info is one sample.
        for (i, data_info) in enumerate(dataset_infos):
            self.keyframe_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_num = data_info['agent_num']
            assert cav_num > 0

            # in one keyframe, loop all agent
            for cav_id in range(agent_start, cav_num+agent_start):

                self.keyframe_database[i][cav_id] = OrderedDict()
                self.keyframe_database[i][cav_id]['lidar'] = data_info[f'lidar_path_{cav_id}']  # maybe add camera in the future
                self.keyframe_database[i][cav_id]['params'] = OrderedDict()
                self.keyframe_database[i][cav_id]['params']['lidar_pose'] = tfm_to_pose(data_info[f"lidar_pose_{cav_id}"]) # tfm in data_info, turn to [x,y,z,roll,yaw,pitch]

                if cav_id == agent_start:
                    # let ego load the gt box, gt box is [x,y,z,dx,dy,dz,w,a,b,c]
                    self.keyframe_database[i][cav_id]['params']['vehicles'] = data_info['gt_boxes_global']
                    self.keyframe_database[i][cav_id]['params']['object_ids'] = data_info['gt_object_ids'].tolist()
                    self.keyframe_database[i][cav_id]['params']['sample_token'] = data_info['token']


    ### rewrite __len__ ###
    def __len__(self):
        return self.len_record

    ### rewrite retrieve_base_data ###
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
            lidar_np: (N, 4)
        """
        # we loop the accumulated length list to see get the scenario index
        keyframe = self.keyframe_database[idx]

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in keyframe.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['params'] = cav_content['params'] # lidar_pose, vehicles(gt_boxes), object_id(token)

            # load the corresponding data into the dictionary
            nbr_dims = 4 # x,y,z,intensity
            scan = np.fromfile(cav_content['lidar'], dtype='float32')
            points = scan.reshape((-1, 5))[:, :nbr_dims] 
            data[cav_id]['lidar_np'] = points


        return data


    def __getitem__(self, index):
        return self.retrieve_base_data(index)



if __name__ == '__main__':
    noisy = False
    dataset = SimpleDataset()
    idx = 0
    
    data_dict = dataset[idx]
    print(data_dict)
