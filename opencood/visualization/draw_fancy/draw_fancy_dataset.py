"""
 write a simple dataset to load data from opv2v
 to draw different pictures.

 it is almost the same with basedataset, but have no params.
 it will retrieve global/local informations.

 it suppose root folder contains only one scene
"""


import time
from torch.utils.data import Dataset
import os
from collections import OrderedDict
import opencood.utils.pcd_utils as pcd_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.transformation_utils import x1_to_x2, x_to_world
from opencood.utils.pose_utils import generate_noise

class SimpleDataset(Dataset):
    def __init__(self,
                 root_dir="/GPFS/rhome/yifanlu/workspace/OpenCOOD/opencood/visualization/draw_fancy/dataset"):
        
        print("Dataset dir:", root_dir)
        self.max_cav =5

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                               if os.path.isdir(
                    os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml')])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False
    def __len__(self):
        return self.len_record[-1]

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

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
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # todo: load camera image in the future version
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = \
                load_yaml(cav_content[timestamp_key]['yaml'])
            data[cav_id]['lidar_np'] = \
                pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

        return data

    def __getitem__(self, idx):
        """ Here we want to read the lidar_np, lidar_pose, 
            return point cloud in the global coordinate and the groundtruth R,T

        Args:
            base_data: dict
                data[cav_id] has key: 'params' and 'lidar_np'
                data[cav_id]['params']['lidar_pose'] : x,y,z,roll,yaw,pitch
                data[cav_id]['lidar_np']: [N, 4]

        Returns:
            To register cav1 to ego.
            lidar_in_ego is already aligned with ego point cloud

            lidar_in_ego_noisy(source) = tfm @ lidar_in_ego(target)
            target = tfm^(-1) @ source
        """        
        base_data_dict = self.retrieve_base_data(idx)


        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break


        # # transform all points to global coordinate
        # processed_data_dict = OrderedDict()

        # for cav_id, cav_content in base_data_dict.items():
        #     # ego serve as target, keep pc unchanged

        #     processed_data_dict[cav_id] = OrderedDict()
        #     lidar_pose = cav_content['params']['lidar_pose']

        #     T_ego_cav = x1_to_x2(lidar_pose, ego_lidar_pose)
        #     lidar_np = cav_content['lidar_np']
        #     lidar_np[:, 3] = 1 
        #     lidar_in_ego = (T_ego_cav @ lidar_np.T).T # [N, 4]

        #     if not cav_content['ego']:
        #         lidar_in_ego_noisy = (tfm @ lidar_in_ego.T).T
        #         processed_data_dict[cav_id]['lidar_projected'] = lidar_in_ego_noisy
        #         processed_data_dict[cav_id]['noise_tfm'] = tfm
        #     else:
        #         processed_data_dict[cav_id]['lidar_projected'] = lidar_in_ego

        #     processed_data_dict[cav_id]['lidar_pose'] = lidar_pose # no use.
        #     processed_data_dict[cav_id]['ego'] = cav_content['ego'] # no use.

        # return processed_data_dict
        return base_data_dict


if __name__ == '__main__':
    noisy = False
    dataset = SimpleDataset()
    idx = 0
    
    data_dict = dataset[idx]
    print(data_dict)
