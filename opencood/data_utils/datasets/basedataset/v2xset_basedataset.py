# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset

# All the same as OPV2V
class V2XSETBaseDataset(OPV2VBaseDataset):
    def __init__(self, params, visulize, train=True):
        super().__init__(params, visulize, train)

        if self.load_camera_file is True: # '2021_09_09_13_20_58'. This scenario has only 3 camera files?
            scenario_folders_new = [x for x in self.scenario_folders if '2021_09_09_13_20_58' not in x]
            self.scenario_folders = scenario_folders_new
            self.reinitialize()

        self.use_hdf5 = False

    def generate_object_center_camera(self, 
                                cav_contents, 
                                reference_lidar_pose):
        """
        Since V2XSet has not release bev_visiblity map, we can only filter object by range.

        Suppose the detection range of camera is within 50m
        """
        return self.post_processor.generate_object_center_v2xset_camera(
            cav_contents, reference_lidar_pose
        )