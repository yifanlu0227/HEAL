"""
    A new version of proposal matcher. 
    It will collect voxel features, instead of keypoint features.
    TODO: Add agent-object pose graph optimization
"""

import torch
from torch import nn
import numpy as np
import spconv
from collections import OrderedDict
import opencood.utils.spconv_utils as spconv_utils
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from opencood.utils.box_utils import corner_to_center_torch, boxes_to_corners_3d, project_box3d, get_mask_for_boxes_within_range_torch
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import limit_period
from icecream import ic


class MatcherV2(nn.Module):

    def __init__(self, cfg, pc_range):
        super(MatcherV2, self).__init__()
        self.order = cfg['order']
        self.voxel_size = cfg['voxel_size']
        self.feature_source = cfg['feature_source'] # ['x_conv2', 'x_conv3', 'x_conv4']
        self.pc_range = pc_range
        self.sp_wraper = spconv_utils.warpSparseTensor()
        self.sp_merger = spconv_utils.MergeDuplicate("max")

    @torch.no_grad()
    def forward(self, data_dict):
        clusters, scores = self.clustering(data_dict)
        data_dict['boxes_fused'], data_dict[
            'scores_fused'] = self.cluster_fusion(clusters, scores)
        self.collect_voxel_feature(data_dict)
        return data_dict

    def clustering(self, data_dict):
        """
        Assign predicted boxes to clusters according to their ious with each other
        """
        clusters_batch = []
        scores_batch = []
        record_len = [int(cavnum) for cavnum in data_dict['record_len']] 
        lidar_poses = data_dict['lidar_pose'].cpu().numpy()

        # iterate each frame
        for i, cavnum in enumerate(record_len):
            cur_boxes_list = data_dict['det_boxes'][sum(record_len[:i]):sum(record_len[:i]) + cavnum]
            cur_boxes_list_ego = []
            # preserve ego boxes
            cur_boxes_list_ego.append(cur_boxes_list[0])
            # transform box to ego coordinate. [x,y,z,h,w,l,yaw]
            for agent_id in range(1, cavnum):
                tfm = x1_to_x2(lidar_poses[sum(record_len[:i])+agent_id], 
                                lidar_poses[sum(record_len[:i])])
                tfm = torch.from_numpy(tfm).to(cur_boxes_list[0].device).float()
                cur_boxes = cur_boxes_list[agent_id]
                cur_corners = boxes_to_corners_3d(cur_boxes, order=self.order)
                cur_corners_ego = project_box3d(cur_corners, tfm)
                cur_boxes_ego = corner_to_center_torch(cur_corners_ego, order=self.order)
                cur_boxes_list_ego.append(cur_boxes_ego)

            cur_boxes_list = cur_boxes_list_ego

            cur_scores_list = data_dict['det_scores'][sum(record_len[:i]):sum(record_len[:i]) + cavnum]
            cur_boxes_list = [b for b in cur_boxes_list if len(b) > 0]
            cur_scores_list = [s for s in cur_scores_list if len(s) > 0]

            if len(cur_scores_list) == 0:
                clusters_batch.append([torch.Tensor([0.0, 0.0, 0.0, 1.6, 2.0, 4.0, 0]). #
                                      to(torch.device('cuda:0')).view(1, 7)])
                scores_batch.append([torch.Tensor([0.01]).to(torch.device('cuda:0')).view(-1)])
                continue

            pred_boxes_cat = torch.cat(cur_boxes_list, dim=0)
            pred_boxes_cat[:, -1] = limit_period(pred_boxes_cat[:, -1])
            pred_scores_cat = torch.cat(cur_scores_list, dim=0)

            ious = boxes_iou3d_gpu(pred_boxes_cat, pred_boxes_cat)
            cluster_indices = torch.zeros(len(ious)).int()
            cur_cluster_id = 1

            # cluster proposals
            while torch.any(cluster_indices == 0):
                cur_idx = torch.where(cluster_indices == 0)[0][0] # find the idx of the first pred which is not assigned yet
                cluster_indices[torch.where(ious[cur_idx] > 0.1)[0]] = cur_cluster_id
                cur_cluster_id += 1

            clusters = []
            scores = []

            for j in range(1, cur_cluster_id):
                clusters.append(pred_boxes_cat[cluster_indices==j])
                scores.append(pred_scores_cat[cluster_indices==j])

            clusters_batch.append(clusters)
            scores_batch.append(scores)

        return clusters_batch, scores_batch

    def cluster_fusion(self, clusters, scores):
        """
        Merge boxes in each cluster with scores as weights for merging.
        TODO: change to select the proposal with highest score? And then adjust the proposal
        """
        boxes_fused = []
        scores_fused = []
        for cl, sl in zip(clusters, scores): # cl, sl are clusters and scores within one sample
            for c, s in zip(cl, sl): # one sample (cl) has many clusters (c), c,s,a correspond to one cluster
                # reverse direction for non-dominant direction of boxes
                dirs = c[:, -1]
                max_score_idx = torch.argmax(s)
                dirs_diff = torch.abs(dirs - dirs[max_score_idx].item())
                lt_pi = (dirs_diff > np.pi).int()
                dirs_diff = dirs_diff * (1 - lt_pi) + (
                            2 *  np.pi - dirs_diff) * lt_pi
                score_lt_half_pi = s[dirs_diff > np.pi / 2].sum()  # larger than
                score_set_half_pi = s[
                    dirs_diff <= np.pi / 2].sum()  # small equal than
                # select larger scored direction as final direction
                if score_lt_half_pi <= score_set_half_pi:
                    dirs[dirs_diff > np.pi / 2] +=  np.pi
                else:
                    dirs[dirs_diff <= np.pi / 2] +=  np.pi
                    
                dirs = limit_period(dirs)
                s_normalized = s / s.sum()
                sint = torch.sin(dirs) * s_normalized
                cost = torch.cos(dirs) * s_normalized
                theta = torch.atan2(sint.sum(), cost.sum()).view(1, )
                center_dim = c[:, :-1] * s_normalized[:, None]
                
                boxes_fused.append(torch.cat([center_dim.sum(dim=0), theta]))
                s_sorted = torch.sort(s, descending=True).values
                s_fused = 0
                for i, ss in enumerate(s_sorted):
                    s_fused += ss ** (i + 1)
                s_fused = torch.tensor([min(s_fused, 1.0)], device=s.device)
                scores_fused.append(s_fused)

        assert len(boxes_fused) > 0
        boxes_fused = torch.stack(boxes_fused, dim=0)
        box_num_sample = [len(c) for c in clusters] # in a batch, each sample has how many boxes
        boxes_fused = [
            boxes_fused[sum(box_num_sample[:i]):sum(box_num_sample[:i]) + l] for i, l
            in enumerate(box_num_sample)]
        scores_fused = torch.stack(scores_fused, dim=0)
        scores_fused = [
            scores_fused[sum(box_num_sample[:i]):sum(box_num_sample[:i]) + l] for
            i, l in enumerate(box_num_sample)]

        for i in range(len(boxes_fused)):
            corners3d = boxes_to_corners_3d(boxes_fused[i], order=self.order)
            mask = get_mask_for_boxes_within_range_torch(corners3d, self.pc_range)
            boxes_fused[i] = boxes_fused[i][mask]
            scores_fused[i] = scores_fused[i][mask]

        return boxes_fused, scores_fused

    def retrieve_cav_sp_feature(self, sp_feature, agent_pos):
        features = sp_feature.features
        indices = sp_feature.indices
        mask = indices[:, 0] == agent_pos

        new_indices = indices.clone()
        new_indices[:, 0] = 0

        return spconv.SparseConvTensor(features[mask], new_indices[mask], sp_feature.spatial_shape, batch_size=1)

    def collect_voxel_feature(self, data_dict):
        """
        1. collect features by feauture_source
        2. convert sparse features to dense features
        3. warp dense feature map and merge them
        4. convert dense feature map to sparse 
        """
        
        multi_scale_3d_features = data_dict['multi_scale_3d_features'] # sum(record_len), but SparseConvTensor
        multi_scale_3d_stride = data_dict['multi_scale_3d_strides']
        data_dict['multi_scale_3d_features_fused'] = OrderedDict()
        lidar_poses = data_dict['lidar_pose'].cpu().numpy()
        device = data_dict['lidar_pose'].device

        for srcname in self.feature_source:
            start_agent_pos = 0
            sp_feature = multi_scale_3d_features[srcname]
            stride = multi_scale_3d_stride[srcname]
            voxel_size = torch.tensor(self.voxel_size).to(device)
            voxel_size *= stride
            sp_tensor_fused_list = [] # each sample
            # ic(srcname)
            # ic(sp_feature.indices)

            for idx, cavnum in enumerate(data_dict['record_len']):
                # each sample
                sp_tensor_cav_list = [self.retrieve_cav_sp_feature(sp_feature, start_agent_pos)] # each cav
            
                for agent_id in range(1, cavnum):
                    sp_tensor_cav = self.retrieve_cav_sp_feature(sp_feature, start_agent_pos+agent_id)

                    if data_dict['proj_first'] is False:
                        tfm = x1_to_x2(lidar_poses[start_agent_pos+agent_id], lidar_poses[start_agent_pos])
                        tfm = torch.from_numpy(tfm).to(device) # cav_to_ego
                        sp_tensor_warp = self.sp_wraper(sp_tensor_cav, tfm, voxel_size, self.pc_range)
                        sp_tensor_cav = sp_tensor_warp

                    sp_tensor_cav_list.append(sp_tensor_cav)
                    
                sp_tensor_fused = spconv_utils.fuseSparseTensor(sp_tensor_cav_list) # only fuse
                sp_tensor_fused = self.sp_merger(sp_tensor_fused)
                sp_tensor_fused.indices[:, 0] = idx # batch_idx set to sample idx

                # sp_tensor_fused = self.retrieve_cav_sp_feature(sp_feature, start_agent_pos)
                # sp_tensor_fused.indices[:, 0] = idx

                sp_tensor_fused_list.append(sp_tensor_fused)

                start_agent_pos += cavnum


        

            new_features = torch.cat([x.features for x in sp_tensor_fused_list], dim=0)
            new_indice = torch.cat([x.indices for x in sp_tensor_fused_list], dim=0)
            features_fused = spconv.SparseConvTensor(new_features, new_indice, sp_tensor_fused_list[0].spatial_shape,
                                            len(data_dict['record_len']),  sp_tensor_fused_list[0].grid)

            data_dict['multi_scale_3d_features_fused'][srcname] = features_fused

            # ic("test dense feature")
            
            # # ic(features_fused.features)
            # ic(features_fused.indices.shape)
            # ic(features_fused.features.shape)
            # ic(features_fused.spatial_shape)
            # ic(features_fused.indices[:,1].min())
            # ic(features_fused.indices[:,2].min())
            # ic(features_fused.indices[:,3].min())
            # ic(features_fused.indices[:,1].max())
            # ic(features_fused.indices[:,2].max())
            # ic(features_fused.indices[:,3].max())
            # # dense_feature = features_fused.dense()
            # # ic(dense_feature.shape)
