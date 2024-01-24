"""
    A new version of proposal matcher. 
    It will collect BEV features, instead of keypoint features.
    TODO: Add agent-object pose graph optimization

"""

import torch
from torch import nn
import numpy as np
import spconv
from collections import OrderedDict
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from opencood.utils.box_utils import corner_to_center_torch, boxes_to_corners_3d, project_box3d, get_mask_for_boxes_within_range_torch
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import limit_period
from icecream import ic
from itertools import compress


class MatcherV3(nn.Module):
    def __init__(self, cfg, pc_range):
        super(MatcherV3, self).__init__()
        self.order = cfg['order']
        self.pc_range = pc_range
        self.enlarge_ratio = cfg.get("enlarge_ratio", 1)

    @torch.no_grad()
    def forward(self, data_dict):
        clusters, scores, agentid_batch, view_vector_batch = self.clustering(data_dict)
        data_dict['boxes_fused'], data_dict['scores_fused'], data_dict['agentid_fused'], data_dict['view_vector_fused'] \
                = self.cluster_fusion(clusters, scores, agentid_batch, view_vector_batch)

        self.get_roi_from_box(data_dict) # ['roi_fused']
        return data_dict

    def clustering(self, data_dict):
        """
        Assign predicted boxes to clusters according to their ious with each other.
        Assign the order to boxes, belong to which agent 

        Returns:
            clusters_batch: [batch1, batch2, batch3, ...], 
                where batch1 = [[box1_in_cluster1, box2_in_cluster1, box3_in_cluster1], [box1_in_cluster2, box2_in_cluster2], ...]
        """
        clusters_batch = []
        scores_batch = []
        agentid_batch = []
        view_vector_batch = []

        record_len = [int(cavnum) for cavnum in data_dict['record_len']] 
        lidar_poses = data_dict['lidar_pose'].cpu().numpy()

        # iterate each frame
        for i, cavnum in enumerate(record_len):
            cur_boxes_list = data_dict['det_boxes'][sum(record_len[:i]):sum(record_len[:i]) + cavnum] # within one sample, different cav
            cur_boxes_list_ego = []
            cur_agentid_list = []
            cur_view_vector_list = []
            # preserve ego boxes
            cur_boxes_list_ego.append(cur_boxes_list[0])
            cur_agentid_list.append(torch.tensor([sum(record_len[:i]) + 0] * len(cur_boxes_list[0])))

            ### view vector ####
            cur_boxes = cur_boxes_list[0]
            view_angle = torch.atan2(cur_boxes[:, 1], cur_boxes[:, 0]) - cur_boxes[:,6] # view angle
            view_angle = limit_period(view_angle) # normalized view angle
            distance = (cur_boxes[:, 0] ** 2 + cur_boxes[:, 1] ** 2) ** 0.5
            view_vector = torch.stack([view_angle, distance], dim=-1) # [proposalnum, 2]
            cur_view_vector_list.append(view_vector)
            ####################

            # transform box to ego coordinate. [x,y,z,h,w,l,yaw]
            # especially proj first is false
            for agent_id in range(1, cavnum):
                tfm = x1_to_x2(lidar_poses[sum(record_len[:i])+agent_id], 
                                lidar_poses[sum(record_len[:i])])
                tfm = torch.from_numpy(tfm).to(cur_boxes_list[0].device).float()
                cur_boxes = cur_boxes_list[agent_id]
                cur_corners = boxes_to_corners_3d(cur_boxes, order=self.order)

                ### view vector ####
                view_angle = torch.atan2(cur_boxes[:, 1], cur_boxes[:, 0]) - cur_boxes[:,6] # view angle
                view_angle = limit_period(view_angle) # normalized view angle
                distance = (cur_boxes[:, 0] ** 2 + cur_boxes[:, 1] ** 2) ** 0.5
                view_vector = torch.stack([view_angle, distance], dim=-1) # [proposalnum, 2]
                ####################

                cur_corners_ego = project_box3d(cur_corners, tfm)
                cur_boxes_ego = corner_to_center_torch(cur_corners_ego, order=self.order)
                cur_boxes_list_ego.append(cur_boxes_ego)
                cur_agentid_list.append(torch.tensor([sum(record_len[:i]) + agent_id] * len(cur_boxes_ego)))
                cur_view_vector_list.append(view_vector)


            cur_boxes_list = cur_boxes_list_ego
            cur_scores_list = data_dict['det_scores'][sum(record_len[:i]):sum(record_len[:i]) + cavnum]

            
            cur_boxes_list = [b for b in cur_boxes_list if len(b) > 0]
            cur_scores_list = [s for s in cur_scores_list if len(s) > 0]
            cur_agentid_list = [a for a in cur_agentid_list if len(a) > 0]
            cur_view_vector_list = [v for v in cur_view_vector_list if len(v) > 0]

            if len(cur_scores_list) == 0:
                clusters_batch.append([torch.Tensor([0.0, 0.0, 0.0, 1.6, 2.0, 4.0, 0]). # hwl
                                      to(torch.device('cuda')).view(1, 7)])
                scores_batch.append([torch.Tensor([0.01]).to(torch.device('cuda')).view(-1)])
                agentid_batch.append([torch.tensor([0]).to(torch.device('cuda')).view(-1)])
                view_vector_batch.append([torch.tensor([[0, 0]]).to(torch.device('cuda'))])
                continue

            pred_boxes_cat = torch.cat(cur_boxes_list, dim=0)
            pred_boxes_cat[:, -1] = limit_period(pred_boxes_cat[:, -1])
            pred_scores_cat = torch.cat(cur_scores_list, dim=0)
            agentid_cat = torch.cat(cur_agentid_list, dim=0).to(torch.long)
            view_vector_cat = torch.cat(cur_view_vector_list, dim=0)

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
            agentid = []
            view_vector = []

            for j in range(1, cur_cluster_id):
                clusters.append(pred_boxes_cat[cluster_indices==j])  # shape: [[num_in_cluster, 7], ... ]
                scores.append(pred_scores_cat[cluster_indices==j])  # shape: [[num_in_cluster,], ...]
                agentid.append(agentid_cat[cluster_indices==j])  # shape: [[num_in_cluster,], ...]
                view_vector.append(view_vector_cat[cluster_indices==j])  # shape [[num_in_cluster, 2],...]

            clusters_batch.append(clusters) # shape: [[[num_in_cluster, 7], ...], ... ]
            scores_batch.append(scores)  # shape: [[[num_in_cluster,], ...], ... ]
            agentid_batch.append(agentid)   # shape: [[[num_in_cluster,], ...], ...]
            view_vector_batch.append(view_vector)    # shape [[[num_in_cluster, 2], ...], ...]

        return clusters_batch, scores_batch, agentid_batch, view_vector_batch

    def cluster_fusion(self, clusters, scores, agentid, view_vector):
        """
        Merge boxes in each cluster with scores as weights for merging.
        """
        boxes_fused = []
        scores_fused = []
        agentid_fused = []
        view_vector_fused = []
        for cl, sl, al, vl in zip(clusters, scores, agentid, view_vector): # cl, sl are clusters and scores within one sample
            for c, s, a, v in zip(cl, sl, al, vl): # one sample (cl) has many clusters (c), c,s,a correspond to one cluster.
                # reverse direction for non-dominant direction of boxes
                dirs = c[:, -1]
                max_score_idx = torch.argmax(s)
                dirs_diff = torch.abs(dirs - dirs[max_score_idx].item())
                lt_pi = (dirs_diff > np.pi).int()
                dirs_diff = dirs_diff * (1 - lt_pi) + (
                            2 *  np.pi - dirs_diff) * lt_pi
                score_lt_half_pi = s[dirs_diff > np.pi / 2].sum()  # larger than
                score_set_half_pi = s[dirs_diff <= np.pi / 2].sum()  # small equal than
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
                # s_sorted = torch.sort(s, descending=True).values
                # s_fused = 0
                # for i, ss in enumerate(s_sorted):
                #     s_fused += ss ** (i + 1)
                # s_fused = torch.tensor([min(s_fused, 1.0)], device=s.device)
                s_fused = torch.max(s)

                scores_fused.append(s_fused) # content: [s_cluster0, s_cluster1, ...]
                agentid_fused.append(a) # content [[id1,id2], [id1, id2, id3], ...]
                view_vector_fused.append(v) # shape [[2, 2], [3, 2], ...]

        assert len(boxes_fused) > 0
        boxes_fused = torch.stack(boxes_fused, dim=0)
        box_num_sample = [len(c) for c in clusters] # in a batch, each sample has how many clusters

        boxes_fused = [boxes_fused[sum(box_num_sample[:i]):sum(box_num_sample[:i]) + l] for 
                            i, l in enumerate(box_num_sample)] # shape [[num_of_cluster_in_sample1, 7], [num_of_cluster_in_sample2, 7], ...]

        scores_fused = torch.stack(scores_fused, dim=0)
        scores_fused = [scores_fused[sum(box_num_sample[:i]):sum(box_num_sample[:i]) + l] for
                            i, l in enumerate(box_num_sample)]  # shape [[num_of_cluster_in_sample1,], [num_of_cluster_in_sample2,], ...]

        agentid_fused = [agentid_fused[sum(box_num_sample[:i]):sum(box_num_sample[:i]) + l] for 
                            i, l in enumerate(box_num_sample)]  # content [[[id1,id2], [id1, id2, id3], ... ], [sample2 content], ...]

        view_vector_fused = [view_vector_fused[sum(box_num_sample[:i]):sum(box_num_sample[:i]) + l] for 
                            i, l in enumerate(box_num_sample)]  # shape [[ [2,2], [3,2], ...], [sample2 content], ...]

        for i in range(len(boxes_fused)):
            corners3d = boxes_to_corners_3d(boxes_fused[i], order=self.order)
            mask = get_mask_for_boxes_within_range_torch(corners3d, self.pc_range)
            boxes_fused[i] = boxes_fused[i][mask]
            scores_fused[i] = scores_fused[i][mask]
            agentid_fused[i] = list(compress(agentid_fused[i], mask))
            view_vector_fused[i] = list(compress(view_vector_fused[i], mask))

        return boxes_fused, scores_fused, agentid_fused, view_vector_fused

    def get_roi_from_box(self, data_dict):
        feature_shape = data_dict['feature_shape'] # [H,W]
        grid_size_H = (self.pc_range[4] - self.pc_range[1]) / feature_shape[0]
        grid_size_W = (self.pc_range[3] - self.pc_range[0]) / feature_shape[1]

        boxes_fused_list = data_dict['boxes_fused'] # [sample1, sample2, ...]
        roi_list = []

        for boxes_fused in boxes_fused_list:
            # boxes_fused shape [N, 7], hwl order
            # we omit the angle in the naive version
            grid_center_x = (boxes_fused[:,0] - self.pc_range[0]) / grid_size_W
            grid_center_y = (boxes_fused[:,1] - self.pc_range[1]) / grid_size_H
            grid_offset_x =  boxes_fused[:, -2] / 2 / grid_size_W 
            grid_offset_y =  boxes_fused[:, -3] / 2 / grid_size_H + 1  # enlarge

            
            xmin = (grid_center_x - grid_offset_x * self.enlarge_ratio).clamp(min=0)
            xmax = (grid_center_x + grid_offset_x * self.enlarge_ratio).clamp(max=feature_shape[1] - 1)
            ymin = (grid_center_y - grid_offset_y * self.enlarge_ratio).clamp(min=0)
            ymax = (grid_center_y + grid_offset_y * self.enlarge_ratio).clamp(max=feature_shape[0] - 1)

            roi = torch.stack([xmin, xmax, ymin, ymax], dim=-1).to(torch.long) # [boxnum, 4]
            
            roi_list.append(roi)

        data_dict['roi_fused'] = roi_list # shape [[num_of_cluster_in_sample1, 4], [num_of_cluster_in_sample2, 4], ...]