# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# This module is designed for box alignment
# We will use g2o for pose graph optimization.



from opencood.models.sub_modules.pose_graph_optim import PoseGraphOptimization2D
from opencood.utils.transformation_utils import pose_to_tfm
from opencood.utils.common_utils import check_torch_to_numpy
from opencood.utils import box_utils
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import g2o
from icecream import ic
import copy
import os
import matplotlib.pyplot as plt

DEBUG = False

def vis_pose_graph(poses, pred_corner3d, save_dir_path, vis_agent=False):
    """
    Args:
        poses: list of np.ndarray
            each item is a pose . [pose_before, ..., pose_refined]

        pred_corner3d: list
            predicted box for each agent.

        vis_agent: bool
            whether draw the agent's box

    """
    COLOR = ['red','springgreen','dodgerblue', 'darkviolet', 'orange']
    from opencood.utils.transformation_utils import get_relative_transformation

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    for iter, pose in enumerate(poses):
        box_idx = 0
        # we first transform other agents' box to ego agent's coordinate
        relative_t_matrix = get_relative_transformation(pose)
        N = pose.shape[0]
        nonempty_indices = [idx for (idx, corners) in enumerate(pred_corner3d) if len(corners)!=0]
        pred_corners3d_in_ego = [box_utils.project_box3d(pred_corner3d[i], relative_t_matrix[i]) for i in nonempty_indices]

        for agent_id in range(len(pred_corners3d_in_ego)):
            if agent_id not in nonempty_indices:
                continue
            corner3d = pred_corners3d_in_ego[agent_id]
            agent_pos = relative_t_matrix[agent_id][:2,3] # agent's position in ego's coordinate

            if vis_agent:
                plt.scatter(agent_pos[0], agent_pos[1], s=4, c=COLOR[agent_id])

            corner2d = corner3d[:,:4,:2]
            center2d = np.mean(corner2d, axis=1)
            for i in range(corner2d.shape[0]):
                plt.scatter(corner2d[i,[0,1],0], corner2d[i,[0,1], 1], s=2, c=COLOR[agent_id])
                plt.plot(corner2d[i,[0,1,2,3,0],0], corner2d[i,[0,1,2,3,0], 1], linewidth=1, c=COLOR[agent_id])
                plt.text(corner2d[i,0,0], corner2d[i,0,1], s=str(box_idx), fontsize="xx-small")
                # add a line connecting box center and agent.
                box_center = center2d[i] # [2,]
                connection_x = [agent_pos[0], box_center[0]]
                connection_y = [agent_pos[1], box_center[1]]

                plt.plot(connection_x, connection_y,'--', linewidth=0.5, c=COLOR[agent_id], alpha=0.3)
                box_idx += 1
        
        filename = os.path.join(save_dir_path, f"{iter}.png")
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.savefig(filename, dpi=400)
        plt.clf()


def all_pair_l2(A, B):
    """ All pair L2 distance for A and B
    Args:
        A : np.ndarray
            shape [N_A, D]
        B : np.ndarray
            shape [N_B, D]
    Returns:
        C : np.ndarray
            shape [N_A, N_B]
    """
    TwoAB = 2*A@B.T  # [N_A, N_B]
    C = np.sqrt(
              np.sum(A * A, 1, keepdims=True).repeat(TwoAB.shape[1], axis=1) \
            + np.sum(B * B, 1, keepdims=True).T.repeat(TwoAB.shape[0], axis=0) \
            - TwoAB
        )
    return C




def box_alignment_relative_sample_np(
            pred_corners_list,
            noisy_lidar_pose, 
            uncertainty_list=None, 
            landmark_SE2=True,
            adaptive_landmark=False,
            normalize_uncertainty=False,
            abandon_hard_cases = False,
            drop_hard_boxes = False,
            drop_unsure_edge = False,
            use_uncertainty = True,
            thres = 1.5,
            yaw_var_thres = 0.2,
            max_iterations = 1000):
    """ Perform box alignment for one sample. 
    Correcting the relative pose.

    Args:
        pred_corners_list: in each ego coordinate
            [[N_1, 8, 3], ..., [N_cav1, 8, 3]]

        clean_lidar_poses:
            [N_cav1, 6], in degree
        
        noisy_lidar_poses:
            [N_cav1, 6], in degree

        uncertainty_list:
            [[N_1, 3], [N_2, 3], ..., [N_cav1, 3]]

        landmark_SE2:
            if True, the landmark is SE(2), otherwise R^2
        
        adaptive_landmark: (when landmark_SE2 = True)
            if True, landmark will turn to R^2 if yaw angles differ a lot

        normalize_uncertainty: bool
            if True, normalize the uncertainty
        
        abandon_hard_cases: bool
            if True, algorithm will just return original poses for hard cases

        drop_unsure_edge: bool

    Returns: 
        refined_lidar_poses: np.ndarray
            [N_cav1, 3], 
    """
    if not use_uncertainty:
        uncertainty_list = None
    ## first transform point from ego coordinate to world coordinate, using lidar_pose.
    order = 'lwh'  # hwl
    N = noisy_lidar_pose.shape[0]
    lidar_pose_noisy_tfm = pose_to_tfm(noisy_lidar_pose)

    nonempty_indices = [idx for (idx, corners) in enumerate(pred_corners_list) if len(corners)!=0] # if one agent detects no boxes, its corners is just [].
    
    pred_corners_world_list = \
        [box_utils.project_box3d(pred_corners_list[i], lidar_pose_noisy_tfm[i]) for i in nonempty_indices]  # [[N1, 8, 3], [N2, 8, 3],...]
    pred_box3d_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_list if len(corner)!=0]   # [[N1, 7], [N2, 7], ...], angle in radian
    pred_box3d_world_list = \
        [box_utils.corner_to_center(corner, order) for corner in pred_corners_world_list]   # [[N1, 7], [N2, 7], ...], angle in radian
    pred_center_list = \
        [np.mean(corners, axis=1) for corners in pred_corners_list if len(corners)!=0] # [[N1,3], [N2,3], ...]

    pred_center_world_list = \
        [pred_box3d_world[:,:3] for pred_box3d_world in pred_box3d_world_list]
    pred_yaw_world_list = \
        [pred_box3d[:, 6] for pred_box3d in pred_box3d_world_list]
    pred_len = \
        [len(corners) for corners in pred_corners_list] 


    box_idx_to_agent = []
    for i in range(N):
        box_idx_to_agent += [i] * pred_len[i] 
    
    pred_center_cat = np.concatenate(pred_center_list, axis=0)   # [sum(pred_box), 3]
    pred_center_world_cat =  np.concatenate(pred_center_world_list, axis=0)  # [sum(pred_box), 3]
    pred_box3d_cat =  np.concatenate(pred_box3d_list, axis=0)  # [sum(pred_box), 7]
    pred_yaw_world_cat = np.concatenate(pred_yaw_world_list, axis=0)  # [sum(pred_box)]

    # hard-coded currently
    w_a = 1.6 # width of anchor
    l_a = 3.9 # length of anchor
    d_a_square = w_a ** 2 + l_a ** 2 # anchor's diag


    if uncertainty_list is not None:
        pred_log_sigma2_cat = np.concatenate([i for i in uncertainty_list if len(i)!=0], axis=0)
        # Since the regression target is x_t = (x_g - x_a)/d_a, 
        # var(x) = d_a^2 * var(x_t)
        # so we 1/var(x) = 1/var(x_t) / d_a^2  
        # sigma_{delta_x}^2 -> sigma_x^2. 
        pred_certainty_cat = np.exp(-pred_log_sigma2_cat)
        pred_certainty_cat[:,:2] /= d_a_square 


        if normalize_uncertainty:
            pred_certainty_cat = np.sqrt(pred_certainty_cat)


    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat, pred_center_world_cat) # [sum(pred_box), sum(pred_box)]

    # let pair from one vehicle be max distance
    MAX_DIST = 10000
    cum = 0
    for i in range(N):
        pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum +pred_len[i]] = MAX_DIST   # do not include itself
        cum += pred_len[i]


    cluster_id = N # let the vertex id of object start from N
    cluster_dict = OrderedDict()
    remain_box = set(range(cum))

    for box_idx in range(cum): 

        if box_idx not in remain_box:  # already assigned
            continue
        
        within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero()[0]
        within_thres_idx_list = within_thres_idx_tensor.tolist()

        if len(within_thres_idx_list) == 0:  # if it's a single box
            continue

        # start from within_thres_idx_list, find new box added to the cluster
        explored = [box_idx]
        unexplored = [idx for idx in within_thres_idx_list if idx in remain_box]

        while unexplored:
            idx = unexplored[0]
            within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero()[0]
            within_thres_idx_list = within_thres_idx_tensor.tolist()
            for newidx in within_thres_idx_list:
                if (newidx not in explored) and (newidx not in unexplored) and (newidx in remain_box):
                    unexplored.append(newidx)
            unexplored.remove(idx)
            explored.append(idx)
        
        if len(explored) == 1: # it's a single box, neighbors have been assigned
            remain_box.remove(box_idx)
            continue
        
        cluster_box_idxs = explored

        cluster_dict[cluster_id] = OrderedDict()
        cluster_dict[cluster_id]['box_idx'] = [idx for idx in cluster_box_idxs]
        cluster_dict[cluster_id]['box_center_world'] = [pred_center_world_cat[idx] for idx in cluster_box_idxs]  # coordinate in world, [3,]
        cluster_dict[cluster_id]['box_yaw'] = [pred_yaw_world_cat[idx] for idx in cluster_box_idxs]

        yaw_var = np.var(cluster_dict[cluster_id]['box_yaw'])
        cluster_dict[cluster_id]['box_yaw_varies'] = yaw_var > yaw_var_thres
        cluster_dict[cluster_id]['active'] = True


        ########### adaptive_landmark ##################
        if landmark_SE2:
            if adaptive_landmark and yaw_var > yaw_var_thres:
                landmark = pred_center_world_cat[box_idx][:2]
                for _box_idx in cluster_box_idxs:
                    pred_certainty_cat[_box_idx] *= 2
            else:
                landmark = copy.deepcopy(pred_center_world_cat[box_idx])
                landmark[2] = pred_yaw_world_cat[box_idx]
        else:
            landmark = pred_center_world_cat[box_idx][:2]
        ##################################################


        cluster_dict[cluster_id]['landmark'] = landmark  # [x, y, yaw] or [x, y]
        cluster_dict[cluster_id]['landmark_SE2'] = True if landmark.shape[0] == 3 else False

        DEBUG = False
        if DEBUG:
            from icecream import ic
            ic(cluster_dict[cluster_id]['box_idx'])
            ic(cluster_dict[cluster_id]['box_center_world'])
            ic(cluster_dict[cluster_id]['box_yaw'])
            ic(cluster_dict[cluster_id]['landmark'])
        

        cluster_id += 1
        for idx in cluster_box_idxs:
            remain_box.remove(idx)

    
    vertex_num = cluster_id
    agent_num = N
    landmark_num = cluster_id - N


    ########### abandon_hard_cases ##########
    """
        We should think what is hard cases for agent-object pose graph optimization
            1. Overlapping boxes are rare (landmark_num <= 3)
            2. Yaw angles differ a lot
    """

    if abandon_hard_cases:
        # case1: object num is smaller than 3
        if landmark_num <= 3:
            return noisy_lidar_pose[:,[0,1,4]]
        
        # case2: more than half of the landmarks yaw varies 
        yaw_varies_cnt = sum([cluster_dict[i]["box_yaw_varies"] for i in range(agent_num, vertex_num)])
        if yaw_varies_cnt >= 0.5 * landmark_num:
            return noisy_lidar_pose[:,[0,1,4]]

    ########### drop hard boxes ############

    if drop_hard_boxes:
        for landmark_id in range(agent_num, vertex_num):
            if cluster_dict[landmark_id]['box_yaw_varies']:
                cluster_dict[landmark_id]['active'] = False




    """
        Now we have clusters for objects. we can create pose graph.
        First we consider center as landmark.
        Maybe set corner as landmarks in the future.
    """
    pgo = PoseGraphOptimization2D()

    # Add agent to vertexs
    for agent_id in range(agent_num):
        v_id = agent_id
        # notice lidar_pose use degree format, translate it to radians.
        pose_np = noisy_lidar_pose[agent_id, [0,1,4]]
        pose_np[2] = np.deg2rad(pose_np[2])  # radians
        v_pose = g2o.SE2(pose_np)
        
        if agent_id == 0:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        else:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

    # Add object to vertexs
    for landmark_id in range(agent_num, vertex_num):
        v_id = landmark_id
        landmark = cluster_dict[landmark_id]['landmark'] # (3,) or (2,)
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']

        if landmark_SE2:
            v_pose = g2o.SE2(landmark)
        else:
            v_pose = landmark

        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

    # Add agent-object edge to edge set
    for landmark_id in range(agent_num, vertex_num):
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']

        if not cluster_dict[landmark_id]['active']:
            continue

        for box_idx in cluster_dict[landmark_id]['box_idx']:
            agent_id = box_idx_to_agent[box_idx]
            if landmark_SE2:
                e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0,1,6]].astype(np.float64))
                info = np.identity(3, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1,2],[0,1,2]] = pred_certainty_cat[box_idx]

                    ############ drop_unsure_edge ###########
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:
                        continue

            else:
                e_pose = pred_box3d_cat[box_idx][[0,1]].astype(np.float64)
                info = np.identity(2, dtype=np.float64)
                if uncertainty_list is not None:
                    info[[0,1],[0,1]] = pred_certainty_cat[box_idx][:2]

                    ############ drop_unsure_edge ############
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:
                        continue

            pgo.add_edge(vertices=[agent_id, landmark_id], measurement=e_pose, information=info, SE2=landmark_SE2)
    
    pgo.optimize(max_iterations)

    pose_new_list = []
    for agent_id in range(agent_num):
        # print(pgo.get_pose(agent_id).vector())
        pose_new_list.append(pgo.get_pose(agent_id).vector())

    refined_pose = np.array(pose_new_list)
    refined_pose[:,2] = np.rad2deg(refined_pose[:,2])  # rad -> degree, same as source

    return refined_pose

def box_alignment_relative_np(pred_corner3d_list, 
                              uncertainty_list, 
                              lidar_poses, 
                              record_len, 
                              **kwargs):
    """
    Args:
        pred_corner3d_list: list of tensors, with shape [[N1_object, 8, 3], [N2_object, 8, 3], ...,[N_sumcav_object, 8, 3]]
            box in each agent's coordinate. (proj_first=False)
        
        pred_box3d_list: not necessary
            list of tensors, with shape [[N1_object, 7], [N2_object, 7], ...,[N_sumcav_object, 7]]

        scores_list: list of tensor, [[N1_object,], [N2_object,], ...,[N_sumcav_object,]]
            box confidence score.

        lidar_poses: torch.Tensor [sum(cav), 6]

        record_len: torch.Tensor
    Returns:
        refined_lidar_pose: torch.Tensor [sum(cav), 6]
    """
    refined_lidar_pose = []
    start_idx = 0
    for b in record_len:
        refined_lidar_pose.append(
            box_alignment_relative_sample_np(
                pred_corner3d_list[start_idx: start_idx + b],
                lidar_poses[start_idx: start_idx + b],
                uncertainty_list= None if uncertainty_list is None else uncertainty_list[start_idx: start_idx + b],
                **kwargs
            )
        )
        start_idx += b

    return np.cat(refined_lidar_pose, axis=0)


