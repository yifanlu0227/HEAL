
from email.mime import base
from tkinter import Y

from black import left_hand_split
from opencood.utils.transformation_utils import x_to_world, x1_to_x2
from opencood.utils.box_utils import create_bbx, mask_boxes_outside_range_numpy, corner_to_center
from torch.utils.data import Subset
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import numpy as np
import os
import copy
from pyquaternion import Quaternion

v2x = True
if v2x:
    from opencood.visualization.draw_fancy.draw_fancy_datasetv2x import SimpleDataset
else:
    from opencood.visualization.draw_fancy.draw_fancy_dataset import SimpleDataset

COLOR = ['red','springgreen','dodgerblue', 'darkviolet']
COLOR_RGB = [ tuple([int(cc * 255) for cc in matplotlib.colors.to_rgb(c)]) for c in COLOR]
COLOR_PC = [tuple([int(cc*0.2 + 255*0.8) for cc in c]) for c in COLOR_RGB]
classes = ['agent1', 'agent2', 'agent3', 'agent4']


def generate_object_center_v2x(cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            In fact, only the ego vehile needs to generate object center

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (1, 8, 3).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        # from opencood.data_utils.datasets import GT_RANGE

        order = 'hwl'
        max_num = 200
        gt_boxes = cav_contents[0]['params']['vehicles'] # notice [N,10], 10 includes [x,y,z,dx,dy,dz,w,a,b,c]
        object_ids = cav_contents[0]['params']['object_ids']
        
        object_dict = {"gt_boxes": gt_boxes, "object_ids":object_ids}

        output_dict = {}
        lidar_range = (-64,-64,-3,64,64,2)
        
        gt_boxes = object_dict['gt_boxes']
        object_ids = object_dict['object_ids']
        for i, object_content in enumerate(gt_boxes):
            x,y,z,dx,dy,dz,w,a,b,c = object_content

            q = Quaternion([w,a,b,c])
            T_world_object = q.transformation_matrix
            T_world_object[:3,3] = object_content[:3]

            T_world_lidar = x_to_world(reference_lidar_pose)

            object2lidar = np.linalg.solve(T_world_lidar, T_world_object) # T_lidar_object


            # shape (3, 8)
            # hopefully this is correct? 
            x_corners = dx / 2 * np.array([ 1,  1, -1, -1,  1,  1, -1, -1]) # (8,)
            y_corners = dy / 2 * np.array([-1,  1,  1, -1, -1,  1,  1, -1])
            z_corners = dz / 2 * np.array([-1, -1, -1, -1,  1,  1,  1,  1])

            bbx = np.vstack((x_corners, y_corners, z_corners)) # (3, 8)

            # bounding box under ego coordinate shape (4, 8)
            bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

            # project the 8 corners to world coordinate
            bbx_lidar = np.dot(object2lidar, bbx).T # (8, 4)
            bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0) # (1, 8, 3)

            bbox_corner = copy.deepcopy(bbx_lidar)

            bbx_lidar = corner_to_center(bbx_lidar, order=order)
            bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar,
                                                    lidar_range,
                                                    order)


            if bbx_lidar.shape[0] > 0:
                output_dict.update({object_ids[i]: bbox_corner})


        object_np = np.zeros((max_num, 8, 3))
        mask = np.zeros(max_num)
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        # should not appear repeated items
        object_np = object_np[:len(object_ids)]

        return object_np, object_ids

def generate_object_center(cav_contents,
                            reference_lidar_pose):
    """
    Retrieve all objects in a format of (n, 7), where 7 represents
    x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

    Parameters
    ----------
    cav_contents : list
        List of dictionary, save all cavs' information.

    reference_lidar_pose : list
        The final target lidar pose with length 6.

    Returns
    -------
    object_np : np.ndarray
        Shape is (n, 8, 3). n is number of xxx

    object_ids : list
        Length is number of bbx in current sample.
    """
    
    order = 'hwl'
    max_num = 200


    tmp_object_dict = {}
    for cav_content in cav_contents:
        tmp_object_dict.update(cav_content['params']['vehicles'])

    output_dict = {}
    filter_range = [-140, -60, -3, 140, 60, 2]

    for object_id, object_content in tmp_object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, reference_lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)

        bbox_corner = copy.deepcopy(bbx_lidar)

        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar,
                                                   filter_range,
                                                   order)

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: bbox_corner})

    object_np = np.zeros((max_num, 8, 3))
    mask = np.zeros(max_num)
    object_ids = []

    for i, (object_id, object_bbx) in enumerate(output_dict.items()):
        object_np[i] = object_bbx[0, :]
        mask[i] = 1
        object_ids.append(object_id)

    unique_indices = \
                [object_ids.index(x) for x in set(object_ids)]
    near_indices = [idx for idx in unique_indices if (object_np[idx][0][0]**2 + object_np[idx][0][1]**2) < 45**2]
    print(len(unique_indices), len(near_indices))
    object_np = object_np[near_indices]

    return object_np, near_indices

def main():
    ## basic setting
    dataset = SimpleDataset()
    data_dict_demo = dataset[0]
    cav_ids = list(data_dict_demo.keys())
    cav_invert_dict = dict() # cav_id -> 0/1/2
    for (idx, cav_id) in enumerate(cav_ids):
        cav_invert_dict[cav_id] = idx

    recs = []
    for i in range(0,len(cav_ids)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=COLOR[i]))

    ## matplotlib setting
    plt.figure()
    plt.style.use('dark_background')

    ## box setting
    # ego coord
    dx = 4.9
    dy = 2
    dz = 1.5
    x_corners = dx / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])  # (8,)
    y_corners = dy / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = dz / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    box_corners = np.stack((x_corners, y_corners, z_corners), axis=-1) # (8, 3)
    # box_corners = np.pad(box_corners,((0,0),(0,1)), constant_values=1) # (8, 4)
    box_corners = box_corners[np.newaxis,...]
    if v2x:
        box_corners[:,:,0] -= 2.2


    ## draw
    print("loop over dataset")
    dataset_len = len(dataset)
    for idx in range(dataset_len):
        print(idx)
        base_data_dict = dataset[idx]
        
        
        # retrieve all bbox, under world coordinate
        for cav_id, cav_content in base_data_dict.items():
            lidar_np_ego_agg = np.zeros((0, 4))
            cav_box_agg = dict()
            cav_lidar_agg = dict()
            ego_pose = cav_content['params']['lidar_pose']
            ego_id = cav_id

            if v2x:
                cav_contents = [base_data_dict[1]]
            else:
                cav_contents = [cav_content]

            if v2x:
                object_np, object_ids = generate_object_center_v2x(cav_contents, ego_pose)
            else:
                object_np, object_ids = generate_object_center(cav_contents, ego_pose)

            if (not v2x) and (not cav_id in object_ids):
                object_np = np.concatenate((object_np, box_corners), axis=0)
                object_ids.append(cav_id)

            lidar_np_ego_agg = cav_content['lidar_np']


            ## setting canvas and extransic
            # drawing include 2 things, point cloud and bbox
            # since it's collaboration view, bbox are shared across each cav
            canvas_shape=(800, 1200)
            camera_center_coords=(-10, 0, 10)
            camera_focus_coords=(-10 + 0.5396926, 0, 10 - 0.34202014)

            if v2x:
                left_hand = False
            else:
                left_hand = True

            canvas = canvas_3d.Canvas_3D(canvas_shape, camera_center_coords, camera_focus_coords, left_hand=left_hand) 
            # canvas_xy, valid_mask = canvas.get_canvas_coords(lidar_np_world_agg)
            # canvas.draw_canvas_points(canvas_xy[valid_mask], colors=COLOR_PC[cav_invert_dict[cav_id]])
            

            canvas_xy, valid_mask = canvas.get_canvas_coords(lidar_np_ego_agg)
            canvas.draw_canvas_points(canvas_xy[valid_mask], colors=COLOR_PC[cav_invert_dict[cav_id]])
            # draw bbox for each cav
            if cav_id == 3:
                object_np = np.concatenate((object_np, box_corners), axis=0)
            canvas.draw_boxes(object_np, colors=COLOR_RGB[cav_invert_dict[cav_id]])

            plt.axis("off")
            plt.imshow(canvas.canvas)

            plt.tight_layout()


            if v2x:
                save_path = f"./result_v2x/single_view_{classes[cav_invert_dict[cav_id]]}"
            else:
                save_path = f"./result/single_view_{classes[cav_invert_dict[cav_id]]}"

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            plt.savefig(f"{save_path}/{idx:02d}.png", transparent=False, dpi=300)
            plt.clf()

if __name__ == "__main__":
    main()