
from tkinter import Y
from turtle import left
from opencood.utils.transformation_utils import x_to_world
from torch.utils.data import Subset
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import numpy as np
import os

v2x = True
if v2x:
    from opencood.visualization.draw_fancy.draw_fancy_datasetv2x import SimpleDataset
else:
    from opencood.visualization.draw_fancy.draw_fancy_dataset import SimpleDataset

COLOR = ['red','springgreen','dodgerblue', 'darkviolet']
COLOR_RGB = [ tuple([int(cc * 255) for cc in matplotlib.colors.to_rgb(c)]) for c in COLOR]
COLOR_PC = [tuple([int(cc*0.2 + 255*0.8) for cc in c]) for c in COLOR_RGB]
classes = ['agent1', 'agent2', 'agent3', 'agent4']

def main():
    ## basic setting
    dataset = SimpleDataset()
    data_dict_demo = dataset[0]
    cav_ids = list(data_dict_demo.keys())
    cav_invert_dict = dict() # cav_id -> o/1/2
    for (idx, cav_id) in enumerate(cav_ids):
        cav_invert_dict[cav_id] = idx

    recs = []
    for i in range(0,len(cav_ids)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=COLOR[i]))


    
    ## matplotlib setting
    plt.figure()
    plt.style.use('dark_background')
    # plt.xticks([])
    # plt.yticks([])
    

    ## box setting
    ## ego coord
    dx = 4.5
    dy = 2
    dz = 1.6
    x_corners = dx / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])  # (8,)
    y_corners = dy / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = dz / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    box_corners = np.stack((x_corners, y_corners, z_corners), axis=-1) # (8, 3)
    box_corners = np.pad(box_corners,((0,0),(0,1)), constant_values=1) # (8, 4)


    ## draw
    print("loop over dataset")
    dataset_len = len(dataset)
    for idx in range(dataset_len):
        print(idx)
        base_data_dict = dataset[idx]

        lidar_np_world_agg = np.zeros((0, 4))
        cav_box_agg = dict()
        cav_lidar_agg = dict()

        for cav_id, cav_content in base_data_dict.items():
            lidar_pose = cav_content['params']['lidar_pose'] # list [6,]
            T_world_lidar = x_to_world(lidar_pose)
            lidar_np_lidar = cav_content['lidar_np'] # [N, 4], ego coord
            lidar_np_lidar[:, 3] = 1
            lidar_np_world = (T_world_lidar @ lidar_np_lidar.T).T # [N, 4], world coord
            cav_lidar_agg[cav_id] = lidar_np_world
            lidar_np_world_agg = np.concatenate((lidar_np_world_agg, lidar_np_world), axis=0)

            # get bbox for each one.
            if v2x:
                ego_pose = lidar_pose
            else:
                ego_pose = cav_content['params']['true_ego_pos']
            T_world_ego = x_to_world(ego_pose)
            cav_box_agg[cav_id] = ((T_world_ego @ box_corners.T).T)[np.newaxis,:,:3] # (1,8,3)


        ## setting canvas and extransic
        if v2x:
            canvas_shape=(800, 1200)
            camera_center_coords=(10, 32, 50)
            camera_focus_coords=(10 , 32 + 0.8396926, 50 - 0.84202014)
            focal_length = 400
            left_hand = False
        else:
            canvas_shape=(800, 1200)
            camera_center_coords=(-210, 185, 40)
            camera_focus_coords=(-210 , 185 + 0.9396926, 40 - 0.84202014)
            focal_length = 400
            left_hand = True
        point_color = "Mixed2"

        canvas = canvas_3d.Canvas_3D(canvas_shape, camera_center_coords, camera_focus_coords, focal_length, left_hand=left_hand) 
        # canvas_xy, valid_mask = canvas.get_canvas_coords(lidar_np_world_agg)
        # canvas.draw_canvas_points(canvas_xy[valid_mask], colors=COLOR_PC[cav_invert_dict[cav_id]])
        
        for cav_id in cav_ids:
            # draw point cloud for each cav
            canvas_xy, valid_mask = canvas.get_canvas_coords(cav_lidar_agg[cav_id])
            canvas.draw_canvas_points(canvas_xy[valid_mask], colors=COLOR_PC[cav_invert_dict[cav_id]])
            # draw bbox for each cav
            canvas.draw_boxes(cav_box_agg[cav_id], colors=COLOR_RGB[cav_invert_dict[cav_id]]) #, texts=classes[cav_invert_dict[cav_id]])

        plt.legend(recs,classes,loc='lower left')
        plt.axis("off")
        plt.imshow(canvas.canvas)

        plt.tight_layout()

        if not v2x:
            plt.gca().invert_xaxis() 

        # plt.scatter(lidar_np_world_agg[:,0], lidar_np_world_agg[:,1], s=1, c='w')

        
        if v2x:
            save_path = f"./result_v2x/scene_overview_{point_color}"
        else:
            save_path = f"./result/scene_overview_{point_color}"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        plt.savefig(f"{save_path}/overview{idx:02d}.png", transparent=False, dpi=500)
        plt.clf()

if __name__ == "__main__":
    main()