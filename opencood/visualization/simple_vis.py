# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from matplotlib import pyplot as plt
import numpy as np
import copy

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

def visualize(infer_result, pcd, pc_range, save_path, method='3d', left_hand=False):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        infer_result:
            pred_box_tensor : torch.Tensor
                (N, 8, 3) prediction.

            gt_tensor : torch.Tensor
                (N, 8, 3) groundtruth bbx
            
            uncertainty_tensor : optional, torch.Tensor
                (N, ?)

            lidar_agent_record: optional, torch.Tensor
                (N_agnet, )


        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
        pc_range = [int(i) for i in pc_range]
        pcd_np = pcd.cpu().numpy()

        pred_box_tensor = infer_result.get("pred_box_tensor", None)
        gt_box_tensor = infer_result.get("gt_box_tensor", None)

        if pred_box_tensor is not None:
            pred_box_np = pred_box_tensor.cpu().numpy()
            pred_name = ['pred'] * pred_box_np.shape[0]

            score = infer_result.get("score_tensor", None)
            if score is not None:
                score_np = score.cpu().numpy()
                pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

            uncertainty = infer_result.get("uncertainty_tensor", None)
            if uncertainty is not None:
                uncertainty_np = uncertainty.cpu().numpy()
                uncertainty_np = np.exp(uncertainty_np)
                d_a_square = 1.6**2 + 3.9**2
                
                if uncertainty_np.shape[1] == 3:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) 
                    # yaw angle is in radian, it's the same in g2o SE2's setting.

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 2:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 7:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]                    

        if gt_box_tensor is not None:
            gt_box_np = gt_box_tensor.cpu().numpy()
            gt_name = ['gt'] * gt_box_np.shape[0]

        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand) 

            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
            canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
                # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=['']*len(gt_name), box_line_thickness=4) # paper visualization
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
                # canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=['']*len(pred_name), box_line_thickness=4) # paper visualization

            # heterogeneous
            agent_modality_list = infer_result.get("agent_modality_list", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if agent_modality_list is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, modality_name in enumerate(agent_modality_list):
                    if modality_name == "m1":
                        color = (0,191,255)
                    elif modality_name == "m2":
                        color = (255,185,15)
                    elif modality_name == "m3":
                        color = (138,211,222)
                    elif modality_name == 'm4':
                        color = (32, 60, 160)
                    else:
                        color = (66,66,66)

                    render_dict = {'m1': 'L1', 'm2':"C1", 'm3':'L2', 'm4':'C2'}
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[modality_name])
                    # canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[render_dict[modality_name]], box_text_size=1.5, box_line_thickness=5) # paper visualization



        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            canvas.draw_canvas_points(canvas_xy[valid_mask])
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)

            # heterogeneous
            agent_modality_list = infer_result.get("agent_modality_list", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if agent_modality_list is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, modality_name in enumerate(agent_modality_list):
                    if modality_name == "m1":
                        color = (0,191,255)
                    elif modality_name == "m2":
                        color = (255,185,15)
                    elif modality_name == "m3":
                        color = (123,0,70)
                    else:
                        color = (66,66,66)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=[modality_name])

        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()


