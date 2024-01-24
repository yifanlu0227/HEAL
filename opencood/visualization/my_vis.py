from operator import gt
import numpy as np
import pickle
from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from icecream import ic
from torch import margin_ranking_loss



def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """

        pcd_np = pcd.cpu().numpy()
        pred_box_np = pred_box_tensor.cpu().numpy()
        gt_box_np = gt_tensor.cpu().numpy()

        plt.figure(dpi=400)
        # draw point cloud. It's in lidar coordinate
        plt.scatter(pcd_np[:,0], pcd_np[:,1], s=0.5)

        N = gt_tensor.shape[0]
        for i in range(N):
            plt.plot(gt_box_np[i,:,0], gt_box_np[i,:,1], c= "r", marker='.', linewidth=1, markersize=1.5)

        N = pred_box_tensor.shape[0]
        for i in range(N):
            plt.plot(pred_box_np[i,:,0], pred_box_np[i,:,1], c= "g", marker='.', linewidth=1, markersize=1.5)
        

        plt.savefig(save_path)
        plt.clf()