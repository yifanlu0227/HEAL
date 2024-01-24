# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from PIL import Image
import numpy as np
import torch
import torchvision
import cv2
import math
from shapely.geometry import Point, MultiPoint

def load_camera_data(camera_files, preload=True):
    """
    Args:
        camera_files: list, 
            store camera path
        shape : tuple
            (width, height), resize the image, and overcoming the lazy loading.
    Returns:
        camera_data_list: list,
            list of Image, RGB order
    """
    camera_data_list = []
    for camera_file in camera_files:
        camera_data = Image.open(camera_file)
        if preload:
            camera_data = camera_data.copy()
        camera_data_list.append(camera_data)
    return camera_data_list


def sample_augmentation(data_aug_conf, is_train):
    """
    https://github.com/nv-tlabs/lift-splat-shoot/blob/d74598cb51101e2143097ab270726a561f81f8fd/src/data.py#L96
    """
    H, W = data_aug_conf['H'], data_aug_conf['W']
    fH, fW = data_aug_conf['final_dim']
    if is_train:
        resize = np.random.uniform(*data_aug_conf['resize_lim'])
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.random.uniform(*data_aug_conf['bot_pct_lim']))*newH) - fH
        crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH) # [x_start, y_start, x_end, y_end]
        flip = False
        if data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
            flip = True
        rotate = np.random.uniform(*data_aug_conf['rot_lim'])
    else:
        resize = max(fH/H, fW/W)
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(data_aug_conf['bot_pct_lim']))*newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
    return resize, resize_dims, crop, flip, rotate


def img_transform(imgs, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    imgs_output = []
    for img in imgs:
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        imgs_output.append(img)


    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])

    if flip: 
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2 # [x_start, y_start, x_end, y_end]
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return imgs_output, post_rot, post_tran

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))

img_to_tensor = torchvision.transforms.ToTensor() # [0,255] -> [0,1]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=True):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        # mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        # indices[mask] = num_bins
        indices[indices < 0] = 0
        indices[indices >= num_bins] = num_bins - 1
        indices[~torch.isfinite(indices)] = num_bins - 1

        # Convert to integer
        indices = indices.type(torch.int64)
        return indices, None
    else:
        # mask indices outside of bounds
        mask = (indices < 0) | (indices >= num_bins) | (~torch.isfinite(indices))
        indices[indices < 0] = 0
        indices[indices >= num_bins] = num_bins - 1
        indices[~torch.isfinite(indices)] = num_bins - 1

        # Convert to integer
        indices = indices.type(torch.int64)
        return indices, ~mask

def depth_discretization(depth_min, depth_max, num_bins, mode):
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        depth_discre = depth_min + bin_size * np.arange(num_bins)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        depth_discre = depth_min + bin_size * (np.arange(num_bins) * np.arange(1, 1+num_bins)) / 2
    else:
        raise NotImplementedError
    return depth_discre

def indices_to_depth(indices, depth_min, depth_max, num_bins, mode):
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        depth = indices * bin_size + depth_min
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        depth = depth_min + bin_size * (indices * (indices+1)) / 2
    else:
        raise NotImplementedError
    return depth

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

def coord_3d_to_2d(gt_box3d, int_matrix, ext_matrix, image_H=600, image_W=800, image=None, idx=None):
    """
    Projects XYZ points onto the canvas and returns the projected canvas
    coordinates.

    Args:
        gt_box3d : np.ndarray
            shape (N, 8, 3). point coord in world (LiDAR) coordinate. 
        int_matrix : np.ndarray
            shape (4, 4)
        ext_matrix : np.ndarray
            shape (4, 4), T_wc, transform point in camera coord to world coord.

    Returns:
        gt_box2d : np.ndarray
            shape (N, 8, 2). pixel coord (u, v) in the image. You may want to flip them for image data indexing. 
        gt_box2d_mask : np.ndarray (bool)
            shape (N,). If false, this box is out of image boundary
        fg_mask : np.ndarray 
            shape (image_H, image_W), 1 means foreground, 0 means background
    """
    N = gt_box3d.shape[0]
    xyz = gt_box3d.reshape(-1, 3) # (N*8, 3)

    xyz_hom = np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)

    ext_matrix = np.linalg.inv(ext_matrix)[:3,:4]
    img_pts = (int_matrix @ ext_matrix @ xyz_hom.T).T

    depth = img_pts[:, 2]
    uv = img_pts[:, :2] / depth[:, None]
    uv_int = uv.round().astype(np.int32) # [N*8, 2]


    # o--------> u
    # |
    # |
    # |
    # v v


    valid_mask1 = ((uv_int[:, 0] >= 0) & (uv_int[:, 0] < image_W) & 
                    (uv_int[:, 1] >= 0) & (uv_int[:, 1] < image_H)).reshape(N, 8)
    
    valid_mask2 = ((depth > 0.5) & (depth < 100)).reshape(N, 8)
    gt_box2d_mask = valid_mask1.any(axis=1) & valid_mask2.all(axis=1) # [N, ]
    
    gt_box2d = uv_int.reshape(N, 8, 2) # [N, 8, 2]
    gt_box2d_u = np.clip(gt_box2d[:,:,0], 0, image_W-1)
    gt_box2d_v = np.clip(gt_box2d[:,:,1], 0, image_H-1)
    gt_box2d = np.stack((gt_box2d_u, gt_box2d_v), axis=-1) # [N, 8, 2]

    # create fg/bg mask
    fg_mask = np.zeros((image_H, image_W))
    for gt_box in gt_box2d[gt_box2d_mask]:
        u_min = gt_box[:,0].min()
        v_min = gt_box[:,1].min()
        u_max = gt_box[:,0].max()
        v_max = gt_box[:,1].max()
        fg_mask[v_min:v_max, u_min:u_max] = 1
        # poly = MultiPoint(gt_box).convex_hull
        # cv2.fillConvexPoly(fg_mask, np.array(list(zip(*poly.exterior.coords.xy)), dtype=np.int32), 1)

    DEBUG = False
    if DEBUG:
        from matplotlib import pyplot as plt
        plt.imshow(image)
        for i in range(N):
            if gt_box2d_mask[i]:
                coord2d = gt_box2d[i]
                for start, end in [(0, 1), (1, 2), (2, 3), (3, 0),
                               (0, 4), (1, 5), (2, 6), (3, 7),
                               (4, 5), (5, 6), (6, 7), (7, 4)]:
                    plt.plot(coord2d[[start,end]][:,0], coord2d[[start,end]][:,1], marker="o", c='g')
        plt.savefig(f"/GPFS/rhome/yifanlu/OpenCOOD/vis_result/dairv2x_lss_vehonly/image_gt_box2d_{idx}.png", dpi=300)
        plt.clf()
        plt.imshow(fg_mask)
        plt.savefig(f"/GPFS/rhome/yifanlu/OpenCOOD/vis_result/dairv2x_lss_vehonly/image_gt_box2d_{idx}_mask.png", dpi=300)
        plt.clf()

    
    return gt_box2d, gt_box2d_mask, fg_mask


def load_intrinsic_DAIR_V2X(int_dict):
    # cam_D : [5, ], what'is this...
    # cam_K : [9, ]
    cam_D = int_dict['cam_D']
    cam_K = int_dict['cam_K']
    return np.array(cam_K).reshape(3,3)