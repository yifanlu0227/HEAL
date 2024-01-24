"""
https://github.com/AmnonDrory/BestBuddiesRegistration/blob/main/code/bb_pc/utils/subsampling.py
"""

import numpy as np
import open3d as o3d
import pandas as pd
from copy import deepcopy

num_features = 3

def calc_bin_inds(PC, n_bins, axis, mode):
    N = PC.shape[0]
    if "adaptive" in mode:
        inds = np.round(np.linspace(0, N, n_bins + 1)).astype(int)
        s = np.sort(PC[:, axis])
        thresh = s[inds[1:]-1]
    else: # "equally_spaced"
        thresh = np.linspace(np.min(PC[:,axis]), np.max(PC[:,axis]),  n_bins + 1)
        thresh = thresh[1:]

    bin_ind = np.zeros(N) + np.nan
    for i in range(n_bins):
        is_cur = (PC[:, axis] <= thresh[i]) & np.isnan(bin_ind)
        bin_ind[is_cur] = i

    assert np.sum(np.isnan(bin_ind)) == 0, "Error: not all samples were assigned to a bin"

    return bin_ind

def voxelGrid_filter_inner(PC, num_samples, mode):

    if "equal_nbins_per_axis" in mode:
        n_bins = int(np.ceil(num_samples ** (1. / 3)))
        n_bins_x = n_bins
        n_bins_y = n_bins
        n_bins_z = n_bins
    else:
        span = []
        for axis in range(3):
            span.append( np.max(PC[:,axis])-np.min(PC[:,axis]) )
        normalized_num_samples = num_samples * (span[0]**2 / (span[1]*span[2]))
        n_bins_x = int(np.ceil(normalized_num_samples ** (1. / 3)))
        n_bins_y = int(np.ceil(n_bins_x * (span[1]/span[0])))
        n_bins_z = int(np.ceil(n_bins_x * (span[2] / span[0])))
        assert (n_bins_x * n_bins_y * n_bins_z) >= num_samples, "Error"
    x_bin_inds = calc_bin_inds(PC, n_bins_x, 0, mode)
    y_bin_inds = calc_bin_inds(PC, n_bins_y, 1, mode)
    z_bin_inds = calc_bin_inds(PC, n_bins_z, 2, mode)

    data = np.hstack([x_bin_inds.reshape([-1,1]),
                      y_bin_inds.reshape([-1,1]),
                      z_bin_inds.reshape([-1,1]),
                      PC])

    df = pd.DataFrame(data, columns=['x_ind', 'y_ind', 'z_ind', 'x', 'y', 'z'])
    newPC = np.array(df.groupby(['x_ind', 'y_ind', 'z_ind']).mean())

    return newPC

def voxelGrid_filter(PC, num_requested_samples, mode):
    """
    Sub-sample a point cloud by defining a grid of voxels, and returning the average point in each one.

    :param PC: Nx3 array, point cloud, each row is a sample
    :param num_samples: numbver of requested samples
    :param mode: list of strings, can contain any of the following:
                 "exact_number" - return exactly num_requested_samples, otherwise may return more than requested number (but never less)
                 "equal_nbins_per_axis" - same number of bins for each axis (x,y,z). Otherwise the bins are cube shaped, and usually a different number of bins fits in each of the dimensions.
                 "adaptive" - smaller bins where there is more data. Otherwise, all bins are the same size.
    :return: newPC - a point cloud with approximately num_requested_samples
    """
    num_samples = num_requested_samples
    N = PC.shape[0]
    done = False
    MAX_ATTEMPTS = 40
    ACCELERATION_FACTOR = 2
    MAX_DIVERGENCE_TIME = 4
    TOLERANCE = 0.05
    rel_history = []
    newPC_history = []
    while not done:
        newPC = voxelGrid_filter_inner(PC, num_samples, mode)
        new_N = newPC.shape[0]
        newPC_history.append(newPC)
        relative_error_in_size = (new_N/float(num_requested_samples)) -1
        rel_history.append(relative_error_in_size)
        if (relative_error_in_size < 0) or (relative_error_in_size > TOLERANCE):
            best_ind = np.argmin(np.abs(rel_history))
            if (len(rel_history) - best_ind > MAX_DIVERGENCE_TIME) and (np.max(rel_history) > 0):
                    done = True
            else:
                num_samples = int(np.ceil(num_samples*float(num_requested_samples)/new_N))
                if (np.max(rel_history) < 0):
                    num_samples = int(ACCELERATION_FACTOR*num_samples)

        else:
            done = True

        if len(rel_history) >= MAX_ATTEMPTS:
            done = True

    if len(rel_history) >= MAX_ATTEMPTS:
        assert False, "voxelGrid_filter could not supply required number of samples"
        print("Error: voxelGrid_filter could not supply required number of samples, recovering")
        best_ind = np.argmax(rel_history)
        return newPC_history[best_ind]

    rel_history_above_only = np.array(rel_history)
    rel_history_above_only[rel_history_above_only<0] = np.inf
    best_ind_above = np.argmin(rel_history_above_only)

    newPC = newPC_history[best_ind_above]
    if 'exact_number' in mode:
        p = np.random.permutation(newPC.shape[0])
        inds = p[:num_requested_samples]
        newPC = newPC[inds,:]

    return newPC

def voxel_filter(pcd, N):
    # pcd is of open3d point cloud class
    if "numpy" in str(type(pcd)):
        tmp = o3d.geometry.PointCloud()
        tmp.points = o3d.utility.Vector3dVector(pcd)
        pcd = tmp
    K = np.shape(pcd.points)[0]
    vs = 1e-3
    while K>N:
        pcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=vs)
        vs *= 2
        K = np.shape(pcd.points)[0]
    return pcd

def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)

def fps_from_given_pc(pts, K, given_pc):
    """
    copied from https://github.com/orendv/learning_to_sample/blob/master/reconstruction/src/sample_net_point_net_ae.py
    :param self:
    :param pts:
    :param K:
    :param given_pc:
    :return:
    """
    farthest_pts = np.zeros((K, 3))
    t = given_pc.shape[0]
    farthest_pts[0:t,:] = given_pc

    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, t):
        distances = np.minimum(distances, calc_distances(farthest_pts[i,:], pts))

    for i in range(t, K):
        farthest_pts[i,:] = pts[np.argmax(distances),:]
        distances = np.minimum(distances, calc_distances(farthest_pts[i,:], pts))
    return farthest_pts

def get_random_subset(PC, num_samples, mode="farthest", submode=None, allow_overask=False):
    """
    Subsample a point cloud, using either of various methods

    :param PC:
    :param num_samples:
    :param mode:
    :param n_bins:
    :param submode: Relevant for the "r_normalized" and "r_squared_normalized" methods.
    :return:
    """
    if num_samples > PC.shape[0]:
        if allow_overask:
            return PC
        else:
            assert False, "Error: requesting more samples than there are"

    if PC.shape[0] == num_samples:
        result = PC
    if mode == "uniform":
        inds = np.random.permutation(PC.shape[0])[:num_samples]
        result = PC[inds, :]
    elif mode == "farthest":
        first_ind = np.random.permutation(PC.shape[0])[0]
        result = fps_from_given_pc(PC, num_samples, PC[first_ind:(first_ind+1), :])
    elif "voxel" in mode:
        if submode is None:
            submode = ["equal_nbins_per_axis"]

        # The voxelGrid subsampling algorithm has no randomality.
        # we force it to have some by rendomly removing a small subset of the points

        keep_fraction = 0.9
        num_keep = int(PC.shape[0]*keep_fraction)
        if num_samples < num_keep:
            PC = get_random_subset(PC, num_keep, mode="uniform")
        result = voxelGrid_filter(PC, num_samples, submode)

    else:
        assert False, "unknown mode"

    return result

def subsample_fraction(PC, fraction):
    N = PC.shape[0]
    subset_size = int(np.round(N * fraction))
    inds = np.random.permutation(N)[:subset_size]
    return PC[inds,:]


def keep_closest(PC, max_dist):
    R = np.sqrt(np.sum(PC ** 2, axis=1))
    return PC[R <= max_dist, :]


def fit_plane(PC):
    xy1 = deepcopy(PC)
    xy1[:, 2] = 1
    z = PC[:, 2]
    abc, _, _, _ = np.linalg.lstsq(xy1, z, rcond=None)
    return abc


def is_on_plane(PC, abc, thickness):
    all_xy1 = deepcopy(PC)
    all_xy1[:, 2] = 1
    predicted_road_z = np.matmul(all_xy1, abc.reshape([-1, 1])).flatten()
    res = np.abs(PC[:, 2] - predicted_road_z) <= thickness
    return res

def remove_road(PC):
    mode = "plane"  # "constant_height"
    local_PC = keep_closest(PC, 10)
    count, bin_edges = np.histogram(local_PC[:, 2], 100)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ind_of_most_frequent = np.argmax(count)
    road_z = bin_centers[ind_of_most_frequent]
    road_thickness = 0.5  # meters
    if mode == "constant_height":
        is_road = np.abs(PC[:, 2] - road_z) <= road_thickness
    elif mode == "plane":
        raw_is_road = np.abs(local_PC[:, 2] - road_z) <= road_thickness
        raw_road_points = local_PC[raw_is_road, :]
        xy1 = deepcopy(raw_road_points)
        xy1[:, 2] = 1
        z = raw_road_points[:, 2]
        abc, _, _, _ = np.linalg.lstsq(xy1, z, rcond=None)
        all_xy1 = deepcopy(PC)
        all_xy1[:, 2] = 1
        predicted_road_z = np.matmul(all_xy1, abc.reshape([-1, 1])).flatten()
        is_road = np.abs(PC[:, 2] - predicted_road_z) <= road_thickness
    else:
        assert False, "unknown mode"

    return PC[~is_road, :]