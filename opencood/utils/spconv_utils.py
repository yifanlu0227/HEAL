import spconv
import torch
import numpy as np
from spconv.modules import SparseModule
from opencood.utils.box_utils import project_points_by_matrix_torch
from torch_scatter import scatter
from icecream import ic 

class RemoveDuplicate(SparseModule):
    """
    Only keep one when duplicated
    """
    def forward(self, x: spconv.SparseConvTensor):
        inds = x.indices
        spatial_shape = [x.batch_size, *x.spatial_shape]
        spatial_stride = [0] * len(spatial_shape)
        val = 1
        for i in range(inds.shape[1] - 1, -1, -1):
            spatial_stride[i] = val
            val *= spatial_shape[i]
        indices_index = inds[:, -1].clone()

        for i in range(len(spatial_shape) - 1):
            indices_index += spatial_stride[i] * inds[:, i]

        _, unique_inds = torch.unique(indices_index, return_inverse=True)
        unique_inds = torch.unique(unique_inds)
        new_inds = inds[unique_inds]
        new_features = x.features[unique_inds]
        res = spconv.SparseConvTensor(new_features, new_inds, x.spatial_shape,
                                      x.batch_size, x.grid)
        return res

class MergeDuplicate(SparseModule):
    def __init__(self, reduce="max"):
        super().__init__()
        self.reduce = reduce
    def forward(self, x: spconv.SparseConvTensor):
        inds = x.indices
        spatial_shape = [x.batch_size, *x.spatial_shape]
        spatial_stride = [0] * len(spatial_shape)
        val = 1
        for i in range(inds.shape[1] - 1, -1, -1):
            spatial_stride[i] = val
            val *= spatial_shape[i]
        indices_index = inds[:, -1].clone()

        for i in range(len(spatial_shape) - 1):
            indices_index += spatial_stride[i] * inds[:, i]

        _, unique_inds = torch.unique(indices_index, return_inverse=True) # [0, 1, 0]

        scatter_feature = x.features # [N_point, features]
        scatter_indices = unique_inds # [N_point, ]

        out_feature = scatter(scatter_feature, scatter_indices, dim=0, reduce=self.reduce)  # [N', num_features]
        out_indices = scatter(scatter_indices, scatter_indices, dim=0, reduce="mean")
        out_indices = inds[out_indices] # [N', ndim+1] 

        res = spconv.SparseConvTensor(out_feature, out_indices, x.spatial_shape,
                                      x.batch_size, x.grid)
        return res


def fuseSparseTensor(x_list):
    """
        Suppose same spatial shape.
        Need eliminate same pos tensor later
    """
    new_features = torch.cat([x.features for x in x_list], dim=0)
    new_indice = torch.cat([x.indices for x in x_list], dim=0)
    res = spconv.SparseConvTensor(new_features, new_indice, x_list[0].spatial_shape,
                                        x_list[0].batch_size,  x_list[0].grid)
    return res


class warpSparseTensor(SparseModule):
    """
    warp the sparse tensor.
    1. Retrieve the indices
    2. turn indices to grid point
    3. transform grid point
    4. turn back to indices
    5. construct new sparse tensor
    Args:
        x: SparseTensor,
            spatial_shape:(z,y,x)
        transformation: torch.Tensor
            [4,4]
        voxel_size: torch.Tensor
            [v_x, v_y, v_z]
        range3d: list
            [xmin, xmax, ymin, ymax, zmin, zmax]

    """
    def indices_to_point(self, indices, transformation_matrix, voxel_size, range3d):
        """
            indices: [batch_id, z, y, x]
        """
        indices_xyz = indices[:,[3,2,1]].clone().double() # [x, y, z]
        indices_xyz[:,0] += torch.div(range3d[0], voxel_size[0])
        indices_xyz[:,1] += torch.div(range3d[1], voxel_size[1])
        indices_xyz[:,2] += torch.div(range3d[2], voxel_size[2])
        indices_xyz += 0.5

        points_xyz = indices_xyz * voxel_size # [N_points, 3]
        points_xyz_new = project_points_by_matrix_torch(points_xyz, transformation_matrix)

        return points_xyz_new

    def construct_new_tensor(self, x, points_xyz, voxel_size, range3d):
        """
            points_new: tensor
                [N_points, ndim + 1], first dim is batch id
        """
        mask = (points_xyz[:, 0] > range3d[0]) & (points_xyz[:, 0] < range3d[3])\
                & (points_xyz[:, 1] > range3d[1]) & (points_xyz[:, 1] < range3d[4]) \
                & (points_xyz[:, 2] > range3d[2]) & (points_xyz[:, 2] < range3d[5])

        features_new = x.features[mask]
        points_xyz = points_xyz[mask]
        new_indices = x.indices[mask].clone()

        new_indices_xyz = torch.div(points_xyz, voxel_size) 

        new_indices_xyz[:,0] -= torch.div(range3d[0], voxel_size[0])
        new_indices_xyz[:,1] -= torch.div(range3d[1], voxel_size[1])
        new_indices_xyz[:,2] -= torch.div(range3d[2], voxel_size[2])

        new_indices[:,1:] = new_indices_xyz[:,[2,1,0]].long()

        return spconv.SparseConvTensor(features_new, new_indices, x.spatial_shape, x.batch_size, x.grid)


    def forward(self, x, transformation_matrix, voxel_size, range3d):
        points_new = self.indices_to_point(x.indices, transformation_matrix, voxel_size, range3d)
        return self.construct_new_tensor(x, points_new, voxel_size, range3d)
        



def test():
    feature1 = torch.randn(2,8)
    feature2 = torch.randn(2,8)
    indices1 = torch.Tensor([[0,0,1,2],[0,0,2,3]])
    indices2 = torch.Tensor([[0,0,1,3],[0,0,2,4]])
    spatial_shape = (4,3,5) # z,y,x
    batch_size = 1
    
    voxel_size = (0.4, 0.4, 1)
    pc_range = [-40, -40, -3, 40, 40, 1]
    tfm = torch.eye(4)
    tfm[1,3] += 2
    warpsp = warpSparseTensor()
    sp1 = spconv.SparseConvTensor(feature1, indices1,spatial_shape, batch_size)
    sp2 = warpsp(sp1, tfm, voxel_size, pc_range)
    ic(sp1.features)
    ic(sp1.indices)
    ic(sp2.features)
    ic(sp2.indices)


if __name__ == "__main__":
    test()