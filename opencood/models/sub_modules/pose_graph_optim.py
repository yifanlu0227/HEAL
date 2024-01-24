# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# This is pose graph optimizer, using g2o (bind to C++)

import g2o
import numpy as np

class PoseGraphOptimization2D(g2o.SparseOptimizer):
    def __init__(self, verbose=False):
        super().__init__()
        # solver = g2o.BlockSolverSE2(g2o.LinearSolverCholmodSE2())
        solver = g2o.BlockSolverSE2(g2o.LinearSolverDenseSE2())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        super().set_verbose(verbose)

    def optimize(self, max_iterations=1000):
        super().initialize_optimization()
        super().optimize(max_iterations)


    def add_vertex(self, id, pose, fixed=False, SE2=True):
        if SE2:
            v = g2o.VertexSE2()
        else:
            v = g2o.VertexPointXY()
        v.set_estimate(pose)
        v.set_id(id)
        v.set_fixed(fixed)
        super().add_vertex(v)


    def add_edge(self, vertices, measurement, 
            information=np.identity(3),
            robust_kernel=None, SE2 = True):
        """
        Args:
            measurement: g2o.SE2
        """
        if SE2:
            edge = g2o.EdgeSE2()
        else:
            edge = g2o.EdgeSE2PointXY()

        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose shape [3, 1] / [2, 1]
        edge.set_information(information)  # importance of each component shape [3, 3] / [2, 2]
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        super().set_verbose(True)

    def optimize(self, max_iterations=50):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_estimate(pose)
        v_se3.set_id(id)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose, shape [4, 4]
        edge.set_information(information)  # importance of each component, shape [6, 6]
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


if __name__ == "__main__":
    pgo = PoseGraphOptimization()

    with open("/GPFS/rhome/yifanlu/workspace/g2o_test/noise.g2o","r") as f:
        for line in f:
            if line.startswith("VERTEX_SE3:QUAT"):
                vertex_content = line.split(" ",1)[1]
                vertex_content_array = np.fromstring(vertex_content, dtype=float, sep=" ")
                ids = int(vertex_content_array[0])
                index = [0,1,2,6,3,4,5]
                pose_array = vertex_content_array[1:][index]

                pose = np.eye(4)
                pose[:3,3] = pose_array[:3]
                pose[:3,:3] = g2o.Quaternion(pose_array[3:]).matrix()
                pose = g2o.Isometry3d(pose)

                fixed = True if ids==6 else False
                # fixed = False
                pgo.add_vertex(id=ids, pose=pose, fixed=fixed)

            elif line.startswith("EDGE_SE3:QUAT"):
                edge_content = line.split(" ", 1)[1]
                edge_content_array = np.fromstring(edge_content, dtype=float, sep=" ")
                
                edge = [int(v) for v in edge_content_array[:2]]
                index = [0,1,2,6,3,4,5]
                pose_array = edge_content_array[2:2+7][index]
                information_array = edge_content_array[2+7:]

                pose = np.eye(4)
                pose[:3,3] = pose_array[:3]
                pose[:3,:3] = g2o.Quaternion(pose_array[3:]).matrix()
                pose = g2o.Isometry3d(pose)

                information = np.eye(6)
                information[0,0] = information_array[0]
                information[1,1] = information_array[6]
                information[2,2] = information_array[11]
                information[3,3] = information_array[15]
                information[4,4] = information_array[18]
                information[5,5] = information_array[20]

                pgo.add_edge(edge, pose, information)


    print('num vertices:', len(pgo.vertices()))
    print('num edges:', len(pgo.edges()), end='\n\n')
    pgo.optimize()

    # pgo.save("out_pose_graph2.g2o")
