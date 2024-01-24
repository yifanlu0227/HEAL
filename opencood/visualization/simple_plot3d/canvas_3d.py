"""
Written by Jinhyung Park

Simple 3D visualization for 3D points & boxes. Intended as a simple, hackable
alternative to mayavi for certain point cloud tasks.
"""

import numpy as np
import cv2
import copy
from functools import partial
import matplotlib

class Canvas_3D(object):
    def __init__(self,
                 canvas_shape=(500, 1000),
                 camera_center_coords=(-15, 0, 10),
                 camera_focus_coords=(-15 + 0.9396926, 0, 10 - 0.44202014),
                #  camera_center_coords=(-25, 0, 20),
                #  camera_focus_coords=(-25 + 0.9396926, 0, 20 - 0.64202014),
                 focal_length=None,
                 canvas_bg_color=(0, 0, 0), 
                 left_hand=True):
        """
        Args:
            canvas_shape (Tuple[Int]): Canvas image size - height & width.
            camera_center_coords (Tuple[Float]): Location of camera center in
                3D space.
            camera_focus_coords (Tuple[Float]): Intuitively, what point in 3D 
                space is the camera pointed at? These are absolute coordinates,
                *not* relative to camera center.
            focal_length (None | Int):
                None: Half of the max of height & width of canvas_shape. This
                    seems to be a decent default.
                Int: Specified directly.
            canvas_bg_color (Tuple[Int]): RGB (0 ~ 255) of canvas background
                color.
            left_hand: bool
                Since this algorithm is designed for right hand coord. We take -y if True
        """
        
        self.canvas_shape = canvas_shape
        self.H, self.W = self.canvas_shape
        self.canvas_bg_color = canvas_bg_color
        self.left_hand = left_hand
        if left_hand:
            camera_center_coords = list(camera_center_coords)
            camera_center_coords[1] = - camera_center_coords[1]
            camera_center_coords = tuple(camera_center_coords)
            
            camera_focus_coords = list(camera_focus_coords)
            camera_focus_coords[1] = - camera_focus_coords[1]
            camera_focus_coords = tuple(camera_focus_coords)

        self.camera_center_coords = camera_center_coords
        self.camera_focus_coords = camera_focus_coords

        if focal_length is None:
            self.focal_length = max(self.H, self.W) // 2
        else:
            self.focal_length = focal_length

        # Setup extrinsics and intrinsics of this virtual camera.
        self.ext_matrix = self.get_extrinsic_matrix(
            self.camera_center_coords, self.camera_focus_coords)
        self.int_matrix = np.array([
            [self.focal_length, 0, self.W // 2, 0],
            [0, self.focal_length, self.H // 2, 0],
            [0, 0, 1, 0],
        ])
        
        self.clear_canvas()
    
    def get_canvas(self):
        return self.canvas

    def clear_canvas(self):
        self.canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.canvas[..., :] = self.canvas_bg_color

    def get_canvas_coords(self, 
                          xyz, 
                          depth_min=0.1,
                          return_depth=False):
        """
        Projects XYZ points onto the canvas and returns the projected canvas
        coordinates.

        Args:
            xyz (ndarray): (N, 3+) array of coordinates. Additional columns
                beyond the first three are ignored.
            depth_min (Float): Only points with a projected depth larger
                than this value are "valid".
            return_depth (Boolean): Whether to additionally return depth of
                projected points.
        Returns:
            canvas_xy (ndarray): (N, 2) array of projected canvas coordinates.
                "x" is dim0, "y" is dim1 of canvas.
            valid_mask (ndarray): (N,) boolean mask indicating which of 
                canvas_xy fits into canvas (are visible from virtual camera).
            depth (ndarray): Optionally returned (N,) array of depth values
        """
        if self.left_hand:
            xyz[:,1] = - xyz[:,1]

        xyz = xyz[:,:3]
        xyz_hom = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
        img_pts = (self.int_matrix @ self.ext_matrix @ xyz_hom.T).T

        depth = img_pts[:, 2]
        xy = img_pts[:, :2] / depth[:, None]
        xy_int = xy.round().astype(np.int32)

        # Flip X and Y so "x" is dim0, "y" is dim1 of canvas
        xy_int = xy_int[:, ::-1] 

        valid_mask = ((depth > depth_min) &
                      (xy_int[:, 0] >= 0) & (xy_int[:, 0] < self.H) & 
                      (xy_int[:, 1] >= 0) & (xy_int[:, 1] < self.W))
        
        if return_depth:
            return xy_int, valid_mask, depth
        else:
            return xy_int, valid_mask
                                      

    def draw_canvas_points(self, 
                           canvas_xy,
                           radius=-1,
                           colors=None,
                           colors_operand=None):
        """
        Draws canvas_xy onto self.canvas.

        Args:
            canvas_xy (ndarray): (N, 2) array of *valid* canvas coordinates.
                "x" is dim0, "y" is dim1 of canvas.
            radius (Int): 
                -1: Each point is visualized as a single pixel.
                r: Each point is visualized as a circle with radius r.
            colors: 
                None: colors all points white.
                Tuple: RGB (0 ~ 255), indicating a single color for all points.
                ndarray: (N, 3) array of RGB values for each point.
                String: Such as "Spectral", uses a matplotlib cmap, with the
                    operand (the value cmap is called on for each point) being 
                    colors_operand.
            colors_operand (ndarray): (N,) array of values cooresponding to 
                canvas_xy, to be used only if colors is a cmap. Unlike 
                Canvas_BEV, cannot be None if colors is a String.
        """
        if len(canvas_xy) == 0:
            return 
            
        if colors is None:
            colors = np.full(
                (len(canvas_xy), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(canvas_xy), 3), dtype=np.uint8)
            colors_tmp[..., :len(colors)] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(canvas_xy)
            colors = colors.astype(np.uint8)
        elif isinstance(colors, str):
            assert colors_operand is not None
            colors = matplotlib.cm.get_cmap(colors)
                    
            # Normalize 0 ~ 1 for cmap
            colors_operand = colors_operand - colors_operand.min()
            colors_operand = colors_operand / colors_operand.max()
        
            # Get cmap colors - note that cmap returns (*input_shape, 4), with
            # colors scaled 0 ~ 1
            colors = (colors(colors_operand)[:, :3] * 255).astype(np.uint8)
        else:
            raise Exception(
                "colors type {} was not an expected type".format(type(colors)))

        if radius == -1:
            self.canvas[canvas_xy[:, 0], canvas_xy[:, 1], :] = colors
        else:
            for color, (x, y) in zip(colors.tolist(), canvas_xy.tolist()):
                self.canvas = cv2.circle(self.canvas, (y, x), radius, color, 
                                         -1, lineType=cv2.LINE_AA)
        
    def draw_lines(self,
                   canvas_xy, 
                   start_xyz,
                   end_xyz,
                   colors=(255, 255, 255),
                   thickness=1):
        """
        Draws lines between provided 3D points.
        
        Args:
            # added from original repo
            canvas_xy (ndarray): (N, 2) array of *valid* canvas coordinates.
                    "x" is dim0, "y" is dim1 of canvas.

            start_xyz (ndarray): Shape (N, 3) of 3D points to start from.
            end_xyz (ndarray): Shape (N, 3) of 3D points to end at. Same length
                as start_xyz.
            colors: 
                None: colors all points white.
                Tuple: RGB (0 ~ 255), indicating a single color for all points.
                ndarray: (N, 3) array of RGB values for each point.
            thickness (Int):
                Thickness of drawn cv2 line.            
        """
        if colors is None:
            colors = np.full(
                (len(canvas_xy), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(canvas_xy), 3), dtype=np.uint8)
            colors_tmp[..., :len(colors)] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(canvas_xy)
            colors = colors.astype(np.uint8)
        else:
            raise Exception(
                "colors type {} was not an expected type".format(type(colors)))
        
        start_pts_xy, start_pts_valid_mask, start_pts_d = \
            self.get_canvas_coords(start_xyz, True)
        end_pts_xy, end_pts_valid_mask, end_pts_d = \
            self.get_canvas_coords(end_xyz, True)

        for idx, (color, start_pt_xy, end_pt_xy) in enumerate(
                zip(colors.tolist(), start_pts_xy.tolist(), 
                    end_pts_xy.tolist())):
                
            if start_pts_valid_mask[idx] and end_pts_valid_mask[idx]:
                self.canvas = cv2.line(self.canvas,
                                    tuple(start_pt_xy[::-1]),
                                    tuple(end_pt_xy[::-1]),
                                    color=color,
                                    thickness=thickness, 
                                    lineType=cv2.LINE_AA)

    def draw_boxes(self,
                   boxes,
                   colors=None,
                   texts=None,
                   depth_min=0.1,
                   draw_incomplete_boxes=False,
                   box_line_thickness=2,
                   box_text_size=0.5,
                   text_corner=1):
        """
        Draws 3D boxes.

        Args:
            boxes (ndarray): Shape (N, 8, 3), corners in 3d
                modified from original repo

            colors: 
                None: colors all points white.
                Tuple: RGB (0 ~ 255), indicating a single color for all points.
                ndarray: (N, 3) array of RGB values for each point.

            texts (List[String]): Length N; text to write next to boxes.

            depth_min (Float): Only box corners with a projected depth larger
                than this value are drawn if draw_incomplete_boxes is True.

            draw_incomplete_boxes (Boolean): If any boxes are incomplete,
                meaning it has a corner out of view based on depth_min, decide
                whether to draw them at all.

            thickness (Int):
                Thickness of drawn cv2 box lines. 

            box_line_thickness (int): cv2 line/text thickness
            box_text_size (float): cv2 putText size
            text_corner (int): 0 ~ 7. Which corner of 3D box to write text at.
        """
        # Setup colors
        if colors is None:
            colors = np.full(
                (len(boxes), 3), fill_value=255, dtype=np.uint8)
        elif isinstance(colors, tuple):
            assert len(colors) == 3
            colors_tmp = np.zeros((len(boxes), 3), dtype=np.uint8)
            colors_tmp[..., :len(colors)] = np.array(colors)
            colors = colors_tmp
        elif isinstance(colors, np.ndarray):
            assert len(colors) == len(boxes)
            colors = colors.astype(np.uint8)
        else:
            raise Exception(
                "colors type {} was not an expected type".format(type(colors)))
        
        

        corners = boxes # N x 8 x 3

        # Now we have corners. Need them on the canvas 2D space.
        corners_xy, valid_mask = self.get_canvas_coords(
            corners.reshape(-1, 3), depth_min=depth_min)
        corners_xy = corners_xy.reshape(-1, 8, 2)
        valid_mask = valid_mask.reshape(-1, 8)

        # Now draw them with lines in correct places
        for i, (color, curr_corners_xy, curr_valid_mask) in enumerate(
            zip(colors.tolist(), corners_xy.tolist(), valid_mask.tolist())):

            if not draw_incomplete_boxes and sum(curr_valid_mask) != 8:
                # Some corner is invalid, don't draw the box at all.
                continue

            for start, end in [(0, 1), (1, 2), (2, 3), (3, 0),
                               (0, 4), (1, 5), (2, 6), (3, 7),
                               (4, 5), (5, 6), (6, 7), (7, 4)]:
                if not (curr_valid_mask[start] and curr_valid_mask[end]):
                    continue # start or end is not valid
                    
                self.canvas = cv2.line(
                    self.canvas, 
                    (curr_corners_xy[start][1], curr_corners_xy[start][0]),
                    (curr_corners_xy[end][1], curr_corners_xy[end][0]),
                    color=color,
                    thickness=box_line_thickness, 
                    lineType=cv2.LINE_AA)
            
            # If even a single line was drawn, add text as well.
            if sum(curr_valid_mask) > 0:
                if texts is not None:
                    self.canvas = cv2.putText(self.canvas,
                                            str(texts[i]),
                                            (curr_corners_xy[text_corner][1], 
                                             curr_corners_xy[text_corner][0]),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            box_text_size,
                                            color,
                                            thickness=box_line_thickness)


    @staticmethod
    def cart2sph(xyz):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        
        depth = np.linalg.norm(xyz, 2, axis=1)
        az = -np.arctan2(y, x)
        el = np.arcsin(z / depth)
        return az, el, depth

    @staticmethod
    def get_extrinsic_matrix(
        camera_center_coords,
        camera_focus_coords,
    ):
        """
        Args:
            camera_center_coords: (x, y, z) of where camera should be located 
                in 3D space.
            camera_focus_coords: (x, y, z) of where camera should look at from 
                camera_center_coords
            
        Thoughts:
            Remember that in camera coordiantes, pos x is right, pos y is up, 
                pos z is forward.
        """
        center_x, center_y, center_z = camera_center_coords
        focus_x, focus_y, focus_z = camera_focus_coords
        az, el, depth = Canvas_3D.cart2sph(np.array([
            [focus_x - center_x, focus_y - center_y, focus_z - center_z]
        ]))
        az = float(az)
        el = float(el)
        depth = float(depth)
        
        ### First, construct extrinsics
        ## Rotation matrix
        
        z_rot = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1]
        ])
        
        # el is rotation around y axis. 
        y_rot = np.array([
            [np.cos(-el), 0, -np.sin(-el)],
            [0, 1, 0],
            [np.sin(-el), 0, np.cos(-el)],
        ])
        
        
        ## Now, how the z_rot and y_rot work (spherical coordiantes), is it 
        ## computes rotations starting from the positive x axis and rotates 
        ## positive x axis to the desired direction. The desired direction is 
        ## the "looking direction" of the camera, which should actually be the 
        ## z-axis. So should convert the points so that the x axis is the new z 
        ## axis, and after the transformations.
        ## Why x -> z for points? If we think about rotating the camera, z 
        ## should become x, so reverse when moving points.
        last_rot = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0] # x -> z
        ])
        
        # Put them together. Order matters. Make it hom.
        rot_matrix = np.eye(4, dtype=np.float32)
        rot_matrix[:3, :3] = last_rot @ y_rot @ z_rot
        
        ## Translation matrix
        trans_matrix = np.array([
            [1, 0, 0, -center_x],
            [0, 1, 0, -center_y],
            [0, 0, 1, -center_z],
            [0, 0, 0, 1],
        ])
        
        ## Finally, extrinsics matrix. Order matters - do trans then rot
        ext_matrix =  rot_matrix @ trans_matrix
        
        return ext_matrix