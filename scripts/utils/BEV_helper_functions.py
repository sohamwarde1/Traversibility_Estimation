import open3d as o3d
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import glob
import os
import time
import re
import numpy as np

def get_total_transform(
    quat=(
        -0.45720033826192996,
        -0.4523183402260819,
        -0.5455934690231464,
        0.5373115821826182,
    ),
    trans=(-0.012769940823864567, -0.07826319215076401, -0.39482353374563656),
    apply_optical_to_body: bool = True,
) -> np.ndarray:

    T_base = np.eye(4)
    T_base[:3, :3] = R.from_quat(quat).as_matrix()
    T_base[:3, 3] = trans

    if not apply_optical_to_body:
        return T_base

    # ROS Optical (+X right, +Y down, +Z forward) to Body frame alignment
    T_optical_to_body = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

    return T_base @ T_optical_to_body

def build_pcd(rgb_path, depth_path):

    fx, fy, cx, cy = 1052.19970703125, 1052.19970703125, 956.3457641601562, 553.95703125

    depth_scale = 1000.0   # mm -> meters
    depth_trunc = 30.0

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.array(Image.open(depth_path))  # uint16, mm

    h, w = depth.shape
    rgb_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    # Transform from camera optical frame to box_base frame
    T_TOTAL = get_total_transform()
    pcd.transform(T_TOTAL)
    return pcd

def generate_bev_grid(
    points: np.ndarray,
    heights: np.ndarray,
    cell_size: float = 0.05,
    x_range: tuple = (0.0, 10.0),
    y_range: tuple = (-5.0, 5.0),
    colors: np.ndarray = None,
    normals: np.ndarray = None,
) -> np.ndarray:

    x_min, x_max = x_range
    y_min, y_max = y_range

    # Calculate grid dimensions
    H = int(np.round((x_max - x_min) / cell_size))  # Forward cells (rows)
    W = int(np.round((y_max - y_min) / cell_size))  # Lateral cells (cols)
    num_cells = H * W

    # Filter points within bounds
    x, y = points[:, 0], points[:, 1]
    valid_mask = (
        (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max) & ~np.isnan(heights)
    )

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = heights[valid_mask]

    # Map coordinates to 2D grid cell indices
    row_idx = np.floor((x_valid - x_min) / cell_size).astype(np.int64)
    col_idx = np.floor((y_valid - y_min) / cell_size).astype(np.int64)

    # row_idx = np.clip(row_idx, 0, H - 1)
    # col_idx = np.clip(col_idx, 0, W - 1)

    row_idx = np.floor((x_max - x_valid) / cell_size).astype(np.int64)

    # Column index: y_min (left side) -> Col 0 (left of image)
    #               y_max (right side) -> Col W - 1 (right of image)
    col_idx = np.floor((y_valid - y_min) / cell_size).astype(np.int64)

    row_idx = np.clip(row_idx, 0, H - 1)
    col_idx = np.clip(col_idx, 0, W - 1)
    

    flat_idx = row_idx * W + col_idx

    # Initialize cell statistics
    counts = np.bincount(flat_idx, minlength=num_cells)
    occupied = counts > 0

    z_min = np.full(num_cells, np.nan)
    z_max = np.full(num_cells, np.nan)
    z_mean = np.full(num_cells, np.nan)
    z_std = np.full(num_cells, np.nan)

    # Vectorized accumulation for height statistics
    z_sum = np.bincount(flat_idx, weights=z_valid, minlength=num_cells)
    z_sq_sum = np.bincount(flat_idx, weights=z_valid**2, minlength=num_cells)

    z_min_tmp = np.full(num_cells, np.inf)
    z_max_tmp = np.full(num_cells, -np.inf)
    np.minimum.at(z_min_tmp, flat_idx, z_valid)
    np.maximum.at(z_max_tmp, flat_idx, z_valid)

    # Compute stats for occupied cells
    z_mean[occupied] = z_sum[occupied] / counts[occupied]
    z_var = (z_sq_sum[occupied] / counts[occupied]) - (z_mean[occupied] ** 2)
    z_std[occupied] = np.sqrt(np.maximum(z_var, 0.0))
    z_min[occupied] = z_min_tmp[occupied]
    z_max[occupied] = z_max_tmp[occupied]

    channels = [
        z_min.reshape(H, W),
        z_max.reshape(H, W),
        z_mean.reshape(H, W),
        z_std.reshape(H, W),
        counts.astype(np.float32).reshape(H, W),
        occupied.astype(np.float32).reshape(H, W),
    ]

    # Optional Channel Aggregations (RGB / Surface Normals)
    if colors is not None:
        colors_valid = colors[valid_mask]
        for c in range(colors_valid.shape[1]):
            c_sum = np.bincount(
                flat_idx, weights=colors_valid[:, c], minlength=num_cells
            )
            c_mean = np.full(num_cells, np.nan)
            c_mean[occupied] = c_sum[occupied] / counts[occupied]
            channels.append(c_mean.reshape(H, W))

    if normals is not None:
        normals_valid = normals[valid_mask]
        for n in range(normals_valid.shape[1]):
            n_sum = np.bincount(
                flat_idx, weights=normals_valid[:, n], minlength=num_cells
            )
            n_mean = np.full(num_cells, np.nan)
            n_mean[occupied] = n_sum[occupied] / counts[occupied]
            channels.append(n_mean.reshape(H, W))

    return np.stack(channels, axis=0)

def generate_bev_from_o3d(
    pcd: o3d.geometry.PointCloud, x_range, y_range
) -> np.ndarray:
    pts_3d = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    # Open3D Camera Coordinates:
    # forward_dist = Z (pts_3d[:, 2])
    # lateral_dist = X (pts_3d[:, 0])
    # height       = -Y (pts_3d[:, 1], inverted because +Y points down in camera frame)

    bev_points = np.column_stack((pts_3d[:, 2], pts_3d[:, 0]))  # Forward (Z), Lateral (X)
    heights = pts_3d[:, 1]  # Invert Y so up is positive

    return generate_bev_grid(
        points=bev_points,
        heights=heights,
        x_range=x_range,
        y_range=y_range,
        colors=colors,
        normals=normals,
    )

