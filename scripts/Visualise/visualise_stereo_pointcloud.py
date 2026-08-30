import numpy as np
import open3d as o3d
import os
import time

folder = "../../../figure_8_turnpike_2023-09-13-17-47-52/stereo_colored_point_cloud"

files = sorted([f for f in os.listdir(folder) if f.endswith(".npy") and "_color" not in f])

vis = o3d.visualization.Visualizer()
vis.create_window()

# --- initialize with first frame ---
idx0 = files[0].replace(".npy", "")
points = np.load(os.path.join(folder, f"{idx0}.npy"))
colors = np.load(os.path.join(folder, f"{idx0}_color.npy"))

if colors.max() > 1:
    colors = colors / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

vis.add_geometry(pcd)

# --- loop ---
for f in files:
    idx = f.replace(".npy", "")

    points = np.load(os.path.join(folder, f"{idx}.npy"))
    colors = np.load(os.path.join(folder, f"{idx}_color.npy"))

    if colors.max() > 1:
        colors = colors / 255.0

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    time.sleep(0.05)

vis.destroy_window()