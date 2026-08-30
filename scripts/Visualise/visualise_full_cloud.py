import numpy as np
import open3d as o3d
import os
import time

folder = "../../../figure_8_turnpike_2023-09-13-17-47-52/full_cloud"

# get all frames
files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])

# --- initialize visualizer ---
vis = o3d.visualization.Visualizer()
vis.create_window()

# --- load first frame ---
points = np.load(os.path.join(folder, files[0]))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# assign default color (gray)
colors = np.ones_like(points) * 0.7
pcd.colors = o3d.utility.Vector3dVector(colors)

vis.add_geometry(pcd)

# optional: add coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
vis.add_geometry(frame)

# optional: set camera zoom
ctr = vis.get_view_control()
ctr.set_zoom(0.8)

# --- loop through frames ---
for f in files:
    points = np.load(os.path.join(folder, f))

    pcd.points = o3d.utility.Vector3dVector(points)

    # keep same color
    colors = np.ones_like(points) * 0.7
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    time.sleep(0.05)

vis.destroy_window()