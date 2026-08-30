import open3d as o3d

# Load the PLY file (reads point cloud data)
file_path = "../../../2024-10-01-11-29-55_dlio.ply"
geometry = o3d.io.read_point_cloud(file_path)

# Fallback to mesh loading if the PLY contains a polygonal 3D mesh instead of points
if not geometry.has_points():
    geometry = o3d.io.read_triangle_mesh(file_path)
    geometry.compute_vertex_normals()  # Computes surface lighting details

# Open the viewer window
o3d.visualization.draw_geometries(
    [geometry],
    window_name="Open3D PLY Viewer",
    width=1024,
    height=768
)