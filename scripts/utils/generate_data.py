import os
import shutil
import subprocess
import yaml
import zarr
import numpy as np

def get_repo_root():
    """Finds the absolute root directory of the current git repository (traversibility)."""
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], 
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
        return root
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("Could not determine repository root. Ensure you are inside the git repository.")

# 1. Resolve paths anchored to git root (traversibility)
repo_root = get_repo_root()
id3100_dir = os.path.abspath(os.path.join(repo_root, ".."))

# Config file: repo_root/scripts/config/data_config.yaml
config_path = os.path.join(repo_root, "scripts/config/data_config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

folder_name = config.get("folder") 

# 2. Input and Output Paths
odom_path = os.path.join(id3100_dir, folder_name, "zed2i_vio_map")
img_timestamps_path = os.path.join(id3100_dir, folder_name, "zed2i_right_images_data")
images_dir = os.path.join(id3100_dir, folder_name, "zed2i_right_images")
output_dir = os.path.join(id3100_dir, f"{folder_name}_train")

os.makedirs(output_dir, exist_ok=True)

# 3. Load Timestamps and Odometry Positions
odom_root = zarr.open(odom_path, mode='r')
odo_timestamps = odom_root['timestamp'][:]
odo_pos = odom_root['pose_pos'][:]  # Shape: (N, 3) -> [x, y, z]

img_root = zarr.open(img_timestamps_path, mode='r')
image_timestamps = img_root['timestamp'][:]

# 4. Filter Odometry by 0.5m Displacement Threshold
threshold = 0.5  # meters
saved_indices = [0]
last_pos = odo_pos[0]

for i in range(1, len(odo_pos)):
    dist = np.linalg.norm(odo_pos[i] - last_pos)
    if dist >= threshold:
        saved_indices.append(i)
        last_pos = odo_pos[i]

print(f"Found {len(saved_indices)} reference poses with >={threshold}m displacement.")

# 5. Find Closest Timestamps & Copy the JPEG Image File
saved_count = 0
for idx in saved_indices:
    target_time = odo_timestamps[idx]
    
    # Nearest neighbor matching in time domain
    img_idx = np.argmin(np.abs(image_timestamps - target_time))
    
    # Map index to zero-padded filename (e.g., 5 -> 000005.jpeg)
    src_image_name = f"{img_idx:06d}.jpeg"
    src_image_path = os.path.join(images_dir, src_image_name)
    
    # Output name combining odom index and image index
    dst_image_name = f"{img_idx:06d}.jpeg"
    dst_image_path = os.path.join(output_dir, dst_image_name)
    
    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dst_image_path)
        saved_count += 1
    else:
        print(f"Warning: Source image missing at {src_image_path}")

print(f"Successfully copied {saved_count} images to: {output_dir}")