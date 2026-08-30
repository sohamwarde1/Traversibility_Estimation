"""
Show BEV (bird's-eye-view) grid maps from TartanDrive 2.0 style exports.

Assumes:
  - metadata.yaml, timestamps.txt in data_dir
  - one grid_map_XXXX.npy per timestep, each shaped (16, 480, 480),
    matched in order to lines in timestamps.txt

Usage:
    python show_bev.py /path/to/data_dir --frame 0                  # single layer BEV
    python show_bev.py /path/to/data_dir --frame 0 --layer terrain  # pick layer
    python show_bev.py /path/to/data_dir --frame 0 --all-layers     # 4x4 grid of all layers
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_metadata(data_dir):
    with open(os.path.join(data_dir, "metadata.yaml")) as f:
        meta = yaml.safe_load(f)
    meta["n_cells_x"] = int(round(meta["width"] / meta["resolution"]))
    meta["n_cells_y"] = int(round(meta["height"] / meta["resolution"]))
    return meta


def load_timestamps(data_dir):
    with open(os.path.join(data_dir, "timestamps.txt")) as f:
        lines = [line.strip() for line in f if line.strip()]
    try:
        return [float(x) for x in lines]
    except ValueError:
        return lines


def find_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "grid_map_*.npy")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    return files


def extent_from_meta(meta):
    """Real-world (x_min, x_max, y_min, y_max) extent in meters for imshow."""
    x0, y0 = meta["origin"]
    x1 = x0 + meta["width"]
    y1 = y0 + meta["height"]
    return [x0, x1, y0, y1]


def robust_vrange(arr, low=1, high=99):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [low, high])
    if vmin == vmax:
        vmin, vmax = finite.min(), finite.max()
    return vmin, vmax


def show_single_bev(data_dir, frame_idx, layer="terrain"):
    """Single top-down BEV image for one layer, axes in real-world meters."""
    meta = load_metadata(data_dir)
    timestamps = load_timestamps(data_dir)
    files = find_files(data_dir)

    feature_keys = meta["feature_keys"]
    stack = np.load(files[frame_idx])              # (16, 480, 480)
    layer_dict = dict(zip(feature_keys, stack))
    bev = layer_dict[layer]

    vmin, vmax = robust_vrange(bev)
    extent = extent_from_meta(meta)
    ts = timestamps[frame_idx] if frame_idx < len(timestamps) else "?"

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        bev, cmap="terrain", origin="lower",
        extent=extent, vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"BEV: {layer} | frame {frame_idx} | t={ts}")
    plt.colorbar(im, ax=ax, label=layer, fraction=0.046)
    plt.tight_layout()
    plt.show()


def show_all_layers_bev(data_dir, frame_idx):
    """4x4 grid: every channel of the BEV stack for one timestep."""
    meta = load_metadata(data_dir)
    timestamps = load_timestamps(data_dir)
    files = find_files(data_dir)

    feature_keys = meta["feature_keys"]
    stack = np.load(files[frame_idx])               # (16, 480, 480)
    layer_dict = dict(zip(feature_keys, stack))
    extent = extent_from_meta(meta)
    ts = timestamps[frame_idx] if frame_idx < len(timestamps) else "?"

    ncols = 4
    nrows = int(np.ceil(len(feature_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, key in zip(axes_flat, feature_keys):
        data = layer_dict[key]
        vmin, vmax = robust_vrange(data)
        im = ax.imshow(data, cmap="terrain", origin="lower",
                        extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(key, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_flat[len(feature_keys):]:
        ax.axis("off")

    fig.suptitle(f"Full BEV stack | frame {frame_idx} | t={ts}", fontsize=14)
    plt.tight_layout()
    plt.show()


def step_through_bev(data_dir, start_frame=0, layer="terrain"):
    """
    Interactive single-layer BEV viewer.
    Controls:
      -> or 'd'  : next frame
      <- or 'a'  : previous frame
      up/down    : jump +/- 10 frames
      'q'        : quit
    """
    meta = load_metadata(data_dir)
    timestamps = load_timestamps(data_dir)
    files = find_files(data_dir)
    feature_keys = meta["feature_keys"]
    extent = extent_from_meta(meta)
    n_frames = len(files)

    state = {"idx": start_frame}

    def get_layer(idx):
        stack = np.load(files[idx])  # (16, 480, 480)
        return dict(zip(feature_keys, stack))[layer]

    fig, ax = plt.subplots(figsize=(7, 7))
    data0 = get_layer(state["idx"])
    vmin, vmax = robust_vrange(data0)
    im = ax.imshow(data0, cmap="terrain", origin="lower",
                    extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label=layer, fraction=0.046)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    def redraw():
        idx = state["idx"]
        data = get_layer(idx)
        vmin, vmax = robust_vrange(data)
        im.set_data(data)
        im.set_clim(vmin, vmax)
        ts = timestamps[idx] if idx < len(timestamps) else "?"
        ax.set_title(f"BEV: {layer} | frame {idx}/{n_frames - 1} | t={ts}")
        fig.canvas.draw_idle()

    def on_key(event):
        idx = state["idx"]
        if event.key in ("right", "d"):
            idx += 1
        elif event.key in ("left", "a"):
            idx -= 1
        elif event.key == "up":
            idx += 10
        elif event.key == "down":
            idx -= 10
        elif event.key == "q":
            plt.close(fig)
            return
        else:
            return
        state["idx"] = max(0, min(n_frames - 1, idx))
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    print("Controls: -> / d = next, <- / a = prev, up/down = +/-10 frames, q = quit")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show BEV grid map(s)")
    parser.add_argument("data_dir")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--layer", type=str, default="terrain",
                         help="Layer to show (ignored if --all-layers)")
    parser.add_argument("--all-layers", action="store_true",
                         help="Show all 16 channels in a grid instead of one")
    parser.add_argument("--step", action="store_true",
                         help="Interactive step-through viewer (arrow keys / a,d, q to quit)")
    args = parser.parse_args()

    if args.step:
        step_through_bev(args.data_dir, start_frame=args.frame, layer=args.layer)
    elif args.all_layers:
        show_all_layers_bev(args.data_dir, args.frame)
    else:
        show_single_bev(args.data_dir, args.frame, layer=args.layer)