# data_generation/replicator_image_data.py
"""
Replicator-style image + metadata pipeline for Isaac Sim.

Run this script inside Isaac Sim / Isaac Lab (where omni.* is available).
Edit configs in configs/env_config.py.
"""
import os
import time
import json
import numpy as np
from PIL import Image
from pathlib import Path
from configs.env_config import ENV_CFG

OUTDIR = Path(ENV_CFG["output_root"]) / "sdr"
OUTDIR.mkdir(parents=True, exist_ok=True)
for sub in ("rgb", "depth", "seg", "meta", "proprio"):
    (OUTDIR / sub).mkdir(exist_ok=True)

NUM_IMAGES = ENV_CFG["replicator_num_images"]
W = ENV_CFG["camera_width"]
H = ENV_CFG["camera_height"]

# ----------------------
# Helper writers
# ----------------------
def write_rgb(img_arr, path):
    Image.fromarray(img_arr).save(path)

def write_depth(depth_arr, path):
    # save as float32 numpy (meters)
    np.save(path, depth_arr.astype(np.float32))

def write_seg(seg_arr, path):
    Image.fromarray(seg_arr.astype(np.uint8)).save(path)

def write_meta(meta, path):
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

# ----------------------
# Isaac-specific hooks (EDIT THESE)
# ----------------------
def setup_scene():
    """
    TODO: add stage / scene loading code if required.
    Example: add_reference_to_stage("/path/to/scene.usd", prim_path="/World/Scene")
    """
    print("[replicator] setup_scene() called. Make sure your scene is loaded in Isaac Sim.")

def get_camera_buffers():
    """
    Replace with code to read camera buffers from your Isaac Sim camera prim.
    Should return:
      rgb: HxWx3 uint8 array
      depth: HxW float32 array in meters (0 for invalid)
      seg:  HxW int array of instance ids (optional)
    Many Isaac examples use a RenderProduct or RenderBuffer readback.
    """
    # Placeholder: returns dummy images (replace with real readback)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.ones((H, W), dtype=np.float32) * 1.0
    seg = np.zeros((H, W), dtype=np.uint8)
    return rgb, depth, seg

def get_robot_proprio():
    """
    Replace with code to read robot joint positions/velocities/gripper state.
    Return a 1D numpy array (float32).
    """
    # Placeholder example: zeros
    return np.zeros((14,), dtype=np.float32)

# ----------------------
# Main loop
# ----------------------
def main():
    print("[replicator] starting.")
    setup_scene()
    for frame in range(NUM_IMAGES):
        # In actual replicator you would randomize scene parameters here
        # (lighting, textures, object poses). Add a randomizer call if needed.
        rgb, depth, seg = get_camera_buffers()
        proprio = get_robot_proprio()

        rgb_path = OUTDIR / "rgb" / f"frame_{frame:06d}.png"
        depth_path = OUTDIR / "depth" / f"frame_{frame:06d}.npy"
        seg_path = OUTDIR / "seg" / f"frame_{frame:06d}.png"
        meta_path = OUTDIR / "meta" / f"frame_{frame:06d}.json"
        proprio_path = OUTDIR / "proprio" / f"frame_{frame:06d}.npy"

        write_rgb(rgb, rgb_path)
        write_depth(depth, depth_path)
        write_seg(seg, seg_path)
        np.save(proprio_path, proprio)

        meta = {
            "frame": frame,
            "time": time.time(),
            "camera": {
                "width": W,
                "height": H,
                "fov": ENV_CFG["camera_fov"]
            },
            # include scene seed / randomizer params here if you randomize
        }
        write_meta(meta, meta_path)

        if frame % 50 == 0:
            print(f"[replicator] wrote frame {frame}/{NUM_IMAGES}")

        # Keep a realistic rate if desired (not required for SDR)
        # time.sleep(0.01)

    print("[replicator] finished. Dataset at:", OUTDIR)

if __name__ == "__main__":
    main()
