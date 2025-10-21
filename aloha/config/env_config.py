# configs/env_config.py
# Edit these strings to match the prim paths / camera names in your USD stage or Lab config.

ENV_CFG = {
    "camera_prim": "/World/CameraMain",      # prim path for your camera
    "camera_width": 640,
    "camera_height": 480,
    "camera_fov": 60.0,
    "robot_prim": "/World/ALOHA",            # robot articulation prim path
    "output_root": "/tmp/aloha_dataset",     # where datasets will be written
    "replicator_num_images": 2000,
    "teleop_max_steps": 1000,
    "control_rate_hz": 20,
}
