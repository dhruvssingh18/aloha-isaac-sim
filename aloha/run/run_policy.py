# inference/run_policy.py
"""
Run trained diffusion policy inside Isaac Lab / Isaac Sim.

This script should be executed inside Isaac Sim environment so that you can
read camera buffer and apply robot commands.
"""
import torch
import time
import numpy as np
from pathlib import Path
from configs.env_config import ENV_CFG
from models.diffusion_policy import DiffusionPolicy, sinusoidal_embedding
from training.train_diffusion_policy import SimpleEncoder  # reuse encoder class
import torchvision.transforms as T

CHECKPOINT = Path("/tmp/aloha_checkpoints/checkpoint_epoch_40.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 4
H = 6
ACTION_DIM = 7

transform = T.Compose([T.ToTensor()])

# ----------------------
# Isaac-specific hooks (ADAPT THESE)
# ----------------------
def get_camera_stack():
    """
    Return stacked RGB frames of shape (3*K, H, W) as torch.FloatTensor 0..1
    Implement by reading camera buffers or using pre-rendered frames.
    """
    # TODO: replace with actual readback
    import torch
    return torch.zeros((3 * K, ENV_CFG["camera_height"], ENV_CFG["camera_width"])).float()

def get_proprio():
    """
    Return current proprio vector as torch tensor.
    """
    return torch.zeros((14,), dtype=torch.float32)

def apply_action(action_vec):
    """
    action_vec: numpy or torch vector of size ACTION_DIM, representing the single step command.
    Convert to joint commands or EE target and send to your robot controller.
    """
    # TODO: replace with real apply call
    print("[run_policy] applying action:", action_vec[:6])

# ----------------------
# Load models
# ----------------------
def load_models():
    ck = torch.load(CHECKPOINT, map_location=DEVICE)
    encoder = SimpleEncoder(K=K, pretrained=False, feature_dim=512).to(DEVICE)
    diffusion = DiffusionPolicy(cond_dim=512 + 16, action_dim=H * ACTION_DIM, T=200).to(DEVICE)
    proprio_mlp = torch.nn.Sequential(torch.nn.Linear(14, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16)).to(DEVICE)

    encoder.load_state_dict(ck["encoder"])
    diffusion.load_state_dict(ck["diffusion"])
    proprio_mlp.load_state_dict(ck["proprio_mlp"])
    encoder.eval(); diffusion.eval(); proprio_mlp.eval()
    return encoder, proprio_mlp, diffusion

def main():
    encoder, proprio_mlp, diffusion = load_models()
    print("[run_policy] models loaded. Starting control loop.")
    try:
        while True:
            # 1) read sensors
            rgb_stack = get_camera_stack().unsqueeze(0).to(DEVICE)  # [1, 3K, H, W]
            proprio = get_proprio().unsqueeze(0).to(DEVICE)         # [1, proprio_dim]

            # 2) build cond
            with torch.no_grad():
                feat = encoder(rgb_stack)                      # [1,512]
                pfeat = proprio_mlp(proprio)                  # [1,16]
                cond = torch.cat([feat, pfeat], dim=1)        # [1, cond_dim]

                # 3) sample action chunk
                action_chunk = diffusion.sample(cond, device=DEVICE, steps=200)  # [1, H*action_dim]
                action_chunk = action_chunk.cpu().numpy().reshape(H, ACTION_DIM)
                first_action = action_chunk[0]

            # 4) apply only first action (receding horizon)
            apply_action(first_action)

            # control rate
            time.sleep(1.0 / ENV_CFG["control_rate_hz"])
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    main()
