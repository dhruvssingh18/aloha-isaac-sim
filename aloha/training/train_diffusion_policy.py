# training/train_diffusion_policy.py
import os
import torch
from torch.utils.data import DataLoader
from training.dataset_loader import MultimodalActionChunkDataset
from models.diffusion_policy import DiffusionPolicy
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Config
ROOT = "/tmp/aloha_dataset/sdr"   # same as replicator output root
BATCH = 32
EPOCHS = 40
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 4
H = 6
ACTION_DIM = 7   # change to your action size (e.g., Δx(3)+Δquat(4) or Δx+Δr+grip)
OUT_CHECKPOINT = Path("/tmp/aloha_checkpoints")
OUT_CHECKPOINT.mkdir(parents=True, exist_ok=True)

# Simple image encoder to produce cond vector (frozen or learned)
class SimpleEncoder(nn.Module):
    def __init__(self, K=4, pretrained=False, feature_dim=512):
        super().__init__()
        # Use a small resnet backbone from torchvision
        self.backbone = models.resnet18(pretrained=pretrained)
        # adjust to accept 3*K channels (stacked RGB frames)
        self.backbone.conv1 = nn.Conv2d(3 * K, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, rgb_stack):
        # rgb_stack: [B, 3*K, H, W] (float tensor 0..1)
        feat = self.backbone(rgb_stack)
        return feat

def collate_fn(batch):
    conds = [b[0] for b in batch]
    actions = [b[1] for b in batch]
    # stack rgb stacks: each cond["rgb"] is tensor (3*K, H, W)
    rgbs = torch.stack([c["rgb"] for c in conds], dim=0)
    proprios = torch.stack([c["proprio"] for c in conds], dim=0)
    actions = torch.stack(actions, dim=0)
    return rgbs, proprios, actions

def main():
    ds = MultimodalActionChunkDataset(ROOT, context_K=K, future_H=H, action_dim=ACTION_DIM,
                                      transform=None)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, collate_fn=collate_fn)

    encoder = SimpleEncoder(K=K, pretrained=False, feature_dim=512).to(DEVICE)
    diffusion = DiffusionPolicy(cond_dim=512 + 16, action_dim=H * ACTION_DIM, T=200).to(DEVICE)
    # Note: we append proprio into cond; here we map proprio to 16-dim via small MLP
    proprio_mlp = nn.Sequential(nn.Linear(14, 64), nn.ReLU(), nn.Linear(64, 16)).to(DEVICE)

    opt = torch.optim.AdamW(list(diffusion.parameters()) + list(encoder.parameters()) + list(proprio_mlp.parameters()),
                            lr=LR, weight_decay=1e-4)

    for epoch in range(EPOCHS):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0
        for rgbs, proprios, actions in pbar:
            rgbs = rgbs.to(DEVICE)            # [B, 3K, H, W]
            proprios = proprios.to(DEVICE)
            actions = actions.to(DEVICE)      # [B, H*action_dim]

            # forward cond
            feats = encoder(rgbs)             # [B,512]
            pfeat = proprio_mlp(proprios)     # [B,16]
            cond = torch.cat([feats, pfeat], dim=1)  # [B, cond_dim=528]

            # target x0 is action chunk (H*action_dim)
            loss = diffusion.p_losses(actions, cond)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

        # save checkpoint each epoch
        torch.save({
            "encoder": encoder.state_dict(),
            "proprio_mlp": proprio_mlp.state_dict(),
            "diffusion": diffusion.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
        }, OUT_CHECKPOINT / f"checkpoint_epoch_{epoch+1}.pth")
        print(f"[train] epoch {epoch+1} saved.")

if __name__ == "__main__":
    main()
