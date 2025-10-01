# training/dataset_loader.py
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T

class MultimodalActionChunkDataset(Dataset):
    """
    Expects structure:
      root/
        rgb/frame_000000.png
        depth/frame_000000.npy
        proprio/frame_000000.npy
        action/frame_000000.npy      # single-step expert action
    This dataset returns:
      cond: images (stacked or single) + proprio fused externally
      action_chunk: flattened vector action[t:t+H] of shape (H * action_dim,)
    """
    def __init__(self, root, context_K=4, future_H=6, action_dim=7, transform=None):
        self.root = root
        self.rgb_glob = sorted(glob.glob(os.path.join(root, "rgb", "frame_*.png")))
        self.depth_glob = sorted(glob.glob(os.path.join(root, "depth", "frame_*.npy")))
        self.proprio_glob = sorted(glob.glob(os.path.join(root, "proprio", "frame_*.npy")))
        self.action_glob = sorted(glob.glob(os.path.join(root, "action", "frame_*.npy")))
        length = min(len(self.rgb_glob), len(self.depth_glob), len(self.proprio_glob), len(self.action_glob))
        self.rgb_glob = self.rgb_glob[:length]
        self.depth_glob = self.depth_glob[:length]
        self.proprio_glob = self.proprio_glob[:length]
        self.action_glob = self.action_glob[:length]
        self.transform = transform if transform is not None else T.Compose([T.ToTensor()])
        self.K = context_K
        self.H = future_H
        self.action_dim = action_dim
        self.length = length

    def __len__(self):
        # drop last frames that don't have a full future chunk
        return max(0, self.length - self.H - self.K)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)  # returns CxHxW (float tensor 0..1)

    def _load_depth(self, path):
        d = np.load(path)
        # normalize or keep meters — keep as float32 numpy
        # convert to tensor [1,H,W]
        return torch.from_numpy(d).unsqueeze(0).float()

    def __getitem__(self, idx):
        # choose window start such that we can return K context images and H future actions
        # context images: frames idx .. idx+K-1
        # action chunk: frames idx+K .. idx+K+H-1 (each action file corresponds to that timestep)
        start = idx
        imgs = []
        depths = []
        for i in range(start, start + self.K):
            imgs.append(self._load_image(self.rgb_glob[i]))
            depths.append(self._load_depth(self.depth_glob[i]))
        # stack along channel
        imgs = torch.cat(imgs, dim=0)        # (3*K, H, W)
        depths = torch.cat(depths, dim=0)    # (1*K, H, W) but we concatenated 1 per frame
        proprio = torch.from_numpy(np.load(self.proprio_glob[start + self.K - 1])).float()
        # action chunk:
        actions = []
        for j in range(start + self.K, start + self.K + self.H):
            a = np.load(self.action_glob[j]).astype(np.float32)
            actions.append(a)
        actions = np.stack(actions, axis=0)  # H x action_dim
        actions_flat = torch.from_numpy(actions.reshape(-1)).float()
        cond = {
            "rgb": imgs,
            "depth": depths,
            "proprio": proprio
        }
        return cond, actions_flat
