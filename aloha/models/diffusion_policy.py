# models/diffusion_policy.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding(timesteps, dim):
    """
    timesteps: tensor (B,) of integers
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb

class MLPDenoiser(nn.Module):
    def __init__(self, cond_dim, action_dim, t_emb_dim=128, hidden=512):
        super().__init__()
        self.t_proj = nn.Linear(t_emb_dim, t_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(cond_dim + action_dim + t_emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        # If you want deeper, swap to SiLU layers etc.

    def forward(self, noisy_action, t_emb, cond):
        # noisy_action: [B, action_dim]
        # t_emb: [B, t_emb_dim]
        x = torch.cat([noisy_action, t_emb, cond], dim=1)
        return self.net(x)

class DiffusionPolicy(nn.Module):
    def __init__(self, cond_dim, action_dim, T=200, t_emb_dim=128):
        super().__init__()
        self.cond_dim = cond_dim
        self.action_dim = action_dim
        self.T = T
        betas = torch.linspace(1e-4, 2e-2, T)
        alphas = 1.0 - betas
        alpha_cum = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cum", alpha_cum)
        self.t_emb_dim = t_emb_dim
        self.denoiser = MLPDenoiser(cond_dim=cond_dim, action_dim=action_dim, t_emb_dim=t_emb_dim)

    def q_sample(self, x0, t):
        # x0: [B, action_dim]
        # t: [B] integer timesteps
        device = x0.device
        a = self.alpha_cum[t].unsqueeze(1).to(device)  # [B,1]
        noise = torch.randn_like(x0)
        xt = torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise
        return xt, noise

    def p_losses(self, x0, cond):
        """
        x0: [B, action_dim]
        cond: [B, cond_dim]
        """
        device = x0.device
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=device)
        xt, noise = self.q_sample(x0, t)
        t_emb = sinusoidal_embedding(t, self.t_emb_dim).to(device)
        pred = self.denoiser(xt, t_emb, cond)
        loss = F.mse_loss(pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, cond, device, steps=None, verbose=False):
        """
        Cond: [B, cond_dim]
        returns: x0_pred [B, action_dim]
        """
        B = cond.shape[0]
        action_dim = self.action_dim
        x = torch.randn(B, action_dim, device=device)  # start from noise
        steps = self.T if steps is None else steps
        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            t_emb = sinusoidal_embedding(t, self.t_emb_dim).to(device)
            # predict noise
            pred_noise = self.denoiser(x, t_emb, cond)
            beta_t = self.betas[i].to(device)
            alpha_t = self.alphas[i].to(device)
            alpha_cum_t = self.alpha_cum[i].to(device)
            # compute posterior mean as in DDPM
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1-alpha_cum_t) * pred_noise) + sigma_t * z
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1.0 - alpha_cum_t)
            mean = coef1 * (x - coef2 * pred_noise)
            if i > 0:
                sigma = torch.sqrt(beta_t)
                z = torch.randn_like(x)
                x = mean + sigma * z
            else:
                x = mean
            if verbose and (i % max(1, steps // 5) == 0):
                print(f"[diffusion] sampling step {i}")
        # After full reverse process, x is an estimate of x0
        return x
