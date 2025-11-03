
import torch
import torch.nn as nn
from torch.distributions import Normal

# -------------------------------
# Actor Network
# -------------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        z = dist.rsample()  # reparameterization trick
        action = torch.tanh(z) * self.max_action
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

# -------------------------------
# Critic Network
# -------------------------------
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

