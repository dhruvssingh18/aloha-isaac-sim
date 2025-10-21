"""
sac_isaac_reeman.py

Soft Actor-Critic (SAC) training for a Reeman chassis in Isaac Sim
(no Gym). Uses sensor inputs, domain randomization, and off-policy replay buffer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque

# -------------------------------
# Networks
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


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

# -------------------------------
# Replay Buffer
# -------------------------------

class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = map(np.array, zip(*batch))
        return (torch.tensor(obs, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(next_obs, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(-1))

    def __len__(self):
        return len(self.buffer)

# -------------------------------
# Domain randomization
# -------------------------------

def sample_domain_randomization():
    params = {
        "mass_mult": random.uniform(0.8, 1.2),
        "wheel_friction": random.uniform(0.6, 1.4),
        "motor_gain": random.uniform(0.9, 1.1),
        "lidar_noise": random.uniform(0.0, 0.05),
        "imu_noise": random.uniform(0.0, 0.1),
        "wheel_base": random.uniform(0.28, 0.35),
        "max_wheel_speed": random.uniform(3.0, 6.0),
        "obstacle_radius": random.uniform(0.15, 0.5),
        "collision_threshold": random.uniform(0.15, 0.5),
    }
    return params

# -------------------------------
# Isaac Sim placeholder
# -------------------------------

class ReemanSim:
    def __init__(self, n_lidar=64, max_lidar_range=10.0):
        self.n_lidar = n_lidar
        self.max_lidar_range = max_lidar_range
        self.state = None
        self.domain_params = None

    def reset(self, domain_params):
        self.domain_params = domain_params
        # Replace with Isaac Sim reset & domain parameter application
        self.state = {
            "pose": np.array([0.0, 0.0, 0.0]),
            "vel": np.array([0.0, 0.0]),
            "joints": np.array([0.0, 0.0]),
            "lidar": np.ones(self.n_lidar) * self.max_lidar_range,
            "imu": np.zeros(3)
        }
        return self.get_observation()

    def step(self, action):
        # Replace with Isaac Sim step call applying wheel commands
        left_cmd, right_cmd = action
        max_wheel_speed = self.domain_params["max_wheel_speed"]
        left_speed = left_cmd * max_wheel_speed
        right_speed = right_cmd * max_wheel_speed

        x, y, yaw = self.state["pose"]
        linear_vel = 0.5 * (left_speed + right_speed)
        angular_vel = (right_speed - left_speed) / (self.domain_params["wheel_base"] + 1e-9)
        dt = 0.05
        x += linear_vel * np.cos(yaw) * dt
        y += linear_vel * np.sin(yaw) * dt
        yaw += angular_vel * dt

        self.state["pose"] = np.array([x, y, yaw])
        self.state["vel"] = np.array([linear_vel, angular_vel])
        self.state["joints"] = np.array([left_speed, right_speed])

        lidar = self.state["lidar"] + np.random.normal(0, self.domain_params["lidar_noise"], self.n_lidar)
        lidar = np.clip(lidar, 0, self.max_lidar_range) / self.max_lidar_range
        imu = self.state["imu"] + np.random.normal(0, self.domain_params["imu_noise"], 3)

        self.state["lidar"] = lidar
        self.state["imu"] = imu

        obs = self.get_observation()
        reward = linear_vel * dt
        done = lidar.min() < self.domain_params["collision_threshold"]

        return obs, reward, done

    def get_observation(self):
        obs = np.concatenate([
            self.state["lidar"].astype(np.float32),
            self.state["imu"].astype(np.float32),
            np.array([*self.state["pose"], *self.state["vel"]], dtype=np.float32),
            self.state["joints"].astype(np.float32)
        ])
        return obs

# -------------------------------
# SAC Training
# -------------------------------

def soft_q_update(actor, critic, critic_target, buffer, actor_opt, critic_opt, gamma=0.99, alpha=0.2, tau=0.005, batch_size=256):
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)

    # Q-target
    with torch.no_grad():
        next_action, next_log_prob = actor.sample(next_obs)
        q1_next, q2_next = critic_target(next_obs, next_action)
        q_target = rewards + gamma * (1 - dones) * (torch.min(q1_next, q2_next) - alpha * next_log_prob)

    q1, q2 = critic(obs, actions)
    critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # Actor loss
    new_action, log_prob = actor.sample(obs)
    q1_new, q2_new = critic(obs, new_action)
    actor_loss = (alpha * log_prob - torch.min(q1_new, q2_new)).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # Soft update target network
    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# -------------------------------
# Main loop
# -------------------------------

def main():
    n_lidar = 64
    obs_dim = n_lidar + 3 + 5 + 2
    action_dim = 2

    actor = Actor(obs_dim, action_dim)
    critic = Critic(obs_dim, action_dim)
    critic_target = Critic(obs_dim, action_dim)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-4)

    buffer = ReplayBuffer(max_size=1_000_000)
    sim = ReemanSim(n_lidar=n_lidar)

    num_episodes = 5000
    max_steps = 1000
    batch_size = 256

    for ep in range(num_episodes):
        domain_params = sample_domain_randomization()
        obs = sim.reset(domain_params)
        ep_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, _ = actor.sample(obs_tensor)
            action_np = action.detach().numpy()

            next
