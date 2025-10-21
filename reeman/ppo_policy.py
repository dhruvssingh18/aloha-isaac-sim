"""
ppo_isaac_reeman.py

PPO training for a Reeman chassis in Isaac Sim using direct API (no Gym),
with sensor inputs (LIDAR, IMU, odometry, joints) and domain randomization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random

# -------------------------------
# Policy & Value networks
# -------------------------------
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        return mu, std


class Value(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------------
# Domain randomization
# -------------------------------
def sample_domain_randomization():
    """Return a dictionary of randomized parameters for this episode"""
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
# Isaac Sim placeholders
# -------------------------------

class ReemanSim:
    """
    Replace these methods with your Isaac Sim Python API calls
    """

    def __init__(self, n_lidar=64, max_lidar_range=10.0):
        self.n_lidar = n_lidar
        self.max_lidar_range = max_lidar_range
        self.domain_params = None
        # internal state placeholder
        self.state = None

    def reset(self, domain_params):
        self.domain_params = domain_params
        # Reset your robot and environment here, apply domain randomization
        # Return initial observation vector
        # Example placeholder:
        self.state = {
            "pose": np.array([0.0, 0.0, 0.0]),
            "vel": np.array([0.0, 0.0]),
            "joints": np.array([0.0, 0.0]),
            "lidar": np.ones(self.n_lidar) * self.max_lidar_range,
            "imu": np.zeros(3)
        }
        return self.get_observation()

    def step(self, action):
        """
        Apply wheel commands to robot in Isaac Sim
        Return next observation, reward, done
        """
        # TODO: Replace this with Isaac Sim API call to move chassis one step
        left_cmd, right_cmd = action
        max_wheel_speed = self.domain_params.get("max_wheel_speed", 5.0)
        left_speed = left_cmd * max_wheel_speed
        right_speed = right_cmd * max_wheel_speed

        # Update pose kinematics (placeholder)
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

        # Lidar + IMU noise
        lidar = self.state["lidar"] + np.random.normal(0, self.domain_params["lidar_noise"], self.n_lidar)
        lidar = np.clip(lidar, 0, self.max_lidar_range) / self.max_lidar_range
        imu = self.state["imu"] + np.random.normal(0, self.domain_params["imu_noise"], 3)

        self.state["lidar"] = lidar
        self.state["imu"] = imu

        obs = self.get_observation()
        reward = linear_vel * dt  # simple forward reward
        done = False
        if lidar.min() < self.domain_params["collision_threshold"]:
            reward -= 5.0
            done = True

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
# PPO Training Loop
# -------------------------------

class PPOBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.done = []

    def add(self, obs, action, reward, log_prob, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.done.append(done)

    def clear(self):
        self.__init__()

# PPO update function
def ppo_update(policy, value_net, optimizer_policy, optimizer_value, buffer, gamma=0.99, eps_clip=0.2, n_epochs=10):
    obs = torch.tensor(buffer.obs, dtype=torch.float32)
    actions = torch.tensor(buffer.actions, dtype=torch.float32)
    rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
    values = torch.tensor(buffer.values, dtype=torch.float32)
    done = torch.tensor(buffer.done, dtype=torch.float32)

    # Compute advantages (simple discounted returns - generalized advantage can be added)
    returns = []
    R = 0
    for r, d in zip(reversed(rewards.numpy()), reversed(done.numpy())):
        R = r + gamma * R * (1 - d)
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - values.squeeze()

    for _ in range(n_epochs):
        mu, std = policy(obs)
        dist = Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(-1)
        ratio = torch.exp(new_log_probs - old_log_probs.sum(-1))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(value_net(obs).squeeze(), returns)

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

# -------------------------------
# Main training
# -------------------------------

def main():
    n_lidar = 64
    obs_dim = n_lidar + 3 + 5 + 2  # lidar + imu + pose/vel + joints
    action_dim = 2  # left/right wheel commands

    policy = Policy(obs_dim, action_dim)
    value_net = Value(obs_dim)
    optimizer_policy = optim.Adam(policy.parameters(), lr=3e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=3e-4)

    sim = ReemanSim(n_lidar=n_lidar)

    buffer = PPOBuffer()
    max_steps_per_episode = 1000
    num_episodes = 5000

    for ep in range(num_episodes):
        domain_params = sample_domain_randomization()
        obs = sim.reset(domain_params)
        done = False
        step = 0
        ep_reward = 0

        while not done and step < max_steps_per_episode:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            mu, std = policy(obs_tensor)
            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum().item()
            value = value_net(obs_tensor).item()

            action_np = action.detach().numpy()
            next_obs, reward, done = sim.step(action_np)

            buffer.add(obs, action_np, reward, log_prob, value, done)

            obs = next_obs
            ep_reward += reward
            step += 1

        # PPO update after each episode
        ppo_update(policy, value_net, optimizer_policy, optimizer_value, buffer)
        buffer.clear()
        print(f"Episode {ep}, Reward: {ep_reward:.2f}, Steps: {step}")

        # Optional: save model periodically
        if ep % 100 == 0:
            torch.save(policy.state_dict(), f"ppo_policy_ep{ep}.pt")
            torch.save(value_net.state_dict(), f"ppo_value_ep{ep}.pt")

if __name__ == "__main__":
    main()
