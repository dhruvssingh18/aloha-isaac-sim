
# scripts/rl_train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import yaml
from policy import Actor, Critic
from environment import Environment
import os

# -------------------------------
# Load YAML configs
# -------------------------------
with open("config/env.yaml") as f:
    env_config = yaml.safe_load(f)

with open("config/agent.yaml") as f:
    agent_config = yaml.safe_load(f)

with open("config/experiment.yaml") as f:
    experiment_config = yaml.safe_load(f)

# -------------------------------
# Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        return (torch.tensor(obs, dtype=torch.float32, device=device),
                torch.tensor(actions, dtype=torch.float32, device=device),
                torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1),
                torch.tensor(next_obs, dtype=torch.float32, device=device),
                torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1))

    def __len__(self):
        return len(self.buffer)

# -------------------------------
# SAC Soft Q Update
# -------------------------------
def soft_q_update(actor, critic, critic_target, buffer, actor_opt, critic_opt,
                  gamma=0.99, alpha=0.2, tau=0.005, batch_size=256):
    if len(buffer) < batch_size:
        return

    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)

    # Q-target
    with torch.no_grad():
        next_action, next_log_prob = actor.sample(next_obs)
        q1_next, q2_next = critic_target(next_obs, next_action)
        q_target = rewards + gamma * (1 - dones) * (torch.min(q1_next, q2_next) - alpha * next_log_prob)

    # Critic update
    q1, q2 = critic(obs, actions)
    critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # Actor update
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
# Domain Randomization
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
        "collision_threshold": random.uniform(0.15, 0.5),
    }
    return params

# -------------------------------
# Main Training Loop
# -------------------------------
def main():
    obs_dim = agent_config['obs_dim']
    action_dim = agent_config['action_dim']

    actor = Actor(obs_dim, action_dim).to(device)
    critic = Critic(obs_dim, action_dim).to(device)
    critic_target = Critic(obs_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=agent_config['actor_lr'])
    critic_opt = optim.Adam(critic.parameters(), lr=agent_config['critic_lr'])

    buffer = ReplayBuffer(max_size=agent_config['buffer_size'])
    env = Environment(env_config)

    num_episodes = experiment_config['num_episodes']
    max_steps = experiment_config['max_steps']
    batch_size = experiment_config['batch_size']
    gamma = experiment_config['gamma']
    alpha = experiment_config['alpha']
    tau = experiment_config['tau']

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(num_episodes):
        domain_params = sample_domain_randomization()
        obs = env.reset()
        ep_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            action, _ = actor.sample(obs_tensor)
            action_np = action.detach().cpu().numpy()

            next_obs, reward, done, info = env.step(action_np)
            buffer.add(obs, action_np, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            step += 1

            soft_q_update(actor, critic, critic_target, buffer, actor_opt, critic_opt,
                          gamma=gamma, alpha=alpha, tau=tau, batch_size=batch_size)

        print(f"Episode {ep+1}/{num_episodes} | Total Reward: {ep_reward:.2f}")

        # Save models every 100 episodes
        if (ep + 1) % 100 == 0:
            torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_ep{ep+1}.pth"))
            torch.save(critic.state_dict(), os.path.join(save_dir, f"critic_ep{ep+1}.pth"))

if __name__ == "__main__":
    main()

