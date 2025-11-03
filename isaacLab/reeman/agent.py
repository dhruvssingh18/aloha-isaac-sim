
# scripts/agent.py
import torch
import torch.optim as optim
from policy import Actor, Critic
from rl_train import soft_q_update  # import your soft_q_update function
from rl_train import ReplayBuffer

class Agent:
    def __init__(self, obs_dim, action_dim, actor_lr=3e-4, critic_lr=3e-4, buffer_size=1_000_000):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = Critic(obs_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(max_size=buffer_size)

        self.gamma = 0.99
        self.alpha = 0.2
        self.tau = 0.005

    def get_action(self, observation):
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        action, _ = self.actor.sample(obs_tensor)
        return action.detach().numpy()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self, batch_size=256):
        soft_q_update(
            self.actor, self.critic, self.critic_target,
            self.buffer, self.actor_opt, self.critic_opt,
            gamma=self.gamma, alpha=self.alpha, tau=self.tau,
            batch_size=batch_size
        )

