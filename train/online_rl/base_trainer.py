"""
Base Online RL Trainer

Abstract base class for online reinforcement learning trainers.
Provides common functionality for environment interaction and learning.
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import numpy as np
from collections import deque

from train.utils.buffers import RolloutBuffer, ReplayBuffer
from core.device_utils import get_device


class OnlineRLTrainer(ABC):
    """
    Abstract base class for online RL trainers.

    Provides common functionality:
    - Environment interaction
    - Logging
    - Checkpointing
    - Evaluation
    """

    def __init__(
        self,
        env,
        policy: nn.Module,
        output_dir: str = "./online_rl_output",
        device: str = "auto",
        seed: int = 42,
        gamma: float = 0.99,
        total_timesteps: int = 1000000,
        eval_episodes: int = 10,
        eval_freq: int = 10000,
        log_freq: int = 1000,
        save_freq: int = 50000,
    ):
        self.env = env
        self.policy = policy
        self.output_dir = output_dir
        self.gamma = gamma
        self.total_timesteps = total_timesteps
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.save_freq = save_freq

        # Set device using shared utility
        self.device = get_device(device)
        self.policy = self.policy.to(self.device)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        """Run training."""
        pass

    @abstractmethod
    def learn_step(self) -> Dict[str, float]:
        """Perform one learning step."""
        pass

    def collect_rollout(self, num_steps: int) -> Tuple[float, int]:
        """Collect experience by interacting with environment."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for _ in range(num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                action, value, log_prob = self.policy.get_action_value(obs_tensor)

            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            self._store_transition(obs, action, reward, done, value, log_prob)

            episode_reward += reward
            episode_length += 1

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs

        return np.mean(self.episode_rewards), np.mean(self.episode_lengths)

    @abstractmethod
    def _store_transition(self, obs, action, reward, done, value, log_prob):
        """Store transition in buffer."""
        pass

    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """Evaluate the policy."""
        if num_episodes is None:
            num_episodes = self.eval_episodes

        self.policy.eval()
        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    action = self.policy.get_action(obs_tensor, deterministic=True)

                action_np = action.cpu().numpy()
                obs, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        self.policy.train()

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
        }

    def save(self, path: str = None):
        """Save the policy."""
        if path is None:
            path = os.path.join(self.output_dir, "policy.pt")

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "episode_rewards": list(self.episode_rewards),
        }, path)
        print(f"Saved policy to {path}")

    def load(self, path: str):
        """Load the policy."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded policy from {path}")


class ActorCritic(nn.Module):
    """Simple Actor-Critic network for continuous control."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(obs)
        return self.actor_mean(features), self.critic(features)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, _ = self.forward(obs)
        if deterministic:
            return mean
        std = self.actor_log_std.exp()
        return torch.distributions.Normal(mean, std).sample()

    def get_action_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action, value.squeeze(-1), dist.log_prob(action).sum(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return value.squeeze(-1), dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


# Re-export buffers for backward compatibility
__all__ = ["OnlineRLTrainer", "ActorCritic", "RolloutBuffer", "ReplayBuffer"]
