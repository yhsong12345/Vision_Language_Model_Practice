"""
Base Imitation Learning Trainer - Abstract base class for IL trainers.
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple
from torch.utils.data import Dataset
import numpy as np

from core.device_utils import get_device


class ExpertDataset(Dataset):
    """Dataset for expert demonstrations."""

    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class PolicyNetwork(nn.Module):
    """Simple MLP policy for imitation learning."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256], continuous: bool = True):
        super().__init__()
        self.continuous = continuous

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)

        if continuous:
            self.mean_head = nn.Linear(prev_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        return self.mean_head(features) if self.continuous else self.action_head(features)

    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            if self.continuous:
                mean = self.forward(state)
                action = mean if deterministic else mean + torch.randn_like(mean) * self.log_std.exp()
            else:
                logits = self.forward(state)
                action = torch.argmax(logits, dim=-1) if deterministic else torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
        return action.squeeze(0)

    def get_action_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.continuous:
            mean, std = self.forward(state), self.log_std.exp()
            return torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
        else:
            return torch.log_softmax(self.forward(state), dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)


class ILTrainer(ABC):
    """Abstract base class for Imitation Learning trainers."""

    def __init__(self, env, policy: Optional[nn.Module] = None, output_dir: str = "./il_output", device: str = "auto", seed: int = 42):
        self.env = env
        self.output_dir = output_dir

        # Get dimensions
        self.state_dim = env.observation_space.shape[0]
        self.continuous = hasattr(env.action_space, 'shape')
        self.action_dim = env.action_space.shape[0] if self.continuous else env.action_space.n

        # Create policy if not provided
        self.policy = policy or PolicyNetwork(self.state_dim, self.action_dim, continuous=self.continuous)
        self.device = get_device(device)
        self.policy = self.policy.to(self.device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        pass

    def collect_expert_demonstrations(self, expert_policy, num_episodes: int = 100, max_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Collect demonstrations from an expert policy."""
        states, actions, episode_rewards = [], [], []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            ep_reward = 0
            for _ in range(max_steps):
                action = expert_policy(state)
                states.append(state)
                actions.append(action)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
                state = next_state
            episode_rewards.append(ep_reward)

        print(f"Collected {len(states)} transitions from {num_episodes} episodes")
        print(f"Expert mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32 if self.continuous else np.int64)

    def evaluate(self, num_episodes: int = 20, max_steps: int = 1000, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate the learned policy."""
        self.policy.eval()
        rewards, lengths = [], []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            ep_reward, ep_length = 0, 0
            for _ in range(max_steps):
                action = self.policy.get_action(torch.tensor(state, dtype=torch.float32).to(self.device), deterministic)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                ep_length += 1
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
            lengths.append(ep_length)

        self.policy.train()
        return {"mean_reward": np.mean(rewards), "std_reward": np.std(rewards), "mean_length": np.mean(lengths), "min_reward": np.min(rewards), "max_reward": np.max(rewards)}

    def save(self, path: str = None):
        path = path or os.path.join(self.output_dir, "policy.pt")
        torch.save({"policy_state_dict": self.policy.state_dict(), "state_dim": self.state_dim, "action_dim": self.action_dim, "continuous": self.continuous}, path)
        print(f"Saved policy to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"Loaded policy from {path}")
