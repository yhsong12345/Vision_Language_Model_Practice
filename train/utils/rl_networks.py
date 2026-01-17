"""
Shared RL Network Architectures

Common neural network components for RL trainers:
- GaussianPolicy: Stochastic policy with tanh squashing for SAC/CQL
- TwinQNetwork: Double Q-networks for SAC/CQL/IQL
- ValueNetwork: State value function for IQL
- QNetwork: Single Q-network
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    activation: nn.Module = nn.ReLU,
) -> nn.Sequential:
    """Build a simple MLP with ReLU activations."""
    layers = []
    prev_dim = input_dim

    for _ in range(num_layers):
        layers.extend([nn.Linear(prev_dim, hidden_dim), activation()])
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy with tanh squashing for continuous control.

    Used by SAC, CQL, and other algorithms requiring stochastic policies
    with bounded actions.
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        log_std_min: Optional[float] = None,
        log_std_max: Optional[float] = None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min if log_std_min is not None else self.LOG_STD_MIN
        self.log_std_max = log_std_max if log_std_max is not None else self.LOG_STD_MAX

        # Build backbone
        layers = []
        prev_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        features = self.network(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with tanh squashing and compute log prob."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)

        # Log prob with tanh squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action (without tanh squashing)."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        return normal.log_prob(action).sum(-1, keepdim=True)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for evaluation."""
        mean, log_std = self.forward(obs)

        if deterministic:
            return torch.tanh(mean)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        return torch.tanh(normal.sample())


class QNetwork(nn.Module):
    """Single Q-network Q(s, a)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.network = build_mlp(
            input_dim=obs_dim + action_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)


class TwinQNetwork(nn.Module):
    """
    Twin Q-networks for double Q-learning.

    Used by SAC, CQL, TD3, and other algorithms to reduce
    overestimation bias.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        self.q1 = QNetwork(obs_dim, action_dim, hidden_dim, num_layers)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dim, num_layers)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Q1 and Q2 values."""
        return self.q1(obs, action), self.q2(obs, action)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 value (for actor update)."""
        return self.q1(obs, action)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return minimum of Q1 and Q2."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    """State value network V(s) for IQL."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.network = build_mlp(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class DeterministicPolicy(nn.Module):
    """Deterministic policy for TD3 and TD3+BC."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_action: float = 1.0,
    ):
        super().__init__()

        self.max_action = max_action
        self.network = build_mlp(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_layers=num_layers,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.network(obs))

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.forward(obs)


__all__ = [
    "build_mlp",
    "GaussianPolicy",
    "QNetwork",
    "TwinQNetwork",
    "ValueNetwork",
    "DeterministicPolicy",
]
