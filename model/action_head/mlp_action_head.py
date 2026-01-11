"""
MLP Action Head for VLA Models

Standard MLP-based action prediction head with optional action chunking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


class MLPActionHead(nn.Module):
    """
    MLP-based Action Prediction Head.

    Predicts robot actions from fused vision-language features.
    Supports action chunking for temporal consistency.

    Args:
        input_dim: Input feature dimension (from LLM)
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers
        chunk_size: Number of future actions to predict
        dropout: Dropout probability
        activation: Activation function ("gelu", "relu", "silu")
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        chunk_size: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Select activation
        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        act_fn = activations.get(activation, nn.GELU)

        # Build MLP layers
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [action_dim * chunk_size]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # No activation/norm on last layer
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act_fn())
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict actions from features.

        Args:
            features: (batch, input_dim) - fused features from LLM
            actions: (batch, [chunk_size,] action_dim) - ground truth for training

        Returns:
            Dict with:
                - predicted_actions: (batch, [chunk_size,] action_dim)
                - loss: MSE loss if actions provided
        """
        out = self.mlp(features)

        if self.chunk_size > 1:
            predicted_actions = out.view(-1, self.chunk_size, self.action_dim)
        else:
            predicted_actions = out

        outputs = {"predicted_actions": predicted_actions}

        if actions is not None:
            loss = F.mse_loss(predicted_actions, actions)
            outputs["loss"] = loss

        return outputs

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Simple prediction interface.

        Args:
            features: (batch, input_dim) or (input_dim,)

        Returns:
            actions: (batch, [chunk_size,] action_dim) or ([chunk_size,] action_dim)
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
            outputs = self.forward(features)
            return outputs["predicted_actions"].squeeze(0)

        outputs = self.forward(features)
        return outputs["predicted_actions"]


class GaussianMLPActionHead(nn.Module):
    """
    Gaussian MLP Action Head with mean and variance prediction.

    Outputs a Gaussian distribution over actions, useful for:
    - Stochastic policies
    - Uncertainty estimation
    - RL training (PPO, SAC)

    Args:
        input_dim: Input feature dimension
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers
        min_std: Minimum standard deviation
        max_std: Maximum standard deviation
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        min_std: float = 0.01,
        max_std: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        # Shared backbone
        backbone_layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)

        for i in range(len(dims) - 1):
            backbone_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        self.backbone = nn.Sequential(*backbone_layers)

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning action distribution.

        Args:
            features: (batch, input_dim)
            deterministic: If True, return mean action

        Returns:
            Dict with:
                - action: Sampled or mean action
                - mean: Mean of distribution
                - std: Standard deviation
                - log_prob: Log probability of sampled action (if not deterministic)
        """
        hidden = self.backbone(features)

        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(
            log_std,
            min=np.log(self.min_std),
            max=np.log(self.max_std),
        )
        std = torch.exp(log_std)

        outputs = {
            "mean": mean,
            "std": std,
        }

        if deterministic:
            outputs["action"] = mean
        else:
            # Reparameterization trick
            noise = torch.randn_like(mean)
            action = mean + std * noise
            outputs["action"] = action

            # Log probability (using pre-computed constant)
            log_prob = -0.5 * (
                ((action - mean) / std) ** 2
                + 2 * log_std
                + 1.8378770664093453  # log(2 * pi)
            )
            outputs["log_prob"] = log_prob.sum(dim=-1)

        return outputs

    def log_prob(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of given actions.

        Args:
            features: (batch, input_dim)
            actions: (batch, action_dim)

        Returns:
            log_prob: (batch,)
        """
        hidden = self.backbone(features)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(
            log_std,
            min=np.log(self.min_std),
            max=np.log(self.max_std),
        )
        std = torch.exp(log_std)

        log_prob = -0.5 * (
            ((actions - mean) / std) ** 2
            + 2 * log_std
            + 1.8378770664093453  # log(2 * pi)
        )
        return log_prob.sum(dim=-1)


if __name__ == "__main__":
    print("=" * 60)
    print("MLP Action Head Test")
    print("=" * 60)

    batch_size = 4
    input_dim = 1536
    action_dim = 7

    # Test standard MLP head
    print("\nStandard MLP Action Head:")
    mlp_head = MLPActionHead(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=512,
        num_layers=3,
        chunk_size=1,
    )

    features = torch.randn(batch_size, input_dim)
    actions_gt = torch.randn(batch_size, action_dim)

    outputs = mlp_head(features, actions_gt)
    print(f"  Input: {features.shape}")
    print(f"  Output: {outputs['predicted_actions'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")

    # Test chunked MLP head
    print("\nChunked MLP Action Head (chunk_size=10):")
    chunked_head = MLPActionHead(
        input_dim=input_dim,
        action_dim=action_dim,
        chunk_size=10,
    )

    chunked_outputs = chunked_head(features)
    print(f"  Output: {chunked_outputs['predicted_actions'].shape}")

    # Test Gaussian head
    print("\nGaussian MLP Action Head:")
    gaussian_head = GaussianMLPActionHead(
        input_dim=input_dim,
        action_dim=action_dim,
    )

    gaussian_outputs = gaussian_head(features)
    print(f"  Mean: {gaussian_outputs['mean'].shape}")
    print(f"  Std: {gaussian_outputs['std'].shape}")
    print(f"  Action: {gaussian_outputs['action'].shape}")
    print(f"  Log prob: {gaussian_outputs['log_prob'].shape}")

    print("\n" + "=" * 60)
    print("All action head tests passed!")
    print("=" * 60)
