"""
Common Neural Network Layers

Reusable layer components used across the VLA framework:
- PositionalEncoding: Sinusoidal positional encoding for transformers
- MLP: Configurable multi-layer perceptron
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer architectures.

    Adds position information to input sequences using fixed sinusoidal patterns.
    Compatible with variable sequence lengths up to max_len.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    output_activation: Optional[str] = None,
    dropout: float = 0.0,
    layer_norm: bool = False,
) -> nn.Sequential:
    """
    Build a multi-layer perceptron.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'gelu', 'tanh', 'silu')
        output_activation: Optional activation for output layer
        dropout: Dropout probability
        layer_norm: Whether to use layer normalization

    Returns:
        nn.Sequential MLP module
    """
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }

    act_fn = activations.get(activation, nn.ReLU)

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    if output_activation and output_activation in activations:
        layers.append(activations[output_activation]())

    return nn.Sequential(*layers)


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    # Test PositionalEncoding
    pe = PositionalEncoding(d_model=256, max_len=100, dropout=0.1)
    x = torch.randn(4, 50, 256)
    out = pe(x)
    print(f"PositionalEncoding: {x.shape} -> {out.shape}")

    # Test MLP
    mlp = MLP(input_dim=64, output_dim=10, hidden_dims=[128, 64])
    x = torch.randn(8, 64)
    out = mlp(x)
    print(f"MLP: {x.shape} -> {out.shape}")

    print("All layer tests passed!")
