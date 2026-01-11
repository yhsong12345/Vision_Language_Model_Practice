"""
Vision Projectors for VLM/VLA Models

Projects vision encoder features to LLM embedding space.
Supports multiple projection strategies:
- MLP projector (LLaVA-style)
- Attention pooling with learnable queries
- Perceiver-style cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VisionProjector(nn.Module):
    """
    MLP-based Vision Projector (LLaVA-style).

    Projects vision features to LLM embedding dimension using
    a two-layer MLP with GELU activation.

    Args:
        vision_dim: Input dimension from vision encoder
        llm_dim: Output dimension matching LLM embeddings
        hidden_dim: Hidden layer dimension (defaults to llm_dim)
        dropout: Dropout probability
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = hidden_dim or llm_dim

        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, llm_dim),
        )

        self.vision_dim = vision_dim
        self.llm_dim = llm_dim

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to LLM space.

        Args:
            vision_features: (batch, num_patches, vision_dim)

        Returns:
            projected: (batch, num_patches, llm_dim)
        """
        return self.projector(vision_features)


class AttentionPoolingProjector(nn.Module):
    """
    Attention-based Vision Projector with learnable query tokens.

    Uses cross-attention to compress variable-length vision features
    into a fixed number of tokens. Similar to Perceiver/Q-Former.

    Args:
        vision_dim: Input dimension from vision encoder
        llm_dim: Output dimension matching LLM embeddings
        num_tokens: Number of output tokens (compressed representation)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_tokens: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim

        # Project vision features to llm_dim first
        self.input_proj = nn.Linear(vision_dim, llm_dim)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_tokens, llm_dim) * 0.02
        )

        # Cross-attention: queries attend to vision features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm and feedforward
        self.norm1 = nn.LayerNorm(llm_dim)
        self.norm2 = nn.LayerNorm(llm_dim)

        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(llm_dim * 4, llm_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project and compress vision features using attention pooling.

        Args:
            vision_features: (batch, num_patches, vision_dim)

        Returns:
            pooled: (batch, num_tokens, llm_dim)
        """
        B = vision_features.shape[0]

        # Project to llm_dim
        kv = self.input_proj(vision_features)

        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)

        # Cross-attention
        attended, _ = self.cross_attention(queries, kv, kv)
        attended = self.norm1(attended + queries)

        # Feedforward
        output = self.ffn(attended)
        output = self.norm2(output + attended)

        return output


class PerceiverProjector(nn.Module):
    """
    Perceiver-style projector with multiple cross-attention layers.

    More powerful than single-layer attention pooling, can capture
    more complex vision-language relationships.

    Args:
        vision_dim: Input dimension from vision encoder
        llm_dim: Output dimension matching LLM embeddings
        num_tokens: Number of latent tokens
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_tokens: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(vision_dim, llm_dim)

        # Learnable latent tokens
        self.latent_tokens = nn.Parameter(
            torch.randn(1, num_tokens, llm_dim) * 0.02
        )

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList()
        self.cross_norms = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
        self.self_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(num_layers):
            # Cross-attention (latents attend to vision)
            self.cross_attention_layers.append(
                nn.MultiheadAttention(llm_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.cross_norms.append(nn.LayerNorm(llm_dim))

            # Self-attention (latents attend to latents)
            self.self_attention_layers.append(
                nn.MultiheadAttention(llm_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.self_norms.append(nn.LayerNorm(llm_dim))

            # FFN
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(llm_dim, llm_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(llm_dim * 4, llm_dim),
            ))
            self.ffn_norms.append(nn.LayerNorm(llm_dim))

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features through Perceiver layers.

        Args:
            vision_features: (batch, num_patches, vision_dim)

        Returns:
            latents: (batch, num_tokens, llm_dim)
        """
        B = vision_features.shape[0]

        # Project vision features
        kv = self.input_proj(vision_features)

        # Initialize latents
        latents = self.latent_tokens.expand(B, -1, -1)

        # Process through layers
        for i in range(self.num_layers):
            # Cross-attention
            cross_out, _ = self.cross_attention_layers[i](latents, kv, kv)
            latents = self.cross_norms[i](latents + cross_out)

            # Self-attention
            self_out, _ = self.self_attention_layers[i](latents, latents, latents)
            latents = self.self_norms[i](latents + self_out)

            # FFN
            ffn_out = self.ffn_layers[i](latents)
            latents = self.ffn_norms[i](latents + ffn_out)

        return latents


def create_projector(
    projector_type: str,
    vision_dim: int,
    llm_dim: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create vision projectors.

    Args:
        projector_type: One of "mlp", "attention", "perceiver"
        vision_dim: Vision encoder output dimension
        llm_dim: LLM embedding dimension
        **kwargs: Additional arguments for specific projector types

    Returns:
        Vision projector module
    """
    projectors = {
        "mlp": VisionProjector,
        "attention": AttentionPoolingProjector,
        "perceiver": PerceiverProjector,
    }

    if projector_type not in projectors:
        raise ValueError(f"Unknown projector type: {projector_type}. "
                        f"Available: {list(projectors.keys())}")

    return projectors[projector_type](vision_dim, llm_dim, **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("Vision Projector Test")
    print("=" * 60)

    batch_size = 2
    num_patches = 196  # 14x14 patches
    vision_dim = 768
    llm_dim = 1536

    dummy_features = torch.randn(batch_size, num_patches, vision_dim)

    # Test MLP projector
    print("\nMLP Projector:")
    mlp_proj = VisionProjector(vision_dim, llm_dim)
    mlp_out = mlp_proj(dummy_features)
    print(f"  Input: {dummy_features.shape}")
    print(f"  Output: {mlp_out.shape}")

    # Test Attention Pooling projector
    print("\nAttention Pooling Projector:")
    attn_proj = AttentionPoolingProjector(vision_dim, llm_dim, num_tokens=64)
    attn_out = attn_proj(dummy_features)
    print(f"  Input: {dummy_features.shape}")
    print(f"  Output: {attn_out.shape}")

    # Test Perceiver projector
    print("\nPerceiver Projector:")
    perc_proj = PerceiverProjector(vision_dim, llm_dim, num_tokens=64, num_layers=2)
    perc_out = perc_proj(dummy_features)
    print(f"  Input: {dummy_features.shape}")
    print(f"  Output: {perc_out.shape}")

    print("\n" + "=" * 60)
    print("All projector tests passed!")
    print("=" * 60)
