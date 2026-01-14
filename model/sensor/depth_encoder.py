"""
Depth Camera Encoders

Encoders for processing depth camera data:
- Single-channel depth images
- RGB-D fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class DepthEncoder(nn.Module):
    """
    CNN-based Depth Camera Encoder.

    Processes single-channel depth images using 2D convolutions.

    Args:
        input_channels: Number of input channels (1 for depth, can be more for multi-frame)
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_tokens: int = 16,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Projection to tokens
        self.token_proj = nn.Linear(output_dim * 16, output_dim * num_tokens)

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        Encode depth image.

        Args:
            depth_image: (batch, channels, height, width) depth image

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B = depth_image.shape[0]

        x = self.backbone(depth_image)
        x = x.view(B, -1)
        x = self.token_proj(x)
        x = x.view(B, self.num_tokens, self.output_dim)

        return x


class DepthTransformerEncoder(nn.Module):
    """
    Transformer-based Depth Encoder.

    Uses ViT-style patch embedding for depth images with
    self-attention and cross-attention pooling.

    Args:
        input_channels: Number of input channels
        patch_size: Size of each patch
        image_size: Input image size
        hidden_dim: Transformer hidden dimension
        output_dim: Output feature dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        input_channels: int = 1,
        patch_size: int = 16,
        image_size: int = 224,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_tokens: int = 16,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            input_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Learnable query tokens for cross-attention pooling
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

        # Cross-attention for output
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        Encode depth image using transformer.

        Args:
            depth_image: (batch, channels, height, width)

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B = depth_image.shape[0]

        # Patch embedding: (B, hidden_dim, H/P, W/P) -> (B, num_patches, hidden_dim)
        x = self.patch_embed(depth_image)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer processing
        x = self.transformer(x)

        # Project to output dimension
        x = self.output_proj(x)

        # Cross-attention pooling
        query = self.query_tokens.expand(B, -1, -1)
        pooled, _ = self.cross_attention(query, x, x)
        pooled = self.norm(pooled + query)

        return pooled


class RGBDEncoder(nn.Module):
    """
    RGB-D Fusion Encoder.

    Processes RGB and depth images with separate branches,
    then fuses them together.

    Args:
        rgb_channels: Number of RGB channels (3)
        depth_channels: Number of depth channels (1)
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_tokens: Number of output tokens
        fusion_type: Type of fusion ('concat', 'add', 'cross_attention')
    """

    def __init__(
        self,
        rgb_channels: int = 3,
        depth_channels: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_tokens: int = 16,
        fusion_type: str = "concat",
    ):
        super().__init__()

        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.fusion_type = fusion_type

        # RGB branch
        self.rgb_backbone = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Depth branch
        self.depth_backbone = nn.Sequential(
            nn.Conv2d(depth_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion layers
        if fusion_type == "concat":
            fusion_input_dim = hidden_dim * 2
        else:
            fusion_input_dim = hidden_dim

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Cross-attention fusion (if selected)
        if fusion_type == "cross_attention":
            self.rgb_to_depth_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
            )
            self.depth_to_rgb_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Projection to tokens
        self.token_proj = nn.Linear(output_dim * 16, output_dim * num_tokens)

    def forward(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode RGB-D images.

        Args:
            rgb_image: (batch, 3, height, width) RGB image
            depth_image: (batch, 1, height, width) depth image

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B = rgb_image.shape[0]

        # Process branches
        rgb_feat = self.rgb_backbone(rgb_image)
        depth_feat = self.depth_backbone(depth_image)

        # Fusion
        if self.fusion_type == "concat":
            fused = torch.cat([rgb_feat, depth_feat], dim=1)
        elif self.fusion_type == "add":
            fused = rgb_feat + depth_feat
        elif self.fusion_type == "cross_attention":
            # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
            H, W = rgb_feat.shape[2], rgb_feat.shape[3]
            rgb_flat = rgb_feat.flatten(2).transpose(1, 2)
            depth_flat = depth_feat.flatten(2).transpose(1, 2)

            # Cross-modal attention
            rgb_enhanced, _ = self.rgb_to_depth_attn(rgb_flat, depth_flat, depth_flat)
            depth_enhanced, _ = self.depth_to_rgb_attn(depth_flat, rgb_flat, rgb_flat)

            # Combine
            combined = self.fusion_norm(rgb_enhanced + depth_enhanced)
            fused = combined.transpose(1, 2).view(B, -1, H, W)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Final processing
        x = self.fusion(fused)
        x = x.view(B, -1)
        x = self.token_proj(x)
        x = x.view(B, self.num_tokens, self.output_dim)

        return x


class MultiScaleDepthEncoder(nn.Module):
    """
    Multi-scale Depth Encoder.

    Processes depth images at multiple scales and fuses features.
    Useful for capturing both local details and global structure.

    Args:
        input_channels: Number of input channels
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_tokens: Number of output tokens
        scales: List of scale factors (e.g., [1.0, 0.5, 0.25])
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_tokens: int = 16,
        scales: list = None,
    ):
        super().__init__()

        if scales is None:
            scales = [1.0, 0.5, 0.25]

        self.scales = scales
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.num_scales = len(scales)

        # Create encoder for each scale
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            for _ in scales
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 16 * self.num_scales, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, output_dim * num_tokens),
        )

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        Encode depth image at multiple scales.

        Args:
            depth_image: (batch, channels, height, width)

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B, C, H, W = depth_image.shape

        scale_features = []
        for scale, encoder in zip(self.scales, self.scale_encoders):
            if scale != 1.0:
                scaled = F.interpolate(
                    depth_image,
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled = depth_image

            feat = encoder(scaled)
            feat = feat.view(B, -1)
            scale_features.append(feat)

        # Concatenate multi-scale features
        combined = torch.cat(scale_features, dim=-1)

        # Fuse and reshape to tokens
        x = self.fusion(combined)
        x = x.view(B, self.num_tokens, self.output_dim)

        return x


if __name__ == "__main__":
    print("=" * 60)
    print("Depth Encoder Test")
    print("=" * 60)

    batch_size = 2
    height, width = 224, 224

    # Test DepthEncoder
    print("\nDepth CNN Encoder:")
    depth_image = torch.randn(batch_size, 1, height, width)
    depth_encoder = DepthEncoder(input_channels=1, output_dim=256, num_tokens=16)
    out = depth_encoder(depth_image)
    print(f"  Input: {depth_image.shape}")
    print(f"  Output: {out.shape}")

    # Test DepthTransformerEncoder
    print("\nDepth Transformer Encoder:")
    depth_transformer = DepthTransformerEncoder(
        input_channels=1,
        output_dim=256,
        num_tokens=16,
    )
    out = depth_transformer(depth_image)
    print(f"  Input: {depth_image.shape}")
    print(f"  Output: {out.shape}")

    # Test RGBDEncoder
    print("\nRGB-D Encoder:")
    rgb_image = torch.randn(batch_size, 3, height, width)
    rgbd_encoder = RGBDEncoder(
        rgb_channels=3,
        depth_channels=1,
        output_dim=256,
        num_tokens=16,
        fusion_type="concat",
    )
    out = rgbd_encoder(rgb_image, depth_image)
    print(f"  RGB Input: {rgb_image.shape}")
    print(f"  Depth Input: {depth_image.shape}")
    print(f"  Output: {out.shape}")

    # Test MultiScaleDepthEncoder
    print("\nMulti-Scale Depth Encoder:")
    ms_encoder = MultiScaleDepthEncoder(
        input_channels=1,
        output_dim=256,
        num_tokens=16,
        scales=[1.0, 0.5, 0.25],
    )
    out = ms_encoder(depth_image)
    print(f"  Input: {depth_image.shape}")
    print(f"  Output: {out.shape}")

    print("\n" + "=" * 60)
    print("All depth encoder tests passed!")
    print("=" * 60)
