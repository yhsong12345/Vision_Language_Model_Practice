"""
Radar Encoders

Encoders for processing radar data:
- Range-Doppler maps (2D images)
- Radar point clouds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class RadarEncoder(nn.Module):
    """
    CNN-based Radar Encoder for range-doppler maps.

    Processes 2D radar images (range-doppler, range-azimuth, etc.).

    Args:
        input_channels: Number of input channels (e.g., 2 for magnitude+velocity)
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        input_channels: int = 2,
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

    def forward(self, radar_data: torch.Tensor) -> torch.Tensor:
        """
        Encode radar data.

        Args:
            radar_data: (batch, channels, height, width) range-doppler map

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B = radar_data.shape[0]

        x = self.backbone(radar_data)
        x = x.view(B, -1)
        x = self.token_proj(x)
        x = x.view(B, self.num_tokens, self.output_dim)

        return x


class RangeDopplerEncoder(nn.Module):
    """
    Specialized encoder for Range-Doppler maps.

    Uses separate processing for range and doppler dimensions,
    then fuses them together.

    Args:
        input_channels: Number of input channels
        output_dim: Output feature dimension
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        input_channels: int = 2,
        output_dim: int = 256,
        num_tokens: int = 16,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Range processing (vertical)
        self.range_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Doppler processing (horizontal)
        self.doppler_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Token projection
        self.token_proj = nn.Linear(output_dim * 16, output_dim * num_tokens)

    def forward(self, radar_data: torch.Tensor) -> torch.Tensor:
        """
        Encode range-doppler map.

        Args:
            radar_data: (batch, channels, range_bins, doppler_bins)

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B = radar_data.shape[0]

        # Separate processing
        range_feat = self.range_conv(radar_data)
        doppler_feat = self.doppler_conv(radar_data)

        # Align spatial dimensions and concatenate
        min_h = min(range_feat.shape[2], doppler_feat.shape[2])
        min_w = min(range_feat.shape[3], doppler_feat.shape[3])

        range_feat = F.adaptive_avg_pool2d(range_feat, (min_h, min_w))
        doppler_feat = F.adaptive_avg_pool2d(doppler_feat, (min_h, min_w))

        fused = torch.cat([range_feat, doppler_feat], dim=1)

        # Final processing
        x = self.fusion(fused)
        x = x.view(B, -1)
        x = self.token_proj(x)
        x = x.view(B, self.num_tokens, self.output_dim)

        return x


class RadarPointCloudEncoder(nn.Module):
    """
    Encoder for radar point cloud detections.

    Processes radar detections as point clouds with features like:
    range, azimuth, elevation, velocity, RCS (radar cross-section).

    Args:
        input_dim: Point feature dimension (e.g., 5 for range, az, el, vel, rcs)
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_tokens: int = 16,
        max_detections: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.max_detections = max_detections

        # Per-detection MLP
        self.detection_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Self-attention for detection relationships
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(
        self,
        radar_points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode radar point detections.

        Args:
            radar_points: (batch, num_detections, input_dim)
            mask: (batch, num_detections) - True for valid detections

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B, N, _ = radar_points.shape

        # Per-detection features
        x = self.detection_mlp(radar_points)

        # Self-attention with optional mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # True means ignore

        attn_out, _ = self.self_attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)

        # Project to output dim
        x = self.output_proj(x)

        # Cross-attention pooling
        query = self.query_tokens.expand(B, -1, -1)
        pooled, _ = self.cross_attention(query, x, x, key_padding_mask=key_padding_mask)
        pooled = self.norm2(pooled + query)

        return pooled


if __name__ == "__main__":
    print("=" * 60)
    print("Radar Encoder Test")
    print("=" * 60)

    batch_size = 2

    # Test Range-Doppler CNN Encoder
    print("\nRadar CNN Encoder:")
    radar_data = torch.randn(batch_size, 2, 64, 256)  # 2 channels, 64 range, 256 doppler
    radar_encoder = RadarEncoder(input_channels=2, output_dim=256, num_tokens=16)
    out = radar_encoder(radar_data)
    print(f"  Input: {radar_data.shape}")
    print(f"  Output: {out.shape}")

    # Test Range-Doppler Specialized Encoder
    print("\nRange-Doppler Encoder:")
    rd_encoder = RangeDopplerEncoder(input_channels=2, output_dim=256, num_tokens=16)
    out = rd_encoder(radar_data)
    print(f"  Input: {radar_data.shape}")
    print(f"  Output: {out.shape}")

    # Test Radar Point Cloud Encoder
    print("\nRadar Point Cloud Encoder:")
    radar_points = torch.randn(batch_size, 128, 5)  # 128 detections, 5 features
    mask = torch.ones(batch_size, 128, dtype=torch.bool)
    mask[:, 100:] = False  # Some invalid detections

    rpc_encoder = RadarPointCloudEncoder(input_dim=5, output_dim=256, num_tokens=16)
    out = rpc_encoder(radar_points, mask)
    print(f"  Input: {radar_points.shape}")
    print(f"  Output: {out.shape}")

    print("\n" + "=" * 60)
    print("All radar encoder tests passed!")
    print("=" * 60)
