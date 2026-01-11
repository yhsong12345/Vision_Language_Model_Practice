"""
LiDAR Point Cloud Encoders

Encoders for processing LiDAR point cloud data:
- PointNet-style encoder
- Point Transformer encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for LiDAR point clouds.

    Processes raw point clouds using per-point MLPs and global max pooling.

    Args:
        input_dim: Point feature dimension (e.g., 4 for x,y,z,intensity)
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_points: Expected number of points (for batch processing)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_points: int = 4096,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_points = num_points

        # Per-point MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Global feature extraction
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Encode point cloud.

        Args:
            point_cloud: (batch, num_points, input_dim)

        Returns:
            features: (batch, output_dim)
        """
        B, N, _ = point_cloud.shape

        # Per-point features
        x = point_cloud.view(B * N, -1)
        x = self.mlp1(x)
        x = x.view(B, N, -1)

        # Global max and average pooling
        global_max = torch.max(x, dim=1)[0]  # (B, hidden_dim)
        global_avg = torch.mean(x, dim=1)    # (B, hidden_dim)

        # Concatenate
        global_feat = torch.cat([global_max, global_avg], dim=-1)

        # Final projection
        output = self.mlp2(global_feat)

        return output


class PointCloudEncoder(nn.Module):
    """
    Point Cloud Encoder with attention-based pooling.

    Uses PointNet-style processing with learnable query tokens
    for fixed-size output representation.

    Args:
        input_dim: Point feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_tokens: int = 32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Per-point processing
        self.point_embed = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )

        # Global context
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Learnable query tokens for compression
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

        # Cross-attention for pooling
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Encode point cloud to fixed-size tokens.

        Args:
            point_cloud: (batch, num_points, input_dim)

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B, N, _ = point_cloud.shape

        # Per-point features
        x = self.point_embed(point_cloud)  # (B, N, hidden_dim)

        # Global max pooling for context
        global_feat = torch.max(x, dim=1, keepdim=True)[0]
        global_feat = global_feat.expand(-1, N, -1)

        # Combine local and global
        x = x + global_feat

        # Project to output dim
        x = self.global_mlp(x)  # (B, N, output_dim)

        # Cross-attention pooling
        query = self.query_tokens.expand(B, -1, -1)
        pooled, _ = self.cross_attention(query, x, x)
        pooled = self.norm(pooled + query)

        return pooled


class PointTransformerEncoder(nn.Module):
    """
    Point Transformer Encoder.

    Uses self-attention on point clouds with positional encoding.
    More powerful but computationally expensive.

    Args:
        input_dim: Point feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_tokens: int = 32,
        max_points: int = 4096,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Input embedding
        self.input_embed = nn.Linear(input_dim, hidden_dim)

        # Learnable positional encoding (based on spatial position)
        self.spatial_embed = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Transformer encoder layers
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

        # Learnable query tokens
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

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Encode point cloud using transformer.

        Args:
            point_cloud: (batch, num_points, input_dim)
                         First 3 dims assumed to be x, y, z

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B, N, _ = point_cloud.shape

        # Feature embedding
        feat = self.input_embed(point_cloud)

        # Spatial positional encoding (from xyz coordinates)
        pos = self.spatial_embed(point_cloud[:, :, :3])
        x = feat + pos

        # Transformer processing
        x = self.transformer(x)

        # Project to output dim
        x = self.output_proj(x)

        # Cross-attention pooling
        query = self.query_tokens.expand(B, -1, -1)
        pooled, _ = self.cross_attention(query, x, x)
        pooled = self.norm(pooled + query)

        return pooled


class VoxelEncoder(nn.Module):
    """
    Voxel-based LiDAR Encoder.

    Converts point cloud to voxel representation and uses 3D convolutions.
    Efficient for dense point clouds.

    Args:
        input_dim: Point feature dimension
        voxel_size: Size of each voxel
        spatial_range: Spatial range [x_min, x_max, y_min, y_max, z_min, z_max]
        output_dim: Output feature dimension
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        input_dim: int = 4,
        voxel_size: float = 0.1,
        spatial_range: list = [-50, 50, -50, 50, -5, 5],
        output_dim: int = 512,
        num_tokens: int = 32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.voxel_size = voxel_size
        self.spatial_range = spatial_range
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Calculate grid size
        self.grid_size = [
            int((spatial_range[1] - spatial_range[0]) / voxel_size),
            int((spatial_range[3] - spatial_range[2]) / voxel_size),
            int((spatial_range[5] - spatial_range[4]) / voxel_size),
        ]

        # 3D CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv3d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool3d((4, 4, 2)),
        )

        # Projection to tokens
        self.token_proj = nn.Linear(output_dim * 4 * 4 * 2, output_dim * num_tokens)

    def forward(self, voxel_features: torch.Tensor) -> torch.Tensor:
        """
        Encode voxelized point cloud.

        Args:
            voxel_features: (batch, input_dim, D, H, W) voxel grid

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B = voxel_features.shape[0]

        x = self.backbone(voxel_features)
        x = x.view(B, -1)
        x = self.token_proj(x)
        x = x.view(B, self.num_tokens, self.output_dim)

        return x


if __name__ == "__main__":
    print("=" * 60)
    print("LiDAR Encoder Test")
    print("=" * 60)

    batch_size = 2
    num_points = 4096
    input_dim = 4  # x, y, z, intensity
    output_dim = 512
    num_tokens = 32

    # Create dummy point cloud
    point_cloud = torch.randn(batch_size, num_points, input_dim)

    # Test PointNet Encoder
    print("\nPointNet Encoder:")
    pointnet = PointNetEncoder(input_dim=input_dim, output_dim=output_dim)
    out = pointnet(point_cloud)
    print(f"  Input: {point_cloud.shape}")
    print(f"  Output: {out.shape}")

    # Test Point Cloud Encoder (with tokens)
    print("\nPoint Cloud Encoder (with attention pooling):")
    pc_encoder = PointCloudEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        num_tokens=num_tokens,
    )
    out = pc_encoder(point_cloud)
    print(f"  Input: {point_cloud.shape}")
    print(f"  Output: {out.shape}")

    # Test Point Transformer Encoder
    print("\nPoint Transformer Encoder:")
    pt_encoder = PointTransformerEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim,
        num_layers=2,
        num_tokens=num_tokens,
    )
    out = pt_encoder(point_cloud)
    print(f"  Input: {point_cloud.shape}")
    print(f"  Output: {out.shape}")

    print("\n" + "=" * 60)
    print("All LiDAR encoder tests passed!")
    print("=" * 60)
