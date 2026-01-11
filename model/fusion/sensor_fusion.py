"""
Sensor Fusion Modules

Multi-modal sensor fusion strategies for VLA models:
- Simple concatenation with self-attention
- Cross-modal attention fusion
- Hierarchical fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class SensorFusion(nn.Module):
    """
    Multi-modal Sensor Fusion Module.

    Fuses features from multiple sensors using self-attention.
    Each modality gets a learnable modality embedding.

    Args:
        hidden_dim: Feature dimension
        num_heads: Number of attention heads
        num_layers: Number of self-attention layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Self-attention layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        # Modality embeddings
        self.modality_embeddings = nn.ParameterDict({
            "camera": nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02),
            "lidar": nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02),
            "radar": nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02),
            "imu": nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02),
            "gps": nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02),
        })

    def forward(
        self,
        sensor_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse multi-modal sensor features.

        Args:
            sensor_features: Dict mapping sensor name to features
                Each tensor has shape (batch, num_tokens, hidden_dim)

        Returns:
            fused: (batch, total_tokens, hidden_dim)
        """
        features_list = []

        for name, feat in sensor_features.items():
            if name in self.modality_embeddings:
                B, N, _ = feat.shape
                modality_emb = self.modality_embeddings[name].expand(B, N, -1)
                feat = feat + modality_emb
            features_list.append(feat)

        # Concatenate all sensor features
        fused = torch.cat(features_list, dim=1)

        # Self-attention fusion
        for layer in self.layers:
            fused = layer(fused)

        return fused


class CrossModalFusion(nn.Module):
    """
    Cross-Modal Attention Fusion.

    Uses cross-attention between modalities for more expressive fusion.
    Each modality attends to all other modalities.

    Args:
        hidden_dim: Feature dimension
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Cross-attention layers for each modality pair
        self.cross_attention = nn.ModuleDict()
        self.cross_norms = nn.ModuleDict()
        self.ffns = nn.ModuleDict()
        self.ffn_norms = nn.ModuleDict()

        modalities = ["camera", "lidar", "radar", "imu"]

        for layer_idx in range(num_layers):
            for mod in modalities:
                key = f"{mod}_layer{layer_idx}"
                self.cross_attention[key] = nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                )
                self.cross_norms[key] = nn.LayerNorm(hidden_dim)
                self.ffns[key] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                self.ffn_norms[key] = nn.LayerNorm(hidden_dim)

        # Modality embeddings
        self.modality_embeddings = nn.ParameterDict({
            mod: nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            for mod in modalities
        })

    def forward(
        self,
        sensor_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse features using cross-modal attention.

        Args:
            sensor_features: Dict mapping sensor name to features

        Returns:
            fused: (batch, total_tokens, hidden_dim)
        """
        # Add modality embeddings
        features = {}
        for name, feat in sensor_features.items():
            if name in self.modality_embeddings:
                B, N, _ = feat.shape
                modality_emb = self.modality_embeddings[name].expand(B, N, -1)
                features[name] = feat + modality_emb
            else:
                features[name] = feat

        # Cross-modal attention layers
        for layer_idx in range(self.num_layers):
            # Concatenate all features as context
            all_features = torch.cat(list(features.values()), dim=1)

            updated_features = {}
            for mod, feat in features.items():
                key = f"{mod}_layer{layer_idx}"

                if key in self.cross_attention:
                    # Cross-attention to all modalities
                    attn_out, _ = self.cross_attention[key](feat, all_features, all_features)
                    feat = self.cross_norms[key](feat + attn_out)

                    # FFN
                    ffn_out = self.ffns[key](feat)
                    feat = self.ffn_norms[key](feat + ffn_out)

                updated_features[mod] = feat

            features = updated_features

        # Concatenate all fused features
        fused = torch.cat(list(features.values()), dim=1)

        return fused


class HierarchicalFusion(nn.Module):
    """
    Hierarchical Sensor Fusion.

    Fuses sensors in a hierarchical manner:
    1. Similar sensors first (e.g., camera + lidar for spatial)
    2. Then combine spatial with temporal (IMU)
    3. Finally combine all with language

    Args:
        hidden_dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Spatial fusion (camera + lidar + radar)
        self.spatial_fusion = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # Temporal fusion (spatial + IMU)
        self.temporal_fusion = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # Final fusion
        self.final_fusion = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # Modality embeddings
        self.spatial_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.temporal_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(
        self,
        sensor_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Hierarchical fusion of sensor features.

        Args:
            sensor_features: Dict with "camera", "lidar", "radar", "imu" keys

        Returns:
            fused: (batch, total_tokens, hidden_dim)
        """
        spatial_features = []
        temporal_features = []

        # Separate spatial and temporal modalities
        for name, feat in sensor_features.items():
            if name in ["camera", "lidar", "radar"]:
                spatial_features.append(feat)
            elif name in ["imu", "gps"]:
                temporal_features.append(feat)

        # Fuse spatial features
        if spatial_features:
            spatial = torch.cat(spatial_features, dim=1)
            B, N, _ = spatial.shape
            spatial = spatial + self.spatial_emb.expand(B, N, -1)
            spatial = self.spatial_fusion(spatial)
        else:
            spatial = None

        # Fuse temporal features
        if temporal_features:
            temporal = torch.cat(temporal_features, dim=1)
            B, N, _ = temporal.shape
            temporal = temporal + self.temporal_emb.expand(B, N, -1)

        # Combine spatial and temporal
        if spatial is not None and temporal_features:
            combined = torch.cat([spatial, temporal], dim=1)
            combined = self.temporal_fusion(combined)
        elif spatial is not None:
            combined = spatial
        elif temporal_features:
            combined = temporal
        else:
            raise ValueError("No sensor features provided")

        # Final fusion
        fused = self.final_fusion(combined)

        return fused


class GatedFusion(nn.Module):
    """
    Gated Sensor Fusion.

    Uses learned gates to weight different modalities.
    Useful when some sensors may be more reliable than others.

    Args:
        hidden_dim: Feature dimension
        num_modalities: Maximum number of modalities
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_modalities: int = 4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1),
        )

        # Modality projections
        self.modality_projs = nn.ModuleDict({
            "camera": nn.Linear(hidden_dim, hidden_dim),
            "lidar": nn.Linear(hidden_dim, hidden_dim),
            "radar": nn.Linear(hidden_dim, hidden_dim),
            "imu": nn.Linear(hidden_dim, hidden_dim),
        })

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        sensor_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Gated fusion of sensor features.

        Args:
            sensor_features: Dict mapping sensor name to features

        Returns:
            fused: (batch, hidden_dim)
        """
        # Pool each modality to single vector
        pooled = {}
        for name, feat in sensor_features.items():
            if name in self.modality_projs:
                # Mean pooling over tokens
                pooled_feat = feat.mean(dim=1)  # (B, hidden_dim)
                pooled[name] = self.modality_projs[name](pooled_feat)

        # Compute gates
        modality_order = ["camera", "lidar", "radar", "imu"]
        gate_input = []

        for mod in modality_order:
            if mod in pooled:
                gate_input.append(pooled[mod])
            else:
                B = list(pooled.values())[0].shape[0]
                gate_input.append(torch.zeros(B, self.hidden_dim, device=list(pooled.values())[0].device))

        gate_input = torch.cat(gate_input, dim=-1)
        gates = self.gate_network(gate_input)  # (B, num_modalities)

        # Weighted combination
        fused = torch.zeros_like(list(pooled.values())[0])
        for i, mod in enumerate(modality_order):
            if mod in pooled:
                fused = fused + gates[:, i:i+1] * pooled[mod]

        fused = self.output_proj(fused)

        return fused


if __name__ == "__main__":
    print("=" * 60)
    print("Sensor Fusion Test")
    print("=" * 60)

    batch_size = 2
    hidden_dim = 512

    # Create dummy sensor features
    sensor_features = {
        "camera": torch.randn(batch_size, 64, hidden_dim),
        "lidar": torch.randn(batch_size, 32, hidden_dim),
        "radar": torch.randn(batch_size, 16, hidden_dim),
        "imu": torch.randn(batch_size, 8, hidden_dim),
    }

    # Test SensorFusion
    print("\nSensor Fusion (Self-Attention):")
    fusion1 = SensorFusion(hidden_dim=hidden_dim)
    out1 = fusion1(sensor_features)
    print(f"  Output: {out1.shape}")

    # Test CrossModalFusion
    print("\nCross-Modal Fusion:")
    fusion2 = CrossModalFusion(hidden_dim=hidden_dim)
    out2 = fusion2(sensor_features)
    print(f"  Output: {out2.shape}")

    # Test HierarchicalFusion
    print("\nHierarchical Fusion:")
    fusion3 = HierarchicalFusion(hidden_dim=hidden_dim)
    out3 = fusion3(sensor_features)
    print(f"  Output: {out3.shape}")

    # Test GatedFusion
    print("\nGated Fusion:")
    fusion4 = GatedFusion(hidden_dim=hidden_dim)
    out4 = fusion4(sensor_features)
    print(f"  Output: {out4.shape}")

    print("\n" + "=" * 60)
    print("All fusion tests passed!")
    print("=" * 60)
