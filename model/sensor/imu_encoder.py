"""
IMU Encoders

Encoders for processing Inertial Measurement Unit (IMU) data:
- Accelerometer (3-axis)
- Gyroscope (3-axis)
- Magnetometer (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, : x.size(1)]


class IMUEncoder(nn.Module):
    """
    Transformer-based IMU Encoder.

    Processes temporal sequences of IMU readings using self-attention.

    Args:
        input_dim: IMU feature dimension (6 for accel+gyro, 9 with mag)
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        seq_len: Expected sequence length
        num_tokens: Number of output tokens
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
        seq_len: int = 100,
        num_tokens: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)

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

        # Learnable query tokens for pooling
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02
        )

        # Cross-attention for compression
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Encode IMU sequence.

        Args:
            imu_data: (batch, seq_len, input_dim)

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B = imu_data.shape[0]

        # Project and add positional encoding
        x = self.input_proj(imu_data)
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Cross-attention pooling
        query = self.query_tokens.expand(B, -1, -1)
        compressed, _ = self.cross_attention(query, x, x)
        compressed = self.norm(compressed + query)

        # Final projection
        output = self.output_proj(compressed)

        return output


class TemporalIMUEncoder(nn.Module):
    """
    Temporal Convolutional IMU Encoder.

    Uses 1D convolutions for efficient temporal processing.
    Suitable for real-time applications.

    Args:
        input_dim: IMU feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        kernel_size: Convolution kernel size
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
        kernel_size: int = 5,
        num_tokens: int = 8,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # 1D Convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_tokens)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Encode IMU sequence.

        Args:
            imu_data: (batch, seq_len, input_dim)

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        # Transpose for Conv1d: (B, input_dim, seq_len)
        x = imu_data.transpose(1, 2)

        # Convolutional processing
        x = self.backbone(x)

        # Pool to fixed number of tokens
        x = self.adaptive_pool(x)

        # Transpose back and project
        x = x.transpose(1, 2)  # (B, num_tokens, hidden_dim)
        x = self.output_proj(x)

        return x


class LSTMIMUEncoder(nn.Module):
    """
    LSTM-based IMU Encoder.

    Uses bidirectional LSTM for temporal modeling.
    Good for capturing long-range dependencies.

    Args:
        input_dim: IMU feature dimension
        hidden_dim: LSTM hidden dimension
        output_dim: Output feature dimension
        num_layers: Number of LSTM layers
        num_tokens: Number of output tokens
        bidirectional: Whether to use bidirectional LSTM
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 2,
        num_tokens: int = 8,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.bidirectional = bidirectional

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Output projection
        self.output_proj = nn.Linear(lstm_output_dim, output_dim)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Encode IMU sequence.

        Args:
            imu_data: (batch, seq_len, input_dim)

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        B = imu_data.shape[0]

        # Project input
        x = self.input_proj(imu_data)

        # LSTM processing
        x, _ = self.lstm(x)

        # Project to output dim
        x = self.output_proj(x)

        # Cross-attention pooling
        query = self.query_tokens.expand(B, -1, -1)
        compressed, _ = self.cross_attention(query, x, x)
        compressed = self.norm(compressed + query)

        return compressed


class MultiModalIMUEncoder(nn.Module):
    """
    Multi-Modal IMU Encoder.

    Separately processes accelerometer, gyroscope, and optionally
    magnetometer data, then fuses them.

    Args:
        accel_dim: Accelerometer dimension (usually 3)
        gyro_dim: Gyroscope dimension (usually 3)
        mag_dim: Magnetometer dimension (0 if not used)
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_tokens: Number of output tokens
    """

    def __init__(
        self,
        accel_dim: int = 3,
        gyro_dim: int = 3,
        mag_dim: int = 0,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_tokens: int = 8,
    ):
        super().__init__()

        self.accel_dim = accel_dim
        self.gyro_dim = gyro_dim
        self.mag_dim = mag_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Separate encoders for each modality
        self.accel_encoder = TemporalIMUEncoder(
            input_dim=accel_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_tokens=num_tokens,
        )

        self.gyro_encoder = TemporalIMUEncoder(
            input_dim=gyro_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_tokens=num_tokens,
        )

        if mag_dim > 0:
            self.mag_encoder = TemporalIMUEncoder(
                input_dim=mag_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_tokens=num_tokens,
            )
            num_modalities = 3
        else:
            self.mag_encoder = None
            num_modalities = 2

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(
        self,
        accel: torch.Tensor,
        gyro: torch.Tensor,
        mag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode multi-modal IMU data.

        Args:
            accel: (batch, seq_len, accel_dim) accelerometer
            gyro: (batch, seq_len, gyro_dim) gyroscope
            mag: (batch, seq_len, mag_dim) magnetometer (optional)

        Returns:
            features: (batch, num_tokens, output_dim)
        """
        # Encode each modality
        accel_feat = self.accel_encoder(accel)  # (B, num_tokens, hidden_dim)
        gyro_feat = self.gyro_encoder(gyro)     # (B, num_tokens, hidden_dim)

        if self.mag_encoder is not None and mag is not None:
            mag_feat = self.mag_encoder(mag)    # (B, num_tokens, hidden_dim)
            combined = torch.cat([accel_feat, gyro_feat, mag_feat], dim=-1)
        else:
            combined = torch.cat([accel_feat, gyro_feat], dim=-1)

        # Fuse modalities
        output = self.fusion(combined)

        return output


if __name__ == "__main__":
    print("=" * 60)
    print("IMU Encoder Test")
    print("=" * 60)

    batch_size = 2
    seq_len = 100
    input_dim = 6  # 3 accel + 3 gyro
    output_dim = 256
    num_tokens = 8

    # Create dummy IMU data
    imu_data = torch.randn(batch_size, seq_len, input_dim)

    # Test Transformer IMU Encoder
    print("\nTransformer IMU Encoder:")
    trans_encoder = IMUEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_tokens=num_tokens,
    )
    out = trans_encoder(imu_data)
    print(f"  Input: {imu_data.shape}")
    print(f"  Output: {out.shape}")

    # Test Temporal Conv IMU Encoder
    print("\nTemporal Conv IMU Encoder:")
    conv_encoder = TemporalIMUEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        num_tokens=num_tokens,
    )
    out = conv_encoder(imu_data)
    print(f"  Input: {imu_data.shape}")
    print(f"  Output: {out.shape}")

    # Test LSTM IMU Encoder
    print("\nLSTM IMU Encoder:")
    lstm_encoder = LSTMIMUEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        num_tokens=num_tokens,
    )
    out = lstm_encoder(imu_data)
    print(f"  Input: {imu_data.shape}")
    print(f"  Output: {out.shape}")

    # Test Multi-Modal IMU Encoder
    print("\nMulti-Modal IMU Encoder:")
    accel = torch.randn(batch_size, seq_len, 3)
    gyro = torch.randn(batch_size, seq_len, 3)
    mm_encoder = MultiModalIMUEncoder(
        accel_dim=3,
        gyro_dim=3,
        output_dim=output_dim,
        num_tokens=num_tokens,
    )
    out = mm_encoder(accel, gyro)
    print(f"  Accel: {accel.shape}, Gyro: {gyro.shape}")
    print(f"  Output: {out.shape}")

    print("\n" + "=" * 60)
    print("All IMU encoder tests passed!")
    print("=" * 60)
