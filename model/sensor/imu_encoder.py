"""
IMU Encoders

Encoders for processing Inertial Measurement Unit (IMU) data:
- Accelerometer (3-axis)
- Gyroscope (3-axis)
- Magnetometer (optional)
"""

import torch
import torch.nn as nn
from typing import Optional

from model.utils.layers import PositionalEncoding


class IMUEncoder(nn.Module):
    """
    Transformer-based IMU Encoder.

    Processes temporal sequences of IMU readings using self-attention.
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

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """Encode IMU sequence. Input: (batch, seq_len, input_dim) -> (batch, num_tokens, output_dim)"""
        B = imu_data.shape[0]

        x = self.input_proj(imu_data)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        query = self.query_tokens.expand(B, -1, -1)
        compressed, _ = self.cross_attention(query, x, x)
        compressed = self.norm(compressed + query)

        return self.output_proj(compressed)


class TemporalIMUEncoder(nn.Module):
    """Temporal Convolutional IMU Encoder for real-time applications."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
        kernel_size: int = 5,
        num_tokens: int = 8,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_tokens = num_tokens

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
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        x = imu_data.transpose(1, 2)  # (B, input_dim, seq_len)
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = x.transpose(1, 2)  # (B, num_tokens, hidden_dim)
        return self.output_proj(x)


class LSTMIMUEncoder(nn.Module):
    """LSTM-based IMU Encoder for long-range dependencies."""

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
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(lstm_output_dim, output_dim)

        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, output_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        B = imu_data.shape[0]

        x = self.input_proj(imu_data)
        x, _ = self.lstm(x)
        x = self.output_proj(x)

        query = self.query_tokens.expand(B, -1, -1)
        compressed, _ = self.cross_attention(query, x, x)
        return self.norm(compressed + query)


class MultiModalIMUEncoder(nn.Module):
    """Multi-Modal IMU Encoder for separate accelerometer, gyroscope, magnetometer processing."""

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
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        self.accel_encoder = TemporalIMUEncoder(accel_dim, hidden_dim, hidden_dim, num_tokens=num_tokens)
        self.gyro_encoder = TemporalIMUEncoder(gyro_dim, hidden_dim, hidden_dim, num_tokens=num_tokens)

        self.mag_encoder = None
        num_modalities = 2
        if mag_dim > 0:
            self.mag_encoder = TemporalIMUEncoder(mag_dim, hidden_dim, hidden_dim, num_tokens=num_tokens)
            num_modalities = 3

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
        accel_feat = self.accel_encoder(accel)
        gyro_feat = self.gyro_encoder(gyro)

        if self.mag_encoder is not None and mag is not None:
            mag_feat = self.mag_encoder(mag)
            combined = torch.cat([accel_feat, gyro_feat, mag_feat], dim=-1)
        else:
            combined = torch.cat([accel_feat, gyro_feat], dim=-1)

        return self.fusion(combined)


if __name__ == "__main__":
    print("IMU Encoder Test")
    batch_size, seq_len, input_dim = 2, 100, 6

    imu_data = torch.randn(batch_size, seq_len, input_dim)

    trans_encoder = IMUEncoder(input_dim=input_dim, output_dim=256, seq_len=seq_len, num_tokens=8)
    print(f"Transformer IMU: {imu_data.shape} -> {trans_encoder(imu_data).shape}")

    conv_encoder = TemporalIMUEncoder(input_dim=input_dim, output_dim=256, num_tokens=8)
    print(f"Temporal Conv IMU: {imu_data.shape} -> {conv_encoder(imu_data).shape}")

    lstm_encoder = LSTMIMUEncoder(input_dim=input_dim, output_dim=256, num_tokens=8)
    print(f"LSTM IMU: {imu_data.shape} -> {lstm_encoder(imu_data).shape}")

    accel, gyro = torch.randn(batch_size, seq_len, 3), torch.randn(batch_size, seq_len, 3)
    mm_encoder = MultiModalIMUEncoder(accel_dim=3, gyro_dim=3, output_dim=256, num_tokens=8)
    print(f"Multi-Modal IMU: accel={accel.shape}, gyro={gyro.shape} -> {mm_encoder(accel, gyro).shape}")

    print("All IMU encoder tests passed!")
