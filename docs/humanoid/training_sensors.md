# Humanoid Sensor Training Guide

This document covers training pipelines for humanoid sensor processing, including joint encoders, IMU processing, force/torque sensors, foot contact detection, and proprioception fusion.

## Table of Contents

1. [Overview](#overview)
2. [Proprioception Encoder](#proprioception-encoder)
3. [Joint State Encoding](#joint-state-encoding)
4. [IMU Processing](#imu-processing)
5. [Foot Contact Sensing](#foot-contact-sensing)
6. [Force/Torque Sensors](#forcetorque-sensors)
7. [Sensor Fusion for Humanoid](#sensor-fusion-for-humanoid)
8. [Training Pipelines](#training-pipelines)
9. [Sensor Calibration](#sensor-calibration)
10. [Real-Time Considerations](#real-time-considerations)

---

## Overview

Humanoid robots rely on rich proprioceptive and exteroceptive sensing for balance, locomotion, and manipulation. The sensor stack includes:

| Sensor Type | Dimensions | Update Rate | Primary Use |
|-------------|------------|-------------|-------------|
| Joint Encoders | 32 joints × 3 (pos, vel, torque) | 1000 Hz | State estimation |
| IMU | 9-dim (orientation, angular vel, accel) | 400 Hz | Balance control |
| Foot Contact | 4 binary (2 feet × 2 sensors) | 1000 Hz | Gait detection |
| Force/Torque | 12-dim (6 per wrist/ankle) | 500 Hz | Manipulation/Balance |
| Vision | 2 cameras (head-mounted) | 30 Hz | Task guidance |
| Depth | 1 depth sensor | 30 Hz | Obstacle avoidance |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sensor Input Layer                           │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│   Joints    │     IMU     │   Contact   │    F/T      │ Vision │
│  (32×3)     │    (9)      │    (4)      │   (12)      │ (RGB)  │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴───┬────┘
       │             │             │             │          │
       ▼             ▼             ▼             ▼          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Proprioception Encoder                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │  Joint  │  │   IMU   │  │ Contact │  │   F/T   │            │
│  │ Encoder │  │ Encoder │  │ Encoder │  │ Encoder │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       └────────────┴────────────┴────────────┘                  │
│                          │                                       │
│                    ┌─────▼─────┐                                │
│                    │  Fusion   │                                │
│                    │    MLP    │                                │
│                    └─────┬─────┘                                │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Proprioception Feature │
              │      (hidden_dim)       │
              └────────────────────────┘
```

---

## Proprioception Encoder

The `ProprioceptionEncoder` fuses all proprioceptive sensors into a unified representation.

**Reference:** `model/embodiment/humanoid.py:46-126`

### Architecture

```python
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class HumanoidConfig:
    """Configuration for humanoid embodiment."""
    num_joints: int = 32
    proprioception_dim: int = 128
    hidden_dim: int = 256
    action_type: str = "position"  # or "torque"
    use_imu: bool = True
    use_foot_contact: bool = True
    use_force_torque: bool = True


class ProprioceptionEncoder(nn.Module):
    """
    Encode proprioceptive state from multiple sensors.

    Fuses:
    - Joint positions, velocities, torques (32 joints each)
    - IMU data (orientation, angular velocity, acceleration)
    - Foot contact binary flags (4 contact points)
    - Force/torque sensors (optional)
    """

    def __init__(self, config: HumanoidConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        num_joints = config.num_joints

        # Joint state encoder: position + velocity + torque
        joint_input_dim = num_joints * 3  # 32 * 3 = 96
        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # IMU encoder: orientation (3) + angular_vel (3) + accel (3) = 9
        if config.use_imu:
            self.imu_encoder = nn.Sequential(
                nn.Linear(9, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
            )

        # Foot contact encoder: 4 binary contact flags
        if config.use_foot_contact:
            self.contact_encoder = nn.Sequential(
                nn.Linear(4, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim // 4),
            )

        # Force/torque encoder: 6-dim per sensor (2 wrists + 2 ankles)
        if config.use_force_torque:
            self.ft_encoder = nn.Sequential(
                nn.Linear(24, hidden_dim // 2),  # 6 × 4 sensors
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
            )

        # Fusion layer
        fusion_input_dim = hidden_dim  # Joint encoder output
        if config.use_imu:
            fusion_input_dim += hidden_dim // 2
        if config.use_foot_contact:
            fusion_input_dim += hidden_dim // 4
        if config.use_force_torque:
            fusion_input_dim += hidden_dim // 4

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.proprioception_dim),
        )

    def forward(
        self,
        joint_positions: torch.Tensor,    # (batch, num_joints)
        joint_velocities: torch.Tensor,   # (batch, num_joints)
        joint_torques: torch.Tensor,      # (batch, num_joints)
        imu_data: Optional[torch.Tensor] = None,      # (batch, 9)
        foot_contacts: Optional[torch.Tensor] = None,  # (batch, 4)
        force_torque: Optional[torch.Tensor] = None,   # (batch, 24)
    ) -> torch.Tensor:
        """
        Encode proprioceptive state.

        Returns:
            Proprioception feature vector (batch, proprioception_dim)
        """
        # Encode joint state
        joint_state = torch.cat([
            joint_positions,
            joint_velocities,
            joint_torques
        ], dim=-1)
        joint_features = self.joint_encoder(joint_state)

        features = [joint_features]

        # Encode IMU
        if self.config.use_imu and imu_data is not None:
            imu_features = self.imu_encoder(imu_data)
            features.append(imu_features)

        # Encode foot contacts
        if self.config.use_foot_contact and foot_contacts is not None:
            contact_features = self.contact_encoder(foot_contacts.float())
            features.append(contact_features)

        # Encode force/torque
        if self.config.use_force_torque and force_torque is not None:
            ft_features = self.ft_encoder(force_torque)
            features.append(ft_features)

        # Fuse all features
        fused = torch.cat(features, dim=-1)
        return self.fusion(fused)
```

---

## Joint State Encoding

### Joint Configuration

Standard humanoid joint layout (32 DoF):

| Joint Group | Joints | DoF | Index Range |
|-------------|--------|-----|-------------|
| Head | neck_yaw, neck_pitch | 2 | 0-1 |
| Left Arm | shoulder_pitch/roll/yaw, elbow, wrist_pitch/roll/yaw | 7 | 2-8 |
| Right Arm | shoulder_pitch/roll/yaw, elbow, wrist_pitch/roll/yaw | 7 | 9-15 |
| Torso | waist_yaw, waist_pitch, waist_roll | 3 | 16-18 |
| Left Leg | hip_yaw/roll/pitch, knee, ankle_pitch/roll | 6 | 19-24 |
| Right Leg | hip_yaw/roll/pitch, knee, ankle_pitch/roll | 6 | 25-30 |
| Base | virtual_x, virtual_y (floating base) | 1 | 31 |

### Joint Encoder Architecture

```python
class JointStateEncoder(nn.Module):
    """
    Hierarchical joint state encoder with body part awareness.
    """

    def __init__(
        self,
        num_joints: int = 32,
        hidden_dim: int = 256,
        num_body_parts: int = 6,
    ):
        super().__init__()
        self.num_joints = num_joints

        # Body part indices
        self.body_parts = {
            'head': [0, 1],
            'left_arm': list(range(2, 9)),
            'right_arm': list(range(9, 16)),
            'torso': [16, 17, 18],
            'left_leg': list(range(19, 25)),
            'right_leg': list(range(25, 31)),
        }

        # Per-body-part encoders
        self.part_encoders = nn.ModuleDict()
        for part_name, indices in self.body_parts.items():
            input_dim = len(indices) * 3  # pos + vel + torque
            self.part_encoders[part_name] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // num_body_parts),
                nn.LayerNorm(hidden_dim // num_body_parts),
                nn.ReLU(),
            )

        # Cross-body attention for coordination
        self.cross_body_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // num_body_parts,
            num_heads=4,
            batch_first=True,
        )

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hierarchical encoding with body part structure.
        """
        batch_size = joint_positions.shape[0]

        # Encode each body part
        part_features = []
        for part_name, indices in self.body_parts.items():
            # Extract joint states for this body part
            pos = joint_positions[:, indices]
            vel = joint_velocities[:, indices]
            torque = joint_torques[:, indices]

            part_state = torch.cat([pos, vel, torque], dim=-1)
            part_feat = self.part_encoders[part_name](part_state)
            part_features.append(part_feat)

        # Stack as sequence: (batch, num_parts, feat_dim)
        part_features = torch.stack(part_features, dim=1)

        # Cross-body attention for coordination
        attended, _ = self.cross_body_attention(
            part_features, part_features, part_features
        )

        # Flatten and project
        output = attended.reshape(batch_size, -1)
        return self.output_proj(output)
```

### Training Configuration for Joint Encoders

```yaml
# configs/humanoid/joint_encoder.yaml
joint_encoder:
  num_joints: 32
  hidden_dim: 256
  use_hierarchical: true
  body_part_attention: true

  # Normalization ranges (for standardization)
  position_range: [-3.14, 3.14]  # radians
  velocity_range: [-10.0, 10.0]  # rad/s
  torque_range: [-100.0, 100.0]  # Nm

  # Noise augmentation during training
  noise:
    position_std: 0.01  # radians
    velocity_std: 0.05  # rad/s
    torque_std: 1.0     # Nm
```

---

## IMU Processing

The IMU provides crucial balance information for bipedal locomotion.

**Reference:** `model/sensor/imu_encoder.py`

### IMU Encoder Variants

```python
import torch
import torch.nn as nn
from typing import Optional


class IMUEncoder(nn.Module):
    """
    Transformer-based IMU encoder for temporal sequences.

    Processes accelerometer (3-axis) + gyroscope (3-axis) data
    over time sequences for motion estimation.
    """

    def __init__(
        self,
        input_dim: int = 6,  # accel (3) + gyro (3)
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding for temporal sequences
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection with attention pooling
        self.attention_pool = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        imu_sequence: torch.Tensor,  # (batch, seq_len, 6)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode IMU sequence.

        Returns:
            IMU features (batch, output_dim)
        """
        batch_size, seq_len, _ = imu_sequence.shape

        # Project input
        x = self.input_proj(imu_sequence)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Attention pooling over sequence
        attn_weights = torch.softmax(
            self.attention_pool(x).squeeze(-1), dim=-1
        )
        pooled = torch.einsum('bs,bsd->bd', attn_weights, x)

        return self.output_proj(pooled)


class TemporalConvIMUEncoder(nn.Module):
    """
    Temporal Convolutional Network for real-time IMU processing.
    Lower latency than transformer for online control.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 4,
    ):
        super().__init__()

        layers = []
        in_channels = input_dim

        for i in range(num_layers):
            out_channels = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
            ])
            in_channels = out_channels

        self.conv_net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, imu_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imu_sequence: (batch, seq_len, input_dim)
        Returns:
            IMU features (batch, output_dim)
        """
        # Transpose for conv1d: (batch, channels, seq_len)
        x = imu_sequence.transpose(1, 2)
        x = self.conv_net(x)
        x = self.pool(x).squeeze(-1)
        return x


class LSTMIMUEncoder(nn.Module):
    """
    LSTM-based IMU encoder for long-range temporal dependencies.
    Good for capturing drift and long-term motion patterns.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, imu_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imu_sequence: (batch, seq_len, input_dim)
        Returns:
            IMU features (batch, output_dim)
        """
        output, (h_n, c_n) = self.lstm(imu_sequence)

        # Use last hidden state (concatenated for bidirectional)
        if self.lstm.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final = h_n[-1]

        return self.output_proj(final)
```

### IMU Training Configuration

```yaml
# configs/humanoid/imu_encoder.yaml
imu_encoder:
  type: "temporal_conv"  # "transformer", "temporal_conv", or "lstm"
  input_dim: 6  # accel (3) + gyro (3)
  hidden_dim: 128
  output_dim: 64
  sequence_length: 50  # 50ms at 1000Hz

  # Sensor specifications
  accelerometer:
    range: [-16.0, 16.0]  # g
    noise_density: 0.001  # g/sqrt(Hz)
    bias_stability: 0.0001  # g

  gyroscope:
    range: [-2000.0, 2000.0]  # deg/s
    noise_density: 0.01  # deg/s/sqrt(Hz)
    bias_stability: 0.001  # deg/s

  # Training augmentation
  augmentation:
    add_noise: true
    noise_scale: 0.1
    random_bias: true
    bias_range: [-0.01, 0.01]
    time_warp: true
    warp_factor: 0.1
```

### IMU-Based Balance Controller

```python
class BalanceController(nn.Module):
    """
    Use IMU data for real-time balance control.
    Estimates body orientation and provides corrective torques.
    """

    def __init__(
        self,
        imu_encoder: nn.Module,
        hidden_dim: int = 128,
        num_joints: int = 32,
    ):
        super().__init__()
        self.imu_encoder = imu_encoder

        self.balance_net = nn.Sequential(
            nn.Linear(imu_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output corrective torques for lower body (12 joints)
        self.torque_head = nn.Linear(hidden_dim, 12)

        # Output desired CoM velocity adjustment
        self.com_adjustment = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        imu_sequence: torch.Tensor,  # (batch, seq_len, 6)
        current_joint_state: torch.Tensor,  # (batch, num_joints * 3)
    ) -> tuple:
        """
        Compute balance corrections.

        Returns:
            corrective_torques: (batch, 12) for legs
            com_velocity_adjustment: (batch, 3) desired CoM vel change
        """
        # Encode IMU history
        imu_features = self.imu_encoder(imu_sequence)

        # Compute balance corrections
        balance_features = self.balance_net(imu_features)

        corrective_torques = self.torque_head(balance_features)
        com_adjustment = self.com_adjustment(balance_features)

        return corrective_torques, com_adjustment
```

---

## Foot Contact Sensing

Foot contact detection is critical for gait phase estimation and stable locomotion.

### Contact Sensor Configuration

```
Foot Contact Layout:
┌─────────────────────────────────────┐
│           LEFT FOOT                 │
│   ┌─────┐           ┌─────┐        │
│   │ LF  │           │ LR  │        │
│   │ (0) │           │ (1) │        │
│   └─────┘           └─────┘        │
│    Toe               Heel          │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│           RIGHT FOOT                │
│   ┌─────┐           ┌─────┐        │
│   │ RF  │           │ RR  │        │
│   │ (2) │           │ (3) │        │
│   └─────┘           └─────┘        │
│    Toe               Heel          │
└─────────────────────────────────────┘

Contact States:
- [1,1,0,0]: Left foot stance, right foot swing
- [0,0,1,1]: Right foot stance, left foot swing
- [1,1,1,1]: Double support (both feet)
- [0,0,0,0]: Flight phase (jumping/running)
```

### Contact Encoder and Gait Phase Estimator

```python
class FootContactEncoder(nn.Module):
    """
    Encode foot contact information with gait phase estimation.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Contact state encoder
        self.contact_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temporal contact history for gait phase
        self.gait_lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Gait phase classifier (4 phases)
        self.phase_classifier = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        current_contacts: torch.Tensor,   # (batch, 4) binary
        contact_history: torch.Tensor,    # (batch, history_len, 4)
    ) -> tuple:
        """
        Encode contacts and estimate gait phase.

        Returns:
            contact_features: (batch, hidden_dim)
            gait_phase: (batch, 4) phase probabilities
        """
        # Encode current contact state
        contact_features = self.contact_encoder(current_contacts.float())

        # Estimate gait phase from history
        _, (h_n, _) = self.gait_lstm(contact_history.float())
        gait_features = h_n[-1]
        gait_phase = torch.softmax(self.phase_classifier(gait_features), dim=-1)

        return contact_features, gait_phase


class GaitPhaseEstimator(nn.Module):
    """
    Estimate continuous gait phase (0 to 2π) for cyclic locomotion.
    Uses foot contacts + IMU for robust phase estimation.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(4 + 6, hidden_dim),  # contacts + IMU
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output sin and cos of phase for continuity
        self.phase_head = nn.Linear(hidden_dim, 2)  # [sin(φ), cos(φ)]

    def forward(
        self,
        contacts: torch.Tensor,  # (batch, 4)
        imu_data: torch.Tensor,  # (batch, 6)
    ) -> torch.Tensor:
        """
        Returns:
            gait_phase: (batch, 1) in [0, 2π]
        """
        x = torch.cat([contacts.float(), imu_data], dim=-1)
        features = self.encoder(x)
        sin_cos = self.phase_head(features)

        # Convert to phase angle
        phase = torch.atan2(sin_cos[:, 0:1], sin_cos[:, 1:2])
        phase = (phase + torch.pi) % (2 * torch.pi)  # Normalize to [0, 2π]

        return phase
```

---

## Force/Torque Sensors

Force/torque (F/T) sensors at wrists and ankles enable force-controlled manipulation and balance.

### F/T Sensor Configuration

| Location | Measured Forces | Measured Torques | Use Case |
|----------|-----------------|------------------|----------|
| Left Wrist | Fx, Fy, Fz | Tx, Ty, Tz | Manipulation force control |
| Right Wrist | Fx, Fy, Fz | Tx, Ty, Tz | Manipulation force control |
| Left Ankle | Fx, Fy, Fz | Tx, Ty, Tz | Ground reaction, ZMP |
| Right Ankle | Fx, Fy, Fz | Tx, Ty, Tz | Ground reaction, ZMP |

### F/T Encoder Implementation

```python
class ForceTorqueEncoder(nn.Module):
    """
    Encode force/torque sensor data from multiple locations.
    """

    def __init__(
        self,
        num_sensors: int = 4,  # 2 wrists + 2 ankles
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()
        self.num_sensors = num_sensors

        # Per-sensor encoder
        self.sensor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(6, hidden_dim // num_sensors),  # 6-dim F/T
                nn.LayerNorm(hidden_dim // num_sensors),
                nn.ReLU(),
            )
            for _ in range(num_sensors)
        ])

        # Cross-sensor attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // num_sensors,
            num_heads=2,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, ft_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ft_data: (batch, num_sensors, 6) F/T readings
        Returns:
            F/T features (batch, output_dim)
        """
        batch_size = ft_data.shape[0]

        # Encode each sensor
        sensor_features = []
        for i, encoder in enumerate(self.sensor_encoders):
            feat = encoder(ft_data[:, i, :])
            sensor_features.append(feat)

        # Stack as sequence
        sensor_features = torch.stack(sensor_features, dim=1)

        # Cross-sensor attention
        attended, _ = self.cross_attention(
            sensor_features, sensor_features, sensor_features
        )

        # Flatten and project
        output = attended.reshape(batch_size, -1)
        return self.output_proj(output)


class ZMPEstimator(nn.Module):
    """
    Estimate Zero Moment Point from ankle F/T sensors.
    Critical for bipedal balance control.
    """

    def __init__(self, foot_length: float = 0.25, foot_width: float = 0.12):
        super().__init__()
        self.foot_length = foot_length
        self.foot_width = foot_width

    def forward(
        self,
        left_ft: torch.Tensor,   # (batch, 6) [Fx, Fy, Fz, Tx, Ty, Tz]
        right_ft: torch.Tensor,  # (batch, 6)
        left_foot_pos: torch.Tensor,   # (batch, 3) foot position in world
        right_foot_pos: torch.Tensor,  # (batch, 3)
    ) -> torch.Tensor:
        """
        Compute ZMP location in world frame.

        Returns:
            zmp_position: (batch, 2) [x, y] in world frame
        """
        # Extract vertical force and torques
        fz_left = left_ft[:, 2:3]
        fz_right = right_ft[:, 2:3]

        tx_left = left_ft[:, 3:4]
        ty_left = left_ft[:, 4:5]
        tx_right = right_ft[:, 3:4]
        ty_right = right_ft[:, 4:5]

        # Total vertical force
        fz_total = fz_left + fz_right + 1e-6  # Avoid division by zero

        # ZMP in each foot frame
        zmp_left_x = -ty_left / (fz_left + 1e-6)
        zmp_left_y = tx_left / (fz_left + 1e-6)
        zmp_right_x = -ty_right / (fz_right + 1e-6)
        zmp_right_y = tx_right / (fz_right + 1e-6)

        # Transform to world and combine weighted by vertical force
        zmp_x = (fz_left * (left_foot_pos[:, 0:1] + zmp_left_x) +
                 fz_right * (right_foot_pos[:, 0:1] + zmp_right_x)) / fz_total
        zmp_y = (fz_left * (left_foot_pos[:, 1:2] + zmp_left_y) +
                 fz_right * (right_foot_pos[:, 1:2] + zmp_right_y)) / fz_total

        return torch.cat([zmp_x, zmp_y], dim=-1)
```

---

## Sensor Fusion for Humanoid

**Reference:** `model/fusion/sensor_fusion.py`

### Multi-Modal Sensor Fusion

```python
class HumanoidSensorFusion(nn.Module):
    """
    Fuse all humanoid sensors into unified state representation.
    """

    def __init__(
        self,
        proprioception_dim: int = 128,
        imu_dim: int = 64,
        contact_dim: int = 32,
        ft_dim: int = 64,
        vision_dim: int = 512,
        output_dim: int = 256,
        fusion_type: str = "attention",  # "concat", "attention", "gated"
    ):
        super().__init__()
        self.fusion_type = fusion_type

        # Input dimensions
        self.modality_dims = {
            'proprioception': proprioception_dim,
            'imu': imu_dim,
            'contact': contact_dim,
            'force_torque': ft_dim,
            'vision': vision_dim,
        }

        total_dim = sum(self.modality_dims.values())

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.ReLU(),
                nn.Linear(output_dim * 2, output_dim),
            )

        elif fusion_type == "attention":
            # Cross-modal attention fusion
            self.modality_projs = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in self.modality_dims.items()
            })
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                batch_first=True,
            )
            self.output_proj = nn.Linear(output_dim, output_dim)

        elif fusion_type == "gated":
            # Gated fusion with learned modality weights
            self.modality_projs = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in self.modality_dims.items()
            })
            self.gate_net = nn.Sequential(
                nn.Linear(total_dim, len(self.modality_dims)),
                nn.Softmax(dim=-1),
            )
            self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        proprioception: torch.Tensor,  # (batch, proprioception_dim)
        imu: torch.Tensor,             # (batch, imu_dim)
        contact: torch.Tensor,         # (batch, contact_dim)
        force_torque: torch.Tensor,    # (batch, ft_dim)
        vision: torch.Tensor,          # (batch, vision_dim)
    ) -> torch.Tensor:
        """
        Fuse all sensor modalities.

        Returns:
            Fused state representation (batch, output_dim)
        """
        features = {
            'proprioception': proprioception,
            'imu': imu,
            'contact': contact,
            'force_torque': force_torque,
            'vision': vision,
        }

        if self.fusion_type == "concat":
            concat_features = torch.cat(list(features.values()), dim=-1)
            return self.fusion(concat_features)

        elif self.fusion_type == "attention":
            # Project all modalities to same dimension
            projected = [
                self.modality_projs[name](feat).unsqueeze(1)
                for name, feat in features.items()
            ]
            stacked = torch.cat(projected, dim=1)  # (batch, num_modalities, output_dim)

            # Self-attention across modalities
            attended, _ = self.cross_attention(stacked, stacked, stacked)

            # Average pool
            pooled = attended.mean(dim=1)
            return self.output_proj(pooled)

        elif self.fusion_type == "gated":
            # Compute gates from concatenated features
            concat_features = torch.cat(list(features.values()), dim=-1)
            gates = self.gate_net(concat_features)  # (batch, num_modalities)

            # Project and weight by gates
            projected = [
                self.modality_projs[name](feat)
                for name, feat in features.items()
            ]
            stacked = torch.stack(projected, dim=1)  # (batch, num_modalities, output_dim)

            # Weighted sum
            weighted = torch.einsum('bm,bmd->bd', gates, stacked)
            return self.output_proj(weighted)
```

### Hierarchical Fusion for Humanoid

```python
class HierarchicalHumanoidFusion(nn.Module):
    """
    Hierarchical fusion matching humanoid control hierarchy:

    Level 1: Low-level proprioception (joints + contacts)
    Level 2: Balance (proprioception + IMU + F/T)
    Level 3: Task-level (balance + vision + language)
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # Level 1: Proprioception fusion
        self.proprioception_fusion = nn.Sequential(
            nn.Linear(128 + 32, hidden_dim),  # joints + contacts
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Level 2: Balance fusion
        self.balance_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64, hidden_dim),  # proprio + imu + ft
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Level 3: Task fusion with attention
        self.task_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )

        self.vision_proj = nn.Linear(512, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        joint_features: torch.Tensor,    # (batch, 128)
        contact_features: torch.Tensor,  # (batch, 32)
        imu_features: torch.Tensor,      # (batch, 64)
        ft_features: torch.Tensor,       # (batch, 64)
        vision_features: torch.Tensor,   # (batch, 512)
    ) -> dict:
        """
        Hierarchical fusion with intermediate outputs.

        Returns:
            Dictionary with features at each hierarchy level.
        """
        # Level 1: Low-level proprioception
        proprio = torch.cat([joint_features, contact_features], dim=-1)
        proprio_fused = self.proprioception_fusion(proprio)

        # Level 2: Balance-level fusion
        balance = torch.cat([proprio_fused, imu_features, ft_features], dim=-1)
        balance_fused = self.balance_fusion(balance)

        # Level 3: Task-level fusion with vision
        vision_proj = self.vision_proj(vision_features)

        # Stack for attention: [balance, vision]
        tokens = torch.stack([balance_fused, vision_proj], dim=1)
        attended, _ = self.task_attention(tokens, tokens, tokens)
        task_fused = self.output_proj(attended.mean(dim=1))

        return {
            'proprioception': proprio_fused,
            'balance': balance_fused,
            'task': task_fused,
        }
```

---

## Training Pipelines

### Stage 1: Individual Sensor Encoder Training

```python
# scripts/train_humanoid_sensors.py

def train_joint_encoder(config):
    """Pre-train joint state encoder with reconstruction."""

    model = JointStateEncoder(
        num_joints=config.num_joints,
        hidden_dim=config.hidden_dim,
    )

    # Reconstruction head
    decoder = nn.Linear(config.hidden_dim, config.num_joints * 3)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
    )

    for epoch in range(config.num_epochs):
        for batch in dataloader:
            joint_pos = batch['joint_positions']
            joint_vel = batch['joint_velocities']
            joint_torque = batch['joint_torques']

            # Forward
            features = model(joint_pos, joint_vel, joint_torque)

            # Reconstruct
            target = torch.cat([joint_pos, joint_vel, joint_torque], dim=-1)
            reconstruction = decoder(features)

            # Loss
            loss = F.mse_loss(reconstruction, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_imu_encoder(config):
    """Pre-train IMU encoder with orientation prediction."""

    model = TemporalConvIMUEncoder(
        input_dim=6,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
    )

    # Orientation prediction head (quaternion)
    orientation_head = nn.Linear(config.output_dim, 4)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(orientation_head.parameters()),
        lr=config.learning_rate,
    )

    for epoch in range(config.num_epochs):
        for batch in dataloader:
            imu_sequence = batch['imu_sequence']  # (batch, seq_len, 6)
            target_orientation = batch['orientation']  # (batch, 4) quaternion

            # Forward
            features = model(imu_sequence)
            pred_orientation = orientation_head(features)

            # Normalize to unit quaternion
            pred_orientation = F.normalize(pred_orientation, dim=-1)

            # Quaternion loss (handle double cover)
            loss = min(
                F.mse_loss(pred_orientation, target_orientation),
                F.mse_loss(pred_orientation, -target_orientation),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Stage 2: Sensor Fusion Training

```python
def train_sensor_fusion(config, pretrained_encoders):
    """Train sensor fusion with frozen or fine-tuned encoders."""

    fusion_model = HumanoidSensorFusion(
        proprioception_dim=128,
        imu_dim=64,
        contact_dim=32,
        ft_dim=64,
        vision_dim=512,
        output_dim=256,
        fusion_type=config.fusion_type,
    )

    # Optionally freeze pretrained encoders
    if config.freeze_encoders:
        for encoder in pretrained_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = False

    optimizer = torch.optim.AdamW(
        fusion_model.parameters(),
        lr=config.learning_rate,
    )

    for epoch in range(config.num_epochs):
        for batch in dataloader:
            # Encode each modality
            proprio_feat = pretrained_encoders['proprio'](
                batch['joint_pos'], batch['joint_vel'], batch['joint_torque']
            )
            imu_feat = pretrained_encoders['imu'](batch['imu_sequence'])
            contact_feat = pretrained_encoders['contact'](batch['contacts'])
            ft_feat = pretrained_encoders['ft'](batch['force_torque'])
            vision_feat = pretrained_encoders['vision'](batch['images'])

            # Fuse
            fused = fusion_model(
                proprio_feat, imu_feat, contact_feat, ft_feat, vision_feat
            )

            # Task-specific loss (e.g., action prediction)
            pred_action = action_head(fused)
            loss = F.mse_loss(pred_action, batch['target_action'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Training Configuration

```yaml
# configs/humanoid/sensor_training.yaml
sensor_training:
  # Stage 1: Individual encoders
  joint_encoder:
    hidden_dim: 256
    learning_rate: 1e-4
    num_epochs: 100
    batch_size: 256

  imu_encoder:
    type: "temporal_conv"
    hidden_dim: 128
    output_dim: 64
    learning_rate: 1e-4
    num_epochs: 100
    sequence_length: 50

  contact_encoder:
    hidden_dim: 64
    learning_rate: 1e-4
    num_epochs: 50

  ft_encoder:
    hidden_dim: 128
    output_dim: 64
    learning_rate: 1e-4
    num_epochs: 100

  # Stage 2: Fusion
  fusion:
    type: "attention"
    output_dim: 256
    learning_rate: 5e-5
    num_epochs: 200
    freeze_encoders: false

  # Data augmentation
  augmentation:
    joint_noise_std: 0.01
    imu_noise_scale: 0.1
    contact_flip_prob: 0.05
    ft_noise_std: 0.5
```

---

## Sensor Calibration

### Joint Encoder Calibration

```python
class JointCalibration:
    """Calibrate joint encoders for offset and scale errors."""

    def __init__(self, num_joints: int = 32):
        self.offsets = np.zeros(num_joints)
        self.scales = np.ones(num_joints)

    def calibrate_from_known_pose(
        self,
        measured_positions: np.ndarray,
        known_positions: np.ndarray,
    ):
        """
        Calibrate using known reference poses.

        Args:
            measured_positions: (num_samples, num_joints) measured values
            known_positions: (num_samples, num_joints) ground truth
        """
        for j in range(self.num_joints):
            # Linear regression: measured = scale * actual + offset
            A = np.column_stack([known_positions[:, j], np.ones(len(known_positions))])
            coeffs, _, _, _ = np.linalg.lstsq(A, measured_positions[:, j], rcond=None)
            self.scales[j] = coeffs[0]
            self.offsets[j] = coeffs[1]

    def apply(self, raw_positions: np.ndarray) -> np.ndarray:
        """Apply calibration to raw measurements."""
        return (raw_positions - self.offsets) / self.scales
```

### IMU Calibration

```python
class IMUCalibration:
    """
    Calibrate IMU for bias, scale, and misalignment.
    """

    def __init__(self):
        self.accel_bias = np.zeros(3)
        self.accel_scale = np.eye(3)
        self.gyro_bias = np.zeros(3)
        self.gyro_scale = np.eye(3)

    def calibrate_stationary(
        self,
        accel_samples: np.ndarray,  # (num_samples, 3)
        gyro_samples: np.ndarray,   # (num_samples, 3)
        gravity: float = 9.81,
    ):
        """
        Calibrate using stationary data.
        Assumes robot is stationary and level.
        """
        # Gyro bias: mean of stationary readings
        self.gyro_bias = np.mean(gyro_samples, axis=0)

        # Accel bias: mean should equal [0, 0, -g] when level
        accel_mean = np.mean(accel_samples, axis=0)
        expected_accel = np.array([0, 0, -gravity])
        self.accel_bias = accel_mean - expected_accel

    def apply(
        self,
        raw_accel: np.ndarray,
        raw_gyro: np.ndarray,
    ) -> tuple:
        """Apply calibration to raw IMU readings."""
        calibrated_accel = self.accel_scale @ (raw_accel - self.accel_bias)
        calibrated_gyro = self.gyro_scale @ (raw_gyro - self.gyro_bias)
        return calibrated_accel, calibrated_gyro
```

### F/T Sensor Calibration

```python
class FTSensorCalibration:
    """
    Calibrate force/torque sensors using known loads.
    """

    def __init__(self):
        self.offset = np.zeros(6)
        self.calibration_matrix = np.eye(6)

    def calibrate_zero(self, samples: np.ndarray):
        """Zero calibration with no load."""
        self.offset = np.mean(samples, axis=0)

    def calibrate_with_known_loads(
        self,
        raw_readings: np.ndarray,  # (num_samples, 6)
        known_loads: np.ndarray,   # (num_samples, 6)
    ):
        """
        Full calibration with known reference loads.

        Uses least squares to find calibration matrix:
        calibrated = C @ (raw - offset)
        """
        raw_centered = raw_readings - self.offset

        # Solve: known = C @ raw_centered^T
        # C = known @ raw_centered^T @ (raw_centered @ raw_centered^T)^-1
        self.calibration_matrix, _, _, _ = np.linalg.lstsq(
            raw_centered, known_loads, rcond=None
        )

    def apply(self, raw_ft: np.ndarray) -> np.ndarray:
        """Apply calibration."""
        return (raw_ft - self.offset) @ self.calibration_matrix.T
```

---

## Real-Time Considerations

### Sensor Update Rates and Synchronization

```python
class SensorSynchronizer:
    """
    Synchronize multi-rate sensor data for real-time control.
    """

    def __init__(self):
        # Sensor update rates
        self.rates = {
            'joint_state': 1000,   # Hz
            'imu': 400,            # Hz
            'foot_contact': 1000,  # Hz
            'force_torque': 500,   # Hz
            'vision': 30,          # Hz
        }

        # Buffers for each sensor
        self.buffers = {name: deque(maxlen=100) for name in self.rates}
        self.timestamps = {name: deque(maxlen=100) for name in self.rates}

    def add_reading(self, sensor_name: str, data: np.ndarray, timestamp: float):
        """Add new sensor reading with timestamp."""
        self.buffers[sensor_name].append(data)
        self.timestamps[sensor_name].append(timestamp)

    def get_synchronized(self, target_time: float) -> dict:
        """
        Get synchronized sensor readings for target time.
        Uses nearest neighbor interpolation for high-rate sensors
        and extrapolation for low-rate sensors.
        """
        synchronized = {}

        for sensor_name in self.rates:
            timestamps = np.array(self.timestamps[sensor_name])

            if len(timestamps) == 0:
                continue

            # Find nearest timestamp
            idx = np.argmin(np.abs(timestamps - target_time))
            synchronized[sensor_name] = self.buffers[sensor_name][idx]

        return synchronized


class RealtimeSensorPipeline:
    """
    Real-time sensor processing pipeline with jitter handling.
    """

    def __init__(
        self,
        encoders: dict,
        control_rate: int = 1000,  # Hz
        max_latency_ms: float = 1.0,
    ):
        self.encoders = encoders
        self.control_period = 1.0 / control_rate
        self.max_latency = max_latency_ms / 1000.0

        self.synchronizer = SensorSynchronizer()

        # Pre-allocate tensors for speed
        self.joint_buffer = torch.zeros(1, 32 * 3)
        self.imu_buffer = torch.zeros(1, 50, 6)  # 50-step history
        self.contact_buffer = torch.zeros(1, 4)
        self.ft_buffer = torch.zeros(1, 4, 6)

    @torch.no_grad()
    def process(self, current_time: float) -> dict:
        """
        Process sensors and return features within latency budget.
        """
        start_time = time.time()

        # Get synchronized data
        sensor_data = self.synchronizer.get_synchronized(current_time)

        # Update buffers
        self._update_buffers(sensor_data)

        # Encode (with latency monitoring)
        features = {}

        # Joint encoding (highest priority)
        features['proprio'] = self.encoders['joint'](self.joint_buffer)

        # Check latency budget
        elapsed = time.time() - start_time
        if elapsed < self.max_latency * 0.5:
            features['imu'] = self.encoders['imu'](self.imu_buffer)

        elapsed = time.time() - start_time
        if elapsed < self.max_latency * 0.7:
            features['contact'] = self.encoders['contact'](self.contact_buffer)

        elapsed = time.time() - start_time
        if elapsed < self.max_latency * 0.9:
            features['ft'] = self.encoders['ft'](self.ft_buffer)

        return features
```

### Latency-Aware Inference

```python
class LatencyAwareSensorEncoder(nn.Module):
    """
    Adaptive sensor encoding that degrades gracefully under time pressure.
    """

    def __init__(self, full_encoder: nn.Module, fast_encoder: nn.Module):
        super().__init__()
        self.full_encoder = full_encoder
        self.fast_encoder = fast_encoder

    def forward(
        self,
        x: torch.Tensor,
        time_budget_ms: float = 1.0,
    ) -> torch.Tensor:
        """
        Use fast encoder if time budget is tight.
        """
        if time_budget_ms < 0.5:
            return self.fast_encoder(x)
        else:
            return self.full_encoder(x)
```

---

## Datasets for Sensor Training

| Dataset | Sensors | Size | Source |
|---------|---------|------|--------|
| CMU MoCap | Joint positions | 2,605 sequences | [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/) |
| AMASS | Full body motion | 40+ hours | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) |
| Human3.6M | Joints + IMU | 3.6M frames | [vision.imar.ro/human3.6m](http://vision.imar.ro/human3.6m/) |
| TotalCapture | IMU + optical | 5 subjects | [cvssp.org/data/totalcapture](https://cvssp.org/data/totalcapture/) |
| DIP-IMU | Sparse IMU | 10 subjects | [dip.is.tue.mpg.de](https://dip.is.tue.mpg.de/) |
| Humanoid Robot Datasets | Full sensor suite | Varies | Simulation (Isaac Gym, MuJoCo) |

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Complete VLA training pipeline
- [Training Locomotion](training_locomotion.md) - Bipedal walking with sensor feedback
- [Training Manipulation](training_manipulation.md) - Force-controlled manipulation
- [Training Whole Body](training_whole_body.md) - Coordinated whole-body control
- [Deployment](deployment.md) - Real-time sensor deployment
