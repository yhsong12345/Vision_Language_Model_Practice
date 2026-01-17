# Autonomous Vehicle Embodiment

This document provides comprehensive documentation for the Autonomous Vehicle embodiment in the VLA framework, covering the `DrivingVLA` model, BEV encoding, trajectory prediction, and the complete training pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Configuration](#configuration)
5. [Model Implementation](#model-implementation)
6. [Training Pipeline](#training-pipeline)
7. [Datasets](#datasets)
8. [Safety Constraints](#safety-constraints)
9. [Deployment](#deployment)
10. [API Reference](#api-reference)

---

## Overview

### Autonomous Vehicle VLA Pipeline

```
+==========================================================================================+
|                        AUTONOMOUS VEHICLE VLA EMBODIMENT                                  |
+==========================================================================================+
|                                                                                           |
|  INPUT                                                                                    |
|  +--------------------------------+  +--------------------------------+                   |
|  |  Multi-Camera Images           |  |  Language Instruction          |                   |
|  |  (6 cameras, surround-view)    |  |  "Turn left at intersection"   |                   |
|  +--------------------------------+  +--------------------------------+                   |
|              |                                    |                                       |
|              v                                    v                                       |
|  +--------------------------------+  +--------------------------------+                   |
|  |  BEV Encoder                   |  |  Language Projector            |                   |
|  |  - Image Encoder (per camera)  |  |  - LLM → hidden_dim            |                   |
|  |  - Depth Estimation            |  +--------------------------------+                   |
|  |  - Camera-to-BEV Projection    |              |                                       |
|  +--------------------------------+              |                                       |
|              |                                   |                                       |
|              v                                   v                                       |
|  +------------------------------------------------+                                      |
|  |  Cross-Modal Fusion (MultiheadAttention)       |                                      |
|  |  BEV features ← Language conditioning          |                                      |
|  +------------------------------------------------+                                      |
|              |                                                                            |
|              v                                                                            |
|  +--------------------------------+                                                       |
|  |  Motion Planner                |                                                       |
|  |  - Cost Map Prediction         |                                                       |
|  |  - Trajectory Decoder (GRU)    |                                                       |
|  |  - Trajectory Scoring          |                                                       |
|  +--------------------------------+                                                       |
|              |                                                                            |
|              v                                                                            |
|  OUTPUT                                                                                   |
|  +--------------------------------+  +--------------------------------+                   |
|  |  Trajectory                    |  |  Vehicle Controls              |                   |
|  |  (20 waypoints in ego frame)   |  |  (throttle, brake, steer)      |                   |
|  |  + speeds, headings            |  |  [0, 1] normalized             |                   |
|  +--------------------------------+  +--------------------------------+                   |
|                                                                                           |
+==========================================================================================+
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Camera Fusion** | 6 surround-view cameras fused into unified BEV representation |
| **BEV Representation** | 200×200 grid with 0.5m resolution (100m × 100m coverage) |
| **Language Conditioning** | Natural language instructions for navigation |
| **Trajectory Prediction** | 20 future waypoints at 10Hz (2 seconds ahead) |
| **Safety Integration** | Built-in safety shield with speed/acceleration limits |

---

## Architecture

### DrivingVLA Model Structure

```
DrivingVLA
├── BEVEncoder
│   ├── image_encoder (shared CNN for all cameras)
│   │   ├── Conv2d(3, 64, 7, stride=2)
│   │   ├── Conv2d(64, 128, 3, stride=2)
│   │   ├── Conv2d(128, 256, 3, stride=2)
│   │   └── Conv2d(256, hidden_dim, 3, stride=2)
│   ├── depth_head (monocular depth estimation)
│   │   └── Conv2d → 64 depth bins (softmax)
│   ├── cam_to_bev (learned projection parameters)
│   │   └── Parameter(num_cameras, hidden_dim, bev_size, bev_size)
│   └── bev_conv (BEV feature refinement)
│       └── Conv2d × 2
│
├── language_projector
│   └── Linear(llm_hidden_dim, hidden_dim)
│
├── fusion (Cross-modal attention)
│   └── MultiheadAttention(hidden_dim, num_heads)
│
├── planner (MotionPlanner)
│   ├── cost_encoder (cost volume prediction)
│   │   └── Conv2d → cost_map
│   ├── trajectory_decoder (TrajectoryDecoder)
│   │   ├── bev_pool (BEV → feature vector)
│   │   ├── trajectory_gru (autoregressive prediction)
│   │   ├── waypoint_head (x, y prediction)
│   │   ├── speed_head
│   │   └── heading_head
│   └── trajectory_scorer (trajectory ranking)
│
└── control_head (direct vehicle control)
    └── Linear → (throttle, brake, steer)
```

### Data Flow

```python
# Forward pass
images: (B, 6, 3, 224, 224)           # 6 surround-view cameras
language_features: (B, seq_len, 4096)  # LLM hidden states

# BEV Encoding
bev_features: (B, 512, 200, 200)       # Bird's Eye View

# Cross-Modal Fusion
fused_features: (B, 512, 200, 200)     # Language-conditioned BEV

# Motion Planning
trajectory: (B, 20, 2)                  # 20 waypoints (x, y)
speeds: (B, 20)                         # Speed at each waypoint
headings: (B, 20)                       # Heading at each waypoint
cost_map: (B, 1, 200, 200)              # Occupancy/cost

# Control Output
controls: (B, 3)                        # (throttle, brake, steer)
```

---

## Core Components

### 1. BEVEncoder

The BEV (Bird's Eye View) encoder transforms multi-camera images into a unified top-down representation.

```python
from model.embodiment.autonomous_vehicle import BEVEncoder, DrivingVLAConfig

config = DrivingVLAConfig(
    num_cameras=6,
    image_size=224,
    bev_size=200,
    bev_resolution=0.5,  # meters per pixel
    hidden_dim=512,
)

bev_encoder = BEVEncoder(config)

# Input: 6 cameras
images = torch.randn(2, 6, 3, 224, 224)

# Optional: camera parameters
intrinsics = torch.randn(2, 6, 3, 3)   # Camera intrinsics
extrinsics = torch.randn(2, 6, 4, 4)   # Camera extrinsics

# Encode to BEV
output = bev_encoder(images, intrinsics, extrinsics)

print(f"BEV features: {output['bev_features'].shape}")      # (2, 512, 200, 200)
print(f"Camera features: {output['camera_features'].shape}") # (2, 6, 512, 14, 14)
print(f"Depth: {output['depth'].shape}")                     # (2, 6, 64, 14, 14)
```

**Key Implementation Details:**

| Component | Description |
|-----------|-------------|
| `image_encoder` | Shared CNN backbone, 16× downsampling |
| `depth_head` | 64-bin softmax depth distribution |
| `cam_to_bev` | Learned projection (LSS-style) |
| `bev_conv` | 2-layer refinement network |

### 2. TrajectoryDecoder

Autoregressively decodes future waypoints from BEV features.

```python
from model.embodiment.autonomous_vehicle import TrajectoryDecoder

traj_decoder = TrajectoryDecoder(config)

# From BEV features
bev_features = torch.randn(2, 512, 200, 200)
language_features = torch.randn(2, 512)  # Pooled language

output = traj_decoder(bev_features, language_features)

print(f"Trajectory: {output['trajectory'].shape}")  # (2, 20, 2)
print(f"Speeds: {output['speeds'].shape}")          # (2, 20)
print(f"Headings: {output['headings'].shape}")      # (2, 20)
```

**Autoregressive Generation:**

```
Step 0: features + (0, 0) → GRU → waypoint_1
Step 1: features + waypoint_1 → GRU → waypoint_2
...
Step 19: features + waypoint_19 → GRU → waypoint_20
```

### 3. MotionPlanner

Combines cost map prediction with trajectory generation.

```python
from model.embodiment.autonomous_vehicle import MotionPlanner

planner = MotionPlanner(config)

output = planner(
    bev_features=bev_features,
    language_features=language_features,
    goal=torch.tensor([[50.0, 0.0], [30.0, 10.0]]),  # Optional goal
)

print(f"Cost map: {output['cost_map'].shape}")           # (2, 1, 200, 200)
print(f"Trajectory cost: {output['trajectory_cost'].shape}")  # (2,)
```

---

## Configuration

### DrivingVLAConfig

```python
from dataclasses import dataclass

@dataclass
class DrivingVLAConfig:
    """Configuration for driving VLA."""

    # Vision
    num_cameras: int = 6          # Typical surround-view setup
    image_size: int = 224         # Input image resolution
    bev_size: int = 200           # BEV grid size (200×200)
    bev_resolution: float = 0.5   # Meters per pixel

    # LLM
    llm_hidden_dim: int = 4096    # LLM output dimension
    use_language_conditioning: bool = True

    # Action
    trajectory_length: int = 20   # Future waypoints
    dt: float = 0.1               # Time between waypoints (10Hz)

    # Architecture
    hidden_dim: int = 512         # Feature dimension
    num_heads: int = 8            # Attention heads
    num_layers: int = 4           # Transformer layers
```

### DrivingTrainConfig

```python
from train.embodiment.train_driving_vla import DrivingTrainConfig

config = DrivingTrainConfig(
    # Model
    num_cameras=6,
    image_size=224,
    bev_size=200,
    bev_resolution=0.5,
    hidden_dim=512,
    llm_hidden_dim=4096,
    trajectory_length=20,

    # Training
    batch_size=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    num_epochs=100,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    warmup_steps=1000,

    # Loss weights
    trajectory_loss_weight=1.0,
    control_loss_weight=0.5,
    cost_map_loss_weight=0.1,
    safety_loss_weight=0.2,

    # Data
    data_path="./data/driving",
    num_workers=8,

    # Checkpointing
    output_dir="./checkpoints/driving_vla",
    save_steps=1000,
    eval_steps=500,

    # Safety
    use_safety_shield=True,
    max_speed=30.0,           # m/s (~108 km/h)
    max_acceleration=5.0,     # m/s²
    min_distance=2.0,         # meters
)
```

---

## Model Implementation

### Complete DrivingVLA

```python
from model.embodiment.autonomous_vehicle import DrivingVLA, DrivingVLAConfig
import torch

# Create model
config = DrivingVLAConfig(
    num_cameras=6,
    image_size=224,
    bev_size=200,
    hidden_dim=256,  # Smaller for demo
)

model = DrivingVLA(config)

# Prepare inputs
batch_size = 2
images = torch.randn(batch_size, 6, 3, 224, 224)
language_features = torch.randn(batch_size, 32, 4096)  # From LLM

# Forward pass
output = model(images, language_features)

# Outputs
print(f"Trajectory shape: {output['trajectory'].shape}")      # (2, 20, 2)
print(f"Controls shape: {output['controls'].shape}")          # (2, 3)
print(f"BEV features shape: {output['bev_features'].shape}")  # (2, 256, 200, 200)
print(f"Cost map shape: {output['cost_map'].shape}")          # (2, 1, 200, 200)
print(f"Speeds shape: {output['speeds'].shape}")              # (2, 20)
```

### Model Parameters

| Component | Parameters (hidden_dim=512) |
|-----------|----------------------------|
| BEVEncoder | ~15M |
| Language Projector | ~2M |
| Cross-Modal Fusion | ~1.5M |
| Motion Planner | ~8M |
| Control Head | ~0.3M |
| **Total** | **~27M** |

---

## Training Pipeline

### DrivingVLATrainer

```python
from train.embodiment.train_driving_vla import (
    DrivingVLATrainer,
    DrivingTrainConfig,
)

# Configuration
config = DrivingTrainConfig(
    num_cameras=6,
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=100,
    data_path="./data/driving",
    output_dir="./checkpoints/driving_vla",
    use_safety_shield=True,
)

# Create trainer
trainer = DrivingVLATrainer(config)

# Train
trainer.train()

# Resume from checkpoint
trainer.load_checkpoint("./checkpoints/driving_vla/checkpoint_epoch_50.pt")
trainer.train()
```

### Loss Functions

The training uses a multi-task loss:

```python
total_loss = (
    trajectory_loss_weight * trajectory_loss +    # L2 loss on waypoints
    control_loss_weight * control_loss +          # L2 loss on controls
    cost_map_loss_weight * cost_smoothness +      # Smoothness regularization
    safety_loss_weight * safety_loss              # Speed constraint violation
)
```

| Loss Component | Weight | Description |
|----------------|--------|-------------|
| `trajectory_loss` | 1.0 | MSE between predicted and ground truth waypoints |
| `control_loss` | 0.5 | MSE between predicted and ground truth controls |
| `cost_smoothness` | 0.1 | Regularizes cost map to be smooth |
| `safety_loss` | 0.2 | Penalizes speed violations above max_speed |

### Training Command

```bash
python train/embodiment/train_driving_vla.py \
    --num-cameras 6 \
    --image-size 224 \
    --bev-size 200 \
    --hidden-dim 512 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --num-epochs 100 \
    --gradient-accumulation-steps 4 \
    --data-path ./data/driving \
    --output-dir ./checkpoints/driving_vla \
    --use-safety-shield \
    --max-speed 30.0
```

---

## Datasets

### DrivingDataset

```python
from train.embodiment.train_driving_vla import DrivingDataset, DrivingTrainConfig

config = DrivingTrainConfig(data_path="./data/driving")
dataset = DrivingDataset(config.data_path, config, split="train")

# Sample format
sample = dataset[0]
print(f"Images: {sample['images'].shape}")              # (6, 3, 224, 224)
print(f"Trajectory: {sample['trajectory'].shape}")      # (20, 2)
print(f"Controls: {sample['controls'].shape}")          # (3,)
print(f"Language features: {sample['language_features'].shape}")  # (32, 4096)
```

### Supported Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| **CARLA Autopilot** | Synthetic driving data | Local/CARLA simulator |
| **nuScenes** | 1000 driving scenes, 1.4M images | [nuscenes.org](https://www.nuscenes.org/) |
| **Waymo Open** | Large-scale driving | [waymo.com](https://waymo.com/open/) |
| **BDD100K** | 100K diverse driving videos | [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/) |

### Data Format

```json
{
  "id": 0,
  "camera_paths": [
    "cam_0/frame_000000.jpg",
    "cam_1/frame_000000.jpg",
    "cam_2/frame_000000.jpg",
    "cam_3/frame_000000.jpg",
    "cam_4/frame_000000.jpg",
    "cam_5/frame_000000.jpg"
  ],
  "trajectory": "trajectories/traj_000000.npy",
  "controls": "controls/ctrl_000000.npy",
  "instruction": "Drive forward safely"
}
```

---

## Safety Constraints

### Safety Shield Integration

```python
from model.safety.safety_shield import SafetyShield, SafetyConfig
from model.safety.rule_checker import TrafficRuleChecker

# Safety configuration
safety_config = SafetyConfig(
    max_velocity=30.0,       # m/s
    max_acceleration=5.0,    # m/s²
    min_distance=2.0,        # meters to obstacles
)

safety_shield = SafetyShield(safety_config)
rule_checker = TrafficRuleChecker()

# Apply safety during inference
def safe_predict(model, images, language_features):
    output = model(images, language_features)

    # Check trajectory safety
    trajectory = output['trajectory']
    speeds = output['speeds']

    # Enforce speed limits
    speeds = torch.clamp(speeds, max=safety_config.max_velocity)

    # Check traffic rules
    violations = rule_checker.check(trajectory)

    return {
        **output,
        'speeds': speeds,
        'rule_violations': violations,
    }
```

### Safety Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_speed` | 30.0 m/s | Maximum allowed speed |
| `max_acceleration` | 5.0 m/s² | Maximum acceleration/deceleration |
| `min_distance` | 2.0 m | Minimum distance to obstacles |

---

## Deployment

### ONNX Export

```python
from model.utils.export import ONNXExporter
from model.embodiment.autonomous_vehicle import DrivingVLA, DrivingVLAConfig

# Create model
config = DrivingVLAConfig()
model = DrivingVLA(config)
model.load_state_dict(torch.load("./checkpoints/driving_vla/best_model.pt")["model_state_dict"])
model.eval()

# Export
exporter = ONNXExporter()
exporter.export(
    model=model,
    output_path="./deployed/driving_vla.onnx",
    input_shapes={
        "images": (1, 6, 3, 224, 224),
        "language_features": (1, 32, 4096),
    },
    opset_version=17,
)
```

### TorchScript Export

```python
from model.utils.export import TorchScriptExporter

exporter = TorchScriptExporter()
exporter.export_traced(
    model=model,
    example_inputs=(
        torch.randn(1, 6, 3, 224, 224),
        torch.randn(1, 32, 4096),
    ),
    output_path="./deployed/driving_vla.pt",
)
```

### ROS Integration

```python
from integration.ros_bridge import ROSBridge

# Initialize ROS bridge
bridge = ROSBridge(
    model_path="./deployed/driving_vla.pt",
    camera_topics=[
        "/camera/front/image_raw",
        "/camera/front_left/image_raw",
        "/camera/front_right/image_raw",
        "/camera/rear/image_raw",
        "/camera/rear_left/image_raw",
        "/camera/rear_right/image_raw",
    ],
    control_topic="/vehicle/cmd_vel",
)

bridge.run()
```

---

## API Reference

### DrivingVLA

```python
class DrivingVLA(nn.Module):
    """Complete VLA model for autonomous driving."""

    def __init__(self, config: DrivingVLAConfig):
        """
        Initialize DrivingVLA.

        Args:
            config: DrivingVLAConfig with model parameters
        """

    def forward(
        self,
        images: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for driving VLA.

        Args:
            images: (B, num_cameras, 3, H, W) surround-view images
            language_features: (B, seq_len, llm_dim) LLM features
            instruction: Optional text instruction

        Returns:
            Dict containing:
                - trajectory: (B, T, 2) predicted waypoints
                - controls: (B, 3) throttle, brake, steer
                - bev_features: (B, C, H, W) BEV representation
                - cost_map: (B, 1, H, W) predicted cost
                - speeds: (B, T) speed at each waypoint
        """
```

### BEVEncoder

```python
class BEVEncoder(nn.Module):
    """Bird's Eye View encoder from multi-camera images."""

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multi-camera images to BEV.

        Args:
            images: (B, num_cameras, 3, H, W)
            intrinsics: (B, num_cameras, 3, 3) camera intrinsics
            extrinsics: (B, num_cameras, 4, 4) camera extrinsics

        Returns:
            Dict containing:
                - bev_features: (B, C, bev_h, bev_w)
                - camera_features: (B, num_cameras, C, h, w)
                - depth: (B, num_cameras, D, h, w) depth distribution
        """
```

### TrajectoryDecoder

```python
class TrajectoryDecoder(nn.Module):
    """Autoregressive trajectory decoder."""

    def forward(
        self,
        bev_features: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode trajectory from BEV features.

        Args:
            bev_features: (B, C, H, W) BEV features
            language_features: (B, C) pooled language features

        Returns:
            Dict containing:
                - trajectory: (B, T, 2) waypoints
                - speeds: (B, T) speed predictions
                - headings: (B, T) heading predictions
        """
```

---

## Related Documents

- [Training VLA Recipe](training_vla_recipe.md) - Complete training pipeline
- [Training Datasets](training_datasets.md) - Dataset documentation
- [Training Sensors](training_sensors.md) - Multi-sensor fusion
- [Deployment Guide](deployment.md) - Export and deployment
- [Architecture Guide](../architecture.md) - Overall VLA architecture
- [Training Embodiment](../training_embodiment.md) - Cross-embodiment training
