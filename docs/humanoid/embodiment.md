# Humanoid Robot Embodiment

This document provides comprehensive documentation for the Humanoid Robot embodiment in the VLA framework, covering the `HumanoidVLA` model, proprioception encoding, locomotion, manipulation, whole-body control, and the complete training pipeline.

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

### Humanoid VLA Pipeline

```
+==========================================================================================+
|                           HUMANOID ROBOT VLA EMBODIMENT                                   |
+==========================================================================================+
|                                                                                           |
|  INPUT                                                                                    |
|  +--------------------------------+  +--------------------------------+                   |
|  |  Visual Input                  |  |  Proprioception               |                   |
|  |  (RGB Image: 224×224)          |  |  (32 joints × 3 + IMU + foot) |                   |
|  +--------------------------------+  +--------------------------------+                   |
|              |                                    |                                       |
|              v                                    v                                       |
|  +--------------------------------+  +--------------------------------+                   |
|  |  Vision Encoder                |  |  Proprioception Encoder        |                   |
|  |  - CNN backbone                |  |  - Joint Encoder               |                   |
|  |  - Feature extraction          |  |  - IMU Encoder                 |                   |
|  |                                |  |  - Contact Encoder             |                   |
|  +--------------------------------+  +--------------------------------+                   |
|              |                                    |                                       |
|              v                                    v                                       |
|  +------------------------------------------------+                                      |
|  |  Language Conditioning                         |                                      |
|  |  - Language Projector (LLM → hidden_dim)       |                                      |
|  |  - Task Decoder (velocity_cmd + target_poses)  |                                      |
|  +------------------------------------------------+                                      |
|              |                                                                            |
|              v                                                                            |
|  +------------------------------------------------+                                      |
|  |  Whole-Body Controller                         |                                      |
|  |  ├── Locomotion Policy (walking/running)       |                                      |
|  |  ├── Manipulation Policy (reaching/grasping)   |                                      |
|  |  ├── Task Coordinator (priority weights)       |                                      |
|  |  └── Balance Controller (stability)            |                                      |
|  +------------------------------------------------+                                      |
|              |                                                                            |
|              v                                                                            |
|  OUTPUT                                                                                   |
|  +--------------------------------+  +--------------------------------+                   |
|  |  Joint Actions                 |  |  Gripper Commands              |                   |
|  |  (32 DoF joint positions)      |  |  (left, right: [0, 1])         |                   |
|  +--------------------------------+  +--------------------------------+                   |
|                                                                                           |
+==========================================================================================+
```

### Key Features

| Feature | Description |
|---------|-------------|
| **32 DoF Control** | Full humanoid body: head (3), torso (3), arms (14), legs (12) |
| **Proprioception Fusion** | Joint pos/vel/torque + IMU (9D) + Foot contacts (4D) |
| **Hierarchical Control** | Separate locomotion, manipulation, and balance policies |
| **Task Prioritization** | Learned priority weights (locomotion, manipulation, balance) |
| **Language Conditioning** | Natural language instructions for task specification |

---

## Architecture

### HumanoidVLA Model Structure

```
HumanoidVLA
├── vision_encoder (CNN)
│   ├── Conv2d(3, 64, 7, stride=2) + BN + ReLU
│   ├── Conv2d(64, 128, 3, stride=2) + BN + ReLU
│   ├── Conv2d(128, 256, 3, stride=2) + BN + ReLU
│   ├── AdaptiveAvgPool2d(7)
│   └── Linear(256×49, hidden_dim)
│
├── language_projector
│   └── Linear(llm_hidden_dim, hidden_dim)
│
├── task_decoder (from language to task parameters)
│   ├── Linear(hidden_dim, hidden_dim) + ReLU
│   └── Linear(hidden_dim, 3 + 12)  # velocity_cmd + target_poses
│
└── controller (WholeBodyController)
    ├── locomotion (LocomotionPolicy)
    │   ├── proprio_encoder (ProprioceptionEncoder)
    │   │   ├── joint_encoder (num_joints × 3 → hidden_dim)
    │   │   ├── imu_encoder (9 → hidden_dim/2)
    │   │   └── contact_encoder (4 → hidden_dim/4)
    │   ├── command_encoder (3 → hidden_dim/2)
    │   ├── policy_net (2-layer MLP)
    │   ├── action_mean (Linear → action_dim)
    │   ├── action_log_std (Parameter)
    │   ├── value_head (Linear → 1)
    │   └── phase_encoder (2 → hidden_dim/4)
    │
    ├── manipulation (ManipulationPolicy)
    │   ├── visual_encoder (CNN)
    │   ├── arm_proprio_encoder (14×2 → hidden_dim/2)
    │   ├── goal_encoder (6×2 → hidden_dim/2)
    │   ├── policy (2-layer MLP)
    │   ├── arm_action_head (Linear → 14)
    │   └── gripper_head (Linear → 2)
    │
    ├── coordinator (task coordination)
    │   └── Linear(hidden_dim×2, hidden_dim)
    │
    ├── balance_net (stability correction)
    │   └── Linear(hidden_dim + 9, action_dim)
    │
    └── priority_net (task weights)
        └── Linear(hidden_dim, 3) + Softmax
```

### Data Flow

```python
# Forward pass
image: (B, 3, 224, 224)               # Visual input
joint_positions: (B, 32)               # Current joint positions (rad)
joint_velocities: (B, 32)              # Joint velocities (rad/s)
joint_torques: (B, 32)                 # Joint torques (Nm)
language_features: (B, seq_len, 4096)  # LLM hidden states
imu_data: (B, 9)                       # Orientation + angular vel + accel
foot_contacts: (B, 4)                  # Binary foot contact flags

# Proprioception Encoding
proprio_features: (B, hidden_dim)      # Encoded robot state

# Language Processing
velocity_command: (B, 3)               # Target [vx, vy, vyaw]
target_poses: (B, 12)                  # End-effector targets

# Whole-Body Control
action: (B, 32)                        # Joint position commands
priorities: (B, 3)                     # [locomotion, manipulation, balance]
gripper_actions: (B, 2)                # Left and right gripper
```

---

## Core Components

### 1. ProprioceptionEncoder

Encodes the robot's internal state (joint positions, velocities, torques, IMU, foot contacts).

```python
from model.embodiment.humanoid import ProprioceptionEncoder, HumanoidConfig

config = HumanoidConfig(
    num_joints=32,
    hidden_dim=512,
)

proprio_encoder = ProprioceptionEncoder(config)

# Inputs
batch_size = 2
joint_pos = torch.randn(batch_size, 32)     # Joint positions
joint_vel = torch.randn(batch_size, 32)     # Joint velocities
joint_torque = torch.randn(batch_size, 32)  # Joint torques
imu_data = torch.randn(batch_size, 9)       # Orientation + angular vel + accel
foot_contacts = torch.randn(batch_size, 4)  # 2 feet × 2 contact points

# Encode
features = proprio_encoder(
    joint_pos, joint_vel, joint_torque,
    imu_data=imu_data,
    foot_contacts=foot_contacts,
)

print(f"Proprioception features: {features.shape}")  # (2, 512)
```

**Encoding Architecture:**

| Component | Input Dim | Output Dim | Description |
|-----------|-----------|------------|-------------|
| `joint_encoder` | num_joints × 3 | hidden_dim | Joint state (pos, vel, torque) |
| `imu_encoder` | 9 | hidden_dim/2 | Orientation + angular vel + accel |
| `contact_encoder` | 4 | hidden_dim/4 | Binary foot contact flags |
| `fusion` | hidden_dim × 1.75 | hidden_dim | Concatenate + project |

### 2. LocomotionPolicy

Generates joint commands for walking/running locomotion.

```python
from model.embodiment.humanoid import LocomotionPolicy

loco_policy = LocomotionPolicy(config)

# Velocity command: [vx, vy, vyaw]
velocity_command = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.0, 0.1]])  # Forward, turn

# Gait phase (sin, cos) for periodic locomotion
phase = torch.tensor([[0.0, 1.0], [0.707, 0.707]])

output = loco_policy(
    joint_pos, joint_vel, joint_torque,
    velocity_command,
    imu_data=imu_data,
    foot_contacts=foot_contacts,
    phase=phase,
)

print(f"Action mean: {output['action_mean'].shape}")  # (2, 32)
print(f"Action std: {output['action_std'].shape}")    # (32,)
print(f"Value: {output['value'].shape}")              # (2,)
```

**Policy Output:**

| Output | Shape | Description |
|--------|-------|-------------|
| `action_mean` | (B, 32) | Mean joint positions |
| `action_std` | (32,) | Learned standard deviation |
| `value` | (B,) | Value estimate (for PPO) |
| `features` | (B, hidden_dim) | Policy features |

### 3. ManipulationPolicy

Generates arm joint commands for manipulation tasks.

```python
from model.embodiment.humanoid import ManipulationPolicy

manip_policy = ManipulationPolicy(config)

# Visual input
image = torch.randn(2, 3, 224, 224)

# Arm state (14 joints: 7 per arm)
arm_joint_pos = torch.randn(2, 14)
arm_joint_vel = torch.randn(2, 14)

# Target poses: 6D pose for each hand
target_poses = torch.randn(2, 12)  # 6 per hand

output = manip_policy(
    image,
    arm_joint_pos,
    arm_joint_vel,
    target_poses=target_poses,
)

print(f"Arm actions: {output['arm_actions'].shape}")       # (2, 14)
print(f"Gripper actions: {output['gripper_actions'].shape}")  # (2, 2)
```

### 4. WholeBodyController

Coordinates locomotion, manipulation, and balance.

```python
from model.embodiment.humanoid import WholeBodyController

controller = WholeBodyController(config)

output = controller(
    joint_positions=joint_pos,
    joint_velocities=joint_vel,
    joint_torques=joint_torque,
    velocity_command=velocity_command,
    image=image,
    target_poses=target_poses,
    imu_data=imu_data,
    foot_contacts=foot_contacts,
)

print(f"Final action: {output['action'].shape}")           # (2, 32)
print(f"Priorities: {output['priorities'].shape}")          # (2, 3)
print(f"Locomotion action: {output['locomotion_action'].shape}")  # (2, 32)
print(f"Balance correction: {output['balance_correction'].shape}")  # (2, 32)
```

**Task Priority Weighting:**

```python
final_action = (
    priorities[:, 0:1] * locomotion_action +    # Walking/running
    priorities[:, 1:2] * manipulation_action +  # Arm control
    priorities[:, 2:3] * balance_correction     # Stability
)
```

---

## Configuration

### HumanoidConfig

```python
from dataclasses import dataclass

@dataclass
class HumanoidConfig:
    """Configuration for humanoid VLA."""

    # Robot structure
    num_joints: int = 32           # Total DoF
    num_body_parts: int = 15       # Head, torso, arms, legs, hands, feet

    # Observation
    proprioception_dim: int = 128  # Proprioception encoding dim
    image_size: int = 224          # Visual input resolution

    # Action
    action_dim: int = 32           # Joint position commands
    control_freq: float = 100.0    # Control frequency (Hz)
    action_type: str = "position"  # "position" or "torque"

    # Architecture
    hidden_dim: int = 512          # Feature dimension
    num_heads: int = 8             # Attention heads
    num_layers: int = 4            # Transformer layers
    llm_hidden_dim: int = 4096     # LLM output dimension
```

### HumanoidTrainConfig

```python
from train.embodiment.train_humanoid_vla import HumanoidTrainConfig

config = HumanoidTrainConfig(
    # Model
    num_joints=32,
    joint_dim=12,
    image_size=224,
    hidden_dim=512,
    llm_hidden_dim=4096,

    # Training
    batch_size=16,
    learning_rate=3e-4,
    weight_decay=0.01,
    num_epochs=200,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    warmup_steps=2000,

    # Loss weights
    action_loss_weight=1.0,
    locomotion_loss_weight=0.5,
    manipulation_loss_weight=0.5,
    stability_loss_weight=0.3,
    smoothness_loss_weight=0.1,

    # Data
    data_path="./data/humanoid",
    num_workers=8,
    sequence_length=16,

    # Checkpointing
    output_dir="./checkpoints/humanoid_vla",
    save_steps=1000,
    eval_steps=500,

    # Safety
    use_safety_constraints=True,
    max_joint_velocity=5.0,    # rad/s
    max_joint_torque=100.0,    # Nm
    min_com_height=0.3,        # meters
)
```

### Joint Configuration (32 DoF)

| Body Part | Joints | DoF | Index Range |
|-----------|--------|-----|-------------|
| Head | pan, tilt, roll | 3 | 0-2 |
| Torso | yaw, pitch, roll | 3 | 3-5 |
| Left Arm | shoulder (3), elbow (1), wrist (3) | 7 | 6-12 |
| Right Arm | shoulder (3), elbow (1), wrist (3) | 7 | 13-19 |
| Left Leg | hip (3), knee (1), ankle (2) | 6 | 20-25 |
| Right Leg | hip (3), knee (1), ankle (2) | 6 | 26-31 |
| **Total** | | **32** | 0-31 |

---

## Model Implementation

### Complete HumanoidVLA

```python
from model.embodiment.humanoid import HumanoidVLA, HumanoidConfig
import torch

# Create model
config = HumanoidConfig(
    num_joints=32,
    action_dim=32,
    hidden_dim=256,  # Smaller for demo
)

model = HumanoidVLA(config)

# Prepare inputs
batch_size = 2
image = torch.randn(batch_size, 3, 224, 224)
joint_pos = torch.randn(batch_size, 32)
joint_vel = torch.randn(batch_size, 32)
joint_torque = torch.randn(batch_size, 32)
language_features = torch.randn(batch_size, 32, 4096)
imu_data = torch.randn(batch_size, 9)
foot_contacts = torch.randn(batch_size, 4)

# Forward pass
output = model(
    image=image,
    joint_positions=joint_pos,
    joint_velocities=joint_vel,
    joint_torques=joint_torque,
    language_features=language_features,
    imu_data=imu_data,
    foot_contacts=foot_contacts,
)

# Outputs
print(f"Action shape: {output['action'].shape}")              # (2, 32)
print(f"Velocity command: {output['velocity_command'].shape}")  # (2, 3)
print(f"Target poses: {output['target_poses'].shape}")          # (2, 12)
print(f"Priorities: {output['priorities'].shape}")              # (2, 3)
print(f"Gripper actions: {output['gripper_actions'].shape}")    # (2, 2)
```

### Model Parameters

| Component | Parameters (hidden_dim=512) |
|-----------|----------------------------|
| Vision Encoder | ~8M |
| Language Projector | ~2M |
| Task Decoder | ~0.5M |
| Locomotion Policy | ~5M |
| Manipulation Policy | ~4M |
| Coordinator + Balance | ~2M |
| **Total** | **~22M** |

---

## Training Pipeline

### HumanoidVLATrainer

```python
from train.embodiment.train_humanoid_vla import (
    HumanoidVLATrainer,
    HumanoidTrainConfig,
)

# Configuration
config = HumanoidTrainConfig(
    num_joints=32,
    batch_size=16,
    learning_rate=3e-4,
    num_epochs=200,
    data_path="./data/humanoid",
    output_dir="./checkpoints/humanoid_vla",
    use_safety_constraints=True,
)

# Create trainer
trainer = HumanoidVLATrainer(config)

# Train
trainer.train()

# Resume from checkpoint
trainer.load_checkpoint("./checkpoints/humanoid_vla/checkpoint_epoch_100.pt")
trainer.train()
```

### Loss Functions

The training uses a multi-task loss:

```python
total_loss = (
    action_loss_weight * action_loss +           # MSE on joint actions
    locomotion_loss_weight * locomotion_loss +   # Lower body joints
    manipulation_loss_weight * manipulation_loss + # Upper body joints
    stability_loss_weight * stability_loss +     # Velocity/COM constraints
    smoothness_loss_weight * smoothness_loss     # Action smoothness
)
```

| Loss Component | Weight | Description |
|----------------|--------|-------------|
| `action_loss` | 1.0 | MSE on all 32 joint positions |
| `locomotion_loss` | 0.5 | MSE on lower body (legs) joints |
| `manipulation_loss` | 0.5 | MSE on upper body (arms) joints |
| `stability_loss` | 0.3 | Velocity limits + COM height constraint |
| `smoothness_loss` | 0.1 | L2 on action differences (temporal smoothness) |

### Stability Loss Details

```python
def _compute_stability_loss(outputs, proprioception):
    stability_loss = 0.0

    # Penalize high joint velocities
    action_pred = outputs["actions"]
    velocity_penalty = F.relu(torch.abs(action_pred) - max_joint_velocity)
    stability_loss += velocity_penalty.mean()

    # COM height constraint (prevent falling)
    if "com_height" in outputs:
        com_violation = F.relu(min_com_height - outputs["com_height"])
        stability_loss += com_violation.mean()

    return stability_loss
```

### Training Command

```bash
python train/embodiment/train_humanoid_vla.py \
    --num-joints 32 \
    --image-size 224 \
    --hidden-dim 512 \
    --batch-size 16 \
    --learning-rate 3e-4 \
    --num-epochs 200 \
    --gradient-accumulation-steps 2 \
    --data-path ./data/humanoid \
    --output-dir ./checkpoints/humanoid_vla \
    --use-safety-constraints \
    --max-joint-velocity 5.0 \
    --max-joint-torque 100.0
```

---

## Datasets

### HumanoidDataset

```python
from train.embodiment.train_humanoid_vla import HumanoidDataset, HumanoidTrainConfig

config = HumanoidTrainConfig(data_path="./data/humanoid")
dataset = HumanoidDataset(config.data_path, config, split="train")

# Sample format
sample = dataset[0]
print(f"Image: {sample['image'].shape}")              # (3, 224, 224)
print(f"Proprioception: {sample['proprioception'].shape}")  # (num_joints×3 + 6,)
print(f"Actions: {sample['actions'].shape}")           # (16, 32)
print(f"Language features: {sample['language_features'].shape}")  # (32, 4096)
print(f"Task type: {sample['task_type']}")             # 0, 1, or 2
```

### Supported Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| **CMU MoCap** | Human motion capture database | [CMU](http://mocap.cs.cmu.edu/) |
| **AMASS** | Large-scale motion capture | [MPI](https://amass.is.tue.mpg.de/) |
| **D4RL Humanoid** | Offline RL humanoid tasks | [HuggingFace](https://huggingface.co/datasets/imone/D4RL) |
| **HumanoidBench** | Humanoid control benchmark | [GitHub](https://humanoid-bench.github.io/) |
| **DROID** | Diverse robot manipulation | [HuggingFace](https://huggingface.co/datasets/cadene/droid) |
| **LeRobot ALOHA** | Bimanual manipulation | [HuggingFace](https://huggingface.co/lerobot) |

### Data Format

```json
{
  "id": 0,
  "image_path": "images/frame_000000.jpg",
  "proprioception_path": "proprio/proprio_000000.npy",
  "action_path": "actions/action_000000.npy",
  "instruction": "Walk forward",
  "task_type": "locomotion"
}
```

**Proprioception format:** `[joint_pos (32), joint_vel (32), joint_torque (32), imu (6)]`

---

## Safety Constraints

### Safety Shield Integration

```python
from model.safety.safety_shield import SafetyShield, SafetyConfig
from model.safety.constraint_handler import ConstraintHandler

# Safety configuration
safety_config = SafetyConfig(
    max_velocity=5.0,        # rad/s (joint velocity)
    max_acceleration=100.0,  # Nm (reused for torque limit)
)

safety_shield = SafetyShield(safety_config)
constraint_handler = ConstraintHandler(
    action_dim=32,
    hidden_dim=512,
)

# Apply safety during inference
def safe_predict(model, inputs):
    output = model(**inputs)
    action = output['action']

    # Clamp joint velocities
    action = torch.clamp(
        action,
        min=-safety_config.max_velocity,
        max=safety_config.max_velocity,
    )

    # Apply constraint optimization
    safe_action = constraint_handler(action, inputs['joint_positions'])

    return {**output, 'action': safe_action}
```

### Safety Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_joint_velocity` | 5.0 rad/s | Maximum joint velocity |
| `max_joint_torque` | 100 Nm | Maximum joint torque |
| `min_com_height` | 0.3 m | Minimum center-of-mass height (fall detection) |
| `control_freq` | 100 Hz | Real-time control frequency |

### Joint Limits (Example)

| Joint | Lower Limit (rad) | Upper Limit (rad) |
|-------|-------------------|-------------------|
| Hip Pitch | -2.0 | 1.5 |
| Hip Roll | -0.5 | 0.5 |
| Hip Yaw | -0.5 | 0.5 |
| Knee | -0.1 | 2.5 |
| Ankle Pitch | -0.8 | 0.8 |
| Ankle Roll | -0.3 | 0.3 |
| Shoulder Pitch | -2.0 | 2.0 |
| Shoulder Roll | -1.5 | 1.5 |
| Shoulder Yaw | -1.5 | 1.5 |
| Elbow | 0.0 | 2.5 |
| Wrist Pitch | -1.0 | 1.0 |
| Wrist Roll | -1.0 | 1.0 |
| Wrist Yaw | -1.0 | 1.0 |

---

## Deployment

### TorchScript Export

```python
from model.utils.export import TorchScriptExporter
from model.embodiment.humanoid import HumanoidVLA, HumanoidConfig

# Create model
config = HumanoidConfig()
model = HumanoidVLA(config)
model.load_state_dict(torch.load("./checkpoints/humanoid_vla/best_model.pt")["model_state_dict"])
model.eval()

# Export
exporter = TorchScriptExporter()
exporter.export_traced(
    model=model,
    example_inputs=(
        torch.randn(1, 3, 224, 224),   # image
        torch.randn(1, 32),             # joint_pos
        torch.randn(1, 32),             # joint_vel
        torch.randn(1, 32),             # joint_torque
    ),
    output_path="./deployed/humanoid_vla.pt",
)
```

### ONNX Export

```python
from model.utils.export import ONNXExporter

exporter = ONNXExporter()
exporter.export(
    model=model,
    output_path="./deployed/humanoid_vla.onnx",
    input_shapes={
        "image": (1, 3, 224, 224),
        "joint_positions": (1, 32),
        "joint_velocities": (1, 32),
        "joint_torques": (1, 32),
    },
    opset_version=17,
)
```

### ROS2 Integration

```python
from integration.ros_bridge import ROSBridge

# Initialize ROS2 bridge
bridge = ROSBridge(
    model_path="./deployed/humanoid_vla.pt",
    camera_topic="/camera/color/image_raw",
    joint_state_topic="/joint_states",
    joint_command_topic="/joint_commands",
    imu_topic="/imu/data",
)

bridge.run()
```

### Real-Time Control Loop

```python
import time

class HumanoidController:
    def __init__(self, model_path, control_freq=100.0):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.dt = 1.0 / control_freq

    def run(self, robot_interface, max_steps=10000):
        for step in range(max_steps):
            start_time = time.time()

            # Get observations
            image = robot_interface.get_camera_image()
            joint_pos = robot_interface.get_joint_positions()
            joint_vel = robot_interface.get_joint_velocities()
            joint_torque = robot_interface.get_joint_torques()
            imu = robot_interface.get_imu()

            # Predict action
            with torch.no_grad():
                output = self.model(
                    image.unsqueeze(0),
                    joint_pos.unsqueeze(0),
                    joint_vel.unsqueeze(0),
                    joint_torque.unsqueeze(0),
                )
            action = output['action'].squeeze(0).numpy()

            # Send command
            robot_interface.send_joint_command(action)

            # Maintain control frequency
            elapsed = time.time() - start_time
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
```

---

## API Reference

### HumanoidVLA

```python
class HumanoidVLA(nn.Module):
    """Complete VLA model for humanoid robots."""

    def __init__(self, config: HumanoidConfig):
        """
        Initialize HumanoidVLA.

        Args:
            config: HumanoidConfig with model parameters
        """

    def forward(
        self,
        image: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
        velocity_command: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for humanoid VLA.

        Args:
            image: (B, 3, H, W) visual input
            joint_positions: (B, num_joints) current positions
            joint_velocities: (B, num_joints) current velocities
            joint_torques: (B, num_joints) current torques
            language_features: (B, seq_len, llm_dim) LLM features
            velocity_command: (B, 3) target velocity [vx, vy, vyaw]
            target_poses: (B, 12) end-effector targets
            imu_data: (B, 9) IMU data
            foot_contacts: (B, 4) foot contact flags

        Returns:
            Dict containing:
                - action: (B, num_joints) joint position commands
                - velocity_command: (B, 3) decoded velocity
                - target_poses: (B, 12) decoded targets
                - priorities: (B, 3) task priorities
                - gripper_actions: (B, 2) gripper commands
                - visual_features: (B, hidden_dim) vision encoding
        """
```

### ProprioceptionEncoder

```python
class ProprioceptionEncoder(nn.Module):
    """Encode robot proprioceptive state."""

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        imu_data: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode proprioceptive state.

        Args:
            joint_positions: (B, num_joints) positions
            joint_velocities: (B, num_joints) velocities
            joint_torques: (B, num_joints) torques
            imu_data: (B, 9) orientation + angular vel + accel
            foot_contacts: (B, 4) contact flags

        Returns:
            features: (B, hidden_dim) encoded proprioception
        """
```

### LocomotionPolicy

```python
class LocomotionPolicy(nn.Module):
    """Locomotion policy for walking/running."""

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        velocity_command: torch.Tensor,
        imu_data: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate locomotion action.

        Args:
            joint_positions: (B, num_joints) positions
            joint_velocities: (B, num_joints) velocities
            joint_torques: (B, num_joints) torques
            velocity_command: (B, 3) target [vx, vy, vyaw]
            imu_data: (B, 9) IMU data
            foot_contacts: (B, 4) contact flags
            phase: (B, 2) gait phase [sin, cos]

        Returns:
            Dict containing:
                - action_mean: (B, action_dim) mean action
                - action_std: (action_dim,) std deviation
                - value: (B,) value estimate
                - features: (B, hidden_dim) policy features
        """

    def sample_action(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        velocity_command: torch.Tensor,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Sample action from policy distribution."""
```

### WholeBodyController

```python
class WholeBodyController(nn.Module):
    """Coordinates locomotion, manipulation, and balance."""

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        velocity_command: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
        language_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate whole-body action.

        Returns:
            Dict containing:
                - action: (B, action_dim) final weighted action
                - locomotion_action: (B, action_dim) locomotion output
                - manipulation_action: (B, action_dim) manipulation output
                - balance_correction: (B, action_dim) balance adjustment
                - priorities: (B, 3) task weights [loco, manip, balance]
                - gripper_actions: (B, 2) gripper commands
        """
```

---

## Related Documents

- [Training VLA Recipe](training_vla_recipe.md) - Complete training pipeline
- [Training Locomotion](training_locomotion.md) - Locomotion training details
- [Training Manipulation](training_manipulation.md) - Manipulation training details
- [Training Whole Body](training_whole_body.md) - Whole-body coordination
- [Training Datasets](training_datasets.md) - Dataset documentation
- [Deployment Guide](deployment.md) - Safety and deployment
- [Architecture Guide](../architecture.md) - Overall VLA architecture
- [Training Embodiment](../training_embodiment.md) - Cross-embodiment training
