# Vision-Language-Action (VLA) Model Training

A comprehensive framework for training Vision-Language-Action models for robotics and autonomous vehicle applications.

## Project Structure

```
Vision_Language_Model_Practice/
├── model/                    # Model components
│   ├── vlm/                  # Vision-Language Model (encoders, projectors)
│   ├── vla/                  # Complete VLA implementations
│   ├── action_head/          # Action prediction heads (MLP, Gaussian MLP, Diffusion, Transformer)
│   ├── sensor/               # Sensor encoders (LiDAR, Radar, IMU)
│   └── fusion/               # Multi-modal sensor fusion (Cross-modal, Hierarchical, Gated)
├── config/                   # Configuration classes
├── train/                    # Training modules
│   ├── pretrain/             # VLM pretraining (alignment, instruction tuning)
│   ├── finetune/             # Supervised fine-tuning
│   ├── rl/                   # Reinforcement learning (PPO, SAC, GRPO)
│   ├── il/                   # Imitation learning (BC, DAgger, GAIL)
│   └── datasets/             # Dataset loaders
├── eval/                     # Evaluation scripts
└── scripts/                  # Utility scripts
```

---

## VLA Training Procedure

Training a Vision-Language-Action model follows a multi-stage approach. Each stage builds upon the previous one to create a capable robot policy.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VLA Training Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage 1: VLM Pretraining (Optional - use pretrained VLM)                  │
│     │                                                                       │
│     ├── 1a. Vision-Language Alignment                                       │
│     │       └── Train projector to align vision encoder with LLM           │
│     │                                                                       │
│     └── 1b. Visual Instruction Tuning                                       │
│             └── Fine-tune on multimodal instruction data                    │
│                                                                             │
│  Stage 2: VLA Supervised Fine-tuning                                        │
│     │                                                                       │
│     └── Train action head on robot demonstrations                           │
│         ├── Behavioral Cloning (BC)                                         │
│         └── Action Chunking (ACT-style)                                     │
│                                                                             │
│  Stage 3: Policy Improvement (Optional)                                     │
│     │                                                                       │
│     ├── 3a. Reinforcement Learning                                          │
│     │       ├── PPO (On-policy)                                             │
│     │       ├── SAC (Off-policy)                                            │
│     │       └── GRPO (LLM-style optimization)                               │
│     │                                                                       │
│     └── 3b. Interactive Imitation Learning                                  │
│             └── DAgger (expert-in-the-loop)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: VLM Pretraining (Optional)

> **Note**: You can skip this stage by using a pretrained VLM like LLaVA, Qwen-VL, or PaliGemma.

### 1a. Vision-Language Alignment

Train a projector to align visual features with the LLM embedding space.

```python
from model import VisionEncoder, VisionProjector
from train.pretrain import AlignmentTrainer
from config import PretrainingConfig

# Configure
config = PretrainingConfig(
    vision_encoder="openai/clip-vit-large-patch14",
    llm_model="meta-llama/Llama-3.2-3B",
    projector_type="mlp",
    freeze_vision=True,
    freeze_llm=True,
    learning_rate=1e-3,
    batch_size=256,
    num_epochs=1,
)

# Dataset: Image-caption pairs (CC3M, LAION, etc.)
# Only the projector is trained

trainer = AlignmentTrainer(config)
trainer.train()
```

**What's trained**: Only the vision projector (MLP or attention-based)
**Dataset**: Image-caption pairs (CC3M, LAION-400M)

### 1b. Visual Instruction Tuning

Fine-tune the model on multimodal instruction-following data.

```python
from train.pretrain import InstructionTrainer

config = PretrainingConfig(
    freeze_vision=True,
    freeze_llm=False,  # Unfreeze LLM
    learning_rate=2e-5,
    batch_size=128,
    num_epochs=1,
)

# Dataset: LLaVA-Instruct, ShareGPT-4V, etc.
trainer = InstructionTrainer(config)
trainer.train()
```

**What's trained**: Projector + LLM (LoRA optional)
**Dataset**: Visual instruction data (LLaVA-Instruct-150K)

---

## Stage 2: VLA Supervised Fine-tuning

This is the core stage where we add action prediction capability.

### Option A: Basic VLA Fine-tuning

```python
from model import VLAModel, MLPActionHead
from train.finetune import VLAFineTuner
from train.datasets import BridgeDataset, create_openx_dataloader
from config import FineTuningConfig

# 1. Create VLA model
model = VLAModel(
    vlm_name_or_path="llava-hf/llava-1.5-7b-hf",
    action_head=MLPActionHead(
        hidden_dim=1024,
        action_dim=7,  # 6 DoF + gripper
        num_layers=3,
    ),
    freeze_vlm=True,  # Only train action head initially
)

# 2. Load dataset
dataset = BridgeDataset(split="train", max_samples=50000)
dataloader = create_openx_dataloader(dataset, batch_size=32)

# 3. Configure training
config = FineTuningConfig(
    learning_rate=1e-4,
    num_epochs=50,
    action_loss="mse",  # or "smooth_l1"
    gradient_accumulation_steps=4,
)

# 4. Train
trainer = VLAFineTuner(model, config)
trainer.train(dataloader)
```

### Option B: Multi-Sensor VLA (for Autonomous Driving)

```python
from model import MultiSensorVLA, PointNetEncoder, RadarEncoder
from model import GatedFusion  # Learned modality weighting
from train.datasets import NuScenesDataset, create_driving_dataloader

# Create multi-sensor VLA
model = MultiSensorVLA(
    vlm_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
    sensor_encoders={
        "lidar": PointNetEncoder(output_dim=512),
        "radar": RadarEncoder(output_dim=256),
    },
    action_dim=3,  # steering, throttle, brake
)

# Load driving dataset
dataset = NuScenesDataset(
    data_root="/path/to/nuscenes",
    version="v1.0-trainval",
    use_lidar=True,
    use_radar=True,
)
dataloader = create_driving_dataloader(dataset, batch_size=16)

trainer = VLAFineTuner(model, config)
trainer.train(dataloader)
```

#### Available Fusion Strategies

```python
from model import SensorFusion, CrossModalFusion, HierarchicalFusion, GatedFusion

# Simple concatenation + self-attention
fusion = SensorFusion(feature_dim=512, num_heads=8)

# Cross-attention between modalities
fusion = CrossModalFusion(query_dim=512, key_dim=256, num_heads=8)

# Hierarchical spatial-temporal fusion
fusion = HierarchicalFusion(
    feature_dim=512,
    num_spatial_layers=2,
    num_temporal_layers=2,
)

# Learned gating for modality weighting (recommended for multi-sensor)
fusion = GatedFusion(
    modality_dims=[512, 256, 128],  # LiDAR, Radar, IMU
    output_dim=512,
)
```

### Option C: Action Chunking (ACT-style)

For temporal action prediction with Transformer action head:

```python
from model import VLAModel, TransformerActionHead

model = VLAModel(
    vlm_name_or_path="HuggingFaceTB/SmolVLM-Instruct",
    action_head=TransformerActionHead(
        action_dim=7,
        chunk_size=16,  # Predict 16 future actions
        num_layers=4,
    ),
)
```

### Option D: Diffusion Action Head with DDIM Sampling

For multi-modal action distributions with fast inference:

```python
from model import VLAModel, DiffusionActionHead

model = VLAModel(
    vlm_name_or_path="llava-hf/llava-1.5-7b-hf",
    action_head=DiffusionActionHead(
        action_dim=7,
        num_diffusion_steps=100,
        hidden_dim=512,
    ),
)

# Fast inference with DDIM (5-10x faster than DDPM)
action = model.action_head.sample(
    features,
    num_inference_steps=20,  # Only 20 steps instead of 100
    use_ddim=True,           # Enable DDIM for deterministic, fast sampling
)
```

### Option E: Stochastic Actions with Gaussian MLP

For uncertainty-aware action prediction:

```python
from model import VLAModel, GaussianMLPActionHead

model = VLAModel(
    vlm_name_or_path="HuggingFaceTB/SmolVLM-Instruct",
    action_head=GaussianMLPActionHead(
        input_dim=1024,
        action_dim=7,
        hidden_dim=512,
        min_std=0.01,
        max_std=1.0,
    ),
)

# During inference, sample actions with learned uncertainty
action, log_prob = model.action_head.sample(features)
```

---

## Stage 3: Policy Improvement

After supervised fine-tuning, improve the policy with RL or interactive IL.

### Option 3a: Reinforcement Learning

#### PPO (Recommended for Online RL)

```python
from train.rl import PPOTrainer
from config import RLConfig

config = RLConfig(
    algorithm="ppo",
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    entropy_coef=0.01,
    num_envs=8,
    steps_per_update=2048,
)

trainer = PPOTrainer(model, env, config)
trainer.train(total_timesteps=1_000_000)
```

#### SAC (For Sample Efficiency)

```python
from train.rl import SACTrainer
from train.datasets import ReplayBuffer

buffer = ReplayBuffer(capacity=1_000_000)
trainer = SACTrainer(model, env, buffer, config)
trainer.train()
```

#### GRPO (For VLA with LLM)

Group Relative Policy Optimization - optimizes VLA like an LLM:

```python
from train.rl import GRPOTrainer

config = RLConfig(
    algorithm="grpo",
    group_size=4,  # Compare 4 rollouts
    kl_coef=0.1,
)

trainer = GRPOTrainer(model, env, config)
trainer.train()
```

### Option 3b: DAgger (Interactive Imitation Learning)

```python
from train.il import DAgger

config = ILConfig(
    beta_schedule="linear",  # Expert intervention decreases over time
    num_iterations=10,
    rollouts_per_iteration=100,
)

trainer = DAgger(model, env, expert_policy, config)
trainer.train()
```

---

## Recommended Training Recipes

### Recipe 1: Quick Start (Robotics)

For a simple manipulation task:

```python
# 1. Use pretrained SmolVLM (smallest, fastest)
# 2. Train on PushT dataset (simple, fast to train)
# 3. Use MLP action head

from model import create_vla_model
from train.datasets import PushTDataset
from train.il import BehavioralCloning

model = create_vla_model(
    vlm="smolvlm",
    action_head="mlp",
    action_dim=2,
)

dataset = PushTDataset(split="train")
trainer = BehavioralCloning(model, dataset)
trainer.train(num_epochs=100)
```

### Recipe 2: Production (Robotics)

For real robot deployment:

```python
# 1. Use OpenVLA or fine-tuned LLaVA
# 2. Train on Bridge V2 + RT-1 combined
# 3. Use Diffusion action head for multi-modal actions
# 4. Fine-tune with GRPO for task success

from model import OpenVLAWrapper, DiffusionActionHead
from train.datasets import BridgeDataset, RT1Dataset
from torch.utils.data import ConcatDataset

model = OpenVLAWrapper(
    action_head=DiffusionActionHead(action_dim=7),
)

dataset = ConcatDataset([
    BridgeDataset(split="train"),
    RT1Dataset(split="train[:10000]"),
])

# Stage 1: BC
bc_trainer = BehavioralCloning(model, dataset)
bc_trainer.train(num_epochs=50)

# Stage 2: RL fine-tuning
grpo_trainer = GRPOTrainer(model, env, config)
grpo_trainer.train(total_timesteps=100_000)
```

### Recipe 3: Autonomous Driving

```python
from model import MultiSensorVLA, PointTransformerEncoder
from train.datasets import NuScenesDataset

model = MultiSensorVLA(
    vlm="qwen2-vl",
    sensor_encoders={
        "lidar": PointTransformerEncoder(output_dim=512),
    },
    action_dim=3,
)

dataset = NuScenesDataset(
    use_lidar=True,
    prediction_horizon=10,  # Predict 10 future waypoints
)

# BC training (offline RL not recommended for driving)
trainer = BehavioralCloning(model, dataset)
trainer.train()
```

---

## Dataset Selection Guide

| Dataset | Best For | Training Method |
|---------|----------|-----------------|
| **LeRobot** (PushT, ALOHA) | Quick testing, Imitation Learning | BC, DAgger |
| **Open X-Embodiment** (Bridge, RT-1) | VLA fine-tuning with language | BC, VLA SFT |
| **D4RL** (MuJoCo, Antmaze) | Offline RL research | CQL, IQL, DT |
| **RoboMimic** | Robot manipulation IL | BC, Action Chunking |
| **nuScenes** | Autonomous driving | End-to-end BC |
| **CARLA** | Driving simulation + RL | PPO, SAC |

---

## Hardware Requirements

| Model Size | GPU Memory | Recommended GPU |
|------------|------------|-----------------|
| SmolVLM (256M) | 8GB | RTX 3080, RTX 4070 |
| LLaVA-1.5 (7B) | 24GB | RTX 3090, RTX 4090 |
| LLaVA-1.5 (13B) | 48GB | A6000, A100 |
| OpenVLA (7B) | 24GB | RTX 4090, A6000 |

**Tips**:
- Use gradient checkpointing for larger models
- Use LoRA for memory-efficient fine-tuning
- Use DeepSpeed ZeRO-3 for multi-GPU training

### Memory-Efficient Training

For limited GPU memory (8-16GB), use the built-in memory-efficient configuration:

```python
from config import get_training_config

# Pre-configured memory-efficient settings
config = get_training_config("finetune-memory-efficient")
# - Batch size: 2 with gradient accumulation (effective batch: 16)
# - LoRA enabled with r=16
# - Mixed precision (bf16)
# - Vision and LLM frozen

# Or manually enable gradient checkpointing
from model import create_vla_model

model = create_vla_model(
    vision_encoder="siglip-base",
    llm="qwen2-1.5b",
    use_gradient_checkpointing=True,  # Reduces memory ~30%
)
```

### Available Training Configurations

```python
from config import get_training_config

# Fine-tuning configs
config = get_training_config("finetune-quick")            # Quick testing
config = get_training_config("finetune-standard")         # Standard training
config = get_training_config("finetune-full")             # Full fine-tuning
config = get_training_config("finetune-lora")             # LoRA fine-tuning
config = get_training_config("finetune-memory-efficient") # Low memory (8-16GB)

# RL configs
config = get_training_config("rl-ppo")   # PPO
config = get_training_config("rl-sac")   # SAC
config = get_training_config("rl-grpo")  # GRPO for VLA

# IL configs
config = get_training_config("il-bc")    # Behavioral Cloning
config = get_training_config("il-gail")  # GAIL
```

---

## Installation

```bash
# Core dependencies
pip install torch torchvision transformers accelerate

# VLM specific
pip install lerobot  # For LeRobot datasets
pip install tensorflow-datasets  # For Open X-Embodiment

# RL specific
pip install gymnasium mujoco

# Driving specific
pip install nuscenes-devkit
```

---

## Quick Start Example

```python
from model import create_vla_model
from train.datasets import PushTDataset, create_lerobot_dataloader
from train.il import BehavioralCloning
from config import ILConfig

# 1. Create model
model = create_vla_model(
    vlm="smolvlm",
    action_head="mlp",
    action_dim=2,
)

# 2. Load dataset
dataset = PushTDataset(split="train[:1000]")
dataloader = create_lerobot_dataloader(dataset, batch_size=64)

# 3. Train with Behavioral Cloning
config = ILConfig(
    learning_rate=1e-4,
    num_epochs=100,
)

trainer = BehavioralCloning(model, config)
trainer.train(dataloader)

# 4. Save model
model.save_pretrained("./my_vla_model")
```

---

## References

- [OpenVLA](https://openvla.github.io/) - Open-source VLA model
- [RT-2](https://robotics-transformer2.github.io/) - Vision-Language-Action model from Google
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace robot learning library
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) - Cross-embodiment dataset

---

## License

MIT License
