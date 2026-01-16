# VLA Training Documentation for Humanoid Robots

This folder contains comprehensive documentation for training Vision-Language-Action (VLA) models for humanoid robot control, including locomotion, manipulation, and whole-body coordination.

## Documentation Index

| Document | Description |
|----------|-------------|
| [training_vla_recipe.md](training_vla_recipe.md) | **Complete VLA Training Pipeline** - End-to-end training from VLM pretraining to deployment |
| [training_locomotion.md](training_locomotion.md) | **Locomotion Training** - Bipedal walking, running, terrain adaptation |
| [training_manipulation.md](training_manipulation.md) | **Manipulation Training** - Reaching, grasping, bimanual coordination |
| [training_whole_body.md](training_whole_body.md) | **Whole-Body Control** - Loco-manipulation, hierarchical control |
| [training_sensors.md](training_sensors.md) | **Sensor Training** - Joint encoders, IMU, F/T sensors, proprioception fusion |
| [training_datasets.md](training_datasets.md) | **Dataset Documentation** - MoCap, teleoperation, simulation datasets |
| [deployment.md](deployment.md) | **Deployment & Safety** - Real robot deployment with safety constraints |

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Humanoid VLA Training Pipeline                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: VLM Foundation                                                     │
│  ├── 1a: Vision-Language Alignment (Projector training)                     │
│  └── 1b: Humanoid Instruction Tuning (LLM fine-tuning)                      │
│                                                                              │
│  Stage 2: Motion Primitive Learning                                          │
│  └── VAE-based primitive extraction from MoCap data                         │
│                                                                              │
│  Stage 3: Locomotion Training                                                │
│  ├── 3a: Standing/Balance (RL with stability rewards)                       │
│  ├── 3b: Walking (PPO with curriculum learning)                             │
│  └── 3c: Terrain Adaptation (Domain randomization)                          │
│                                                                              │
│  Stage 4: Manipulation Training                                              │
│  ├── 4a: Reaching (BC from demonstrations)                                  │
│  ├── 4b: Grasping (BC + RL with success reward)                             │
│  └── 4c: Bimanual Coordination (Multi-task learning)                        │
│                                                                              │
│  Stage 5: Whole-Body Control                                                 │
│  ├── 5a: Loco-Manipulation (Combined locomotion + manipulation)             │
│  └── 5b: Hierarchical Control (Task → Skill → Motion → Torque)              │
│                                                                              │
│  Stage 6: Policy Improvement                                                 │
│  ├── 6a: Online RL (PPO, SAC, AMP)                                          │
│  ├── 6b: Offline RL (IQL, CQL, TD3+BC)                                      │
│  └── 6c: Motion Imitation (GAIL, AMP)                                       │
│                                                                              │
│  Stage 7: Deployment                                                         │
│  └── Safety-critical real robot deployment                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. VLM Pretraining

```bash
# Stage 1a: Vision-Language Alignment
python train/pretrain/vlm_pretrainer.py \
    --vision-model google/siglip-base-patch16-224 \
    --llm-model Qwen/Qwen2-1.5B-Instruct \
    --alignment-dataset liuhaotian/LLaVA-Pretrain \
    --output-dir ./output/humanoid/stage1a

# Stage 1b: Humanoid Instruction Tuning
python train/pretrain/vlm_pretrainer.py \
    --pretrained-vlm ./output/humanoid/stage1a/best \
    --instruction-dataset liuhaotian/LLaVA-Instruct-150K \
    --humanoid-instructions \
    --output-dir ./output/humanoid/stage1b
```

### 2. Locomotion Training

```bash
# Standing/Balance
python train/online_rl/ppo_trainer.py \
    --env humanoid-stand-v0 \
    --num-steps 1000000 \
    --output-dir ./output/humanoid/locomotion/standing

# Walking with Curriculum
python train/online_rl/ppo_trainer.py \
    --env humanoid-walk-v0 \
    --pretrained ./output/humanoid/locomotion/standing/best.pt \
    --curriculum \
    --num-steps 5000000 \
    --output-dir ./output/humanoid/locomotion/walking
```

### 3. Manipulation Training

```bash
# Behavioral Cloning for Reaching
python train/il/behavioral_cloning.py \
    --model humanoid-manipulation \
    --dataset reaching-demonstrations \
    --output-dir ./output/humanoid/manipulation/reaching

# Grasping with RL
python train/online_rl/sac_trainer.py \
    --env humanoid-grasp-v0 \
    --pretrained ./output/humanoid/manipulation/reaching/best.pt \
    --num-steps 1000000 \
    --output-dir ./output/humanoid/manipulation/grasping
```

### 4. Whole-Body Control

```bash
python train/embodiment/train_humanoid_vla.py \
    --pretrained-locomotion ./output/humanoid/locomotion/walking/best.pt \
    --pretrained-manipulation ./output/humanoid/manipulation/grasping/best.pt \
    --train-whole-body \
    --output-dir ./output/humanoid/whole-body
```

### 5. Deployment

```bash
# Export model
python -c "
from model.utils.export import TorchScriptExporter
from model.embodiment import HumanoidVLA

model = HumanoidVLA.from_pretrained('./output/humanoid/whole-body')
exporter = TorchScriptExporter()
exporter.export_traced(model, output_path='./deployed/humanoid_vla.pt')
"

# Run deployment
python deploy/humanoid_deployment.py \
    --model-path ./deployed/humanoid_vla.pt \
    --robot-interface ros2 \
    --enable-safety-shield
```

## Datasets Summary

| Training Stage | Dataset | Source |
|----------------|---------|--------|
| VLM Alignment | LLaVA-Pretrain (558K) | [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |
| Instruction Tuning | LLaVA-Instruct-150K | [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
| Motion Primitives | CMU MoCap | [CMU](http://mocap.cs.cmu.edu/) |
| Motion Primitives | AMASS | [MPI](https://amass.is.tue.mpg.de/) |
| Locomotion | D4RL Humanoid | [HuggingFace](https://huggingface.co/datasets/imone/D4RL) |
| Manipulation | DROID | [HuggingFace](https://huggingface.co/datasets/cadene/droid) |
| Manipulation | LeRobot ALOHA | [HuggingFace](https://huggingface.co/lerobot) |
| Whole-Body | HumanoidBench | [GitHub](https://humanoid-bench.github.io/) |

## Key Files Reference

```
model/
├── embodiment/
│   ├── humanoid.py              # HumanoidVLA, WholeBodyController
│   └── __init__.py
├── action_head/
│   ├── mlp_action_head.py       # MLPActionHead, GaussianMLPActionHead
│   ├── diffusion_action_head.py # DiffusionActionHead
│   └── transformer_action_head.py # TransformerActionHead, GPTActionHead
├── safety/
│   ├── safety_shield.py         # Safety monitoring
│   ├── constraint_handler.py    # Constraint optimization
│   └── rule_checker.py          # Kinematic/collision checking
└── temporal/
    ├── temporal_encoder.py      # TemporalTransformer, TemporalLSTM
    ├── history_encoder.py       # HistoryEncoder
    └── memory_buffer.py         # MemoryBuffer, EpisodicMemory

train/
├── embodiment/
│   └── train_humanoid_vla.py    # HumanoidVLATrainer
├── online_rl/
│   ├── ppo_trainer.py           # PPO for locomotion
│   └── sac_trainer.py           # SAC for manipulation
├── il/
│   └── behavioral_cloning.py    # BC for demonstrations
└── datasets/
    └── mocap_dataset.py         # MoCap data loading

scripts/
└── run_humanoid_vla.sh          # Training launcher
```

## Humanoid Configuration

```python
from model.embodiment.humanoid import HumanoidConfig

config = HumanoidConfig(
    # Robot structure
    num_joints=32,              # Typical humanoid DOF
    num_body_parts=15,          # Head, torso, arms, legs, hands, feet

    # Observation
    proprioception_dim=128,
    image_size=224,

    # Action
    action_dim=32,
    control_freq=100.0,         # Hz
    action_type="position",     # "position" or "torque"

    # Architecture
    hidden_dim=512,
    num_heads=8,
    num_layers=4,
    llm_hidden_dim=4096,
)
```

## Joint Configuration (32 DoF)

| Body Part | Joints | DoF |
|-----------|--------|-----|
| Head | pan, tilt, roll | 3 |
| Torso | yaw, pitch, roll | 3 |
| Left Arm | shoulder (3), elbow, wrist (3) | 7 |
| Right Arm | shoulder (3), elbow, wrist (3) | 7 |
| Left Leg | hip (3), knee, ankle (2) | 6 |
| Right Leg | hip (3), knee, ankle (2) | 6 |
| **Total** | | **32** |

## Safety Requirements

| Requirement | Value | Description |
|-------------|-------|-------------|
| Max Joint Velocity | 5.0 rad/s | Prevent fast movements |
| Max Joint Torque | 100 Nm | Protect actuators |
| Min COM Height | 0.3 m | Detect falls |
| Control Frequency | 50-100 Hz | Real-time control |

## Getting Help

For issues or questions:
- Check the specific document for your training stage
- Review example scripts in `examples/`
- Run tests in `tests/` to verify your setup
- See [autonomous documentation](../autonomous/) for comparison
