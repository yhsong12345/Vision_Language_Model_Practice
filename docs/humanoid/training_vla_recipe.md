# Humanoid VLA Training Recipe

This document provides the complete training pipeline for Vision-Language-Action (VLA) models designed for humanoid robot control.

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
│  Stage 7: Human-Robot Interaction                                            │
│  └── Command following, handover, collaboration                             │
│                                                                              │
│  Stage 8: Deployment                                                         │
│  └── Safety-critical real robot deployment                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: VLM Foundation for Humanoids

### 1a: Vision-Language Alignment

Train the vision-language projector to align visual features with the LLM embedding space.

```bash
python train/pretrain/vlm_pretrainer.py \
    --vision-model google/siglip-base-patch16-224 \
    --llm-model Qwen/Qwen2-1.5B-Instruct \
    --alignment-dataset liuhaotian/LLaVA-Pretrain \
    --output-dir ./output/humanoid/stage1a
```

### 1b: Humanoid Instruction Tuning

Fine-tune the VLM on humanoid-specific instructions.

```bash
python train/pretrain/vlm_pretrainer.py \
    --pretrained-vlm ./output/humanoid/stage1a/best \
    --instruction-dataset liuhaotian/LLaVA-Instruct-150K \
    --humanoid-instructions \
    --output-dir ./output/humanoid/stage1b
```

**Humanoid-specific instruction types:**
- Object manipulation: "Pick up the red cup from the table"
- Navigation: "Walk to the door"
- Gestures: "Wave hello"
- Human interaction: "Hand me the tool"
- Body awareness: "What is your left hand holding?"

### Dataset Requirements for Stage 1

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 1a: VLM Alignment** | LLaVA-Pretrain | [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | 558K image-caption pairs for vision-language alignment |
| **Stage 1b: Instruction Tuning** | LLaVA-Instruct-150K | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Visual instructions for manipulation, navigation, gestures |

---

## Stage 2: Motion Primitive Learning

Learn a library of motion primitives from motion capture data using VAE-based encoding.

### Motion Primitive Extraction

```bash
python train/humanoid/motion_primitive_trainer.py \
    --mocap-dir ./data/mocap \
    --num-primitives 16 \
    --latent-dim 64 \
    --output-dir ./output/humanoid/stage2
```

### Configuration

```python
from train.humanoid.motion_primitive_trainer import MotionPrimitiveConfig

config = MotionPrimitiveConfig(
    num_primitives=16,          # Number of discrete primitives
    primitive_length=50,        # Frames per primitive
    latent_dim=64,              # VAE latent dimension
    learning_rate=1e-4,
    num_epochs=100,
)
```

### Motion Primitive Library

| Primitive ID | Motion Type | Description |
|--------------|-------------|-------------|
| 0 | Stand | Static standing pose |
| 1 | Walk Forward | Forward walking gait |
| 2 | Walk Backward | Backward walking gait |
| 3 | Turn Left | In-place left turn |
| 4 | Turn Right | In-place right turn |
| 5 | Reach Forward | Forward arm reach |
| 6 | Reach Up | Upward arm reach |
| 7 | Reach Down | Downward arm reach |
| 8 | Grasp | Hand closing motion |
| 9 | Release | Hand opening motion |
| 10 | Lift | Lifting motion with arms |
| 11 | Lower | Lowering motion with arms |
| 12 | Push | Pushing motion |
| 13 | Pull | Pulling motion |
| 14 | Wave | Waving gesture |
| 15 | Point | Pointing gesture |

### Dataset Requirements for Stage 2

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 2: Motion Primitives** | CMU MoCap | [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/) | 2600+ motion sequences covering diverse activities |
| **Stage 2: Motion Primitives** | Human3.6M | [vision.imar.ro/human3.6m](http://vision.imar.ro/human3.6m/) | 3.6M+ 3D human poses with action labels |
| **Stage 2: Motion Primitives** | AMASS | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) | 40+ hours unified MoCap, 300+ subjects |

---

## Stage 3: Locomotion Training

Train bipedal locomotion using reinforcement learning with curriculum.

### 3a: Standing/Balance Training

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-stand-v0 \
    --num-steps 1000000 \
    --reward-weights upright=1.0,survival=0.5,smooth=0.1 \
    --output-dir ./output/humanoid/stage3a
```

### 3b: Walking Training with Curriculum

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-walk-v0 \
    --pretrained ./output/humanoid/stage3a/best \
    --curriculum-velocities 0.3,0.5,0.8,1.0,1.5 \
    --num-steps 5000000 \
    --output-dir ./output/humanoid/stage3b
```

### 3c: Terrain Adaptation

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-terrain-v0 \
    --pretrained ./output/humanoid/stage3b/best \
    --domain-randomization \
    --terrain-types flat,stairs,slopes,rough \
    --num-steps 10000000 \
    --output-dir ./output/humanoid/stage3c
```

### Locomotion Reward Components

| Reward Component | Weight | Description |
|------------------|--------|-------------|
| Forward Velocity | 1.0 | Match target velocity |
| Upright Bonus | 0.5 | Maintain upright posture |
| Survival | 0.1 | Bonus for not falling |
| Energy Efficiency | -0.01 | Minimize joint torques |
| Smooth Motion | -0.1 | Minimize action jerk |
| Foot Clearance | 0.2 | Proper foot lift during swing |
| Gait Symmetry | 0.3 | Symmetric left/right motion |

---

## Stage 4: Manipulation Training

Train arm control for reaching and grasping tasks.

### 4a: Reaching Training (BC)

```bash
python train/il/behavioral_cloning.py \
    --model humanoid-manipulation \
    --dataset reaching-demonstrations \
    --num-epochs 100 \
    --output-dir ./output/humanoid/stage4a
```

### 4b: Grasping Training (BC + RL)

```bash
# Phase 1: Approach with BC
python train/il/behavioral_cloning.py \
    --model humanoid-manipulation \
    --dataset grasping-approach \
    --output-dir ./output/humanoid/stage4b-approach

# Phase 2: Grasp execution with RL
python train/online_rl/sac_trainer.py \
    --env humanoid-grasp-v0 \
    --pretrained ./output/humanoid/stage4b-approach/best \
    --num-steps 1000000 \
    --output-dir ./output/humanoid/stage4b
```

### 4c: Bimanual Coordination

```bash
python train/il/behavioral_cloning.py \
    --model humanoid-bimanual \
    --dataset bimanual-demonstrations \
    --coordination-loss 0.1 \
    --output-dir ./output/humanoid/stage4c
```

---

## Stage 5: Whole-Body Control

Train integrated locomotion and manipulation (loco-manipulation).

### 5a: Walking While Carrying

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-carry-v0 \
    --pretrained-locomotion ./output/humanoid/stage3c/best \
    --pretrained-manipulation ./output/humanoid/stage4c/best \
    --curriculum-weights 0.5,1.0,2.0,5.0 \
    --num-steps 5000000 \
    --output-dir ./output/humanoid/stage5a
```

### 5b: Mobile Manipulation

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-mobile-manip-v0 \
    --pretrained ./output/humanoid/stage5a/best \
    --task-phases approach,manipulate,retreat \
    --num-steps 10000000 \
    --output-dir ./output/humanoid/stage5b
```

### 5c: Hierarchical Whole-Body Control

```bash
python train/embodiment/train_humanoid_vla.py \
    --pretrained-vlm ./output/humanoid/stage1b/best \
    --pretrained-locomotion ./output/humanoid/stage3c/best \
    --pretrained-manipulation ./output/humanoid/stage4c/best \
    --hierarchical-control \
    --output-dir ./output/humanoid/stage5c
```

---

## Stage 6: Policy Improvement

Fine-tune policies using advanced RL methods.

### 6a: Online RL with PPO

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-tasks-v0 \
    --pretrained ./output/humanoid/stage5c/best \
    --num-steps 10000000 \
    --output-dir ./output/humanoid/stage6a-ppo
```

### 6b: Adversarial Motion Priors (AMP)

```bash
python train/online_rl/amp_trainer.py \
    --env humanoid-tasks-v0 \
    --pretrained ./output/humanoid/stage5c/best \
    --reference-motions ./data/mocap \
    --style-reward-weight 0.5 \
    --task-reward-weight 0.5 \
    --num-steps 10000000 \
    --output-dir ./output/humanoid/stage6a-amp
```

### 6c: Offline RL with IQL

```bash
python train/offline_rl/iql_trainer.py \
    --model humanoid-vla \
    --dataset humanoid-demonstrations \
    --expectile 0.7 \
    --temperature 3.0 \
    --num-epochs 1000 \
    --output-dir ./output/humanoid/stage6b
```

---

## Stage 7: Human-Robot Interaction

Train for safe interaction with humans.

### Command Following

```bash
python train/embodiment/train_humanoid_vla.py \
    --pretrained ./output/humanoid/stage6a/best \
    --hri-training \
    --dataset command-following \
    --output-dir ./output/humanoid/stage7a
```

### Handover Training

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-handover-v0 \
    --pretrained ./output/humanoid/stage7a/best \
    --safety-constraints \
    --num-steps 2000000 \
    --output-dir ./output/humanoid/stage7b
```

---

## Humanoid Architecture

### HumanoidVLA Model

```python
from model.embodiment import HumanoidVLA, HumanoidConfig

config = HumanoidConfig(
    # Robot structure
    num_joints=32,              # Typical humanoid DOF
    num_body_parts=15,          # Head, torso, arms, legs, hands, feet

    # Observation
    proprioception_dim=128,     # Joint pos, vel, torque
    image_size=224,

    # Action
    action_dim=32,              # Joint position or torque commands

    # Architecture
    hidden_dim=512,
    num_heads=8,
    num_layers=4,

    # Control
    control_freq=100.0,         # Hz
    action_type="position",     # "position" or "torque"

    # LLM
    llm_hidden_dim=4096,
)

model = HumanoidVLA(config)
```

### Joint Configuration (32 DoF)

| Body Part | Joints | DoF |
|-----------|--------|-----|
| Head | pan, tilt, roll | 3 |
| Torso | yaw, pitch, roll | 3 |
| Left Arm | shoulder (3), elbow, wrist (3) | 7 |
| Right Arm | shoulder (3), elbow, wrist (3) | 7 |
| Left Leg | hip (3), knee, ankle (2) | 6 |
| Right Leg | hip (3), knee, ankle (2) | 6 |
| **Total** | | **32** |

### Action Representations

| Type | Description | Use Case |
|------|-------------|----------|
| Joint Position | Target joint angles | Simplest, PD tracking |
| Joint Velocity | Target joint velocities | Smooth motion |
| Joint Torque | Direct torque commands | Most flexible, hardest |
| End-Effector | 6D hand poses with IK | Manipulation tasks |
| Skill Primitives | Primitive ID + parameters | Hierarchical control |

---

## Training Configuration

### Full Training Configuration

```python
from dataclasses import dataclass

@dataclass
class HumanoidTrainConfig:
    # Model
    num_joints: int = 32
    joint_dim: int = 12          # pos, vel, torque for each
    image_size: int = 224
    hidden_dim: int = 512
    llm_hidden_dim: int = 4096

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 200
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000

    # Loss weights
    action_loss_weight: float = 1.0
    locomotion_loss_weight: float = 0.5
    manipulation_loss_weight: float = 0.5
    stability_loss_weight: float = 0.3
    smoothness_loss_weight: float = 0.1

    # Safety
    use_safety_constraints: bool = True
    max_joint_velocity: float = 5.0      # rad/s
    max_joint_torque: float = 100.0      # Nm
    min_com_height: float = 0.3          # meters
```

### Training Script

```bash
# Full humanoid VLA training
python train/embodiment/train_humanoid_vla.py \
    --num-joints 32 \
    --image-size 224 \
    --hidden-dim 512 \
    --batch-size 16 \
    --learning-rate 3e-4 \
    --num-epochs 200 \
    --data-path ./data/humanoid \
    --output-dir ./checkpoints/humanoid_vla \
    --use-safety-constraints
```

---

## Complete Dataset Reference

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 1a** | LLaVA-Pretrain | [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | 558K image-caption pairs |
| **Stage 1b** | LLaVA-Instruct-150K | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Visual instructions |
| **Stage 2** | CMU MoCap | [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/) | 2600+ motion sequences |
| **Stage 2** | Human3.6M | [vision.imar.ro/human3.6m](http://vision.imar.ro/human3.6m/) | 3.6M+ 3D poses |
| **Stage 2** | AMASS | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) | 40+ hours unified MoCap |
| **Stage 3** | D4RL MuJoCo | [imone/D4RL](https://huggingface.co/datasets/imone/D4RL) | Humanoid locomotion |
| **Stage 4** | DROID | [cadene/droid](https://huggingface.co/datasets/cadene/droid) | Teleoperation data |
| **Stage 5** | HumanoidBench | [humanoid-bench.github.io](https://humanoid-bench.github.io/) | Loco-manipulation |
| **Stage 6** | Isaac Gym/MuJoCo | Simulation | Online RL training |
| **Stage 7** | HandoverSim | [handoversim.github.io](https://handoversim.github.io/) | Human-robot handover |

---

## Next Steps

- [Locomotion Training](training_locomotion.md) - Detailed bipedal locomotion guide
- [Manipulation Training](training_manipulation.md) - Arm control and grasping
- [Whole-Body Control](training_whole_body.md) - Loco-manipulation integration
- [Training Datasets](training_datasets.md) - Dataset details and formats
- [Deployment](deployment.md) - Real robot deployment and safety
