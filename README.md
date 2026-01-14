# Vision-Language-Action (VLA) Model Training Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for training **Vision-Language-Action (VLA)** models for robotics and autonomous systems. Train models that understand visual scenes, follow natural language instructions, and predict robot actions.

## Key Features

- **Multiple VLA Architectures**: Custom VLA, OpenVLA-7B, SmolVLA-450M, Multi-Sensor VLA
- **Flexible Action Heads**: MLP, Gaussian, Diffusion, Transformer
- **Rich Training Paradigms**: BC, DAgger, GAIL, PPO, SAC, CQL, IQL, Decision Transformer
- **Multi-Sensor Fusion**: Camera, Depth Camera, LiDAR, Radar, IMU encoders
- **Embodiment Support**: Manipulation, Autonomous Driving, Humanoid Robots
- **Production Ready**: ONNX, TorchScript, OpenVINO, Triton export
- **Comprehensive Logging**: W&B integration, automatic best model saving, structured logs

## Quick Start

### Installation

```bash
git clone https://github.com/yhsong12345/Vision_Language_Action_Model_Practice.git
cd Vision_Language_Action_Model_Practice

conda create -n vla python=3.10 && conda activate vla
pip install -r requirements.txt
```

### Train Your First Model

```python
from model import create_vla_model
from train.il import BehavioralCloning
from train.datasets import create_lerobot_dataloader

# Create model
model = create_vla_model(
    vision_encoder="siglip-base",
    llm="qwen2-1.5b",
    action_dim=7,
)

# Train with W&B logging and best model saving
trainer = BehavioralCloning(
    model,
    learning_rate=1e-4,
    use_wandb=True,  # Enable W&B monitoring
)
trainer.train(dataloader)
# Best model automatically saved to output_dir/checkpoints/best_model.pt
```

### Run Demos

```bash
python run.py demo pusht      # PushT manipulation
python run.py demo mujoco     # MuJoCo RL
python run.py demo carla      # CARLA driving
python run.py demo inference  # Inference pipeline
```

## Training Pipeline (Staged Approach)

The VLA training follows a **multi-stage pipeline** inspired by LLaVA:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VLA Training Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage 1: VLM Pretraining (Optional - use pretrained VLM)                   │
│  │                                                                          │
│  │  ├── 1a. Vision-Language Alignment                                       │
│  │  │       └── Train projector to align vision encoder with LLM            │
│  │  │                                                                       │
│  │  └── 1b. Visual Instruction Tuning                                       │
│  │          └── Fine-tune on multimodal instruction data                    │
│  │                                                                          │
│  Stage 2: Action Head Training (Supervised Fine-tuning)                     │
│  │                                                                          │
│  │  └── Train action head on robot demonstrations                           │
│  │      ├── MLP / Gaussian MLP (simple, fast)                               │
│  │      ├── Diffusion Head (multi-modal actions)                            │
│  │      └── Transformer Head (action sequences)                             │
│  │                                                                          │
│  Stage 3: Policy Improvement                                                │
│  │                                                                          │
│  │  ├── 3a. Imitation-based                                                 │
│  │  │       ├── BC (Behavioral Cloning) - supervised baseline               │
│  │  │       ├── DAgger (Dataset Aggregation) - expert-in-the-loop           │
│  │  │       └── GAIL (Generative Adversarial IL) - learns implicit reward   │
│  │  │                                                                       │
│  │  ├── 3b. RL-based                                                        │
│  │  │       │                                                               │
│  │  │       ├── Online RL (requires simulator/environment)                  │
│  │  │       │   ├── PPO (Proximal Policy Optimization) - on-policy          │
│  │  │       │   ├── SAC (Soft Actor-Critic) - off-policy                    │
│  │  │       │   └── GRPO (Group Relative PO) - LLM-style optimization       │
│  │  │       │                                                               │
│  │  │       └── Offline RL (from static datasets)                           │
│  │  │           ├── CQL (Conservative Q-Learning) - penalizes OOD           │
│  │  │           ├── IQL (Implicit Q-Learning) - stable, no max              │
│  │  │           ├── TD3+BC - TD3 with BC regularization                     │
│  │  │           └── Decision Transformer - sequence modeling                │
│  │  │                                                                       │
│  │  └── 3c. Model-based                                                     │
│  │          │                                                               │
│  │          └── World Model                                                 │
│  │              ├── RSSM (Recurrent State-Space Model)                      │
│  │              ├── Latent Dynamics Learning                                │
│  │              ├── Imagination-based Planning                              │
│  │              └── Dreamer-style Training                                  │
│  │                                                                          │
│  Stage 4: Deployment                                                        │
│  │                                                                          │
│  │  ├── Simulator (CARLA, Isaac Sim, MuJoCo)                                │
│  │  └── Real Robot (ROS/ROS2 integration)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Vision-Language Alignment

**Goal**: Train the vision projector to align visual features with LLM embeddings.

```python
from model.vlm import VLMModel
from train.pretrain import VLMPretrainer, PretrainingDataset
from config import PretrainingConfig

# Create VLM model (no action head)
model = VLMModel(
    vision_model_name="google/siglip-base-patch16-224",
    llm_model_name="Qwen/Qwen2-1.5B-Instruct",
    projector_type="mlp",  # or "attention", "perceiver"
)

# Configure Stage 1
config = PretrainingConfig(
    output_dir="./pretrained_vlm",
    dataset_name="liuhaotian/LLaVA-Pretrain",
    stage1_epochs=1,
    stage1_lr=1e-3,
    freeze_vision=True,   # Freeze vision encoder
    freeze_llm=True,      # Freeze LLM, only train projector
)

# Train alignment
trainer = VLMPretrainer(model, config)
alignment_data = PretrainingDataset(
    "liuhaotian/LLaVA-Pretrain",
    model.image_processor,
    model.tokenizer,
)
trainer.train_stage1_alignment(alignment_data)
```

**What's trained**: Vision Projector only
**Dataset**: Image-caption pairs (e.g., LLaVA-Pretrain 558K)
**Duration**: ~1 epoch

---

### Stage 2: Visual Instruction Tuning

**Goal**: Fine-tune the LLM to follow visual instructions.

```python
# Configure Stage 2 (continuing from Stage 1)
config = PretrainingConfig(
    output_dir="./pretrained_vlm",
    instruction_dataset="liuhaotian/LLaVA-Instruct-150K",
    stage2_epochs=1,
    stage2_lr=2e-5,
    freeze_vision=True,   # Keep vision frozen
    freeze_llm=False,     # Unfreeze LLM for instruction tuning
)

# Train instruction following
instruction_data = PretrainingDataset(
    "liuhaotian/LLaVA-Instruct-150K",
    model.image_processor,
    model.tokenizer,
)
trainer.train_stage2_instruction(instruction_data)

# Save pretrained VLM
model.save_pretrained("./pretrained_vlm/vlm_final.pt")
```

**What's trained**: Vision Projector + LLM
**Dataset**: Visual instruction data (e.g., LLaVA-Instruct 150K)
**Duration**: ~1 epoch

---

### Stage 3: VLA Action Training

**Goal**: Add action head and train for robot control using pretrained VLM.

```python
from model.vla import VLAModel
from train.il import VLABehavioralCloning
from train.datasets import create_lerobot_dataloader

# Load pretrained VLM and add fresh action head
vla_model = VLAModel.from_pretrained_vlm(
    vlm_path="./pretrained_vlm/vlm_final.pt",
    action_dim=7,                    # Robot action dimension
    hidden_dim=512,                  # Action head hidden size
    action_chunk_size=1,             # Actions per prediction
    # Model config (must match pretrained VLM)
    vision_model_name="google/siglip-base-patch16-224",
    llm_model_name="Qwen/Qwen2-1.5B-Instruct",
    # Fine-tuning strategy
    freeze_vision=True,              # Keep vision frozen
    freeze_llm=False,                # Fine-tune LLM for actions
)

# Train with robot demonstrations
dataloader = create_lerobot_dataloader(
    dataset_name="lerobot/pusht",
    batch_size=32,
)

trainer = VLABehavioralCloning(
    model=vla_model,
    learning_rate=1e-4,
    use_wandb=True,
)
trainer.train(dataloader)

# Save final VLA model
vla_model.save_pretrained("./trained_vla/vla_final.pt")
```

**What's trained**: Action Head + LLM (optionally)
**Dataset**: Robot demonstration data (e.g., LeRobot, Open X-Embodiment)
**Methods**: BC, DAgger, RL fine-tuning

---

### Quick Start: Full Pipeline

```python
from train.pretrain import pretrain_vlm
from model.vla import VLAModel

# Run full VLM pretraining (Stage 1 + Stage 2)
vlm_model = pretrain_vlm(
    vision_model="google/siglip-base-patch16-224",
    llm_model="Qwen/Qwen2-1.5B-Instruct",
    alignment_dataset="liuhaotian/LLaVA-Pretrain",
    instruction_dataset="liuhaotian/LLaVA-Instruct-150K",
    output_dir="./pretrained_vlm",
)

# Load as VLA for action training (Stage 3)
vla_model = VLAModel.from_pretrained_vlm(
    vlm_path="./pretrained_vlm/vlm_final.pt",
    action_dim=7,
)
```

---

## Deployment

### Stage 4: Model Export & Deployment

After training, export your VLA model for production deployment.

#### 4.1 Export to Production Formats

```python
from model.utils.export import (
    ONNXExporter,
    TorchScriptExporter,
    OpenVINOExporter,
    TritonExporter,
)

# Load trained VLA
vla_model = VLAModel.from_pretrained("./trained_vla/vla_final.pt")
vla_model.eval()

# Export to ONNX (cross-platform)
onnx_exporter = ONNXExporter(vla_model)
onnx_exporter.export("./deployed/vla_model.onnx")

# Export to TorchScript (PyTorch production)
ts_exporter = TorchScriptExporter(vla_model)
ts_exporter.export("./deployed/vla_model.pt")

# Export to OpenVINO (Intel hardware)
ov_exporter = OpenVINOExporter(vla_model)
ov_exporter.export("./deployed/vla_openvino/")

# Export to Triton (NVIDIA inference server)
triton_exporter = TritonExporter(vla_model)
triton_exporter.export("./deployed/triton_model/")
```

#### 4.2 Quantization for Edge Deployment

```python
from model.utils.export import QuantizationExporter

# INT8 quantization for faster inference
quant_exporter = QuantizationExporter(vla_model)
quant_exporter.export(
    "./deployed/vla_quantized.onnx",
    quantization_type="int8",  # or "fp16"
)
```

#### 4.3 Inference Pipeline

```python
from model.vla import VLAModel
from PIL import Image

# Load model for inference
model = VLAModel.from_pretrained("./trained_vla/vla_final.pt")
model.eval()
model.cuda()

# Run inference
image = Image.open("camera_frame.jpg")
instruction = "Pick up the red block and place it on the blue plate"

action = model.predict_action(image, instruction)
# action: tensor([x, y, z, rx, ry, rz, gripper])
```

#### 4.4 ROS Integration

```python
# See integration/ros/ for ROS node implementation
from integration.ros import VLAActionServer

# Start ROS action server
server = VLAActionServer(
    model_path="./trained_vla/vla_final.pt",
    action_topic="/vla/action",
    image_topic="/camera/rgb/image_raw",
)
server.run()
```

---

## Pipeline Summary

| Stage | Goal | What's Trained | What's Frozen | Dataset |
|-------|------|----------------|---------------|---------|
| **1a** | Vision-Language Alignment | Projector | Vision + LLM | Image-caption pairs |
| **1b** | Visual Instruction Tuning | Projector + LLM | Vision | Visual QA/Instructions |
| **2** | Action Head Training (BC) | Action Head | Vision + Projector + LLM (or LoRA) | Robot demonstrations |
| **3** | Policy Improvement (RL/IL) | Action Head (+LLM lightly) | Vision + Projector | RL env / more demos |
| **4** | Deployment | - | - | - |

---

## Architecture

```
                    Image + Instruction
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    ┌───────────────┐             ┌───────────────┐
    │    Vision     │             │   Language    │
    │   Encoder     │             │   Tokenizer   │
    │ (SigLIP/CLIP) │             │   (Qwen2)     │
    └───────┬───────┘             └───────┬───────┘
            │                             │
            ▼                             │
    ┌───────────────┐                     │
    │    Vision     │                     │
    │   Projector   │                     │
    └───────┬───────┘                     │
            │                             │
            └──────────────┬──────────────┘
                           ▼
                   ┌───────────────┐
                   │  LLM Backbone │
                   │ (Qwen2/LLaMA) │
                   └───────┬───────┘
                           ▼
                   ┌───────────────┐
                   │  Action Head  │
                   │(MLP/Diffusion)│
                   └───────┬───────┘
                           ▼
                     Robot Actions
```

## Project Structure

```
├── config/                  # Configuration management
│   ├── model_config.py      # VLA model configurations
│   ├── training_config.py   # Training hyperparameters
│   ├── dataset_config.py    # Dataset settings
│   └── hydra_config.py      # Hydra/OmegaConf integration
│
├── model/                   # Model components
│   ├── vlm/                 # Vision-Language backbone
│   ├── vla/                 # VLA implementations
│   ├── action_head/         # Action prediction heads
│   ├── sensor/              # Depth, LiDAR, Radar, IMU encoders
│   ├── fusion/              # Multi-modal fusion
│   ├── temporal/            # History & memory modules
│   ├── world_model/         # RSSM, dynamics models
│   ├── safety/              # Safety shield, constraints
│   ├── embodiment/          # Driving, Humanoid models
│   └── utils/               # Utilities & exporters
│       ├── layers.py        # Shared layers (PositionalEncoding, MLP)
│       ├── device_utils.py  # Device management
│       ├── checkpoint_utils.py
│       └── export/          # Model export (ONNX, TorchScript, etc.)
│
├── train/                   # Training pipelines
│   ├── base_trainer.py      # Unified trainer base classes
│   ├── pretrain/            # VLM pretraining
│   ├── finetune/            # Supervised fine-tuning
│   ├── il/                  # BC, DAgger, GAIL
│   ├── online_rl/           # PPO, SAC, GRPO
│   ├── offline_rl/          # CQL, IQL, TD3+BC, DT
│   ├── world_model/         # Dreamer-style training
│   ├── embodiment/          # Task-specific training
│   ├── datasets/            # Dataset loaders
│   └── utils/               # Training utilities
│       ├── logging.py       # ExperimentLogger, W&B, best model tracking
│       ├── buffers.py       # RolloutBuffer, ReplayBuffer
│       └── evaluation.py    # Policy evaluation
│
├── eval/                    # Evaluation & benchmarks
├── integration/             # ROS, simulators, experiment management
├── examples/                # Demo scripts
├── docs/                    # Documentation
├── tests/                   # Unit tests
└── scripts/                 # Utility scripts
```

## Training with Logging & Best Model Saving

All trainers automatically:
- Save **only the best model** based on validation loss or reward
- Log hyperparameters, model info, and training metrics
- Support **W&B** for real-time monitoring

```python
from train.il import VLABehavioralCloning
from config import ILConfig

config = ILConfig(
    bc_epochs=100,
    learning_rate=1e-4,
    batch_size=32,
    output_dir="./experiments/my_run"
)

trainer = VLABehavioralCloning(
    model=model,
    config=config,
    use_wandb=True,              # Enable W&B
    wandb_project="vla-training",
    experiment_name="pusht_bc_v1",
    dataset_name="pusht"
)

trainer.train(train_dataloader, val_dataloader)
```

### Log Files Created

```
experiments/my_run/
├── checkpoints/
│   ├── best_model.pt        # Best model checkpoint
│   └── last_model.pt        # Last epoch checkpoint
├── config.json              # All hyperparameters
├── model_info.json          # Model architecture info
├── training_log.json        # Complete training history
└── experiment.log           # Human-readable log
```

### W&B Metrics Tracked

- `train/loss`, `train/grad_norm`, `train/lr`
- `val/loss`, `val/accuracy`, `val/reward`
- `epoch_time`, `total_time`
- Best model uploaded as artifact

## Training Methods

### Imitation Learning

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **BC** | Behavioral Cloning | High-quality demonstrations |
| **DAgger** | Dataset Aggregation | Online expert queries |
| **GAIL** | Generative Adversarial IL | No reward function |

### Reinforcement Learning

| Algorithm | Type | Use Case |
|-----------|------|----------|
| **PPO** | Online | Stable simulator training |
| **SAC** | Online | Sample-efficient control |
| **GRPO** | Online | VLA fine-tuning |
| **CQL** | Offline | Mixed-quality data |
| **IQL** | Offline | Suboptimal demonstrations |
| **TD3+BC** | Offline | Near-expert data |
| **DT** | Offline | Long-horizon tasks |

### Action Heads

| Type | Benefits | Best For |
|------|----------|----------|
| **MLP** | Fast, simple | Single-mode actions |
| **Gaussian** | Uncertainty estimation | RL training |
| **Diffusion** | Multi-modal distributions | Precise manipulation |
| **Transformer** | Temporal modeling | Action sequences |

## Supported Datasets

| Domain | Datasets |
|--------|----------|
| **Manipulation** | LeRobot (PushT, ALOHA, xArm), Open X-Embodiment |
| **Driving** | nuScenes, Waymo, CARLA |
| **RGB-D / Depth** | NYU Depth V2, ScanNet, SUN RGB-D, GraspNet, ClearGrasp |
| **Offline RL** | D4RL, RoboMimic |

## Model Export

```python
from model.utils import export_model

# Export to multiple formats
exported = export_model(
    model,
    formats=["onnx", "torchscript", "openvino", "triton"],
    output_dir="./exported",
)
```

**Supported formats:**
- ONNX (cross-platform)
- TorchScript (PyTorch production)
- OpenVINO (Intel hardware)
- Triton (NVIDIA inference server)
- Quantized (INT8/FP16)

## CLI Reference

```bash
# Training
python run.py train --preset <name> [--epochs N] [--lr RATE]

# Evaluation
python run.py eval --checkpoint <path> [--episodes N]

# Inference
python run.py infer --image <path> --instruction "text"
python run.py infer --video <path> --instruction "text"

# Export
python run.py export --checkpoint <path> --format onnx,torchscript

# Demos
python run.py demo [pusht|mujoco|carla|inference]

# List presets
python run.py list
```

## Hardware Requirements

| Model | GPU Memory | Recommended |
|-------|------------|-------------|
| SmolVLA (450M) | 8GB | RTX 3080 |
| Custom VLA (1.5B) | 16GB | RTX 4080 |
| OpenVLA (7B) | 24GB | RTX 4090 |
| Multi-Sensor VLA | 32GB | A6000 |

## Documentation

- [Architecture Guide](docs/architecture.md) - Detailed model architecture
- [Usage Guide](docs/usage.md) - Step-by-step tutorials
- [Training Recipes](docs/training_recipes.md) - Task-specific configurations

## References

- [OpenVLA](https://openvla.github.io/) - Open-source VLA model
- [RT-2](https://robotics-transformer2.github.io/) - Google's VLA model
- [LeRobot](https://github.com/huggingface/lerobot) - HuggingFace robot learning
- [Dreamer](https://danijar.com/project/dreamer/) - World model-based RL
- [Decision Transformer](https://arxiv.org/abs/2106.01345) - Sequence modeling for RL

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with PyTorch, Transformers, and LeRobot**
