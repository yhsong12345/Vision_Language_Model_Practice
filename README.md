# Vision-Language-Action (VLA) Model Training Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for training **Vision-Language-Action (VLA)** models for robotics and autonomous systems. Train models that understand visual scenes, follow natural language instructions, and predict robot actions.

## Key Features

- **Multiple VLA Architectures**: Custom VLA, OpenVLA-7B, SmolVLA-450M, Multi-Sensor VLA
- **Flexible Action Heads**: MLP, Gaussian, Diffusion, Transformer
- **Rich Training Paradigms**: BC, DAgger, GAIL, PPO, SAC, CQL, IQL, Decision Transformer
- **Multi-Sensor Fusion**: Camera, LiDAR, Radar, IMU encoders
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
│   ├── sensor/              # LiDAR, Radar, IMU encoders
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
