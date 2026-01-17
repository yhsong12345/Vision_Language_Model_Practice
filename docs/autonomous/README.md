# VLA Training Documentation for Autonomous Driving

This folder contains comprehensive documentation for training Vision-Language-Action (VLA) models for autonomous driving and related tasks.

## Documentation Index

| Document | Description |
|----------|-------------|
| [embodiment.md](embodiment.md) | **Embodiment Documentation** - DrivingVLA architecture, BEV encoding, trajectory prediction |
| [training_vla_recipe.md](training_vla_recipe.md) | **Complete VLA Training Pipeline** - End-to-end training from VLM pretraining to deployment |
| [training_datasets.md](training_datasets.md) | **Dataset Documentation** - All datasets used in training with sources and formats |
| [training_sensors.md](training_sensors.md) | **Sensor Training** - Camera, LiDAR, radar, IMU processing and sensor fusion |
| [training_world_model.md](training_world_model.md) | **World Model Training** - RSSM, latent dynamics, imagination-based planning |
| [training_temporal_module.md](training_temporal_module.md) | **Temporal Module** - Temporal encoders, history encoding, memory buffers |
| [training_imitation_learning.md](training_imitation_learning.md) | **Imitation Learning** - BC, DAgger, GAIL methods |
| [training_reinforcement_learning.md](training_reinforcement_learning.md) | **Reinforcement Learning** - Offline (IQL, CQL, TD3+BC, DT) and Online (PPO, SAC, GRPO) RL |
| [training_online_rl_carla.md](training_online_rl_carla.md) | **Online RL with CARLA** - Docker setup, simulator configuration, PPO/SAC training |
| [deployment.md](deployment.md) | **Deployment & Export** - ONNX, TorchScript, OpenVINO, Triton, quantization |

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VLA Training Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: VLM Foundation                                                     │
│  ├── 1a: Vision-Language Alignment (Projector training)                     │
│  └── 1b: Visual Instruction Tuning (LLM fine-tuning)                        │
│                                                                              │
│  Stage 2: Action Head Training                                               │
│  └── Supervised fine-tuning with robot demonstration data                   │
│                                                                              │
│  Stage 3: Policy Improvement                                                 │
│  ├── 3a: Imitation Learning (BC, DAgger, GAIL)                              │
│  ├── 3b: Offline RL (CQL, IQL, TD3+BC, Decision Transformer)               │
│  ├── 3c: Online RL (PPO, SAC, GRPO)                                         │
│  └── 3d: World Model Training (RSSM, imagination-based planning)            │
│                                                                              │
│  Stage 4: Deployment                                                         │
│  └── Model export (ONNX, TorchScript, OpenVINO, Triton)                     │
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
    --output-dir ./output/stage1a

# Stage 1b: Instruction Tuning
python train/pretrain/vlm_pretrainer.py \
    --instruction-dataset liuhaotian/LLaVA-Instruct-150K \
    --output-dir ./output/stage1b
```

### 2. Action Head Fine-tuning

```bash
python train/finetune/vla_finetuner.py \
    --pretrained-vlm ./output/stage1b/best \
    --dataset nuscenes \
    --action-dim 3 \
    --output-dir ./output/stage2
```

### 3. Policy Improvement

```bash
# Imitation Learning
python train/il/behavioral_cloning.py --output_dir ./output/stage3a

# Offline RL
python train/offline_rl/iql_trainer.py --output_dir ./output/stage3b

# Online RL (requires simulator)
python train/online_rl/ppo_trainer.py --output_dir ./output/stage3c
```

### 4. Deployment

```bash
# ONNX export
python -c "
from model.utils.export import ONNXExporter
exporter = ONNXExporter()
exporter.export('./output/best_model', './exported/model.onnx')
"
```

## Datasets Summary

| Training Stage | Dataset | Source |
|----------------|---------|--------|
| VLM Alignment | LLaVA-Pretrain (558K) | [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |
| Instruction Tuning | LLaVA-Instruct-150K | [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150k) |
| Action Head (Sim) | CARLA Autopilot | Local |
| Action Head (Real) | nuScenes | [nuscenes.org](https://www.nuscenes.org/) |
| Manipulation | LeRobot PushT/ALOHA | [HuggingFace](https://huggingface.co/lerobot) |
| Multi-Robot | Open X-Embodiment | [HuggingFace](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) |
| Offline RL | D4RL | [GitHub](https://github.com/rail-berkeley/d4rl) |

## Training Scripts

All training scripts are located in the `scripts/` directory:

| Script | Description |
|--------|-------------|
| `run_pretrain.sh` | VLM pretraining |
| `run_finetune.sh` | Action head fine-tuning |
| `run_il.sh` | Imitation learning |
| `run_offline_rl_iql.sh` | IQL offline RL |
| `run_offline_rl_cql.sh` | CQL offline RL |
| `run_online_rl_ppo.sh` | PPO online RL |
| `run_online_rl_sac.sh` | SAC online RL |
| `run_world_model.sh` | World model training |
| `run_driving_vla.sh` | End-to-end driving VLA |

## Key Files Reference

```
train/
├── pretrain/
│   ├── vlm_pretrainer.py       # VLM pretraining
│   ├── alignment_trainer.py    # Vision-language alignment
│   └── instruction_trainer.py  # Instruction tuning
├── finetune/
│   └── vla_finetuner.py        # VLA fine-tuning
├── il/
│   ├── behavioral_cloning.py   # BC
│   ├── dagger.py               # DAgger
│   └── gail.py                 # GAIL
├── offline_rl/
│   ├── iql_trainer.py          # IQL
│   ├── cql_trainer.py          # CQL
│   ├── td3_bc_trainer.py       # TD3+BC
│   └── decision_transformer.py # DT
├── online_rl/
│   ├── ppo_trainer.py          # PPO
│   ├── sac_trainer.py          # SAC
│   └── grpo_trainer.py         # GRPO
├── world_model/
│   └── train_world_model.py    # World model
└── datasets/
    ├── lerobot_dataset.py      # LeRobot datasets
    ├── openx_dataset.py        # Open X-Embodiment
    └── driving_dataset.py      # Driving datasets
```

## Getting Help

For issues or questions:
- Check the specific document for your training stage
- Review the example scripts in `examples/`
- Run tests in `tests/` to verify your setup
