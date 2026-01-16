# VLA Training Recipe for Autonomous Driving

This document provides a comprehensive guide to training Vision-Language-Action (VLA) models for autonomous driving tasks. The training pipeline follows a staged approach from VLM foundation to deployment-ready models.

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

---

## Stage 1: VLM Foundation

The VLM (Vision-Language Model) foundation stage creates a model that can understand visual inputs and relate them to language. This follows the LLaVA paradigm with two sub-stages.

### Stage 1a: Vision-Language Alignment

**Purpose**: Train the vision projector to align visual features with the LLM embedding space.

**What's Trained**: Vision Projector only
**What's Frozen**: Vision Encoder + LLM

#### Dataset Requirements

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 1a: VLM Alignment** | LLaVA-Pretrain | [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | 558K image-caption pairs for vision-language alignment |

#### Training Script

```bash
# Using the VLM Pretrainer
python train/pretrain/vlm_pretrainer.py \
    --vision-model google/siglip-base-patch16-224 \
    --llm-model Qwen/Qwen2-1.5B-Instruct \
    --alignment-dataset liuhaotian/LLaVA-Pretrain \
    --alignment-epochs 1 \
    --alignment-lr 1e-3 \
    --batch-size 32 \
    --output-dir ./output/stage1a_alignment \
    --mixed-precision bf16 \
    --use-wandb
```

#### Using SLURM for Distributed Training

```bash
# Submit distributed pretraining job
sbatch scripts/run_pretrain.sh \
    --vision-model google/siglip-base-patch16-224 \
    --llm-model Qwen/Qwen2-1.5B-Instruct \
    --alignment-dataset liuhaotian/LLaVA-Pretrain \
    --output-dir ./output/stage1a_alignment
```

#### Key Configuration

```python
from config.training_config import PretrainingConfig

config = PretrainingConfig(
    # Stage 1a: Alignment
    alignment_epochs=1,
    alignment_lr=1e-3,           # High LR for projector only

    # General settings
    batch_size=32,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    mixed_precision="bf16",

    # Logging
    logging_steps=10,
    save_steps=1000,
)
```

### Stage 1b: Visual Instruction Tuning

**Purpose**: Train the LLM to follow visual instructions and generate appropriate responses.

**What's Trained**: Vision Projector + LLM
**What's Frozen**: Vision Encoder

#### Dataset Requirements

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 1b: Instruction Tuning** | LLaVA-Instruct-150K | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150k) | 150K visual instruction-response pairs |

#### Training Script

```bash
python train/pretrain/vlm_pretrainer.py \
    --vision-model google/siglip-base-patch16-224 \
    --llm-model Qwen/Qwen2-1.5B-Instruct \
    --instruction-dataset liuhaotian/LLaVA-Instruct-150K \
    --instruction-epochs 3 \
    --instruction-lr 2e-5 \
    --batch-size 16 \
    --output-dir ./output/stage1b_instruction \
    --mixed-precision bf16 \
    --use-wandb
```

---

## Stage 2: Action Head Training

**Purpose**: Add and train an action head to predict robot/vehicle controls from VLM features.

**What's Trained**: Action Head (+ optionally Vision Projector and LLM)
**What's Frozen**: Vision Encoder (+ optionally LLM)

### Action Head Types

| Action Head | Use Case | Output | Best For |
|-------------|----------|--------|----------|
| MLP | Simple, fast | Deterministic actions | Fast inference, simple tasks |
| Gaussian | Stochastic | Mean + variance | RL training, exploration |
| Diffusion | Multi-modal | Denoised samples | Precise manipulation, complex actions |
| Transformer | Sequential | Action chunks | Long-horizon, temporal consistency |
| GPT | Autoregressive | Token-by-token | Language-model style action generation |

#### Action Head Details

**1. MLPActionHead** ([mlp_action_head.py](../../model/action_head/mlp_action_head.py))

Simple multi-layer perceptron for deterministic predictions.

```python
from model.action_head import MLPActionHead

head = MLPActionHead(
    input_dim=1536,        # VLM feature dimension
    action_dim=7,          # Robot action dimensions
    hidden_dim=512,        # Hidden layer size
    num_layers=3,          # Number of MLP layers
    activation="gelu",     # Activation function
    dropout=0.1,           # Dropout probability
)
```

**2. GaussianMLPActionHead** ([mlp_action_head.py](../../model/action_head/mlp_action_head.py))

Stochastic predictions with uncertainty estimation. Essential for RL training.

```python
from model.action_head import GaussianMLPActionHead

head = GaussianMLPActionHead(
    input_dim=1536,
    action_dim=7,
    hidden_dim=512,
    min_std=0.01,          # Minimum standard deviation
    max_std=1.0,           # Maximum standard deviation
)
# Output: mean, std, sampled_action, log_prob
```

**3. DiffusionActionHead** ([diffusion_action_head.py](../../model/action_head/diffusion_action_head.py))

DDPM-based multi-modal action generation. Supports complex, multi-modal action distributions.

```python
from model.action_head import DiffusionActionHead

head = DiffusionActionHead(
    input_dim=1536,
    action_dim=7,
    hidden_dim=256,
    num_diffusion_steps=100,  # Diffusion timesteps
    chunk_size=10,            # Action sequence length
)
# Supports DDIM for fast sampling (20-100 steps)
```

**4. TransformerActionHead** ([transformer_action_head.py](../../model/action_head/transformer_action_head.py))

Autoregressive transformer decoder for sequential action generation.

```python
from model.action_head import TransformerActionHead

head = TransformerActionHead(
    input_dim=1536,
    action_dim=7,
    hidden_dim=512,
    num_layers=4,             # Transformer layers
    num_heads=8,              # Attention heads
    chunk_size=10,            # Max sequence length
    use_causal_mask=True,     # Autoregressive generation
)
```

**5. GPTActionHead** ([transformer_action_head.py](../../model/action_head/transformer_action_head.py))

GPT-style decoder-only transformer with learned positional and type embeddings.

```python
from model.action_head import GPTActionHead

head = GPTActionHead(
    input_dim=1536,
    action_dim=7,
    hidden_dim=512,
    num_layers=6,             # Transformer layers
    num_heads=8,              # Attention heads
    chunk_size=10,            # Max sequence length
)
# Uses learned positional + type embeddings
# More similar to language model architecture
```

#### Action Head Selection Guide

| Scenario | Recommended Head | Reason |
|----------|-----------------|--------|
| Real-time inference | MLP | Fastest, deterministic |
| RL training (PPO/SAC) | Gaussian | Provides log_prob for policy gradients |
| Complex manipulation | Diffusion | Multi-modal action distributions |
| Trajectory planning | Transformer/GPT | Temporal consistency in action sequences |
| Behavior cloning | MLP or Transformer | Direct action prediction |

### Dataset Requirements for Autonomous Driving

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 2: Action Head (Sim)** | CARLA Autopilot | Local/Custom | Simulated urban driving with perfect labels |
| **Stage 2: Action Head (Real)** | nuScenes | [nuscenes.org](https://www.nuscenes.org/) | 1000 real-world scenes, 6 cameras, LiDAR, radar |
| **Stage 2: Action Head (Real)** | Waymo Open | [waymo.com/open](https://waymo.com/open) | Large-scale real driving with rich annotations |
| **Stage 2: Action Head (Real)** | comma.ai 2k19 | [commaai/comma2k19](https://huggingface.co/datasets/commaai/comma2k19) | Real driving with CAN bus steering/throttle |

### Training Script

```bash
# Fine-tune VLA on driving data
python train/finetune/vla_finetuner.py \
    --vision-model google/siglip-base-patch16-224 \
    --llm-model Qwen/Qwen2-1.5B-Instruct \
    --pretrained-vlm ./output/stage1b_instruction/best \
    --dataset nuscenes \
    --action-dim 3 \
    --num-epochs 10 \
    --learning-rate 1e-4 \
    --freeze-vision \
    --output-dir ./output/stage2_action_head \
    --use-wandb
```

### Using LoRA for Efficient Fine-tuning

```bash
python train/finetune/vla_finetuner.py \
    --pretrained-vlm ./output/stage1b_instruction/best \
    --dataset nuscenes \
    --use-lora \
    --lora-r 32 \
    --lora-alpha 64 \
    --freeze-vision \
    --freeze-llm \
    --output-dir ./output/stage2_lora
```

### Key Configuration

```python
from config.training_config import FineTuningConfig

config = FineTuningConfig(
    # Fine-tuning settings
    num_epochs=10,
    learning_rate=1e-4,
    batch_size=8,

    # Freezing strategy
    freeze_vision=True,        # Always freeze vision encoder
    freeze_llm=False,          # Fine-tune LLM for driving domain

    # LoRA (optional)
    use_lora=True,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.1,

    # Action head
    action_head_type="gaussian",  # For RL compatibility
    action_dim=3,                 # [steering, throttle, brake]
)
```

---

## Stage 3: Policy Improvement

After basic action head training, improve the policy using various learning paradigms.

### Stage 3a: Imitation Learning

Use expert demonstrations to refine the policy.

#### Behavioral Cloning (BC)

**Best for**: Initial policy, quick training

```bash
python train/il/behavioral_cloning.py \
    --env carla \
    --bc_epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --output_dir ./output/stage3a_bc
```

#### DAgger (Dataset Aggregation)

**Best for**: Addressing distribution shift

```bash
python train/il/dagger.py \
    --env carla \
    --dagger_iterations 10 \
    --bc_epochs 50 \
    --output_dir ./output/stage3a_dagger
```

#### GAIL (Generative Adversarial Imitation Learning)

**Best for**: Learning from imperfect demonstrations

```bash
python train/il/gail.py \
    --env carla \
    --gail_epochs 100 \
    --output_dir ./output/stage3a_gail
```

### Stage 3b: Offline RL

Learn from static datasets without environment interaction.

#### Dataset Requirements

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **Stage 3b: Offline RL** | D4RL | [rail-berkeley/d4rl](https://github.com/rail-berkeley/d4rl) | Standard offline RL benchmarks |
| **Stage 3b: Offline RL** | Visual D4RL | [Visual D4RL](https://github.com/conglu1997/v-d4rl) | Pixel-based variants of D4RL |
| **Stage 3b: Offline RL** | RoboMimic | [robomimic.github.io](https://robomimic.github.io/) | Robot manipulation trajectories |

#### IQL (Implicit Q-Learning)

**Best for**: Conservative offline learning

```bash
python train/offline_rl/iql_trainer.py \
    --dataset hopper-medium-v2 \
    --num_epochs 100 \
    --batch_size 256 \
    --learning_rate 3e-4 \
    --expectile 0.7 \
    --temperature 3.0 \
    --output_dir ./output/stage3b_iql
```

#### CQL (Conservative Q-Learning)

**Best for**: Avoiding OOD actions

```bash
python train/offline_rl/cql_trainer.py \
    --dataset hopper-medium-v2 \
    --num_epochs 100 \
    --cql_alpha 5.0 \
    --output_dir ./output/stage3b_cql
```

#### Decision Transformer

**Best for**: Sequence-based policy, goal-conditioned behavior

```bash
python train/offline_rl/decision_transformer.py \
    --dataset hopper-medium-v2 \
    --context_length 20 \
    --n_layer 3 \
    --n_head 1 \
    --output_dir ./output/stage3b_dt
```

### Stage 3c: Online RL

Improve policy through environment interaction (requires simulator).

#### PPO (Proximal Policy Optimization)

**Best for**: Stable training, general purpose

```bash
python train/online_rl/ppo_trainer.py \
    --env CartPole-v1 \
    --total_timesteps 100000 \
    --rollout_steps 2048 \
    --ppo_epochs 4 \
    --ppo_clip_range 0.2 \
    --learning_rate 3e-4 \
    --output_dir ./output/stage3c_ppo \
    --use-wandb
```

For CARLA driving:

```bash
# Use the CARLA integration
python examples/carla_demo.py \
    --mode train \
    --algorithm ppo \
    --total_timesteps 1000000 \
    --output_dir ./output/stage3c_ppo_carla
```

#### SAC (Soft Actor-Critic)

**Best for**: Sample efficiency, continuous control

```bash
python train/online_rl/sac_trainer.py \
    --env HalfCheetah-v4 \
    --total_timesteps 1000000 \
    --buffer_size 1000000 \
    --learning_rate 3e-4 \
    --output_dir ./output/stage3c_sac
```

#### GRPO (Group Relative Policy Optimization)

**Best for**: Multi-objective optimization, LLM-based policies

```bash
python train/online_rl/grpo_trainer.py \
    --env driving \
    --group_size 4 \
    --num_iterations 100 \
    --output_dir ./output/stage3c_grpo
```

### Stage 3d: World Model Training

Learn environment dynamics for planning and imagination-based training.

See [training_world_model.md](training_world_model.md) for detailed documentation.

```bash
python train/world_model/train_world_model.py \
    --latent_dim 256 \
    --hidden_dim 512 \
    --imagination_horizon 15 \
    --num_epochs 100 \
    --output_dir ./output/stage3d_world_model
```

---

## Stage 4: Deployment

Export trained models for production inference.

See [deployment.md](deployment.md) for detailed documentation.

### Quick Export

```bash
# ONNX export
python -c "
from model.utils.export import ONNXExporter
exporter = ONNXExporter()
exporter.export('./output/best_model', './exported/model.onnx')
"

# TorchScript export
python -c "
from model.utils.export import TorchScriptExporter
exporter = TorchScriptExporter()
exporter.export('./output/best_model', './exported/model.pt')
"
```

---

## Complete Training Pipeline Example

Here's a complete example training a VLA for autonomous driving:

```bash
#!/bin/bash
# complete_training_pipeline.sh

OUTPUT_BASE="./output/autonomous_vla"

# Stage 1a: Vision-Language Alignment
echo "Stage 1a: Vision-Language Alignment"
python train/pretrain/vlm_pretrainer.py \
    --vision-model google/siglip-base-patch16-224 \
    --llm-model Qwen/Qwen2-1.5B-Instruct \
    --alignment-dataset liuhaotian/LLaVA-Pretrain \
    --alignment-epochs 1 \
    --output-dir $OUTPUT_BASE/stage1a \
    --use-wandb

# Stage 1b: Instruction Tuning
echo "Stage 1b: Instruction Tuning"
python train/pretrain/vlm_pretrainer.py \
    --vision-model google/siglip-base-patch16-224 \
    --llm-model Qwen/Qwen2-1.5B-Instruct \
    --instruction-dataset liuhaotian/LLaVA-Instruct-150K \
    --instruction-epochs 3 \
    --output-dir $OUTPUT_BASE/stage1b \
    --use-wandb

# Stage 2: Action Head Fine-tuning
echo "Stage 2: Action Head Training"
python train/finetune/vla_finetuner.py \
    --pretrained-vlm $OUTPUT_BASE/stage1b/best \
    --dataset nuscenes \
    --action-dim 3 \
    --num-epochs 10 \
    --freeze-vision \
    --use-lora \
    --output-dir $OUTPUT_BASE/stage2 \
    --use-wandb

# Stage 3a: Behavioral Cloning Refinement
echo "Stage 3a: Behavioral Cloning"
python train/il/behavioral_cloning.py \
    --model-path $OUTPUT_BASE/stage2/best \
    --dataset nuscenes \
    --bc_epochs 50 \
    --output_dir $OUTPUT_BASE/stage3a_bc

# Stage 3b: Offline RL (IQL)
echo "Stage 3b: Offline RL"
python train/offline_rl/iql_trainer.py \
    --model-path $OUTPUT_BASE/stage3a_bc/best \
    --dataset driving-expert-v1 \
    --num_epochs 100 \
    --output_dir $OUTPUT_BASE/stage3b_iql

# Stage 4: Export
echo "Stage 4: Export"
python -c "
from model.utils.export import ONNXExporter
exporter = ONNXExporter()
exporter.export('$OUTPUT_BASE/stage3b_iql/best', '$OUTPUT_BASE/exported/model.onnx')
"

echo "Training complete!"
```

---

## Recommended Configurations by Task

### Urban Driving (nuScenes-style)

```python
# Multi-camera, LiDAR fusion
config = {
    "vision_model": "google/siglip-large-patch16-256",
    "llm_model": "Qwen/Qwen2-7B-Instruct",
    "action_head": "gaussian",
    "action_dim": 3,  # steering, throttle, brake
    "cameras": ["front", "front_left", "front_right", "back"],
    "use_lidar": True,
    "sensor_fusion": "cross_attention",
}
```

### Highway Driving (comma.ai-style)

```python
# Single front camera
config = {
    "vision_model": "google/siglip-base-patch16-224",
    "llm_model": "Qwen/Qwen2-1.5B-Instruct",
    "action_head": "mlp",
    "action_dim": 2,  # steering, throttle
    "cameras": ["front"],
    "use_lidar": False,
}
```

### Simulation (CARLA)

```python
# Full sensor suite with online RL
config = {
    "vision_model": "google/siglip-base-patch16-224",
    "llm_model": "Qwen/Qwen2-1.5B-Instruct",
    "action_head": "gaussian",
    "action_dim": 3,
    "training_method": "ppo",  # Online RL in simulation
    "world_model": True,       # Learn dynamics
}
```

---

## Monitoring and Debugging

### Weights & Biases Integration

All training scripts support W&B logging:

```bash
# Enable W&B
python train/finetune/vla_finetuner.py \
    --use-wandb \
    --wandb-project vla-autonomous \
    --experiment-name driving_v1
```

### Key Metrics to Monitor

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| train/loss | 0.1 - 2.0 | Should decrease steadily |
| val/loss | 0.2 - 3.0 | Should track train loss |
| train/lr | 1e-5 - 1e-3 | Follows schedule |
| eval/mean_reward | Task-dependent | Higher is better |

### Common Issues

1. **Loss spikes**: Reduce learning rate or increase gradient clipping
2. **Slow convergence**: Increase batch size or learning rate
3. **Overfitting**: Add regularization (dropout, weight decay)
4. **OOM errors**: Reduce batch size, enable gradient checkpointing

---

## Next Steps

- [Training Datasets](training_datasets.md) - Detailed dataset documentation
- [Training World Model](training_world_model.md) - RSSM and imagination-based training
- [Imitation Learning](training_imitation_learning.md) - BC, DAgger, GAIL details
- [Reinforcement Learning](training_reinforcement_learning.md) - Offline and online RL methods
- [Deployment](deployment.md) - Model export and production deployment
