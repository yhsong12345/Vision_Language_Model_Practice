# Reinforcement Learning for VLA

This document covers reinforcement learning methods for training VLA models, including both offline RL (learning from static datasets) and online RL (learning from environment interaction).

## Overview

Reinforcement learning optimizes policies through trial-and-error, maximizing cumulative rewards. For VLA training:

- **Offline RL**: Learn from pre-collected datasets without environment interaction
- **Online RL**: Learn through direct interaction with simulators or environments

```
┌────────────────────────────────────────────────────────────────────┐
│                    Reinforcement Learning Methods                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                        Offline RL                              │ │
│  │  Learn from static datasets without environment interaction   │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────────────┐ │ │
│  │  │   CQL   │ │   IQL   │ │ TD3+BC  │ │Decision Transformer │ │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └──────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                        Online RL                               │ │
│  │  Learn through environment interaction (simulator required)   │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                         │ │
│  │  │   PPO   │ │   SAC   │ │  GRPO   │                         │ │
│  │  └─────────┘ └─────────┘ └─────────┘                         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Requirements

| Training Method | Dataset | Public Source | Description |
|-----------------|---------|---------------|-------------|
| **Offline RL** | D4RL | [rail-berkeley/d4rl](https://github.com/rail-berkeley/d4rl) | Standard benchmarks (Hopper, Walker, etc.) |
| **Offline RL** | Visual D4RL | [v-d4rl](https://github.com/conglu1997/v-d4rl) | Pixel-based offline RL |
| **Offline RL** | RoboMimic | [robomimic.github.io](https://robomimic.github.io/) | Robot manipulation trajectories |
| **Online RL** | CARLA Simulator | [carla.org](https://carla.org/) | Urban driving simulation |
| **Online RL** | MuJoCo | [mujoco.org](https://mujoco.org/) | Physics simulation |
| **Online RL** | Isaac Gym | [NVIDIA](https://developer.nvidia.com/isaac-gym) | GPU-accelerated simulation |

---

## Offline RL Methods

Offline RL learns from static datasets without additional environment interaction. This is crucial for:

- Safety-critical applications where exploration is dangerous
- Real-world robotics where data collection is expensive
- Leveraging existing demonstration datasets

### IQL (Implicit Q-Learning)

IQL avoids querying out-of-distribution actions by using expectile regression. It's simple, stable, and performs well across many tasks.

#### Algorithm

```
1. Learn V(s) using expectile regression on Q(s, a)
2. Learn Q(s, a) with Bellman backup using V(s')
3. Extract policy via advantage-weighted behavioral cloning
```

#### Training Script

```bash
python train/offline_rl/iql_trainer.py \
    --dataset hopper-medium-v2 \
    --num_epochs 100 \
    --batch_size 256 \
    --learning_rate 3e-4 \
    --expectile 0.7 \
    --temperature 3.0 \
    --output_dir ./output/iql
```

#### Python API

```python
from train.offline_rl.iql_trainer import IQLTrainer
from train.offline_rl.base_trainer import OfflineRLConfig, OfflineReplayBuffer

# Configuration
config = OfflineRLConfig(
    num_epochs=100,
    batch_size=256,
    learning_rate=3e-4,
    output_dir="./output/iql",
)

# Create trainer
trainer = IQLTrainer(
    obs_dim=11,
    action_dim=3,
    config=config,
    expectile=0.7,      # Expectile for value regression
    temperature=3.0,    # Temperature for advantage weighting
)

# Load dataset (D4RL format)
import d4rl
import gymnasium as gym

env = gym.make("hopper-medium-v2")
dataset = env.get_dataset()

buffer = OfflineReplayBuffer(obs_dim=11, action_dim=3)
buffer.load_dataset(dataset)

# Train
trainer.train(buffer)
```

#### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expectile` | 0.7 | Expectile τ for value regression (0.5-0.9) |
| `temperature` | 3.0 | Temperature for advantage weighting |
| `learning_rate` | 3e-4 | Learning rate for all networks |
| `hidden_dim` | 256 | Hidden layer dimension |

### CQL (Conservative Q-Learning)

CQL adds a regularization term to prevent overestimation of Q-values on out-of-distribution actions.

#### Training Script

```bash
python train/offline_rl/cql_trainer.py \
    --dataset hopper-medium-v2 \
    --num_epochs 100 \
    --batch_size 256 \
    --cql_alpha 5.0 \
    --output_dir ./output/cql
```

#### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cql_alpha` | 5.0 | CQL penalty weight |
| `cql_samples` | 10 | Number of action samples for CQL loss |
| `min_q_weight` | 1.0 | Weight for min Q in target |

### TD3+BC

Combines TD3 with behavioral cloning regularization to stay close to the data distribution.

#### Training Script

```bash
python train/offline_rl/td3_bc_trainer.py \
    --dataset hopper-medium-v2 \
    --num_epochs 100 \
    --alpha 2.5 \
    --output_dir ./output/td3bc
```

#### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 2.5 | BC regularization weight |
| `noise_clip` | 0.5 | Target policy noise clipping |
| `policy_delay` | 2 | Delayed policy updates |

### Decision Transformer

Treats RL as a sequence modeling problem, conditioning on returns-to-go.

#### Training Script

```bash
python train/offline_rl/decision_transformer.py \
    --dataset hopper-medium-v2 \
    --context_length 20 \
    --n_layer 3 \
    --n_head 1 \
    --embed_dim 128 \
    --output_dir ./output/dt
```

#### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_length` | 20 | Context window for transformer |
| `n_layer` | 3 | Number of transformer layers |
| `n_head` | 1 | Number of attention heads |
| `embed_dim` | 128 | Embedding dimension |

---

## Online RL Methods

Online RL learns through direct interaction with the environment. Requires a simulator for safe exploration.

### PPO (Proximal Policy Optimization)

PPO is a stable, general-purpose on-policy algorithm with clipped surrogate objective.

#### Algorithm

```
1. Collect rollout using current policy
2. Compute advantages using GAE
3. Update policy using clipped surrogate objective:
   L = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
4. Update value function
```

#### Training Script

```bash
python train/online_rl/ppo_trainer.py \
    --env CartPole-v1 \
    --total_timesteps 100000 \
    --rollout_steps 2048 \
    --ppo_epochs 4 \
    --ppo_clip_range 0.2 \
    --learning_rate 3e-4 \
    --output_dir ./output/ppo \
    --use-wandb
```

#### Python API

```python
from train.online_rl.ppo_trainer import PPOTrainer
from config.training_config import RLConfig
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Configuration
config = RLConfig(
    algorithm="ppo",
    total_timesteps=100000,
    rollout_steps=2048,
    ppo_epochs=4,
    ppo_clip_range=0.2,
    ppo_gae_lambda=0.95,
    ppo_value_coef=0.5,
    ppo_entropy_coef=0.01,
    batch_size=64,
    learning_rate=3e-4,
    discount_gamma=0.99,
    output_dir="./output/ppo",
)

# Create trainer
trainer = PPOTrainer(
    env=env,
    config=config,
    use_wandb=True,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate(num_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
```

#### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ppo_clip_range` | 0.2 | Clipping range for policy ratio |
| `ppo_epochs` | 4 | Epochs per policy update |
| `ppo_gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `ppo_value_coef` | 0.5 | Value loss coefficient |
| `ppo_entropy_coef` | 0.01 | Entropy bonus coefficient |
| `rollout_steps` | 2048 | Steps per rollout |

### SAC (Soft Actor-Critic)

SAC is an off-policy algorithm that maximizes entropy-regularized returns, providing better exploration.

#### Training Script

```bash
python train/online_rl/sac_trainer.py \
    --env HalfCheetah-v4 \
    --total_timesteps 1000000 \
    --buffer_size 1000000 \
    --learning_rate 3e-4 \
    --tau 0.005 \
    --output_dir ./output/sac
```

#### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size` | 1000000 | Replay buffer size |
| `batch_size` | 256 | Batch size for updates |
| `tau` | 0.005 | Target network update rate |
| `alpha` | auto | Entropy temperature (or "auto" for automatic) |
| `learning_starts` | 10000 | Steps before learning begins |

### GRPO (Group Relative Policy Optimization)

GRPO optimizes policies using group-relative comparisons, useful for multi-objective optimization.

#### Training Script

```bash
python train/online_rl/grpo_trainer.py \
    --env driving \
    --group_size 4 \
    --num_iterations 100 \
    --output_dir ./output/grpo
```

#### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 4 | Number of samples in each comparison group |
| `kl_coef` | 0.1 | KL penalty coefficient |
| `clip_range` | 0.2 | Clipping range |

---

## Method Comparison

| Method | Type | Sample Efficiency | Stability | Best For |
|--------|------|-------------------|-----------|----------|
| **IQL** | Offline | High | Very High | General offline RL |
| **CQL** | Offline | High | High | Avoiding OOD actions |
| **TD3+BC** | Offline | High | High | Simple implementation |
| **DT** | Offline | High | High | Sequence tasks |
| **PPO** | Online | Medium | Very High | General on-policy |
| **SAC** | Online | High | High | Continuous control |
| **GRPO** | Online | Medium | High | Multi-objective |

### When to Use Each Method

```
                        ┌─────────────────────────────────┐
                        │ Do you have environment access? │
                        └───────────────┬─────────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
                       Yes                             No
                        │                               │
                        ▼                               ▼
              ┌─────────────────┐              ┌─────────────────┐
              │   Online RL     │              │   Offline RL    │
              └────────┬────────┘              └────────┬────────┘
                       │                                │
           ┌───────────┴───────────┐        ┌───────────┴───────────┐
           │                       │        │                       │
     ┌─────┴─────┐           ┌─────┴─────┐  │   Dataset quality?    │
     │  Simple?  │           │  Sample   │  │                       │
     │           │           │ efficient?│  ├───────────────────────┤
     └─────┬─────┘           └─────┬─────┘  │ Expert: IQL/CQL       │
           │                       │        │ Mixed: TD3+BC         │
           ▼                       ▼        │ Diverse: DT           │
        ┌─────┐               ┌─────┐       └───────────────────────┘
        │ PPO │               │ SAC │
        └─────┘               └─────┘
```

---

## Integration with VLA Training

### Stage 3: Policy Improvement

After Stage 2 (action head training), use RL to improve the policy:

```bash
#!/bin/bash
# RL refinement pipeline

# Option A: Offline RL (if you have good demonstration data)
python train/offline_rl/iql_trainer.py \
    --model_path ./output/stage2_action_head/best \
    --dataset driving-expert-v1 \
    --num_epochs 100 \
    --output_dir ./output/stage3b_iql

# Option B: Online RL (if you have a simulator)
python train/online_rl/ppo_trainer.py \
    --model_path ./output/stage2_action_head/best \
    --env carla \
    --total_timesteps 1000000 \
    --output_dir ./output/stage3c_ppo
```

### Combining Offline and Online RL

```python
# Step 1: Initialize with offline RL
offline_config = OfflineRLConfig(num_epochs=50)
iql_trainer = IQLTrainer(obs_dim, action_dim, offline_config)
iql_trainer.train(offline_buffer)

# Step 2: Fine-tune with online RL
online_config = RLConfig(
    total_timesteps=100000,
    learning_rate=1e-5,  # Lower LR for fine-tuning
)
ppo_trainer = PPOTrainer(env, policy=iql_trainer.policy, config=online_config)
ppo_trainer.train()
```

### VLA-Specific RL Training

```python
from train.online_rl.ppo_trainer import PPOTrainer
from model.vla import VLAModel

# Load pre-trained VLA
model = VLAModel.from_pretrained("./output/stage2_action_head/best")

# Wrap VLA as RL policy
class VLAPolicy:
    def __init__(self, vla_model):
        self.vla = vla_model

    def get_action(self, obs, deterministic=False):
        # Convert observation to VLA input format
        pixel_values = obs["image"]
        input_ids = self.tokenize(obs.get("instruction", "drive safely"))

        outputs = self.vla(pixel_values, input_ids)
        return outputs["predicted_actions"]

# Create PPO trainer with VLA policy
trainer = PPOTrainer(
    env=carla_env,
    policy=VLAPolicy(model),
    config=config,
)
trainer.train()
```

---

## Configuration Reference

### OfflineRLConfig

```python
from config.training_config import OfflineRLConfig

config = OfflineRLConfig(
    # Training
    num_epochs=100,
    batch_size=256,
    learning_rate=3e-4,
    weight_decay=1e-6,

    # Networks
    hidden_dim=256,

    # RL parameters
    discount_gamma=0.99,
    tau=0.005,              # Soft target update rate

    # Logging
    save_freq=10,
    eval_episodes=10,

    # Output
    output_dir="./output/offline_rl",
)
```

### RLConfig (Online)

```python
from config.training_config import RLConfig

config = RLConfig(
    # Algorithm
    algorithm="ppo",  # or "sac", "grpo"

    # Training
    total_timesteps=1000000,
    rollout_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    discount_gamma=0.99,

    # PPO specific
    ppo_epochs=4,
    ppo_clip_range=0.2,
    ppo_gae_lambda=0.95,
    ppo_value_coef=0.5,
    ppo_entropy_coef=0.01,
    ppo_clip_range_vf=None,

    # SAC specific
    buffer_size=1000000,
    learning_starts=10000,
    tau=0.005,

    # Logging
    log_freq=1000,
    eval_freq=5000,
    save_freq=10000,

    # Output
    output_dir="./output/online_rl",
)
```

---

## Monitoring and Debugging

### Key Metrics

#### Offline RL

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| v_loss | 0.01 - 1.0 | Value function loss |
| q_loss | 0.01 - 1.0 | Q-function loss |
| policy_loss | -10 to 10 | Negative is better |
| mean_advantage | -5 to 5 | Should center around 0 |

#### Online RL

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| policy_loss | -0.1 to 0.1 | Clipped surrogate objective |
| value_loss | 0.01 - 1.0 | Value prediction error |
| entropy | Task-dependent | Higher = more exploration |
| mean_reward | Task-dependent | Main metric |

### Common Issues

#### Offline RL: Overestimation

**Symptom**: High Q-values but poor evaluation performance

**Solutions**:
- Increase CQL alpha
- Use more conservative expectile in IQL
- Add ensemble of Q-networks

#### Online RL: Sample Inefficiency

**Symptom**: Learning is very slow

**Solutions**:
- Use SAC instead of PPO
- Increase rollout steps
- Use larger batch size
- Pre-train with BC

#### Training Instability

**Symptom**: Loss spikes, performance crashes

**Solutions**:
- Reduce learning rate
- Increase gradient clipping
- Use smaller updates (lower clip range)
- Add entropy regularization

---

## Best Practices

### 1. Reward Shaping

```python
# Example reward shaping for driving
def compute_reward(state, action, next_state):
    # Safety: penalize collisions
    collision_penalty = -100 if is_collision(next_state) else 0

    # Efficiency: reward progress
    progress_reward = compute_progress(state, next_state)

    # Comfort: penalize jerky actions
    jerk_penalty = -0.1 * np.sum(np.abs(action - previous_action))

    # Lane keeping
    lane_reward = -0.5 * abs(lateral_offset(next_state))

    return collision_penalty + progress_reward + jerk_penalty + lane_reward
```

### 2. Curriculum Learning

```python
# Start with simple scenarios, gradually increase difficulty
def get_difficulty(training_step):
    if training_step < 10000:
        return "easy"
    elif training_step < 50000:
        return "medium"
    else:
        return "hard"

# Adjust environment based on difficulty
env.set_difficulty(get_difficulty(step))
```

### 3. Checkpointing and Recovery

```python
# Regular checkpointing
if step % save_freq == 0:
    trainer.save(f"checkpoint_step_{step}.pt")

# Resume from checkpoint
if resume_path:
    trainer.load(resume_path)
    start_step = trainer.global_step
```

---

## Shell Scripts Reference

### Offline RL Scripts

```bash
# IQL training
./scripts/run_offline_rl_iql.sh --dataset hopper-medium-v2

# CQL training
./scripts/run_offline_rl_cql.sh --dataset hopper-medium-v2

# TD3+BC training
./scripts/run_offline_rl_td3bc.sh --dataset hopper-medium-v2

# Decision Transformer training
./scripts/run_offline_rl_dt.sh --dataset hopper-medium-v2
```

### Online RL Scripts

```bash
# PPO training
./scripts/run_online_rl_ppo.sh --env CartPole-v1

# SAC training
./scripts/run_online_rl_sac.sh --env HalfCheetah-v4

# GRPO training
./scripts/run_online_rl_grpo.sh --env driving
```

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Complete training pipeline
- [Training Datasets](training_datasets.md) - Dataset documentation
- [Training World Model](training_world_model.md) - Imagination-based learning
- [Deployment](deployment.md) - Model export and production deployment
