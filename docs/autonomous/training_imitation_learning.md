# Imitation Learning for VLA

This document covers imitation learning methods for training VLA models, including Behavioral Cloning (BC), DAgger, and GAIL.

## Overview

Imitation learning trains policies by learning from expert demonstrations rather than through trial-and-error in the environment. This is particularly valuable for:

- **Safety-critical applications** like autonomous driving where exploration is dangerous
- **High-dimensional action spaces** where RL exploration is inefficient
- **Human-intuitive behaviors** that are difficult to specify via reward functions

```
┌──────────────────────────────────────────────────────────────────┐
│                   Imitation Learning Methods                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Behavioral Cloning (BC)                                     │ │
│  │  - Direct supervised learning on expert data                 │ │
│  │  - Simple and fast to train                                  │ │
│  │  - Suffers from distribution shift                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                       │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  DAgger (Dataset Aggregation)                                │ │
│  │  - Interactive imitation learning                            │ │
│  │  - Addresses distribution shift via expert queries           │ │
│  │  - Requires access to expert during training                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                       │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  GAIL (Generative Adversarial IL)                            │ │
│  │  - Learns reward function from demonstrations                │ │
│  │  - Can exceed expert performance                             │ │
│  │  - Requires environment interaction                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dataset Requirements

| Training Method | Dataset | Public Source | Description |
|-----------------|---------|---------------|-------------|
| **BC / DAgger** | LeRobot PushT | [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) | 2D manipulation demonstrations |
| **BC / DAgger** | LeRobot ALOHA | [lerobot/aloha_sim_*](https://huggingface.co/datasets/lerobot) | Bimanual manipulation |
| **BC / DAgger** | Open X-Embodiment | [jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) | Multi-robot demos with language |
| **BC / DAgger** | nuScenes | [nuscenes.org](https://www.nuscenes.org/) | Real driving demonstrations |
| **GAIL** | Any + Environment | - | Requires simulator for interaction |

---

## Behavioral Cloning (BC)

Behavioral Cloning is the simplest form of imitation learning that treats the problem as supervised learning.

### Algorithm

```
Input: Expert dataset D = {(s₁, a₁), (s₂, a₂), ...}
Output: Policy π_θ

1. Initialize policy parameters θ
2. For epoch = 1 to N:
   a. Sample batch (s, a) from D
   b. Compute loss: L = ||π_θ(s) - a||²
   c. Update θ via gradient descent
3. Return π_θ
```

### Training Script

```bash
# Standard BC for RL environments
python train/il/behavioral_cloning.py \
    --env CartPole-v1 \
    --bc_epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --output_dir ./output/bc

# VLA BC for robot manipulation
python train/il/behavioral_cloning.py \
    --dataset lerobot/pusht \
    --bc_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --output_dir ./output/vla_bc
```

### Python API

```python
from train.il.behavioral_cloning import BehavioralCloning, VLABehavioralCloning
from config.training_config import ILConfig
import gymnasium as gym

# Standard BC
env = gym.make("CartPole-v1")
config = ILConfig.behavioral_cloning()

trainer = BehavioralCloning(
    env=env,
    config=config,
    use_wandb=True,
)

# Train with expert policy
def expert_policy(state):
    return 1 if state[2] + 0.1 * state[3] > 0 else 0

results = trainer.train(expert_policy=expert_policy)
```

### VLA Behavioral Cloning

```python
from train.il.behavioral_cloning import VLABehavioralCloning
from model.vla import VLAModel
from train.datasets.lerobot_dataset import PushTDataset, create_lerobot_dataloader

# Load VLA model
model = VLAModel(
    vision_model_name="google/siglip-base-patch16-224",
    llm_model_name="Qwen/Qwen2-1.5B-Instruct",
    action_dim=2,
)

# Create dataset and dataloader
dataset = PushTDataset(split="train", chunk_size=10)
dataloader = create_lerobot_dataloader(dataset, batch_size=32)

# Train with VLA BC
config = ILConfig.behavioral_cloning()
trainer = VLABehavioralCloning(
    model=model,
    config=config,
    use_wandb=True,
    dataset_name="pusht",
)

trainer.train(dataloader)
```

### Configuration

```python
from config.training_config import ILConfig

config = ILConfig(
    # BC specific
    bc_epochs=100,
    bc_validation_split=0.1,

    # Training
    learning_rate=1e-4,
    batch_size=64,
    weight_decay=0.01,

    # Output
    output_dir="./output/bc",
)
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bc_epochs` | 100 | Training epochs |
| `batch_size` | 64 | Batch size |
| `learning_rate` | 1e-4 | Learning rate |
| `bc_validation_split` | 0.1 | Validation fraction |

---

## DAgger (Dataset Aggregation)

DAgger addresses the distribution shift problem in BC by iteratively collecting data under the learned policy and querying an expert for corrections.

### Algorithm

```
Input: Expert policy π*, Initial dataset D₀
Output: Policy π_θ

1. Initialize D = D₀
2. For iteration i = 1 to N:
   a. Train π_θ on D
   b. Execute π_θ in environment
   c. For each state s visited:
      - Query expert: a* = π*(s)
      - Record (s, a*)
   d. Aggregate: D = D ∪ {new (s, a*) pairs}
3. Return π_θ
```

### The β (Beta) Schedule

DAgger uses a β parameter to control the mix of expert and learned policy during data collection:

- **β = 1.0**: Use only expert (pure demonstration)
- **β = 0.0**: Use only learned policy
- **Linear decay**: β decreases from 1.0 to 0.0 over iterations
- **Exponential decay**: β = β₀ × 0.9^iteration

```python
# Beta schedules
def get_beta(iteration, schedule, initial_beta, num_iterations):
    if schedule == "constant":
        return initial_beta
    elif schedule == "linear":
        return max(0, initial_beta * (1 - iteration / num_iterations))
    elif schedule == "exponential":
        return initial_beta * (0.9 ** iteration)
```

### Training Script

```bash
# Standard DAgger
python train/il/dagger.py \
    --env CartPole-v1 \
    --dagger_iterations 10 \
    --dagger_episodes_per_iter 20 \
    --dagger_beta_schedule linear \
    --dagger_initial_beta 1.0 \
    --bc_epochs 20 \
    --output_dir ./output/dagger

# VLA DAgger
python train/il/dagger.py \
    --model_path ./pretrained_vla \
    --dagger_iterations 5 \
    --dagger_beta_schedule exponential \
    --output_dir ./output/vla_dagger
```

### Python API

```python
from train.il.dagger import DAgger, VLADAgger
from config.training_config import ILConfig
import gymnasium as gym

# Standard DAgger
env = gym.make("CartPole-v1")

def expert_policy(state):
    return 1 if state[2] + 0.1 * state[3] > 0 else 0

config = ILConfig(
    dagger_iterations=10,
    dagger_episodes_per_iter=20,
    dagger_beta_schedule="linear",
    dagger_initial_beta=1.0,
    bc_epochs=20,
)

trainer = DAgger(
    env=env,
    expert_policy=expert_policy,
    config=config,
)

results = trainer.train()
```

### Configuration

```python
from config.training_config import ILConfig

config = ILConfig(
    # DAgger specific
    dagger_iterations=10,           # Number of DAgger iterations
    dagger_episodes_per_iter=20,    # Episodes per iteration
    dagger_beta_schedule="linear",  # Beta decay schedule
    dagger_initial_beta=1.0,        # Starting beta value

    # BC params (used within DAgger)
    bc_epochs=20,
    batch_size=64,
    learning_rate=1e-4,

    # Output
    output_dir="./output/dagger",
)
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dagger_iterations` | 10 | Number of DAgger iterations |
| `dagger_episodes_per_iter` | 20 | Episodes collected per iteration |
| `dagger_beta_schedule` | "linear" | Beta decay: constant, linear, exponential |
| `dagger_initial_beta` | 1.0 | Starting probability of using expert |
| `bc_epochs` | 20 | BC epochs per DAgger iteration |

### DAgger vs BC Comparison

| Aspect | BC | DAgger |
|--------|-----|--------|
| Distribution shift | Suffers | Addresses |
| Expert access | Only initial | Throughout training |
| Sample efficiency | High | Lower (more data collected) |
| Training complexity | Simple | Iterative |
| Best for | Quick prototyping | Production policies |

---

## GAIL (Generative Adversarial Imitation Learning)

GAIL learns a reward function that distinguishes expert behavior from policy behavior, then optimizes the policy using this learned reward.

### Algorithm

```
Input: Expert trajectories τ*, Environment E
Output: Policy π_θ, Discriminator D_φ

1. Initialize π_θ, D_φ
2. For iteration i = 1 to N:
   a. Collect trajectories τ_π using π_θ
   b. Update D_φ to distinguish τ* from τ_π:
      - Maximize: E[log D(s,a)] + E[log(1 - D(s,a))]
                      τ*            τ_π
   c. Update π_θ using PPO with reward r(s,a) = -log(1 - D(s,a))
3. Return π_θ
```

### Training Script

```bash
# Standard GAIL
python train/il/gail.py \
    --env CartPole-v1 \
    --num_expert_episodes 50 \
    --total_timesteps 100000 \
    --gail_disc_hidden_dim 256 \
    --gail_disc_updates 5 \
    --output_dir ./output/gail

# VLA GAIL
python train/il/gail.py \
    --model_path ./pretrained_vla \
    --total_timesteps 10000 \
    --output_dir ./output/vla_gail
```

### Python API

```python
from train.il.gail import GAIL, VLAGAIL
from config.training_config import ILConfig
import gymnasium as gym

# Standard GAIL
env = gym.make("CartPole-v1")

def expert_policy(state):
    return 1 if state[2] + 0.1 * state[3] > 0 else 0

config = ILConfig(
    gail_disc_hidden_dim=256,
    gail_disc_updates=5,
    gail_disc_lr=3e-4,
    gail_reward_scale=1.0,
    learning_rate=3e-4,
)

trainer = GAIL(env=env, config=config)

results = trainer.train(
    expert_policy=expert_policy,
    num_expert_episodes=50,
    total_timesteps=100000,
)
```

### Discriminator Architecture

```python
class Discriminator(nn.Module):
    """Classifies (state, action) pairs as expert or policy."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def get_reward(self, state, action):
        """GAIL reward: -log(1 - D(s,a))"""
        d = self.forward(state, action)
        return -torch.log(1 - d + 1e-8)
```

### Configuration

```python
from config.training_config import ILConfig

config = ILConfig(
    # GAIL specific
    gail_disc_hidden_dim=256,   # Discriminator hidden size
    gail_disc_updates=5,        # Discriminator updates per policy update
    gail_disc_lr=3e-4,          # Discriminator learning rate
    gail_reward_scale=1.0,      # Scale for GAIL reward

    # PPO params (for policy optimization)
    ppo_epochs=4,
    ppo_clip_range=0.2,
    ppo_gae_lambda=0.95,

    # Training
    learning_rate=3e-4,
    batch_size=64,

    # Output
    output_dir="./output/gail",
)
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gail_disc_hidden_dim` | 256 | Discriminator hidden layer size |
| `gail_disc_updates` | 5 | Discriminator updates per iteration |
| `gail_disc_lr` | 3e-4 | Discriminator learning rate |
| `gail_reward_scale` | 1.0 | Multiplier for GAIL reward |
| `total_timesteps` | 100000 | Total training timesteps |

---

## Method Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **BC** | Simple, fast, no env needed | Distribution shift, limited performance | Quick prototyping, offline data |
| **DAgger** | Addresses distribution shift | Needs expert throughout, interactive | Production policies with expert access |
| **GAIL** | Can exceed expert, learns reward | Needs env, unstable training | Complex tasks, limited demos |

### When to Use Each Method

```
                    ┌─────────────────────────────────────┐
                    │    Do you have environment access?  │
                    └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                   Yes                             No
                    │                               │
        ┌───────────┴───────────┐                   │
        │  Do you need to       │                   │
        │  exceed expert?       │                   │
        └───────────┬───────────┘                   │
                    │                               │
        ┌───────────┴───────────┐                   │
        │                       │                   │
       Yes                     No                   │
        │                       │                   │
        ▼                       │                   │
     ┌──────┐                   │                   │
     │ GAIL │                   │                   │
     └──────┘                   │                   │
                    ┌───────────┴───────────┐       │
                    │   Is expert available │       │
                    │   during training?    │       │
                    └───────────┬───────────┘       │
                                │                   │
                    ┌───────────┴───────────┐       │
                    │                       │       │
                   Yes                     No       │
                    │                       │       │
                    ▼                       │       │
               ┌────────┐                   │       │
               │ DAgger │                   │       │
               └────────┘                   │       │
                                            │       │
                                            ▼       ▼
                                         ┌────┐  ┌────┐
                                         │ BC │  │ BC │
                                         └────┘  └────┘
```

---

## Integration with VLA Training Pipeline

### Stage 3a: Using IL for Policy Refinement

After Stage 2 (action head training), use IL methods to refine the policy:

```bash
#!/bin/bash
# Complete IL refinement pipeline

# Step 1: Initial BC training
python train/il/behavioral_cloning.py \
    --model_path ./output/stage2_action_head/best \
    --dataset lerobot/pusht \
    --bc_epochs 50 \
    --output_dir ./output/stage3a_bc

# Step 2: DAgger refinement (if expert available)
python train/il/dagger.py \
    --model_path ./output/stage3a_bc/best \
    --dagger_iterations 5 \
    --output_dir ./output/stage3a_dagger

# Step 3: GAIL polish (if environment available)
python train/il/gail.py \
    --model_path ./output/stage3a_dagger/best \
    --total_timesteps 50000 \
    --output_dir ./output/stage3a_gail
```

### Combining IL with RL

```python
# First: BC initialization
bc_trainer = VLABehavioralCloning(model, config)
bc_trainer.train(dataloader)

# Then: RL fine-tuning (e.g., PPO)
from train.online_rl.ppo_trainer import PPOTrainer

ppo_config = RLConfig(
    total_timesteps=100000,
    learning_rate=1e-5,  # Lower LR for fine-tuning
)

rl_trainer = PPOTrainer(env, policy=model, config=ppo_config)
rl_trainer.train()
```

---

## Monitoring and Debugging

### Key Metrics

| Metric | BC | DAgger | GAIL |
|--------|-----|--------|------|
| train/loss | MSE on expert actions | MSE on aggregated data | Policy + value loss |
| val/loss | Validation MSE | - | - |
| disc_loss | - | - | Discriminator accuracy |
| mean_reward | - | Per-iteration eval | GAIL reward signal |

### Common Issues

#### BC: Distribution Shift

**Symptom**: Good training loss but poor deployment performance

**Solution**: Use DAgger or GAIL, or collect more diverse demonstrations

#### DAgger: Expert Query Cost

**Symptom**: Training is slow due to expert queries

**Solution**: Reduce `dagger_episodes_per_iter`, use cached expert data

#### GAIL: Discriminator Overfitting

**Symptom**: Discriminator accuracy is 100%, policy doesn't improve

**Solution**: Add noise to discriminator input, reduce `gail_disc_updates`

```python
# Add noise for discriminator regularization
def discriminator_forward_with_noise(self, state, action, noise_scale=0.01):
    state_noisy = state + torch.randn_like(state) * noise_scale
    action_noisy = action + torch.randn_like(action) * noise_scale
    return self.forward(state_noisy, action_noisy)
```

---

## Best Practices

### 1. Data Quality

```python
# Filter low-quality demonstrations
def filter_demonstrations(states, actions, rewards):
    # Keep only successful episodes
    episode_rewards = compute_episode_rewards(rewards)
    threshold = np.percentile(episode_rewards, 70)
    good_indices = episode_rewards >= threshold
    return states[good_indices], actions[good_indices]
```

### 2. Action Chunking for BC

```python
# Predict multiple future actions (ACT-style)
config = ILConfig(
    action_chunk_size=10,  # Predict 10 future actions
    temporal_aggregation=True,
)
```

### 3. Warm-Starting GAIL

```python
# Pre-train with BC before GAIL
bc_epochs = 10
for epoch in range(bc_epochs):
    bc_trainer.train_epoch()

# Then switch to GAIL
gail_trainer = GAIL(env, policy=bc_trainer.policy)
gail_trainer.train()
```

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Complete training pipeline
- [Training Datasets](training_datasets.md) - Dataset documentation
- [Reinforcement Learning](training_reinforcement_learning.md) - Offline and online RL
- [Deployment](deployment.md) - Model export
