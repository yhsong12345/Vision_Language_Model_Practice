# Training World Models for Autonomous Driving

This document covers training latent world models for autonomous driving applications. World models learn the dynamics of the environment and enable imagination-based planning without real-world interaction.

## Overview

World models learn to predict future states, rewards, and observations given current states and actions. For autonomous driving, this means learning:

- **Dynamics**: How the vehicle and environment evolve over time
- **Observations**: What the sensors will observe in future states
- **Rewards**: Expected outcomes (safety, efficiency, comfort)

```
┌────────────────────────────────────────────────────────────────┐
│                    World Model Architecture                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Observation (t) ──► Encoder ──► Latent State (t)             │
│         │                              │                        │
│         │         ┌────────────────────┴────────────────────┐  │
│         │         │   RSSM (Recurrent State-Space Model)    │  │
│         │         │   ┌──────────┐  ┌──────────┐            │  │
│         │         │   │Determin- │  │Stochastic│            │  │
│   Action (t) ─────┤   │   istic  │  │  State   │            │  │
│         │         │   │   State  │  │          │            │  │
│         │         │   └────┬─────┘  └────┬─────┘            │  │
│         │         │        └──────┬──────┘                  │  │
│         │         │               ▼                          │  │
│         │         │     Combined State (t+1)                │  │
│         │         └─────────────────────────────────────────┘  │
│         │                         │                            │
│         │         ┌───────────────┼───────────────┐           │
│         │         │               │               │            │
│         ▼         ▼               ▼               ▼            │
│    ┌─────────┐ ┌─────────┐ ┌───────────┐ ┌────────────┐       │
│    │Decoder  │ │ Reward  │ │   Value   │ │ Continue   │       │
│    │(recon.) │ │Predictor│ │ Predictor │ │ Predictor  │       │
│    └────┬────┘ └────┬────┘ └─────┬─────┘ └─────┬──────┘       │
│         │           │            │              │              │
│         ▼           ▼            ▼              ▼              │
│  Observation(t+1) Reward(t+1) Value(t+1)  Terminal(t+1)       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Dataset Requirements

| Training Stage | Dataset | Public Source | Description |
|----------------|---------|---------------|-------------|
| **World Model (Sim)** | CARLA Trajectories | Local/Custom | Simulated driving sequences with full state |
| **World Model (Real)** | nuScenes Sequences | [nuscenes.org](https://www.nuscenes.org/) | Real driving sequences with 6 cameras + LiDAR |
| **World Model (Real)** | Waymo Motion | [waymo.com/open](https://waymo.com/open) | High-quality motion prediction data |
| **World Model (Video)** | KITTI Raw | [cvlibs.net](http://www.cvlibs.net/datasets/kitti/raw_data.php) | Raw driving sequences with camera + LiDAR |

### Data Format

World model training requires sequential data:

```python
# Expected format for each episode file (.npz)
{
    "observations": np.ndarray,  # (T, C, H, W) - image sequences
    "actions": np.ndarray,       # (T, action_dim) - control actions
    "rewards": np.ndarray,       # (T,) - reward signals
    "dones": np.ndarray,         # (T,) - terminal flags
}
```

### Data Collection

For CARLA-based data collection:

```bash
# Run CARLA data collector
python examples/carla_demo.py \
    --mode collect \
    --episodes 1000 \
    --episode-length 500 \
    --output-dir ./data/world_model/train
```

---

## Training Script

### Basic Usage

```bash
python train/world_model/train_world_model.py \
    --state-dim 256 \
    --action-dim 3 \
    --hidden-dim 512 \
    --latent-dim 32 \
    --batch-size 32 \
    --sequence-length 50 \
    --learning-rate 3e-4 \
    --num-epochs 100 \
    --data-path ./data/world_model \
    --output-dir ./output/world_model
```

### Shell Script

```bash
# scripts/run_world_model.sh
#!/bin/bash
#SBATCH --job-name=world_model
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

python train/world_model/train_world_model.py \
    --state-dim 256 \
    --action-dim 3 \
    --hidden-dim 512 \
    --latent-dim 32 \
    --batch-size 64 \
    --sequence-length 50 \
    --learning-rate 3e-4 \
    --num-epochs 100 \
    --kl-weight 1.0 \
    --reconstruction-weight 1.0 \
    --data-path ./data/world_model \
    --output-dir ./output/world_model \
    "$@"
```

---

## Configuration

### WorldModelTrainConfig

```python
from dataclasses import dataclass

@dataclass
class WorldModelTrainConfig:
    # Model architecture
    state_dim: int = 256          # Deterministic state dimension
    action_dim: int = 3           # Action space (steering, throttle, brake)
    hidden_dim: int = 512         # Hidden layer dimension
    latent_dim: int = 32          # Stochastic latent dimension
    num_categories: int = 32      # Categorical distribution size

    # RSSM specific
    deterministic_dim: int = 256  # GRU hidden state
    stochastic_dim: int = 32      # Stochastic state size

    # Training
    batch_size: int = 32
    sequence_length: int = 50     # Steps per training sequence
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    num_epochs: int = 100
    max_grad_norm: float = 100.0
    warmup_steps: int = 1000

    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 1.0
    kl_balance: float = 0.8       # Prior vs posterior balance
    reward_weight: float = 1.0
    value_weight: float = 0.5
    dynamics_weight: float = 1.0

    # KL annealing
    kl_free_bits: float = 1.0     # Free bits for KL
    kl_anneal_steps: int = 10000  # Steps to anneal KL weight

    # Imagination
    imagination_horizon: int = 15
    num_imagination_samples: int = 8
```

---

## Training Pipeline

### 1. Data Preparation

```python
from train.world_model.train_world_model import SequenceDataset, WorldModelTrainConfig

config = WorldModelTrainConfig(
    action_dim=3,
    sequence_length=50,
    data_path="./data/world_model",
)

# Load training data
train_dataset = SequenceDataset(config.data_path, config, split="train")
val_dataset = SequenceDataset(config.data_path, config, split="val")
```

### 2. Model Initialization

```python
from train.world_model.train_world_model import WorldModelTrainer

trainer = WorldModelTrainer(config)

# Check model parameters
print(f"World Model: {sum(p.numel() for p in trainer.world_model.parameters()):,} params")
print(f"Reward Predictor: {sum(p.numel() for p in trainer.reward_predictor.parameters()):,} params")
```

### 3. Training

```python
# Run training
trainer.train()

# Or resume from checkpoint
trainer.load_checkpoint("./output/world_model/checkpoint_epoch_50.pt")
trainer.train()
```

---

## Loss Functions

### 1. Reconstruction Loss

Measures how well the decoder reconstructs observations from latent states:

```python
recon_loss = F.mse_loss(reconstructed_obs, target_obs)
```

### 2. KL Divergence Loss

Regularizes the latent space to match the prior:

```python
# KL between posterior and prior
kl_loss = kl_divergence(posterior_dist, prior_dist)

# With free bits (minimum KL per dimension)
kl_loss = torch.clamp(kl_loss, min=kl_free_bits)

# KL annealing (gradually increase weight)
kl_weight = min(1.0, global_step / kl_anneal_steps)
```

### 3. Reward Prediction Loss

Predicts rewards from latent states:

```python
reward_pred = reward_predictor(latent_state)
reward_loss = F.mse_loss(reward_pred, target_reward)
```

### 4. Dynamics Consistency Loss

Ensures predicted dynamics match actual transitions:

```python
# Predicted next state from dynamics model
predicted_state = dynamics_model(state, action)

# Actual next state from encoder
actual_state = encoder(next_observation)

dynamics_loss = F.mse_loss(predicted_state, actual_state)
```

### Total Loss

```python
total_loss = (
    reconstruction_weight * recon_loss +
    kl_weight * kl_loss +
    reward_weight * reward_loss +
    dynamics_weight * dynamics_loss
)
```

---

## Imagination-Based Planning

Once trained, the world model can be used for imagination-based planning:

### Rollout Imagination

```python
def imagine(initial_state, policy, horizon=15):
    """Generate imagined trajectories."""
    states = [initial_state]
    actions = []
    rewards = []

    state = initial_state
    for t in range(horizon):
        # Get action from policy
        action = policy(state)
        actions.append(action)

        # Predict next state using world model
        next_state = world_model.imagine_step(state, action)
        states.append(next_state)

        # Predict reward
        reward = reward_predictor(next_state)
        rewards.append(reward)

        state = next_state

    return {
        "states": torch.stack(states, dim=1),
        "actions": torch.stack(actions, dim=1),
        "rewards": torch.stack(rewards, dim=1),
    }
```

### Dreamer-Style Policy Learning

```python
# Generate imagined trajectories
imagined_trajectories = imagine(initial_state, actor, horizon=15)

# Compute returns using value predictor
values = value_predictor(imagined_trajectories["states"])
returns = compute_lambda_returns(
    imagined_trajectories["rewards"],
    values,
    gamma=0.99,
    lambda_=0.95,
)

# Update actor to maximize returns
actor_loss = -returns.mean()
actor_optimizer.zero_grad()
actor_loss.backward()
actor_optimizer.step()
```

---

## Model Architecture Details

### Encoder

Encodes observations into latent representations:

```python
class Encoder(nn.Module):
    def __init__(self, image_size=64, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8 -> 4
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
```

### RSSM (Recurrent State-Space Model)

Combines deterministic and stochastic states:

```python
class RSSM(nn.Module):
    def __init__(self, deterministic_dim, stochastic_dim, action_dim):
        super().__init__()
        # GRU for deterministic state
        self.gru = nn.GRUCell(
            stochastic_dim + action_dim,
            deterministic_dim,
        )

        # Prior and posterior networks
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stochastic_dim * 2),
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stochastic_dim * 2),
        )
```

### Decoder

Reconstructs observations from latent states:

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, image_size=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )
```

---

## Recommended Configurations

### Driving World Model (CARLA)

```python
config = WorldModelTrainConfig(
    # Model
    state_dim=256,
    action_dim=3,  # steering, throttle, brake
    hidden_dim=512,
    latent_dim=32,

    # Training
    batch_size=32,
    sequence_length=50,  # 2 seconds at 25 FPS
    learning_rate=3e-4,
    num_epochs=100,

    # Loss weights
    reconstruction_weight=1.0,
    kl_weight=0.5,  # Start lower for stability
    reward_weight=1.0,

    # Imagination
    imagination_horizon=25,  # 1 second lookahead
)
```

### Video Prediction (nuScenes)

```python
config = WorldModelTrainConfig(
    # Model
    state_dim=512,
    action_dim=3,
    hidden_dim=1024,
    latent_dim=64,

    # Training
    batch_size=16,  # Larger images
    sequence_length=20,  # 1 second at 20 FPS
    learning_rate=1e-4,
    num_epochs=200,

    # Multi-camera
    image_size=128,
)
```

---

## Monitoring and Debugging

### Key Metrics

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| recon_loss | 0.01 - 0.1 | Lower is better, should decrease |
| kl_loss | 1.0 - 10.0 | After annealing, should stabilize |
| reward_loss | 0.1 - 1.0 | Depends on reward scale |
| dynamics_loss | 0.01 - 0.1 | Consistency between predicted/actual |

### Visualization

```python
# Visualize imagined trajectories
def visualize_imagination(world_model, initial_obs, policy):
    # Encode initial observation
    initial_state = world_model.encode(initial_obs)

    # Generate trajectory
    trajectory = world_model.imagine(initial_state, policy, horizon=20)

    # Decode states to observations
    decoded_obs = world_model.decode(trajectory["states"])

    # Plot sequence
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        idx = i * 4
        ax.imshow(decoded_obs[0, idx].permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"t={idx}")
        ax.axis("off")
    plt.savefig("imagined_trajectory.png")
```

### Common Issues

1. **KL collapse**: Increase `kl_free_bits` or decrease `kl_weight`
2. **Blurry reconstructions**: Increase reconstruction weight, add perceptual loss
3. **Poor dynamics**: Increase sequence length, add more training data
4. **Training instability**: Reduce learning rate, increase warmup steps

---

## Integration with VLA

The trained world model can be integrated with VLA training:

### Imagination-Based Data Augmentation

```python
# Generate imagined demonstrations
world_model.load_checkpoint("./output/world_model/best_model.pt")

imagined_data = []
for real_trajectory in real_dataset:
    initial_state = world_model.encode(real_trajectory["observations"][0])

    # Generate variations
    for _ in range(num_augmentations):
        imagined = world_model.imagine(initial_state, policy, horizon=50)
        imagined_data.append(imagined)

# Train VLA on combined data
combined_dataset = ConcatDataset([real_dataset, imagined_data])
```

### Model-Based Reinforcement Learning

```python
# Use world model for planning in VLA training
class WorldModelGuidedTrainer:
    def __init__(self, vla_model, world_model, policy):
        self.vla = vla_model
        self.world_model = world_model
        self.policy = policy

    def plan_action(self, observation):
        # Encode current state
        state = self.world_model.encode(observation)

        # Evaluate multiple action sequences
        best_actions = None
        best_return = -float("inf")

        for _ in range(num_samples):
            imagined = self.world_model.imagine(state, self.policy, horizon=10)
            estimated_return = imagined["rewards"].sum()

            if estimated_return > best_return:
                best_return = estimated_return
                best_actions = imagined["actions"]

        return best_actions[0]  # Return first action
```

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Complete VLA training pipeline
- [Training Datasets](training_datasets.md) - Dataset documentation
- [Reinforcement Learning](training_reinforcement_learning.md) - Online and offline RL
- [Deployment](deployment.md) - Model export and deployment
