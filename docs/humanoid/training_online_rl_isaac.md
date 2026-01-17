# Online RL Training with Isaac Sim/Lab in Docker

This document covers setting up and running online reinforcement learning for humanoid robots using NVIDIA Isaac Sim and Isaac Lab in Docker.

## Overview

Online RL for humanoid robots requires high-fidelity physics simulation for locomotion and manipulation. Isaac Sim provides GPU-accelerated simulation with accurate rigid body dynamics, while Isaac Lab (formerly Orbit) offers optimized RL environments.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Online RL Training Architecture                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐         ┌──────────────────────────────┐  │
│  │   Isaac Sim Docker   │         │   Training Process           │  │
│  │   Container          │  Shared │                              │  │
│  │  ┌────────────────┐  │  Memory │  ┌────────────────────────┐  │  │
│  │  │ Isaac Sim      │  │◄───────►│  │  HumanoidVLA Model     │  │  │
│  │  │ - PhysX GPU    │  │         │  │  - Proprioception Enc  │  │  │
│  │  │ - Omniverse    │  │         │  │  - Locomotion Policy   │  │  │
│  │  │ - USD Scenes   │  │         │  │  - Manipulation Policy │  │  │
│  │  └────────────────┘  │         │  └────────────────────────┘  │  │
│  │                      │         │                              │  │
│  │  GPU: Simulation     │         │  GPU: Training               │  │
│  └──────────────────────┘         └──────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup Options

### Option 1: Isaac Lab in Docker (Recommended)

Isaac Lab provides pre-built RL environments optimized for robotics training.

**Pros:**
- Ready-to-use humanoid environments
- Integrated with rl_games, RSL-RL, Stable Baselines3
- GPU-accelerated parallel environments
- Active development and support

### Option 2: Isaac Sim + Custom Environments

For custom robots or environments not in Isaac Lab.

**Pros:**
- Full flexibility
- Custom USD assets
- Omniverse ecosystem integration

## Isaac Sim/Lab Docker Setup

### 1. Prerequisites

```bash
# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Pull Isaac Docker Images

```bash
# Isaac Sim (full simulator)
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# Isaac Lab (RL-focused, recommended)
docker pull nvcr.io/nvidia/isaac-lab:4.2.0

# For older versions
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1
```

### 3. Run Isaac Sim Container

#### Basic Run

```bash
docker run --name isaac-sim --entrypoint bash \
  -it --gpus all \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  --rm --network=host \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/isaac-sim:4.2.0
```

#### Headless Mode for Training

```bash
docker run --gpus all \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  --rm \
  -v $(pwd):/workspace \
  -v ~/docker/isaac-sim/cache:/root/.cache \
  nvcr.io/nvidia/isaac-sim:4.2.0 \
  ./python.sh -c "
from omni.isaac.kit import SimulationApp
app = SimulationApp({'headless': True})
print('Isaac Sim initialized in headless mode')
app.close()
"
```

### 4. Run Isaac Lab Container

```bash
# Interactive mode
docker run --name isaac-lab --entrypoint bash \
  -it --gpus all \
  -e "ACCEPT_EULA=Y" \
  --rm --network=host \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/isaac-lab:4.2.0

# Run humanoid training directly
docker run --gpus all \
  -e "ACCEPT_EULA=Y" \
  --rm \
  -v $(pwd)/output:/workspace/output \
  nvcr.io/nvidia/isaac-lab:4.2.0 \
  ./isaaclab.sh -p source/standalone/workflows/rl_games/train.py \
    --task Isaac-Humanoid-v0 \
    --headless \
    --num_envs 4096
```

## Available Humanoid Environments

### Isaac Lab Environments

| Environment | Description | DoF | Observation |
|-------------|-------------|-----|-------------|
| `Isaac-Humanoid-v0` | Basic humanoid locomotion | 21 | Joint pos/vel, base state |
| `Isaac-Humanoid-Walk-v0` | Walking with velocity commands | 21 | + velocity command |
| `Isaac-Humanoid-Run-v0` | Running locomotion | 21 | + velocity command |
| `Isaac-Humanoid-Stand-v0` | Balance/standing | 21 | Joint pos/vel, base state |
| `Isaac-Humanoid-Reach-v0` | Reaching tasks | 21 | + target position |

### Custom Humanoid Configuration

```python
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_assets import HUMANOID_CFG

class CustomHumanoidEnvCfg(ManagerBasedRLEnvCfg):
    """Custom humanoid environment configuration."""

    # Scene
    scene = HumanoidSceneCfg(num_envs=4096, env_spacing=4.0)

    # Robot
    robot = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Observations
    observations = HumanoidObservationsCfg()

    # Actions
    actions = HumanoidActionsCfg()

    # Rewards
    rewards = HumanoidRewardsCfg(
        # Locomotion rewards
        lin_vel_reward_scale=1.0,
        ang_vel_reward_scale=0.5,

        # Stability rewards
        orientation_reward_scale=0.2,
        base_height_reward_scale=0.1,

        # Energy penalty
        action_rate_penalty_scale=-0.01,
        joint_torque_penalty_scale=-0.0001,
    )

    # Termination
    terminations = HumanoidTerminationsCfg(
        base_height_threshold=0.3,  # Fall detection
        max_episode_length=1000,
    )
```

## Training Configuration

### Using IsaacSimBridge

Your codebase provides `IsaacSimBridge` in `integration/simulator_bridge.py`:

```python
from integration.simulator_bridge import IsaacSimBridge, SimulatorConfig

config = SimulatorConfig(
    # Rendering
    render=False,
    headless=True,
    fps=60,

    # Physics
    physics_dt=1.0 / 120.0,  # 120 Hz physics
    control_dt=1.0 / 60.0,   # 60 Hz control

    # Scene
    scene_path="/path/to/humanoid.usd",
    robot_config="humanoid_32dof",

    # Camera
    image_width=224,
    image_height=224,
)

bridge = IsaacSimBridge(config)
bridge.initialize()
```

### PPO Training with Isaac Lab

```python
from train.online_rl.ppo_trainer import PPOTrainer, PPOConfig
from model.embodiment.humanoid import HumanoidVLA, HumanoidConfig

# Setup model (32 DoF humanoid)
model_config = HumanoidConfig(
    num_joints=32,
    proprioception_dim=128,
    action_dim=32,
    control_freq=60.0,
)
model = HumanoidVLA(model_config)

# PPO config for humanoid
ppo_config = PPOConfig(
    # Learning
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,

    # Batching
    num_envs=4096,        # Parallel environments
    num_steps=24,         # Steps per update
    batch_size=24 * 4096, # Large batch for stability
    num_epochs=5,

    # Regularization
    entropy_coef=0.01,
    value_loss_coef=0.5,
    max_grad_norm=1.0,
)

trainer = PPOTrainer(model, env, ppo_config)
trainer.train(total_timesteps=100_000_000)
```

### Training with rl_games (Isaac Lab Native)

```python
# Using Isaac Lab's built-in rl_games integration
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.manager_based.locomotion.velocity import velocity_env_cfg

# Create environment
env_cfg = velocity_env_cfg.HumanoidEnvCfg()
env_cfg.scene.num_envs = 4096
env = ManagerBasedRLEnv(cfg=env_cfg)

# rl_games config
rl_games_cfg = {
    "params": {
        "algo": {
            "name": "a2c_continuous",
        },
        "model": {
            "name": "continuous_a2c_logstd",
        },
        "network": {
            "name": "actor_critic",
            "separate": False,
            "space": {
                "continuous": {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "fixed_sigma": True,
                }
            },
            "mlp": {
                "units": [512, 256, 128],
                "activation": "elu",
            },
        },
        "config": {
            "name": "Humanoid",
            "env_name": "Isaac-Humanoid-v0",
            "num_actors": 4096,
            "minibatch_size": 32768,
            "gamma": 0.99,
            "tau": 0.95,
            "learning_rate": 3e-4,
            "lr_schedule": "adaptive",
            "max_epochs": 5000,
            "normalize_input": True,
            "normalize_value": True,
        },
    },
}
```

## Docker Compose Setup

### Single GPU Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  isaac-lab:
    image: nvcr.io/nvidia/isaac-lab:4.2.0
    container_name: isaac-lab-training
    runtime: nvidia
    environment:
      - ACCEPT_EULA=Y
      - PRIVACY_CONSENT=Y
    volumes:
      - ./:/workspace
      - ./output:/workspace/output
      - isaac-cache:/root/.cache
    command: >
      ./isaaclab.sh -p source/standalone/workflows/rl_games/train.py
        --task Isaac-Humanoid-v0
        --headless
        --num_envs 4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  isaac-cache:
```

### Multi-GPU Training

```yaml
# docker-compose-multi-gpu.yml
version: '3.8'

services:
  isaac-lab-multi:
    image: nvcr.io/nvidia/isaac-lab:4.2.0
    container_name: isaac-lab-multi-gpu
    runtime: nvidia
    environment:
      - ACCEPT_EULA=Y
      - PRIVACY_CONSENT=Y
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./:/workspace
      - ./output:/workspace/output
    command: >
      ./isaaclab.sh -p source/standalone/workflows/rl_games/train.py
        --task Isaac-Humanoid-v0
        --headless
        --num_envs 8192
        --distributed
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Training Scripts

### Shell Script for Humanoid Training

```bash
#!/bin/bash
# scripts/run_humanoid_online_rl.sh

# Configuration
ISAAC_VERSION="4.2.0"
TASK="Isaac-Humanoid-v0"
NUM_ENVS=4096
MAX_ITERATIONS=5000
OUTPUT_DIR="./output/humanoid_ppo"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run training in Docker
echo "Starting Isaac Lab humanoid training..."
docker run --gpus all \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  --rm \
  -v $(pwd):/workspace \
  -v ${OUTPUT_DIR}:/workspace/output \
  nvcr.io/nvidia/isaac-lab:${ISAAC_VERSION} \
  ./isaaclab.sh -p source/standalone/workflows/rl_games/train.py \
    --task ${TASK} \
    --headless \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERATIONS}

echo "Training complete! Results saved to ${OUTPUT_DIR}"
```

### Python Training Script

```python
#!/usr/bin/env python3
"""
Online RL training for humanoid with Isaac Lab.

Usage:
    python train/online_rl/train_humanoid_isaac.py \
        --task Isaac-Humanoid-v0 \
        --num-envs 4096 \
        --max-iterations 5000
"""

import argparse
import os
import sys
from pathlib import Path

# Isaac Lab imports (inside container)
try:
    from omni.isaac.lab.app import AppLauncher

    # Parse args before launching
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Isaac-Humanoid-v0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()

    # Launch app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now import the rest
    import torch
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg

except ImportError:
    print("This script must be run inside Isaac Lab container")
    print("Use: ./isaaclab.sh -p train/online_rl/train_humanoid_isaac.py")
    sys.exit(1)


def main():
    # Get environment config
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print(f"Environment: {args.task}")
    print(f"Num envs: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Training loop with rl_games
    from rl_games.algos_torch import runner
    from rl_games.common import env_configurations, vecenv

    # Register environment
    vecenv.register(
        "IsaacLab-Humanoid",
        lambda config_name, num_actors, **kwargs: env
    )

    # Run training
    runner_config = {
        "params": {
            "config": {
                "name": "Humanoid",
                "env_name": "IsaacLab-Humanoid",
                "num_actors": args.num_envs,
                "max_epochs": args.max_iterations,
            }
        }
    }

    rl_runner = runner.Runner()
    rl_runner.load(runner_config)
    rl_runner.run({
        "train": True,
        "play": False,
    })

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
```

## Reward Design for Humanoid

### Locomotion Rewards

```python
class HumanoidRewards:
    """Reward functions for humanoid locomotion."""

    def __init__(self, cfg):
        self.cfg = cfg

    def compute_rewards(self, env):
        """Compute all reward components."""
        rewards = {}

        # Velocity tracking
        rewards["lin_vel"] = self._lin_vel_reward(env)
        rewards["ang_vel"] = self._ang_vel_reward(env)

        # Stability
        rewards["orientation"] = self._orientation_reward(env)
        rewards["base_height"] = self._base_height_reward(env)

        # Energy efficiency
        rewards["action_rate"] = self._action_rate_penalty(env)
        rewards["torque"] = self._torque_penalty(env)

        # Foot contact
        rewards["feet_air_time"] = self._feet_air_time_reward(env)

        return rewards

    def _lin_vel_reward(self, env):
        """Reward for tracking commanded velocity."""
        vel_error = env.command_vel[:, :2] - env.base_lin_vel[:, :2]
        return torch.exp(-torch.sum(vel_error**2, dim=1) / 0.25)

    def _ang_vel_reward(self, env):
        """Reward for tracking commanded angular velocity."""
        ang_vel_error = env.command_vel[:, 2] - env.base_ang_vel[:, 2]
        return torch.exp(-ang_vel_error**2 / 0.25)

    def _orientation_reward(self, env):
        """Reward for upright orientation."""
        # Project gravity vector to body frame
        projected_gravity = env.projected_gravity
        return torch.sum(projected_gravity[:, :2]**2, dim=1)

    def _base_height_reward(self, env):
        """Reward for maintaining target height."""
        height_error = env.base_pos[:, 2] - self.cfg.target_height
        return torch.exp(-height_error**2 / 0.1)

    def _action_rate_penalty(self, env):
        """Penalty for rapid action changes."""
        return -torch.sum((env.actions - env.last_actions)**2, dim=1)

    def _torque_penalty(self, env):
        """Penalty for high joint torques."""
        return -torch.sum(env.torques**2, dim=1)

    def _feet_air_time_reward(self, env):
        """Reward for alternating foot contacts (gait)."""
        contact = env.feet_contact
        first_contact = (env.feet_air_time > 0.5) * contact
        reward = torch.sum(first_contact * env.feet_air_time, dim=1)
        env.feet_air_time *= ~contact  # Reset on contact
        return reward
```

### Manipulation Rewards

```python
class ManipulationRewards:
    """Reward functions for humanoid manipulation."""

    def compute_rewards(self, env):
        rewards = {}

        # Reaching
        rewards["reach"] = self._reach_reward(env)

        # Grasping
        rewards["grasp"] = self._grasp_reward(env)

        # Stability during manipulation
        rewards["balance"] = self._balance_reward(env)

        return rewards

    def _reach_reward(self, env):
        """Reward for end-effector reaching target."""
        ee_pos = env.end_effector_pos
        target_pos = env.target_pos
        distance = torch.norm(ee_pos - target_pos, dim=1)
        return torch.exp(-distance / 0.1)

    def _grasp_reward(self, env):
        """Reward for successful grasping."""
        return env.object_grasped.float() * 10.0

    def _balance_reward(self, env):
        """Reward for maintaining balance during manipulation."""
        com_pos = env.center_of_mass
        support_polygon = env.support_polygon_center
        com_error = torch.norm(com_pos[:, :2] - support_polygon[:, :2], dim=1)
        return torch.exp(-com_error / 0.2)
```

## Curriculum Learning

### Velocity Command Curriculum

```python
class VelocityCurriculum:
    """Curriculum for velocity commands."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.current_level = 0
        self.max_level = 5

        # Velocity ranges per level
        self.velocity_ranges = [
            (0.0, 0.5),   # Level 0: Slow walking
            (0.0, 1.0),   # Level 1: Normal walking
            (0.0, 1.5),   # Level 2: Fast walking
            (0.0, 2.0),   # Level 3: Jogging
            (0.0, 3.0),   # Level 4: Running
            (0.0, 4.0),   # Level 5: Fast running
        ]

    def update(self, success_rate):
        """Update curriculum based on success rate."""
        if success_rate > 0.8 and self.current_level < self.max_level:
            self.current_level += 1
            print(f"Curriculum level increased to {self.current_level}")
        elif success_rate < 0.5 and self.current_level > 0:
            self.current_level -= 1
            print(f"Curriculum level decreased to {self.current_level}")

    def sample_command(self, num_envs):
        """Sample velocity commands for current level."""
        vel_range = self.velocity_ranges[self.current_level]
        return torch.rand(num_envs) * (vel_range[1] - vel_range[0]) + vel_range[0]
```

### Terrain Curriculum

```python
class TerrainCurriculum:
    """Curriculum for terrain difficulty."""

    terrain_types = [
        "flat",           # Level 0
        "rough",          # Level 1
        "slopes",         # Level 2
        "stairs",         # Level 3
        "obstacles",      # Level 4
    ]

    def __init__(self):
        self.current_level = 0

    def get_terrain_config(self):
        """Get terrain configuration for current level."""
        terrain = self.terrain_types[self.current_level]

        configs = {
            "flat": {"height_range": 0.0, "roughness": 0.0},
            "rough": {"height_range": 0.05, "roughness": 0.02},
            "slopes": {"height_range": 0.1, "max_slope": 15.0},
            "stairs": {"step_height": 0.1, "step_width": 0.3},
            "obstacles": {"obstacle_height": 0.2, "density": 0.1},
        }

        return configs[terrain]
```

## Performance Optimization

### GPU Memory Management

```python
# For large-scale training
import torch

# Limit GPU memory for Isaac Sim
torch.cuda.set_per_process_memory_fraction(0.8)

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = compute_loss()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Parallel Environment Scaling

| GPU | Recommended num_envs | Memory Usage |
|-----|---------------------|--------------|
| RTX 3080 (10GB) | 2048 | ~8 GB |
| RTX 3090 (24GB) | 4096 | ~16 GB |
| RTX 4090 (24GB) | 8192 | ~20 GB |
| A100 (40GB) | 16384 | ~32 GB |
| A100 (80GB) | 32768 | ~64 GB |

### Training Speed

| Configuration | Steps/Second | Time for 100M steps |
|---------------|--------------|---------------------|
| 2048 envs, RTX 3090 | ~50,000 | ~30 min |
| 4096 envs, RTX 4090 | ~100,000 | ~15 min |
| 8192 envs, A100 | ~200,000 | ~8 min |
| Multi-GPU (4x A100) | ~600,000 | ~3 min |

## Integration with VLA Pipeline

### Full Training Pipeline

```
1. VLM Pretraining (Stage 1)
   └── Vision-language alignment

2. Motion Primitive Learning (Stage 2)
   └── VAE on MoCap data

3. Locomotion Training (Stage 3)
   └── Online RL in Isaac Lab  ← This document

4. Manipulation Training (Stage 4)
   └── BC + Online RL

5. Whole-Body Control (Stage 5)
   └── Combined loco-manipulation

6. Deployment (Stage 6)
   └── Real robot with safety shield
```

### Loading Pretrained Weights

```python
from model.embodiment.humanoid import HumanoidVLA

# Load locomotion policy from Isaac Lab training
locomotion_checkpoint = torch.load("output/isaac_lab/humanoid_walk.pt")

# Load into VLA model
model = HumanoidVLA.from_pretrained("./output/stage2_primitives")
model.locomotion_policy.load_state_dict(locomotion_checkpoint["policy"])

# Continue with manipulation training
# ...
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Too many envs | Reduce num_envs |
| Slow training | CPU bottleneck | Use GPU physics |
| Unstable training | Large gradients | Reduce learning rate |
| Poor convergence | Bad reward design | Check reward scaling |
| Physics instability | dt too large | Reduce physics_dt |

### Debug Commands

```bash
# Check GPU usage inside container
nvidia-smi

# View training logs
tail -f output/logs/train.log

# Check Isaac Lab environment
./isaaclab.sh -p -c "from omni.isaac.lab.envs import *; print('OK')"

# Test environment step
./isaaclab.sh -p source/standalone/demos/sensors/run_sensors.py --task Isaac-Humanoid-v0
```

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)
- [rl_games Documentation](https://github.com/Denys88/rl_games)
- [HumanoidVLA Architecture](embodiment.md)
- [Locomotion Training](training_locomotion.md)
- [Deployment Guide](deployment.md)
