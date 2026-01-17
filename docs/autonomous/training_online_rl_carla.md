# Online RL Training with CARLA in Docker

This document covers setting up and running online reinforcement learning for autonomous driving using CARLA simulator in Docker.

## Overview

Online RL requires a live simulator to interact with the policy during training. CARLA provides realistic urban driving simulation with sensor data (cameras, LiDAR, radar) for training autonomous driving VLA models.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Online RL Training Architecture                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐         ┌──────────────────────────────┐  │
│  │   CARLA Docker       │         │   Training Process           │  │
│  │   Container          │  TCP    │                              │  │
│  │  ┌────────────────┐  │◄───────►│  ┌────────────────────────┐  │  │
│  │  │ CARLA Server   │  │  2000   │  │  DrivingVLA Model      │  │  │
│  │  │ - Physics      │  │         │  │  - Vision Encoder      │  │  │
│  │  │ - Sensors      │  │         │  │  - Language Model      │  │  │
│  │  │ - Traffic      │  │         │  │  - Action Head         │  │  │
│  │  └────────────────┘  │         │  └────────────────────────┘  │  │
│  │                      │         │                              │  │
│  │  GPU: Rendering      │         │  GPU: Training               │  │
│  └──────────────────────┘         └──────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup Options

### Option 1: Simulator in Docker, Training Outside (Recommended for Development)

**Pros:**
- Easier debugging and development
- Simpler dependency management
- Can use existing Python environment
- Faster iteration

**Cons:**
- Less reproducible
- Manual environment setup

### Option 2: Both in Docker (Recommended for Production)

**Pros:**
- Fully reproducible
- Easy cloud/cluster deployment
- Version-controlled environment

**Cons:**
- More complex setup
- Requires multi-GPU for best performance

## CARLA Docker Setup

### 1. Pull CARLA Docker Image

```bash
# Official CARLA image
docker pull carlasim/carla:0.9.15

# For specific versions
docker pull carlasim/carla:0.9.14
docker pull carlasim/carla:0.9.13
```

### 2. Run CARLA Server

#### Basic Run (with Display)

```bash
docker run --privileged --gpus all --net=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  carlasim/carla:0.9.15 \
  /bin/bash ./CarlaUE4.sh -quality-level=Low
```

#### Headless Mode (Recommended for Training)

```bash
docker run --privileged --gpus all \
  -p 2000-2002:2000-2002 \
  carlasim/carla:0.9.15 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port=2000
```

#### With Custom Settings

```bash
docker run --privileged --gpus all \
  -p 2000-2002:2000-2002 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  carlasim/carla:0.9.15 \
  /bin/bash ./CarlaUE4.sh \
    -RenderOffScreen \
    -nosound \
    -carla-port=2000 \
    -quality-level=Low \
    -fps=20
```

### 3. Verify Connection

```python
import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
print(f"Connected to CARLA {client.get_server_version()}")
```

## Training Configuration

### Using CARLABridge

Your codebase provides `CARLABridge` in `integration/simulator_bridge.py`:

```python
from integration.simulator_bridge import CARLABridge, SimulatorConfig

config = SimulatorConfig(
    # Connection
    carla_host="localhost",
    carla_port=2000,
    carla_town="Town01",

    # Rendering
    render=True,
    headless=False,
    fps=30,

    # Physics
    physics_dt=1.0 / 60.0,  # 60 Hz physics
    control_dt=1.0 / 30.0,  # 30 Hz control

    # Camera
    image_width=224,
    image_height=224,
)

bridge = CARLABridge(config)
bridge.initialize()
```

### PPO Training with CARLA

```python
from train.online_rl.ppo_trainer import PPOTrainer, PPOConfig
from model.embodiment.autonomous_vehicle import DrivingVLA, DrivingVLAConfig
from integration.simulator_bridge import CARLABridge, SimulatorConfig

# Setup simulator
sim_config = SimulatorConfig(
    carla_host="localhost",
    carla_port=2000,
    carla_town="Town01",
)
env = CARLABridge(sim_config)
env.initialize()

# Setup model
model_config = DrivingVLAConfig(
    num_cameras=1,
    image_size=224,
    action_dim=3,  # throttle, steer, brake
    trajectory_length=10,
)
model = DrivingVLA(model_config)

# Setup PPO
ppo_config = PPOConfig(
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    num_epochs=10,
    batch_size=64,
    num_steps=2048,
)

trainer = PPOTrainer(
    model=model,
    env=env,
    config=ppo_config,
)

# Train
trainer.train(total_timesteps=1_000_000)
```

### SAC Training with CARLA

```python
from train.online_rl.sac_trainer import SACTrainer, SACConfig

sac_config = SACConfig(
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    batch_size=256,
    buffer_size=1_000_000,
)

trainer = SACTrainer(
    model=model,
    env=env,
    config=sac_config,
)

trainer.train(total_timesteps=1_000_000)
```

## Docker Compose Setup

### Single GPU Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  carla:
    image: carlasim/carla:0.9.15
    container_name: carla-server
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "2000-2002:2000-2002"
    command: /bin/bash ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low
    healthcheck:
      test: ["CMD", "python3", "-c", "import socket; s=socket.socket(); s.connect(('localhost',2000)); s.close()"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Multi-GPU Configuration (Simulator + Training)

```yaml
# docker-compose-multi-gpu.yml
version: '3.8'

services:
  carla:
    image: carlasim/carla:0.9.15
    container_name: carla-server
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    ports:
      - "2000-2002:2000-2002"
    command: /bin/bash ./CarlaUE4.sh -RenderOffScreen -nosound
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  training:
    build:
      context: .
      dockerfile: Dockerfile.training
    container_name: vla-training
    runtime: nvidia
    depends_on:
      carla:
        condition: service_healthy
    environment:
      - CARLA_HOST=carla
      - CARLA_PORT=2000
      - NVIDIA_VISIBLE_DEVICES=1
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./output:/app/output
      - ./checkpoints:/app/checkpoints
    command: >
      python train/online_rl/ppo_trainer.py
        --env carla
        --carla-host carla
        --carla-port 2000
        --total-timesteps 1000000
        --output-dir /app/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

volumes:
  output:
  checkpoints:
```

### Training Dockerfile

```dockerfile
# Dockerfile.training
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CARLA Python API
RUN pip install carla==0.9.15

# Copy source code
COPY . .

# Default command
CMD ["python", "train/online_rl/ppo_trainer.py"]
```

## Training Scripts

### Shell Script for Development

```bash
#!/bin/bash
# scripts/run_carla_online_rl.sh

# Configuration
CARLA_VERSION="0.9.15"
CARLA_PORT=2000
CARLA_TOWN="Town01"
TOTAL_TIMESTEPS=1000000
OUTPUT_DIR="./output/carla_ppo"

# Start CARLA in Docker
echo "Starting CARLA server..."
docker run -d --name carla-server \
  --gpus all \
  -p ${CARLA_PORT}-$((CARLA_PORT+2)):${CARLA_PORT}-$((CARLA_PORT+2)) \
  carlasim/carla:${CARLA_VERSION} \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port=${CARLA_PORT}

# Wait for CARLA to start
echo "Waiting for CARLA to initialize..."
sleep 30

# Verify connection
python -c "
import carla
client = carla.Client('localhost', ${CARLA_PORT})
client.set_timeout(10.0)
print(f'Connected to CARLA {client.get_server_version()}')
"

if [ $? -ne 0 ]; then
    echo "Failed to connect to CARLA"
    docker stop carla-server
    docker rm carla-server
    exit 1
fi

# Run training
echo "Starting training..."
python train/online_rl/ppo_trainer.py \
  --env carla \
  --carla-host localhost \
  --carla-port ${CARLA_PORT} \
  --carla-town ${CARLA_TOWN} \
  --total-timesteps ${TOTAL_TIMESTEPS} \
  --output-dir ${OUTPUT_DIR}

# Cleanup
echo "Stopping CARLA server..."
docker stop carla-server
docker rm carla-server

echo "Training complete!"
```

### Python Training Script

```python
#!/usr/bin/env python3
"""
Online RL training with CARLA.

Usage:
    python train/online_rl/train_carla_ppo.py \
        --carla-host localhost \
        --carla-port 2000 \
        --total-timesteps 1000000
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from integration.simulator_bridge import CARLABridge, SimulatorConfig
from model.embodiment.autonomous_vehicle import DrivingVLA, DrivingVLAConfig
from train.online_rl.ppo_trainer import PPOTrainer, PPOConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-host", type=str, default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-town", type=str, default="Town01")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=str, default="./output/carla_ppo")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Setup environment
    sim_config = SimulatorConfig(
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        carla_town=args.carla_town,
        image_width=224,
        image_height=224,
    )
    env = CARLABridge(sim_config)
    env.initialize()

    # Setup model
    model_config = DrivingVLAConfig(
        num_cameras=1,
        image_size=224,
        action_dim=3,
    )
    model = DrivingVLA(model_config).to(args.device)

    # Setup trainer
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        num_steps=2048,
        batch_size=64,
    )
    trainer = PPOTrainer(model, env, ppo_config)

    # Train
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.train(
        total_timesteps=args.total_timesteps,
        save_dir=args.output_dir,
    )

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
```

## Reward Design

### Driving Reward Components

```python
def compute_driving_reward(env, action, info):
    """
    Compute reward for autonomous driving.

    Components:
    - Speed reward: Encourage target speed
    - Lane keeping: Stay in lane
    - Collision penalty: Avoid collisions
    - Comfort: Smooth driving
    """
    reward = 0.0

    # Speed reward (target: 30 km/h = 8.33 m/s)
    velocity = info.get("velocity", [0, 0, 0])
    speed = np.linalg.norm(velocity[:2])
    target_speed = 8.33
    speed_reward = 1.0 - abs(speed - target_speed) / target_speed
    speed_reward = max(0, speed_reward)
    reward += speed_reward * 1.0

    # Lane keeping reward
    lane_deviation = info.get("lane_deviation", 0.0)
    lane_reward = max(0, 1.0 - abs(lane_deviation) / 2.0)
    reward += lane_reward * 0.5

    # Collision penalty
    if info.get("collision", False):
        reward -= 10.0

    # Comfort reward (smooth steering/acceleration)
    steering = abs(action[1])
    acceleration = abs(action[0] - action[2])
    comfort_reward = 1.0 - 0.5 * steering - 0.3 * acceleration
    reward += comfort_reward * 0.3

    return reward
```

### Custom Reward Wrapper

```python
class CARLARewardWrapper:
    """Wrapper to customize CARLA rewards."""

    def __init__(self, env, reward_config=None):
        self.env = env
        self.config = reward_config or {}

        # Reward weights
        self.speed_weight = self.config.get("speed_weight", 1.0)
        self.lane_weight = self.config.get("lane_weight", 0.5)
        self.collision_penalty = self.config.get("collision_penalty", 10.0)
        self.target_speed = self.config.get("target_speed", 8.33)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Override reward
        custom_reward = self.compute_reward(action, info)

        return obs, custom_reward, terminated, truncated, info

    def compute_reward(self, action, info):
        # ... reward computation
        pass
```

## CARLA Maps and Scenarios

### Available Towns

| Town | Description | Best For |
|------|-------------|----------|
| Town01 | Small town, simple layout | Basic training |
| Town02 | Small town, more junctions | Junction handling |
| Town03 | Larger town, highway | Highway driving |
| Town04 | Small town with highway loop | Mixed scenarios |
| Town05 | Urban grid, multi-lane | Urban driving |
| Town06 | Low density, long roads | Highway training |
| Town07 | Rural, narrow roads | Rural driving |
| Town10 | Urban, complex junctions | Advanced training |

### Scenario Configuration

```python
def setup_training_scenario(world, scenario="urban"):
    """Setup CARLA training scenario."""

    if scenario == "urban":
        # Dense traffic, pedestrians
        traffic_manager = world.get_traffic_manager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        spawn_vehicles(world, num_vehicles=50)
        spawn_pedestrians(world, num_pedestrians=30)

    elif scenario == "highway":
        # Fast traffic, no pedestrians
        world.load_world("Town06")
        spawn_vehicles(world, num_vehicles=30, vehicle_type="fast")

    elif scenario == "night":
        # Night driving
        weather = carla.WeatherParameters.ClearNight
        world.set_weather(weather)

    elif scenario == "rain":
        # Wet conditions
        weather = carla.WeatherParameters.WetCloudySunset
        world.set_weather(weather)
```

## Performance Optimization

### GPU Memory Management

```python
# Limit CARLA GPU memory (environment variable)
import os
os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"  # For CARLA
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # For training

# Or use Docker device mapping
# docker run --gpus '"device=0"' carlasim/carla...
```

### Training Speed Tips

| Optimization | Command/Setting | Impact |
|--------------|-----------------|--------|
| Lower quality | `-quality-level=Low` | 2-3x faster |
| Fixed FPS | `-fps=20` | Consistent timing |
| Offscreen | `-RenderOffScreen` | Required for headless |
| No sound | `-nosound` | Minor speedup |
| Synchronous mode | `settings.synchronous_mode = True` | Stable training |

### Expected Performance

| GPU | CARLA FPS | Training Throughput |
|-----|-----------|---------------------|
| RTX 3080 | 20-30 | ~500 steps/min |
| RTX 3090 | 25-35 | ~700 steps/min |
| RTX 4090 | 30-50 | ~1000 steps/min |
| A100 | 40-60 | ~1500 steps/min |

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused | CARLA not ready | Wait longer, check port |
| GPU out of memory | Both on same GPU | Use separate GPUs or reduce batch |
| Slow training | Synchronous mode off | Enable synchronous mode |
| Sensor lag | Buffer overflow | Reduce sensor frequency |
| Crashes | Memory leak | Restart CARLA periodically |

### Debug Commands

```bash
# Check CARLA container status
docker ps -a | grep carla

# View CARLA logs
docker logs carla-server

# Check GPU usage
nvidia-smi -l 1

# Test connection
python -c "import carla; c=carla.Client('localhost',2000); print(c.get_server_version())"
```

## Integration with VLA Pipeline

### Full Training Pipeline

```
1. Pretrain VLM (Stage 1)
   └── Vision-language alignment

2. Offline IL/RL (Stage 2)
   └── nuScenes, CARLA recordings

3. Online RL with CARLA (Stage 3)  ← This document
   └── PPO/SAC fine-tuning

4. Deployment (Stage 4)
   └── Export to ONNX/TorchScript
```

### Loading Pretrained Weights

```python
from model.embodiment.autonomous_vehicle import DrivingVLA

# Load from offline training
model = DrivingVLA.from_pretrained("./output/stage2_offline/best")

# Continue with online RL
trainer = PPOTrainer(model, env, config)
trainer.train(total_timesteps=500_000)

# Save final model
model.save_pretrained("./output/stage3_online/final")
```

## References

- [CARLA Documentation](https://carla.readthedocs.io/)
- [CARLA Docker Guide](https://carla.readthedocs.io/en/latest/build_docker/)
- [DrivingVLA Architecture](embodiment.md)
- [Offline RL Training](training_reinforcement_learning.md)
- [Deployment Guide](deployment.md)
