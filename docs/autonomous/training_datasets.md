# Training Datasets for VLA

This document provides comprehensive documentation of all datasets used in the VLA training pipeline for autonomous driving and robot manipulation tasks.

## Dataset Overview by Training Stage

| Training Stage | Dataset | Public Source | Size | Description |
|----------------|---------|---------------|------|-------------|
| **Stage 1a: VLM Alignment** | LLaVA-Pretrain | [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | 558K | Image-caption pairs for vision-language alignment |
| **Stage 1b: Instruction Tuning** | LLaVA-Instruct-150K | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150k) | 150K | Visual instruction-response pairs |
| **Stage 2: Action Head (Manipulation)** | LeRobot PushT | [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) | ~25K | 2D manipulation task |
| **Stage 2: Action Head (Manipulation)** | LeRobot ALOHA | [lerobot/aloha_sim_*](https://huggingface.co/datasets/lerobot) | ~50K | Bimanual manipulation |
| **Stage 2: Action Head (Manipulation)** | Open X-Embodiment | [jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) | 1M+ | Multi-robot demonstrations with language |
| **Stage 2: Action Head (Driving)** | nuScenes | [nuscenes.org](https://www.nuscenes.org/) | 1000 scenes | Real-world multi-modal driving |
| **Stage 2: Action Head (Driving)** | CARLA Autopilot | Local/Custom | Variable | Simulated urban driving |
| **Stage 2: Action Head (Driving)** | comma.ai 2k19 | [commaai/comma2k19](https://huggingface.co/datasets/commaai/comma2k19) | 33 hours | Real driving with CAN bus |
| **Stage 3: Offline RL** | D4RL | [rail-berkeley/d4rl](https://github.com/rail-berkeley/d4rl) | Variable | Standard offline RL benchmarks |
| **Stage 3: Offline RL** | Visual D4RL | [v-d4rl](https://github.com/conglu1997/v-d4rl) | Variable | Pixel-based offline RL |
| **Stage 3: Offline RL** | RoboMimic | [robomimic.github.io](https://robomimic.github.io/) | Variable | Robot manipulation trajectories |

---

## VLM Pretraining Datasets

### LLaVA-Pretrain (Stage 1a)

**Purpose**: Vision-language alignment

```python
# Usage in training
from datasets import load_dataset

dataset = load_dataset("liuhaotian/LLaVA-Pretrain", split="train")
```

| Attribute | Value |
|-----------|-------|
| Size | 558K image-caption pairs |
| Source | CC3M filtered |
| Format | Image + caption text |
| HuggingFace | [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |

**Data Structure**:
```python
{
    "image": PIL.Image,           # RGB image
    "conversations": [            # Conversation pairs
        {"from": "human", "value": "<image>\nDescribe this image."},
        {"from": "gpt", "value": "A photo of..."},
    ]
}
```

**Training Script**:
```bash
python train/pretrain/vlm_pretrainer.py \
    --alignment-dataset liuhaotian/LLaVA-Pretrain \
    --alignment-epochs 1 \
    --alignment-lr 1e-3
```

### LLaVA-Instruct-150K (Stage 1b)

**Purpose**: Visual instruction tuning

| Attribute | Value |
|-----------|-------|
| Size | 150K instruction-response pairs |
| Source | GPT-4 generated from COCO |
| Format | Image + multi-turn conversation |
| HuggingFace | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150k) |

**Data Structure**:
```python
{
    "image": PIL.Image,
    "conversations": [
        {"from": "human", "value": "What is happening in this image?"},
        {"from": "gpt", "value": "In this image, we can see..."},
        {"from": "human", "value": "What color is the car?"},
        {"from": "gpt", "value": "The car is red..."},
    ]
}
```

### Additional VLM Datasets

| Dataset | Size | Source | Use Case |
|---------|------|--------|----------|
| LAION-400M | 400M | [laion.ai](https://laion.ai/) | Large-scale pretraining |
| CC3M | 3M | [Google](https://github.com/google-research-datasets/conceptual-captions) | Conceptual captions |
| CC12M | 12M | [Google](https://github.com/google-research-datasets/conceptual-12m) | Extended captions |
| GraspNet-1Billion | 1B | [graspnet.net](https://graspnet.net/) | 6-DoF grasp poses |
| NYU Depth V2 | 1.4K | [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) | RGB-D indoor scenes |

---

## LeRobot Datasets (Manipulation)

LeRobot provides standardized robot learning datasets on HuggingFace. These are excellent for imitation learning and VLA fine-tuning.

### PushT Dataset

**Purpose**: Simple 2D manipulation task for initial testing

```python
from train.datasets.lerobot_dataset import PushTDataset

dataset = PushTDataset(split="train", chunk_size=10)
```

| Attribute | Value |
|-----------|-------|
| Size | ~25K samples |
| Action Dim | 2 (x, y velocity) |
| Image Size | 96x96 or 224x224 |
| FPS | 50 |
| HuggingFace | [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) |

**Data Structure**:
```python
{
    "observation.image": torch.Tensor,    # (C, H, W)
    "observation.state": torch.Tensor,    # (8,) robot state
    "action": torch.Tensor,               # (2,) or (chunk_size, 2)
    "episode_index": int,
    "frame_index": int,
}
```

### ALOHA Bimanual Datasets

**Purpose**: Dual-arm manipulation tasks

```python
from train.datasets.lerobot_dataset import AlohaDataset

dataset = AlohaDataset(task="insertion", split="train", chunk_size=100)
```

| Task | HuggingFace | Description |
|------|-------------|-------------|
| insertion | [lerobot/aloha_sim_insertion_human](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) | Peg-in-hole insertion |
| transfer_cube | [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) | Cube transfer between arms |

| Attribute | Value |
|-----------|-------|
| Action Dim | 14 (7 per arm: 6 joint + gripper) |
| Image Size | 480x640 |
| Chunk Size | 100 (ACT-style) |

### xArm Datasets

**Purpose**: Single-arm manipulation with xArm robot

```python
from train.datasets.lerobot_dataset import XArmDataset

dataset = XArmDataset(task="lift", split="train")
```

| Task | HuggingFace | Action Dim |
|------|-------------|------------|
| lift | [lerobot/xarm_lift_medium](https://huggingface.co/datasets/lerobot/xarm_lift_medium) | 4-7 |
| push | [lerobot/xarm_push_medium](https://huggingface.co/datasets/lerobot/xarm_push_medium) | 4-7 |

### Other LeRobot Datasets

| Dataset | HuggingFace | Description |
|---------|-------------|-------------|
| UCSD Kitchen | [lerobot/ucsd_kitchen_dataset_converted](https://huggingface.co/datasets/lerobot/ucsd_kitchen_dataset_converted_externally_to_rlds) | Kitchen manipulation |
| UMI Cup | [lerobot/umi_cup_in_the_wild](https://huggingface.co/datasets/lerobot/umi_cup_in_the_wild) | Real-world cup manipulation |

---

## Open X-Embodiment Datasets

The Open X-Embodiment dataset is a large-scale collection of robot manipulation data from multiple institutions.

### Overview

```python
from train.datasets.openx_dataset import BridgeDataset, RT1Dataset

# Bridge V2 dataset
bridge_dataset = BridgeDataset(split="train", max_samples=10000)

# RT-1 dataset
rt1_dataset = RT1Dataset(split="train", max_samples=5000)
```

### Available Datasets

| Dataset | Source | Robot | Episodes | Actions |
|---------|--------|-------|----------|---------|
| Bridge V2 | [bridge_dataset](https://rail.eecs.berkeley.edu/datasets/bridge_release/) | WidowX | 60K | 7D EE + gripper |
| RT-1 | fractal20220817_data | Everyday Robots | 130K | 11D (arm + base) |
| TACO Play | taco_play | Various | 50K | 7D |
| JACO Play | jaco_play | JACO | 30K | 7D |
| Berkeley Cable | berkeley_cable_routing | UR5 | 5K | 6D |
| Language Table | language_table | xArm | 100K | 2D |

### Data Structure

```python
{
    "pixel_values": torch.Tensor,       # (C, H, W) main camera
    "wrist_image": torch.Tensor,        # (C, H, W) wrist camera (optional)
    "state": torch.Tensor,              # Robot proprioception
    "action": torch.Tensor,             # Robot action
    "instruction": str,                 # Language instruction
    "is_terminal": bool,
    "is_first": bool,
}
```

### Usage for VLA Training

```python
from train.datasets.openx_dataset import create_openx_dataloader

dataset = BridgeDataset(
    split="train",
    image_size=(224, 224),
    use_language=True,
    use_wrist_camera=True,
)

dataloader = create_openx_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)
```

---

## Autonomous Driving Datasets

### nuScenes

**Purpose**: Multi-modal real-world driving with rich sensor data

```python
from train.datasets.driving_dataset import NuScenesDataset

dataset = NuScenesDataset(
    data_root="/path/to/nuscenes",
    version="v1.0-trainval",
    split="train",
    use_lidar=True,
    use_waypoints=True,
)
```

| Attribute | Value |
|-----------|-------|
| Size | 1000 scenes, 1.4M samples |
| Cameras | 6 (360° coverage) |
| LiDAR | 32-beam, 20 FPS |
| Radar | 5 units |
| Annotations | 3D boxes, maps, CAN bus |
| Website | [nuscenes.org](https://www.nuscenes.org/) |

**Sensor Configuration**:
```python
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
```

**Action Formats**:
```python
# Waypoint prediction (default)
action = torch.Tensor  # (10, 2) - 10 future waypoints (x, y)

# Vehicle control
action = torch.Tensor  # (3,) - [steering, throttle, brake]
```

### CARLA Dataset

**Purpose**: Simulated driving with perfect ground truth

```python
from train.datasets.driving_dataset import CarlaDataset

dataset = CarlaDataset(
    data_root="/path/to/carla_data",
    split="train",
    use_lidar=True,
)
```

| Attribute | Value |
|-----------|-------|
| Format | Episode directories |
| Cameras | Configurable (front, multi-view) |
| LiDAR | Configurable (points per scan) |
| Actions | Steering, throttle, brake |

**Expected Directory Structure**:
```
data_root/
├── train/
│   ├── episode_000/
│   │   ├── camera_front/
│   │   │   ├── 000000.png
│   │   │   └── ...
│   │   ├── lidar/
│   │   │   ├── 000000.npy
│   │   │   └── ...
│   │   └── measurements/
│   │       ├── 000000.json
│   │       └── ...
│   └── episode_001/
│       └── ...
└── val/
    └── ...
```

### comma.ai Dataset

**Purpose**: Real-world driving with CAN bus data

```python
from train.datasets.driving_dataset import CommaAIDataset

dataset = CommaAIDataset(
    split="train",
    max_samples=10000,
)
```

| Attribute | Value |
|-----------|-------|
| Size | 33 hours of driving |
| Camera | Front-facing |
| Actions | Steering angle, gas, brake |
| HuggingFace | [commaai/comma2k19](https://huggingface.co/datasets/commaai/comma2k19) |

### Waymo Open Dataset

**Purpose**: Large-scale real-world driving

| Attribute | Value |
|-----------|-------|
| Size | 1150 scenes |
| Cameras | 5 cameras |
| LiDAR | 64-beam, high-res |
| Website | [waymo.com/open](https://waymo.com/open) |

---

## Offline RL Datasets

### D4RL

**Purpose**: Standard offline RL benchmarks

```python
import d4rl
import gymnasium as gym

env = gym.make("hopper-medium-v2")
dataset = env.get_dataset()
```

| Environment | Variants | Description |
|-------------|----------|-------------|
| HalfCheetah | random, medium, expert, medium-expert, medium-replay | Locomotion |
| Hopper | random, medium, expert, medium-expert, medium-replay | Locomotion |
| Walker2d | random, medium, expert, medium-expert, medium-replay | Locomotion |
| Ant | random, medium, expert, medium-expert, medium-replay | Locomotion |
| Maze2d | umaze, medium, large | Navigation |
| AntMaze | umaze, medium, large | Navigation |

**Data Structure**:
```python
{
    "observations": np.ndarray,        # (N, obs_dim)
    "actions": np.ndarray,             # (N, action_dim)
    "rewards": np.ndarray,             # (N,)
    "next_observations": np.ndarray,   # (N, obs_dim)
    "terminals": np.ndarray,           # (N,)
}
```

### Visual D4RL (VD4RL)

**Purpose**: Pixel-based offline RL

| Attribute | Value |
|-----------|-------|
| Image Size | 84x84 |
| Environments | DMControl suite |
| GitHub | [conglu1997/v-d4rl](https://github.com/conglu1997/v-d4rl) |

### RoboMimic

**Purpose**: Robot manipulation trajectories for offline learning

| Attribute | Value |
|-----------|-------|
| Tasks | Lift, Can, Square, Transport, etc. |
| Data Quality | Human (ph), Machine (mh), Mixed (mg) |
| Website | [robomimic.github.io](https://robomimic.github.io/) |

---

## Dataset Loading Best Practices

### 1. Efficient Data Loading

```python
from torch.utils.data import DataLoader

# Use multiple workers
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    prefetch_factor=2,      # Prefetch batches
    persistent_workers=True, # Keep workers alive
)
```

### 2. Data Augmentation

```python
import torchvision.transforms as T

transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 3. Mixed Dataset Training

```python
from torch.utils.data import ConcatDataset, WeightedRandomSampler

# Combine multiple datasets
datasets = [
    PushTDataset(split="train"),
    AlohaDataset(task="insertion", split="train"),
    XArmDataset(task="lift", split="train"),
]
combined = ConcatDataset(datasets)

# Weight by dataset size
weights = [1.0 / len(d) for d in datasets for _ in range(len(d))]
sampler = WeightedRandomSampler(weights, len(combined))
```

### 4. Streaming Large Datasets

```python
from train.datasets.openx_dataset import OpenXStreamingDataset

# For datasets that don't fit in memory
streaming_dataset = OpenXStreamingDataset(
    dataset_name="bridge_dataset",
    split="train",
    shuffle_buffer=1000,
)
```

---

## Data Format Conversion

### Converting Custom Data to LeRobot Format

```python
import numpy as np
from pathlib import Path

def convert_to_lerobot_format(
    episodes_dir: str,
    output_path: str,
):
    """Convert custom episodes to LeRobot format."""
    all_data = {
        "observation.image": [],
        "observation.state": [],
        "action": [],
        "episode_index": [],
        "frame_index": [],
    }

    for ep_idx, ep_dir in enumerate(sorted(Path(episodes_dir).iterdir())):
        # Load episode data
        images = np.load(ep_dir / "images.npy")
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")

        for frame_idx in range(len(actions)):
            all_data["observation.image"].append(images[frame_idx])
            all_data["observation.state"].append(states[frame_idx])
            all_data["action"].append(actions[frame_idx])
            all_data["episode_index"].append(ep_idx)
            all_data["frame_index"].append(frame_idx)

    # Save as HuggingFace dataset
    from datasets import Dataset
    dataset = Dataset.from_dict(all_data)
    dataset.save_to_disk(output_path)
```

### Converting to World Model Format

```python
def convert_to_world_model_format(
    source_dir: str,
    output_dir: str,
):
    """Convert episodes to world model training format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for ep_path in Path(source_dir).glob("episode_*"):
        # Load data
        images = np.load(ep_path / "images.npy")
        actions = np.load(ep_path / "actions.npy")
        rewards = np.load(ep_path / "rewards.npy")

        # Save in expected format
        np.savez(
            output_path / f"{ep_path.name}.npz",
            observations=images,
            actions=actions,
            rewards=rewards,
            dones=np.zeros(len(rewards)),
        )
```

---

## Downloading Datasets

### From HuggingFace

```bash
# Using datasets library
pip install datasets

# In Python
from datasets import load_dataset
dataset = load_dataset("lerobot/pusht", split="train")

# Or using huggingface-cli
huggingface-cli download lerobot/pusht
```

### nuScenes

```bash
# Download nuScenes
# 1. Register at https://www.nuscenes.org/
# 2. Download mini set for testing
wget https://www.nuscenes.org/data/v1.0-mini.tgz

# Or full dataset (requires login)
pip install nuscenes-devkit
```

### D4RL

```bash
pip install d4rl

# Datasets download automatically on first use
python -c "
import d4rl
import gymnasium as gym
env = gym.make('hopper-medium-v2')
dataset = env.get_dataset()
"
```

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Complete training pipeline
- [Imitation Learning](training_imitation_learning.md) - BC, DAgger, GAIL
- [Reinforcement Learning](training_reinforcement_learning.md) - Offline and online RL
- [Deployment](deployment.md) - Model export
