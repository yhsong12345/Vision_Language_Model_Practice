# Humanoid Training Datasets

This document provides comprehensive documentation for all datasets used in humanoid VLA training, including sources, formats, and usage guidelines.

## Dataset Overview

| Training Stage | Dataset | Size | Format | Source |
|----------------|---------|------|--------|--------|
| VLM Foundation | LLaVA-Pretrain | 558K | JSON + Images | HuggingFace |
| VLM Foundation | LLaVA-Instruct-150K | 150K | JSON + Images | HuggingFace |
| Motion Primitives | CMU MoCap | 2600+ sequences | BVH | CMU |
| Motion Primitives | Human3.6M | 3.6M+ poses | H5/CSV | University |
| Motion Primitives | AMASS | 40+ hours | NPZ | MPI |
| Locomotion | D4RL MuJoCo | 1M+ transitions | HDF5 | HuggingFace |
| Manipulation | DROID | 76K episodes | HDF5 | HuggingFace |
| Manipulation | LeRobot ALOHA | 100K+ frames | Parquet | HuggingFace |
| Whole-Body | HumanoidBench | Simulation | Various | GitHub |
| HRI | HandoverSim | Simulation | Various | GitHub |

---

## VLM Foundation Datasets

### LLaVA-Pretrain

**Purpose**: Vision-language alignment for Stage 1a

| Attribute | Value |
|-----------|-------|
| Source | [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |
| Size | 558K image-caption pairs |
| Format | JSON + Images |
| License | CC-BY-NC-4.0 |

```python
from datasets import load_dataset

dataset = load_dataset("liuhaotian/LLaVA-Pretrain")

# Sample format
sample = {
    "id": "000000",
    "image": <PIL.Image>,
    "conversations": [
        {"from": "human", "value": "<image>\nDescribe this image."},
        {"from": "gpt", "value": "A humanoid robot standing in a room."}
    ]
}
```

### LLaVA-Instruct-150K

**Purpose**: Visual instruction tuning for Stage 1b

| Attribute | Value |
|-----------|-------|
| Source | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
| Size | 150K instruction pairs |
| Format | JSON + Images |
| License | CC-BY-NC-4.0 |

**Humanoid-relevant instruction types:**
- Object manipulation: "Pick up the object"
- Navigation: "Walk to the location"
- Gestures: "Wave your hand"
- Scene understanding: "What objects can you see?"

---

## Motion Capture Datasets

### CMU Motion Capture Database

**Purpose**: Motion primitive learning and reference motion

| Attribute | Value |
|-----------|-------|
| Source | [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/) |
| Size | 2600+ motion sequences |
| Format | BVH, AMC/ASF |
| License | Free for research |

**Categories:**
- Walking, Running, Jumping
- Sports (basketball, soccer, martial arts)
- Daily activities (sitting, reaching, carrying)
- Interactions (handshake, passing objects)

```python
from train.datasets.mocap_dataset import MoCapDataset

dataset = MoCapDataset(
    data_root="./data/cmu_mocap",
    format="bvh",
    retarget_to_robot=True,
    robot_urdf="./robots/humanoid.urdf",
)

# Sample format
sample = {
    "joint_positions": np.array([...]),  # (T, num_joints)
    "joint_velocities": np.array([...]), # (T, num_joints)
    "root_position": np.array([...]),    # (T, 3)
    "root_orientation": np.array([...]), # (T, 4) quaternion
    "motion_label": "walking",
}
```

### Human3.6M

**Purpose**: 3D human pose estimation and motion learning

| Attribute | Value |
|-----------|-------|
| Source | [vision.imar.ro/human3.6m](http://vision.imar.ro/human3.6m/) |
| Size | 3.6M+ 3D poses |
| Subjects | 11 actors |
| Actions | 17 categories |
| Format | H5, CSV |
| License | Research only (registration required) |

**Action categories:**
- Directions, Discussion, Eating, Greeting
- Phoning, Photo, Posing, Purchases
- Sitting, SittingDown, Smoking, Waiting
- Walking, WalkingDog, WalkingTogether

### AMASS (Archive of Motion Capture as Surface Shapes)

**Purpose**: Unified motion capture for humanoid training

| Attribute | Value |
|-----------|-------|
| Source | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) |
| Size | 40+ hours of motion |
| Subjects | 300+ |
| Format | NPZ (SMPL parameters) |
| License | Research only |

**Included datasets:**
- CMU, Eyes Japan, KIT
- BMLrub, BMLmovi
- ACCAD, TotalCapture

```python
import numpy as np

# Load AMASS motion
data = np.load("amass_motion.npz")

# SMPL parameters
poses = data["poses"]        # (T, 72) body pose
trans = data["trans"]        # (T, 3) root translation
betas = data["betas"]        # (10,) body shape

# Convert to joint angles for robot
from train.datasets.mocap_retargeter import SMPLToRobotRetargeter

retargeter = SMPLToRobotRetargeter(robot_urdf="humanoid.urdf")
robot_joints = retargeter.convert(poses, trans)
```

### Habitat Humanoids (AMASS Subset)

**Purpose**: Ready-to-use motion clips for simulation

| Attribute | Value |
|-----------|-------|
| Source | [ai-habitat/habitat_humanoids](https://huggingface.co/datasets/ai-habitat/habitat_humanoids) |
| Format | NPZ |
| License | Research |

```python
from datasets import load_dataset

dataset = load_dataset("ai-habitat/habitat_humanoids")
```

---

## Locomotion Datasets

### D4RL MuJoCo

**Purpose**: Offline RL for locomotion

| Attribute | Value |
|-----------|-------|
| Source | [imone/D4RL](https://huggingface.co/datasets/imone/D4RL) |
| Environments | Ant, Humanoid, HalfCheetah, Walker2d, Hopper |
| Data types | Random, Medium, Expert, Mixed |
| Format | HDF5 |
| License | Apache 2.0 |

**Humanoid-specific subsets:**
- `humanoid-random-v2`: Random policy
- `humanoid-medium-v2`: Medium-level policy
- `humanoid-expert-v2`: Expert policy
- `humanoid-mixed-v2`: Mixed quality

```python
from datasets import load_dataset

# Load D4RL dataset
dataset = load_dataset("imone/D4RL", "humanoid-medium-v2")

# Sample format
sample = {
    "observations": np.array([...]),  # (obs_dim,)
    "actions": np.array([...]),       # (action_dim,)
    "rewards": float,
    "terminals": bool,
    "next_observations": np.array([...]),
}
```

---

## Manipulation Datasets

### DROID (Diverse Real-world Object-Interacting Data)

**Purpose**: Teleoperation demonstrations for manipulation

| Attribute | Value |
|-----------|-------|
| Source | [cadene/droid](https://huggingface.co/datasets/cadene/droid) |
| Size | 76K episodes |
| Collection | VR teleoperation, MoCap suit |
| Format | HDF5 |
| License | Research |

```python
from datasets import load_dataset

dataset = load_dataset("cadene/droid")

# Sample format
sample = {
    "observation.images.cam0": np.array([...]),  # (H, W, 3)
    "observation.state": np.array([...]),         # Robot state
    "action": np.array([...]),                    # Action
    "language_instruction": "Pick up the cup",
}
```

### LeRobot ALOHA

**Purpose**: Bimanual manipulation demonstrations

| Attribute | Value |
|-----------|-------|
| Source | [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) |
| Type | Bimanual teleoperation |
| Format | Parquet |
| License | Apache 2.0 |

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")

# Sample format
sample = {
    "observation.images.top": np.array([...]),
    "observation.state": np.array([...]),          # 14 joints (7 per arm)
    "action": np.array([...]),                     # 14 joints
}
```

### Open X-Embodiment

**Purpose**: Cross-embodiment manipulation

| Attribute | Value |
|-----------|-------|
| Source | [jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) |
| Robots | 22 robot types |
| Tasks | 160K+ demonstrations |
| Format | RLDS (TensorFlow) |
| License | Various |

```python
from train.datasets.openx_dataset import OpenXDataset

dataset = OpenXDataset(
    data_dir="./data/openx",
    embodiments=["bridge", "rt1", "kuka"],
    normalize_actions=True,
)
```

### ContactDB

**Purpose**: Contact maps for grasp planning

| Attribute | Value |
|-----------|-------|
| Source | [contactdb.cc.gatech.edu](https://contactdb.cc.gatech.edu/) |
| Objects | 50 everyday objects |
| Grasps | 2550 human grasps |
| Format | PLY, JSON |
| License | Research |

---

## Whole-Body Control Datasets

### HumanoidBench

**Purpose**: Benchmark for humanoid whole-body control

| Attribute | Value |
|-----------|-------|
| Source | [humanoid-bench.github.io](https://humanoid-bench.github.io/) |
| Tasks | 27 diverse tasks |
| Environment | MuJoCo, Isaac Gym |
| License | MIT |

**Task categories:**
- Locomotion: Walk, Run, Stand, Balance
- Manipulation: Reach, Grasp, Pick, Place
- Loco-manipulation: Carry, Push, Pull
- Agility: Jump, Climb, Dance

```python
import humanoid_bench

# Create environment
env = humanoid_bench.make("walk")

# Collect demonstrations
obs = env.reset()
for _ in range(1000):
    action = expert_policy(obs)
    next_obs, reward, done, info = env.step(action)
    # Store transition
```

---

## Human-Robot Interaction Datasets

### HandoverSim

**Purpose**: Human-robot handover training

| Attribute | Value |
|-----------|-------|
| Source | [handoversim.github.io](https://handoversim.github.io/) |
| Tasks | Robot-to-human, Human-to-robot handover |
| Environment | Isaac Gym |
| License | MIT |

---

## Simulation Environments (Data Generation)

### MuJoCo

**Purpose**: Physics simulation for data generation

| Attribute | Value |
|-----------|-------|
| Source | [mujoco.org](https://mujoco.org/) |
| Physics | Accurate contact, muscle dynamics |
| Speed | Real-time to faster |
| License | Apache 2.0 |

### Isaac Gym / Isaac Sim

**Purpose**: GPU-accelerated training

| Attribute | Value |
|-----------|-------|
| Source | [developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) |
| Parallel Envs | 4096+ simultaneous |
| GPU | CUDA-accelerated physics |
| License | NVIDIA EULA |

---

## Data Format Specifications

### Standard Episode Format

```python
@dataclass
class HumanoidEpisode:
    """Standard format for humanoid training episodes."""

    # Observations
    images: List[np.ndarray]              # (T, H, W, 3)
    joint_positions: np.ndarray           # (T, 32)
    joint_velocities: np.ndarray          # (T, 32)
    joint_torques: np.ndarray             # (T, 32)
    imu_data: np.ndarray                  # (T, 9)
    foot_contacts: np.ndarray             # (T, 4)

    # Actions
    actions: np.ndarray                   # (T, 32)

    # Metadata
    language_instruction: str
    task_type: str                        # locomotion, manipulation, whole_body
    success: bool
    episode_length: int

    # Optional
    rewards: Optional[np.ndarray] = None  # (T,)
    gripper_states: Optional[np.ndarray] = None  # (T, 2)
```

### LeRobot Format

```python
# LeRobot dataset structure
dataset = {
    "observation.images.{camera}": np.array,  # Images from cameras
    "observation.state": np.array,             # Proprioception
    "action": np.array,                        # Actions
    "episode_index": int,                      # Episode ID
    "frame_index": int,                        # Frame within episode
    "timestamp": float,                        # Time
    "next.done": bool,                         # Episode end
}
```

---

## Data Loading

### Unified DataLoader

```python
from train.datasets import HumanoidDataset

class HumanoidDataset(torch.utils.data.Dataset):
    """
    Unified dataset loader for humanoid training.

    Supports multiple data formats and sources.
    """

    def __init__(
        self,
        data_sources: List[str],
        sequence_length: int = 16,
        image_size: int = 224,
        augment: bool = True,
    ):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.augment = augment

        # Load from multiple sources
        self.episodes = []
        for source in data_sources:
            if source.endswith(".hdf5"):
                self.episodes.extend(self._load_hdf5(source))
            elif source.endswith(".parquet"):
                self.episodes.extend(self._load_parquet(source))
            elif "huggingface" in source:
                self.episodes.extend(self._load_huggingface(source))

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode = self.episodes[idx]

        # Sample random window
        start = random.randint(0, len(episode) - self.sequence_length)
        window = episode[start:start + self.sequence_length]

        # Process observations
        images = self._process_images(window["images"])
        proprio = self._process_proprioception(window)
        actions = torch.tensor(window["actions"], dtype=torch.float32)

        return {
            "images": images,
            "proprioception": proprio,
            "actions": actions,
            "language": episode.get("language_instruction", ""),
        }
```

---

## Data Augmentation

### Image Augmentation

```python
import torchvision.transforms as T

image_augmentation = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Action Augmentation

```python
def augment_actions(
    actions: np.ndarray,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Add Gaussian noise to actions."""
    noise = np.random.normal(0, noise_std, actions.shape)
    return actions + noise
```

### Proprioception Augmentation

```python
def augment_proprioception(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    pos_noise: float = 0.01,
    vel_noise: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add sensor noise to proprioception."""
    pos_noise = np.random.normal(0, pos_noise, joint_pos.shape)
    vel_noise = np.random.normal(0, vel_noise, joint_vel.shape)
    return joint_pos + pos_noise, joint_vel + vel_noise
```

---

## Download Scripts

### Download All Datasets

```bash
#!/bin/bash

# Create data directory
mkdir -p ./data/humanoid

# Download LLaVA datasets
python -c "
from datasets import load_dataset
dataset = load_dataset('liuhaotian/LLaVA-Pretrain')
dataset.save_to_disk('./data/humanoid/llava_pretrain')

dataset = load_dataset('liuhaotian/LLaVA-Instruct-150k')
dataset.save_to_disk('./data/humanoid/llava_instruct')
"

# Download D4RL
python -c "
from datasets import load_dataset
dataset = load_dataset('imone/D4RL', 'humanoid-medium-v2')
dataset.save_to_disk('./data/humanoid/d4rl_humanoid')
"

# Download DROID
python -c "
from datasets import load_dataset
dataset = load_dataset('cadene/droid')
dataset.save_to_disk('./data/humanoid/droid')
"

# Download LeRobot ALOHA
python -c "
from datasets import load_dataset
dataset = load_dataset('lerobot/aloha_sim_transfer_cube_human')
dataset.save_to_disk('./data/humanoid/aloha')
"

echo "All datasets downloaded successfully!"
```

---

## Best Practices

### Data Quality

1. **Filter bad episodes**: Remove episodes with falls, collisions
2. **Normalize actions**: Scale actions to [-1, 1] range
3. **Handle missing data**: Interpolate or skip incomplete episodes
4. **Balance dataset**: Ensure diverse task distribution

### Storage

1. **Use HDF5/Parquet**: Efficient for large datasets
2. **Compress images**: Use JPEG for RGB, PNG for depth
3. **Chunk loading**: Load data in chunks for memory efficiency

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Complete training pipeline
- [Locomotion Training](training_locomotion.md) - Use locomotion datasets
- [Manipulation Training](training_manipulation.md) - Use manipulation datasets
- [Deployment](deployment.md) - Deploy trained models
