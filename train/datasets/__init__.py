"""
Dataset Loaders for VLA Training

Recommended datasets for each training method:

1. Supervised Fine-tuning (Imitation Learning):
   - LeRobot datasets: pusht, aloha, xarm (HuggingFace)
   - Open X-Embodiment: Bridge V2, RT-1 (TensorFlow Datasets)
   - BridgeData V2: Real robot manipulation

2. Behavior Cloning:
   - D4RL Expert: hopper-expert-v2, walker2d-expert-v2 (recommended)
   - D4RL Medium-Expert: Mixed quality demonstrations
   - FilteredBCDataset: Automatic top-k% trajectory filtering

3. Reinforcement Learning:
   - D4RL: Offline RL benchmarks (mujoco, antmaze)
   - RoboMimic: Robot manipulation environments
   - MetaWorld: Multi-task robot learning

4. Autonomous Driving:
   - nuScenes: Multi-modal autonomous driving
   - CARLA: Simulated driving
   - Waymo Open Dataset: Real-world driving

5. Pre-training:
   - LAION-400M: Large-scale image-text pairs
   - CC3M/CC12M: Conceptual Captions
   - DataComp: Curated web data
"""

from .lerobot_dataset import (
    LeRobotDataset,
    PushTDataset,
    AlohaDataset,
    XArmDataset,
    create_lerobot_dataloader,
)

from .openx_dataset import (
    OpenXDataset,
    BridgeDataset,
    RT1Dataset,
    create_openx_dataloader,
)

from .driving_dataset import (
    DrivingDataset,
    NuScenesDataset,
    CarlaDataset,
    create_driving_dataloader,
)

from .rl_dataset import (
    RLDataset,
    D4RLDataset,
    RoboMimicDataset,
    create_rl_dataloader,
)

from .bc_dataset import (
    BCDataset,
    FilteredBCDataset,
    WeightedBCDataset,
    create_bc_dataloader,
)

__all__ = [
    # LeRobot
    "LeRobotDataset",
    "PushTDataset",
    "AlohaDataset",
    "XArmDataset",
    "create_lerobot_dataloader",
    # Open X-Embodiment
    "OpenXDataset",
    "BridgeDataset",
    "RT1Dataset",
    "create_openx_dataloader",
    # Driving
    "DrivingDataset",
    "NuScenesDataset",
    "CarlaDataset",
    "create_driving_dataloader",
    # RL
    "RLDataset",
    "D4RLDataset",
    "RoboMimicDataset",
    "create_rl_dataloader",
    # Behavior Cloning
    "BCDataset",
    "FilteredBCDataset",
    "WeightedBCDataset",
    "create_bc_dataloader",
]
