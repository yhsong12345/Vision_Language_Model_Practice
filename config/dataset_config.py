"""
Dataset Configuration

Defines supported datasets and their configurations for VLA training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class DatasetType(Enum):
    LEROBOT = "lerobot"
    OPEN_X_EMBODIMENT = "open_x_embodiment"
    BRIDGE = "bridge"
    CUSTOM = "custom"


# Supported robot datasets
SUPPORTED_DATASETS = {
    # LeRobot format datasets
    "lerobot/pusht": {
        "type": DatasetType.LEROBOT,
        "action_dim": 2,
        "description": "Push-T task with 2D position control",
        "image_key": "observation.image",
        "action_key": "action",
    },
    "lerobot/aloha_sim_insertion_human": {
        "type": DatasetType.LEROBOT,
        "action_dim": 14,
        "description": "ALOHA bimanual insertion task (human demos)",
        "image_key": "observation.images.cam_high",
        "action_key": "action",
    },
    "lerobot/aloha_sim_insertion_scripted": {
        "type": DatasetType.LEROBOT,
        "action_dim": 14,
        "description": "ALOHA bimanual insertion task (scripted)",
        "image_key": "observation.images.cam_high",
        "action_key": "action",
    },
    "lerobot/aloha_sim_transfer_cube_human": {
        "type": DatasetType.LEROBOT,
        "action_dim": 14,
        "description": "ALOHA cube transfer task (human demos)",
        "image_key": "observation.images.cam_high",
        "action_key": "action",
    },
    "lerobot/xarm_push_medium": {
        "type": DatasetType.LEROBOT,
        "action_dim": 4,
        "description": "XArm pushing task",
        "image_key": "observation.image",
        "action_key": "action",
    },
    "lerobot/xarm_lift_medium": {
        "type": DatasetType.LEROBOT,
        "action_dim": 4,
        "description": "XArm lifting task",
        "image_key": "observation.image",
        "action_key": "action",
    },
    "lerobot/unitree_g1_dexterous_manipulation": {
        "type": DatasetType.LEROBOT,
        "action_dim": 29,
        "description": "Unitree G1 humanoid dexterous manipulation",
        "image_key": "observation.image",
        "action_key": "action",
    },

    # Open X-Embodiment datasets
    "berkeley-autolab/bridge_data_v2": {
        "type": DatasetType.BRIDGE,
        "action_dim": 7,
        "description": "Bridge V2 multi-robot dataset",
        "image_key": "observation",
        "action_key": "action",
        "instruction_key": "language_instruction",
    },

    # RT-X datasets
    "google/rt-1-robot-data": {
        "type": DatasetType.OPEN_X_EMBODIMENT,
        "action_dim": 11,
        "description": "RT-1 Everyday Robots dataset",
        "image_key": "observation.image",
        "action_key": "action",
    },

    # Simulation datasets
    "droid_100": {
        "type": DatasetType.CUSTOM,
        "action_dim": 7,
        "description": "DROID 100-hour robot manipulation",
    },
}


@dataclass
class DatasetConfig:
    """
    Dataset configuration for VLA training.

    Supports various robot manipulation datasets from HuggingFace.
    """
    # Dataset selection
    dataset_name: str = "lerobot/pusht"
    dataset_split: str = "train"
    validation_split: float = 0.1

    # Data loading
    max_samples: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Image processing
    image_size: int = 224
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    random_crop: bool = False
    color_jitter: bool = False

    # Text processing
    max_text_length: int = 128
    default_instruction: str = "Perform the manipulation task."

    # Action processing
    action_normalization: str = "minmax"  # minmax, standard, none
    action_clip_range: Optional[float] = None

    # Depth processing
    use_depth: bool = False
    depth_size: int = 224
    depth_normalization: str = "minmax"  # minmax, standard, none
    depth_clip_range: Optional[List[float]] = None  # [min, max] in meters

    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the selected dataset."""
        if self.dataset_name in SUPPORTED_DATASETS:
            return SUPPORTED_DATASETS[self.dataset_name]
        return {
            "type": DatasetType.CUSTOM,
            "action_dim": 7,
            "description": "Custom dataset",
        }

    @classmethod
    def for_dataset(cls, dataset_name: str, **kwargs) -> "DatasetConfig":
        """Create config for a specific dataset."""
        config = cls(dataset_name=dataset_name, **kwargs)
        return config

    @classmethod
    def pusht(cls) -> "DatasetConfig":
        """Config for Push-T dataset."""
        return cls(
            dataset_name="lerobot/pusht",
            image_size=96,  # Push-T uses smaller images
        )

    @classmethod
    def aloha(cls) -> "DatasetConfig":
        """Config for ALOHA simulation datasets."""
        return cls(
            dataset_name="lerobot/aloha_sim_insertion_human",
            image_size=224,
        )

    @classmethod
    def bridge(cls) -> "DatasetConfig":
        """Config for Bridge V2 dataset."""
        return cls(
            dataset_name="berkeley-autolab/bridge_data_v2",
            image_size=224,
        )

    @classmethod
    def with_depth(cls, dataset_name: str = "nyu_depth_v2", **kwargs) -> "DatasetConfig":
        """Config for datasets with depth camera support."""
        return cls(
            dataset_name=dataset_name,
            use_depth=True,
            depth_size=224,
            **kwargs,
        )

    @classmethod
    def rgbd_manipulation(cls) -> "DatasetConfig":
        """Config for RGB-D robot manipulation tasks."""
        return cls(
            dataset_name="graspnet",
            use_depth=True,
            depth_size=224,
            depth_clip_range=[0.0, 2.0],
        )


@dataclass
class MultiDatasetConfig:
    """
    Configuration for training on multiple datasets.

    Useful for:
    - Cross-embodiment training
    - Domain randomization
    - Transfer learning
    """
    datasets: List[DatasetConfig] = field(default_factory=list)
    sampling_strategy: str = "proportional"  # proportional, uniform, temperature
    sampling_temperature: float = 1.0

    def add_dataset(self, config: DatasetConfig, weight: float = 1.0):
        """Add a dataset to the mix."""
        self.datasets.append((config, weight))

    @classmethod
    def cross_embodiment(cls) -> "MultiDatasetConfig":
        """Config for cross-embodiment training."""
        config = cls()
        config.datasets = [
            DatasetConfig.pusht(),
            DatasetConfig.aloha(),
            DatasetConfig.bridge(),
        ]
        return config


# Autonomous driving datasets
DRIVING_DATASETS = {
    "nuScenes": {
        "description": "Urban driving with 6 cameras + LiDAR",
        "sensors": ["camera", "lidar", "radar"],
        "action_dim": 3,  # steering, throttle, brake
    },
    "Waymo": {
        "description": "Large-scale autonomous driving",
        "sensors": ["camera", "lidar"],
        "action_dim": 3,
    },
    "KITTI": {
        "description": "Classic autonomous driving benchmark",
        "sensors": ["camera", "lidar"],
        "action_dim": 2,  # steering, velocity
    },
    "CARLA": {
        "description": "Simulated driving environment",
        "sensors": ["camera", "lidar", "radar", "imu", "gps"],
        "action_dim": 3,
    },
}


# Depth camera datasets
DEPTH_DATASETS = {
    "nyu_depth_v2": {
        "description": "NYU Depth V2 indoor scenes with RGB-D",
        "sensors": ["camera", "depth"],
        "depth_key": "depth",
        "image_key": "image",
        "depth_range": [0.0, 10.0],  # meters
    },
    "scannet": {
        "description": "ScanNet indoor 3D reconstruction dataset",
        "sensors": ["camera", "depth"],
        "depth_key": "depth",
        "image_key": "color",
        "depth_range": [0.0, 10.0],
    },
    "sun_rgbd": {
        "description": "SUN RGB-D indoor scene understanding",
        "sensors": ["camera", "depth"],
        "depth_key": "depth",
        "image_key": "image",
        "depth_range": [0.0, 10.0],
    },
    "matterport3d": {
        "description": "Matterport3D indoor navigation",
        "sensors": ["camera", "depth"],
        "depth_key": "depth",
        "image_key": "rgb",
        "depth_range": [0.0, 20.0],
    },
    "cleargrasp": {
        "description": "ClearGrasp transparent object depth",
        "sensors": ["camera", "depth"],
        "depth_key": "depth",
        "image_key": "rgb",
        "depth_range": [0.0, 2.0],
    },
    "graspnet": {
        "description": "GraspNet-1Billion grasping dataset",
        "sensors": ["camera", "depth"],
        "depth_key": "depth",
        "image_key": "rgb",
        "depth_range": [0.0, 2.0],
    },
}
