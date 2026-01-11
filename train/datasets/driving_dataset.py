"""
Autonomous Driving Dataset Loaders

Datasets for training VLA models for autonomous driving applications.
These datasets include multi-modal sensor data (camera, LiDAR, radar)
and driving actions (steering, throttle, brake).

================================================================================
TRAINING METHODS: This dataset is designed for:
================================================================================
1. END-TO-END DRIVING (Primary use case)
   - Imitation Learning: Learn driving policy from expert demonstrations
   - Behavioral Cloning: Direct sensor-to-control mapping
   - Waypoint prediction: Predict future trajectory points

2. MULTI-MODAL SENSOR FUSION
   - Camera + LiDAR fusion for robust perception
   - Radar integration for velocity estimation
   - Cross-modal representation learning

3. VLA FOR AUTONOMOUS VEHICLES
   - Language-conditioned driving (e.g., "turn left at the intersection")
   - Navigation with natural language instructions
   - Scene understanding with vision-language models

4. WORLD MODEL PRETRAINING
   - Learn predictive models of driving scenarios
   - Video prediction for planning
   - Scene flow estimation

5. OFFLINE RL FOR DRIVING
   - Conservative Q-Learning (CQL) for safe driving
   - Decision Transformer for trajectory optimization
   - Reward shaping from driving metrics (safety, comfort, efficiency)

NOT recommended for:
- Online RL (safety-critical, requires simulator like CARLA)
- Pure language tasks (limited text annotations)
================================================================================

Recommended datasets:
- nuScenes: Multi-modal driving dataset (camera, LiDAR, radar)
- CARLA: Simulated driving with full sensor suite
- Waymo Open Dataset: Large-scale real-world driving
- comma.ai: Real driving data with CAN bus actions

Installation:
    # nuScenes
    pip install nuscenes-devkit

    # For HuggingFace datasets
    pip install datasets

Usage:
    from train.datasets import NuScenesDataset, create_driving_dataloader

    dataset = NuScenesDataset(data_root="/path/to/nuscenes")
    dataloader = create_driving_dataloader(dataset, batch_size=16)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import numpy as np
from pathlib import Path
import json


class DrivingDataset(Dataset):
    """
    Base class for autonomous driving datasets.

    Standard format:
    - camera_front: Front camera RGB image
    - camera_*: Additional camera views (optional)
    - lidar: LiDAR point cloud (N, 3) or (N, 4)
    - radar: Radar points (optional)
    - action: [steering, throttle, brake] or waypoints
    - speed: Current vehicle speed
    - can_bus: Full CAN bus data (optional)

    Args:
        data_root: Path to dataset root
        split: Dataset split
        cameras: List of camera names to load
        use_lidar: Whether to load LiDAR data
        use_radar: Whether to load radar data
        image_size: Target image size
        max_points: Maximum number of LiDAR/radar points
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        cameras: List[str] = ["front"],
        use_lidar: bool = True,
        use_radar: bool = False,
        image_size: Tuple[int, int] = (224, 224),
        max_points: int = 16384,
        transform=None,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.cameras = cameras
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.image_size = image_size
        self.max_points = max_points
        self.transform = transform

        self._load_dataset()

    def _load_dataset(self):
        """Override in subclass to load specific dataset."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _process_image(self, image_path: str) -> torch.Tensor:
        """Load and process image."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.image_size)

        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor

    def _process_pointcloud(
        self,
        points: np.ndarray,
        max_points: Optional[int] = None,
    ) -> torch.Tensor:
        """Process point cloud to fixed size."""
        max_points = max_points or self.max_points

        if len(points) > max_points:
            # Random subsample
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        elif len(points) < max_points:
            # Pad with zeros
            padding = np.zeros((max_points - len(points), points.shape[1]))
            points = np.vstack([points, padding])

        return torch.from_numpy(points).float()


class NuScenesDataset(DrivingDataset):
    """
    nuScenes Dataset for Autonomous Driving.

    nuScenes is a large-scale autonomous driving dataset with:
    - 6 cameras (360° coverage)
    - 1 LiDAR (32 beams)
    - 5 radars
    - Full CAN bus data
    - 3D object annotations

    Action space: Vehicle control or waypoints
    - Steering angle
    - Throttle/brake
    - Or: Future waypoints (x, y, heading)

    Note: Requires nuScenes devkit and downloaded data.

    Usage:
        dataset = NuScenesDataset(
            data_root="/path/to/nuscenes",
            version="v1.0-mini",  # or "v1.0-trainval"
            split="train",
        )
    """

    CAMERA_NAMES = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        data_root: str,
        version: str = "v1.0-mini",
        split: str = "train",
        use_waypoints: bool = True,
        waypoint_steps: int = 10,
        **kwargs,
    ):
        self.version = version
        self.use_waypoints = use_waypoints
        self.waypoint_steps = waypoint_steps
        super().__init__(data_root, split, **kwargs)

    def _load_dataset(self):
        """Load nuScenes dataset."""
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.can_bus.can_bus_api import NuScenesCanBus

            print(f"Loading nuScenes {self.version}...")
            self.nusc = NuScenes(
                version=self.version,
                dataroot=str(self.data_root),
                verbose=True,
            )

            # Load CAN bus data for vehicle controls
            try:
                self.can_bus = NuScenesCanBus(dataroot=str(self.data_root))
                self.has_can = True
            except Exception:
                self.has_can = False
                print("CAN bus data not available")

            # Get samples for split
            self.samples = self._get_split_samples()
            print(f"Loaded {len(self.samples)} samples")

        except ImportError:
            print("nuscenes-devkit not installed. Creating dummy dataset.")
            self._create_dummy_dataset()

        except Exception as e:
            print(f"Could not load nuScenes: {e}")
            self._create_dummy_dataset()

    def _get_split_samples(self) -> List[str]:
        """Get sample tokens for the split."""
        # nuScenes split logic
        scenes = self.nusc.scene

        if self.split == "train":
            split_scenes = scenes[:int(len(scenes) * 0.8)]
        elif self.split == "val":
            split_scenes = scenes[int(len(scenes) * 0.8):]
        else:
            split_scenes = scenes

        sample_tokens = []
        for scene in split_scenes:
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample_tokens.append(sample_token)
                sample = self.nusc.get("sample", sample_token)
                sample_token = sample["next"]

        return sample_tokens

    def _create_dummy_dataset(self):
        """Create dummy data for testing."""
        self.nusc = None
        self.has_can = False

        self.samples = list(range(1000))
        print(f"Created dummy nuScenes dataset with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.nusc is None:
            return self._get_dummy_sample()

        sample_token = self.samples[idx]
        sample = self.nusc.get("sample", sample_token)

        output = {}

        # Load camera images
        for cam_name in self.cameras:
            if cam_name.upper() in self.CAMERA_NAMES or f"CAM_{cam_name.upper()}" in self.CAMERA_NAMES:
                full_cam_name = cam_name if cam_name.startswith("CAM_") else f"CAM_{cam_name.upper()}"
                if full_cam_name in sample["data"]:
                    cam_data = self.nusc.get("sample_data", sample["data"][full_cam_name])
                    img_path = self.data_root / cam_data["filename"]
                    output[f"camera_{cam_name.lower()}"] = self._process_image(str(img_path))

        # Default front camera
        if "camera_front" not in output:
            cam_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            img_path = self.data_root / cam_data["filename"]
            output["pixel_values"] = self._process_image(str(img_path))
        else:
            output["pixel_values"] = output["camera_front"]

        # Load LiDAR
        if self.use_lidar and "LIDAR_TOP" in sample["data"]:
            lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            lidar_path = self.data_root / lidar_data["filename"]
            points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)[:, :3]
            output["lidar"] = self._process_pointcloud(points)

        # Get vehicle pose and compute actions
        if self.use_waypoints:
            output["action"] = self._get_waypoints(sample)
        else:
            output["action"] = self._get_vehicle_controls(sample)

        return output

    def _get_waypoints(self, sample) -> torch.Tensor:
        """Compute future waypoints from poses."""
        # Get ego pose
        ego_pose = self.nusc.get("ego_pose", sample["data"]["LIDAR_TOP"])
        current_pos = np.array(ego_pose["translation"][:2])
        current_rot = ego_pose["rotation"]

        # Get future waypoints
        waypoints = []
        next_token = sample["next"]

        for _ in range(self.waypoint_steps):
            if next_token:
                next_sample = self.nusc.get("sample", next_token)
                next_ego = self.nusc.get("ego_pose", next_sample["data"]["LIDAR_TOP"])
                next_pos = np.array(next_ego["translation"][:2])

                # Convert to local coordinates
                rel_pos = next_pos - current_pos
                waypoints.append(rel_pos)

                next_token = next_sample["next"]
            else:
                # Repeat last waypoint
                waypoints.append(waypoints[-1] if waypoints else np.zeros(2))

        return torch.tensor(np.array(waypoints), dtype=torch.float32)

    def _get_vehicle_controls(self, sample) -> torch.Tensor:
        """Get vehicle control actions from CAN bus."""
        if not self.has_can:
            return torch.zeros(3)

        try:
            scene = self.nusc.get("scene", sample["scene_token"])
            pose = self.can_bus.get_messages(scene["name"], "pose")
            vehicle = self.can_bus.get_messages(scene["name"], "vehicle_monitor")

            # Get closest timestamp
            timestamp = sample["timestamp"]

            # Extract steering, throttle, brake
            steering = 0.0
            throttle = 0.0
            brake = 0.0

            for msg in vehicle:
                if abs(msg["utime"] - timestamp) < 100000:  # 100ms tolerance
                    steering = msg.get("steering", 0.0)
                    throttle = msg.get("throttle", 0.0)
                    brake = msg.get("brake", 0.0)
                    break

            return torch.tensor([steering, throttle, brake], dtype=torch.float32)

        except Exception:
            return torch.zeros(3)

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return dummy sample for testing."""
        output = {
            "pixel_values": torch.randn(3, *self.image_size),
        }

        if self.use_lidar:
            output["lidar"] = torch.randn(self.max_points, 3)

        if self.use_waypoints:
            output["action"] = torch.randn(self.waypoint_steps, 2)
        else:
            output["action"] = torch.randn(3)

        return output


class CarlaDataset(DrivingDataset):
    """
    CARLA Simulated Driving Dataset.

    CARLA provides synthetic driving data with:
    - Multiple camera views
    - Perfect LiDAR/depth
    - Ground truth segmentation
    - Full vehicle state

    Great for pre-training and ablation studies.

    Expected data format (from CARLA data collector):
    data_root/
        episode_000/
            camera_front/
                000000.png
                ...
            lidar/
                000000.npy
            measurements/
                000000.json

    Usage:
        dataset = CarlaDataset(data_root="/path/to/carla_data")
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(data_root, split, **kwargs)

    def _load_dataset(self):
        """Load CARLA dataset from disk."""
        self.episodes = []
        self.samples = []

        # Find all episodes
        if self.data_root.exists():
            for episode_dir in sorted(self.data_root.glob("episode_*")):
                frames = sorted((episode_dir / "camera_front").glob("*.png"))
                for frame_path in frames:
                    frame_id = frame_path.stem
                    self.samples.append({
                        "episode_dir": episode_dir,
                        "frame_id": frame_id,
                    })

            print(f"Found {len(self.samples)} frames in {len(self.episodes)} episodes")
        else:
            print(f"Data root not found: {self.data_root}")
            self._create_dummy_dataset()

    def _create_dummy_dataset(self):
        """Create dummy data."""
        self.samples = [{"dummy": True} for _ in range(1000)]
        print(f"Created dummy CARLA dataset with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        if sample_info.get("dummy"):
            return self._get_dummy_sample()

        episode_dir = sample_info["episode_dir"]
        frame_id = sample_info["frame_id"]

        output = {}

        # Load front camera
        img_path = episode_dir / "camera_front" / f"{frame_id}.png"
        if img_path.exists():
            output["pixel_values"] = self._process_image(str(img_path))
        else:
            output["pixel_values"] = torch.randn(3, *self.image_size)

        # Load other cameras
        for cam in self.cameras:
            cam_path = episode_dir / f"camera_{cam}" / f"{frame_id}.png"
            if cam_path.exists():
                output[f"camera_{cam}"] = self._process_image(str(cam_path))

        # Load LiDAR
        if self.use_lidar:
            lidar_path = episode_dir / "lidar" / f"{frame_id}.npy"
            if lidar_path.exists():
                points = np.load(str(lidar_path))
                output["lidar"] = self._process_pointcloud(points)
            else:
                output["lidar"] = torch.randn(self.max_points, 3)

        # Load measurements (actions)
        meas_path = episode_dir / "measurements" / f"{frame_id}.json"
        if meas_path.exists():
            with open(meas_path) as f:
                meas = json.load(f)

            output["action"] = torch.tensor([
                meas.get("steer", 0.0),
                meas.get("throttle", 0.0),
                meas.get("brake", 0.0),
            ], dtype=torch.float32)

            output["speed"] = torch.tensor(meas.get("speed", 0.0), dtype=torch.float32)
        else:
            output["action"] = torch.zeros(3)
            output["speed"] = torch.tensor(0.0)

        return output

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.randn(3, *self.image_size),
            "lidar": torch.randn(self.max_points, 3) if self.use_lidar else None,
            "action": torch.randn(3),
            "speed": torch.tensor(0.0),
        }


class CommaAIDataset(DrivingDataset):
    """
    comma.ai Driving Dataset.

    Real-world driving data from comma.ai devices:
    - Front camera video
    - CAN bus data (steering, speed)
    - GPS/IMU data

    Available on HuggingFace: commaai/comma2k19

    Usage:
        dataset = CommaAIDataset(split="train")
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        **kwargs,
    ):
        self.max_samples = max_samples
        super().__init__(data_root="", split=split, **kwargs)

    def _load_dataset(self):
        """Load from HuggingFace."""
        try:
            from datasets import load_dataset

            print("Loading comma.ai dataset from HuggingFace...")
            self.hf_dataset = load_dataset(
                "commaai/comma2k19",
                split=self.split,
                trust_remote_code=True,
            )

            if self.max_samples:
                self.hf_dataset = self.hf_dataset.select(
                    range(min(self.max_samples, len(self.hf_dataset)))
                )

            print(f"Loaded {len(self.hf_dataset)} samples")

        except Exception as e:
            print(f"Could not load comma.ai dataset: {e}")
            self._create_dummy_dataset()

    def _create_dummy_dataset(self):
        self.hf_dataset = None
        self.dummy_samples = [{"dummy": True} for _ in range(self.max_samples or 1000)]

    def __len__(self) -> int:
        if self.hf_dataset:
            return len(self.hf_dataset)
        return len(self.dummy_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.hf_dataset is None:
            return {
                "pixel_values": torch.randn(3, *self.image_size),
                "action": torch.randn(2),  # steering, throttle
                "speed": torch.tensor(0.0),
            }

        sample = self.hf_dataset[idx]

        output = {}

        # Process image
        if "image" in sample:
            output["pixel_values"] = self._process_image(sample["image"])

        # Extract steering and speed from CAN
        if "can" in sample:
            can_data = sample["can"]
            output["action"] = torch.tensor([
                can_data.get("steering_angle", 0.0),
                can_data.get("gas", 0.0) - can_data.get("brake", 0.0),
            ], dtype=torch.float32)
            output["speed"] = torch.tensor(can_data.get("speed", 0.0), dtype=torch.float32)
        else:
            output["action"] = torch.zeros(2)
            output["speed"] = torch.tensor(0.0)

        return output


def collate_driving(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for driving datasets."""
    output = {}

    # Stack tensors
    for key in ["pixel_values", "lidar", "action", "speed"]:
        values = [b[key] for b in batch if key in b and b[key] is not None]
        if values:
            if isinstance(values[0], torch.Tensor):
                output[key] = torch.stack(values)

    # Handle multiple cameras
    cam_keys = [k for k in batch[0].keys() if k.startswith("camera_")]
    for key in cam_keys:
        output[key] = torch.stack([b[key] for b in batch if key in b])

    return output


def create_driving_dataloader(
    dataset: DrivingDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for driving dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_driving,
        pin_memory=pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Autonomous Driving Dataset Test")
    print("=" * 60)

    # Test nuScenes (dummy)
    print("\n1. Testing nuScenes Dataset (dummy mode):")
    try:
        dataset = NuScenesDataset(
            data_root="/path/to/nuscenes",
            version="v1.0-mini",
        )
        print(f"   Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Image shape: {sample['pixel_values'].shape}")
        print(f"   Action shape: {sample['action'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test CARLA (dummy)
    print("\n2. Testing CARLA Dataset (dummy mode):")
    try:
        dataset = CarlaDataset(data_root="/path/to/carla")
        print(f"   Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"   Image shape: {sample['pixel_values'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Driving dataset test complete!")
    print("=" * 60)
