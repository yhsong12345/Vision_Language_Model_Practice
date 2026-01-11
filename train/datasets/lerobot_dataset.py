"""
LeRobot Dataset Loaders

LeRobot provides standardized robot learning datasets on HuggingFace.
These are excellent for imitation learning and VLA fine-tuning.

================================================================================
TRAINING METHODS: This dataset is designed for:
================================================================================
1. IMITATION LEARNING (IL) - Primary use case
   - Behavioral Cloning (BC): Direct state-action mapping
   - DAgger: Interactive imitation with expert queries
   - Action Chunking: Predict sequence of future actions (ACT-style)

2. SUPERVISED FINE-TUNING (SFT)
   - VLA Fine-tuning: Fine-tune Vision-Language-Action models
   - Policy distillation: Transfer from larger to smaller models

3. OFFLINE RL (with reward labeling)
   - Can be converted to offline RL by adding reward annotations
   - Useful for IQL, CQL when combined with reward functions

NOT recommended for:
- Online RL (no environment interaction)
- Pure vision-language pretraining (no language annotations by default)
================================================================================

Recommended datasets:
- lerobot/pusht: Push-T manipulation task (simple, good for testing)
- lerobot/aloha_sim_*: ALOHA bimanual manipulation
- lerobot/xarm_*: xArm robot manipulation
- lerobot/ucsd_kitchen_dataset_converted: Kitchen manipulation

Installation:
    pip install lerobot
    # or
    pip install datasets

Usage:
    from train.datasets import PushTDataset, create_lerobot_dataloader

    dataset = PushTDataset(split="train", chunk_size=10)
    dataloader = create_lerobot_dataloader(dataset, batch_size=32)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import numpy as np


class LeRobotDataset(Dataset):
    """
    Base class for LeRobot datasets from HuggingFace.

    LeRobot datasets follow a standardized format:
    - observation.image: RGB images (C, H, W)
    - observation.state: Robot state (optional)
    - action: Robot actions
    - episode_index: Episode identifier
    - frame_index: Frame within episode

    Args:
        repo_id: HuggingFace dataset repository ID
        split: Dataset split ("train", "test")
        chunk_size: Number of future actions to predict (action chunking)
        image_size: Target image size (H, W)
        max_episodes: Maximum number of episodes to load
        delta_timestamps: Time offsets for action chunking
    """

    # Standard LeRobot datasets
    AVAILABLE_DATASETS = {
        "pusht": "lerobot/pusht",
        "aloha_sim_insertion": "lerobot/aloha_sim_insertion_human",
        "aloha_sim_transfer": "lerobot/aloha_sim_transfer_cube_human",
        "xarm_lift": "lerobot/xarm_lift_medium",
        "xarm_push": "lerobot/xarm_push_medium",
        "ucsd_kitchen": "lerobot/ucsd_kitchen_dataset_converted_externally_to_rlds",
        "umi_cup": "lerobot/umi_cup_in_the_wild",
    }

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        chunk_size: int = 10,
        image_size: Tuple[int, int] = (224, 224),
        max_episodes: Optional[int] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        transform=None,
    ):
        super().__init__()

        self.repo_id = repo_id
        self.split = split
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.transform = transform

        # Default delta timestamps for action chunking
        if delta_timestamps is None:
            # 10 future steps at 50Hz = 0.02s per step
            self.delta_timestamps = {
                "action": [i * 0.02 for i in range(chunk_size)]
            }
        else:
            self.delta_timestamps = delta_timestamps

        # Try to load with lerobot first
        self._load_dataset(max_episodes)

    def _load_dataset(self, max_episodes: Optional[int]):
        """Load dataset from HuggingFace."""
        try:
            # Try LeRobot's native loader
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as LRDataset

            print(f"Loading LeRobot dataset: {self.repo_id}")
            self.lerobot_dataset = LRDataset(
                self.repo_id,
                split=self.split,
                delta_timestamps=self.delta_timestamps,
            )
            self.use_lerobot = True

            # Get dataset info
            self.action_dim = self.lerobot_dataset.action_dim if hasattr(self.lerobot_dataset, 'action_dim') else 7
            self.fps = self.lerobot_dataset.fps if hasattr(self.lerobot_dataset, 'fps') else 50

            print(f"Loaded {len(self.lerobot_dataset)} samples")
            print(f"Action dim: {self.action_dim}, FPS: {self.fps}")

        except ImportError:
            print("LeRobot not installed, using HuggingFace datasets directly")
            self._load_with_hf_datasets(max_episodes)

    def _load_with_hf_datasets(self, max_episodes: Optional[int]):
        """Fallback to HuggingFace datasets library."""
        try:
            from datasets import load_dataset

            print(f"Loading from HuggingFace: {self.repo_id}")

            # Load dataset
            hf_dataset = load_dataset(
                self.repo_id,
                split=self.split,
                trust_remote_code=True,
            )

            # Filter by episode if needed
            if max_episodes is not None:
                episode_indices = hf_dataset['episode_index']
                unique_episodes = sorted(set(episode_indices))[:max_episodes]
                hf_dataset = hf_dataset.filter(
                    lambda x: x['episode_index'] in unique_episodes
                )

            self.hf_dataset = hf_dataset
            self.use_lerobot = False

            # Infer action dimension
            sample = hf_dataset[0]
            if 'action' in sample:
                action = sample['action']
                self.action_dim = len(action) if isinstance(action, (list, np.ndarray)) else 7
            else:
                self.action_dim = 7
            self.fps = 50  # Default

            print(f"Loaded {len(hf_dataset)} samples, action_dim: {self.action_dim}")

        except Exception as e:
            print(f"Could not load dataset: {e}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_dataset()

    def _create_dummy_dataset(self):
        """Create dummy data for testing."""
        self.use_lerobot = False
        self.action_dim = 7
        self.fps = 50

        # Create dummy episodes
        self.dummy_data = []
        num_episodes = 10
        episode_length = 100

        for ep in range(num_episodes):
            for frame in range(episode_length):
                self.dummy_data.append({
                    "observation.image": np.random.randint(
                        0, 255, (3, self.image_size[0], self.image_size[1]), dtype=np.uint8
                    ),
                    "observation.state": np.random.randn(8).astype(np.float32),
                    "action": np.random.randn(self.action_dim).astype(np.float32),
                    "episode_index": ep,
                    "frame_index": frame,
                })

        self.hf_dataset = self.dummy_data
        print(f"Created dummy dataset with {len(self.dummy_data)} samples")

    def __len__(self) -> int:
        if self.use_lerobot:
            return len(self.lerobot_dataset)
        return len(self.hf_dataset)

    def _process_image(self, image) -> torch.Tensor:
        """Process image to tensor."""
        if isinstance(image, torch.Tensor):
            img_tensor = image
        elif isinstance(image, np.ndarray):
            if image.shape[0] == 3:  # Already CHW
                img_tensor = torch.from_numpy(image)
            else:  # HWC
                img_tensor = torch.from_numpy(image).permute(2, 0, 1)
        elif isinstance(image, Image.Image):
            img_array = np.array(image.resize(self.image_size))
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Normalize to [0, 1]
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0

        # Apply transform if provided
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_lerobot:
            sample = self.lerobot_dataset[idx]
            return self._process_lerobot_sample(sample)
        else:
            sample = self.hf_dataset[idx]
            return self._process_hf_sample(sample, idx)

    def _process_lerobot_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Process sample from LeRobot dataset."""
        output = {}

        # Image
        if "observation.image" in sample:
            output["pixel_values"] = self._process_image(sample["observation.image"])
        elif "observation.images.top" in sample:
            output["pixel_values"] = self._process_image(sample["observation.images.top"])

        # State
        if "observation.state" in sample:
            state = sample["observation.state"]
            if isinstance(state, torch.Tensor):
                output["state"] = state
            else:
                output["state"] = torch.tensor(state, dtype=torch.float32)

        # Action (with chunking)
        if "action" in sample:
            action = sample["action"]
            if isinstance(action, torch.Tensor):
                output["action"] = action
            else:
                output["action"] = torch.tensor(action, dtype=torch.float32)

        # Episode info
        if "episode_index" in sample:
            output["episode_index"] = sample["episode_index"]
        if "frame_index" in sample:
            output["frame_index"] = sample["frame_index"]

        return output

    def _process_hf_sample(self, sample: Dict, idx: int) -> Dict[str, torch.Tensor]:
        """Process sample from HuggingFace dataset."""
        output = {}

        # Find image key
        for key in ["observation.image", "image", "observation.images.top", "rgb"]:
            if key in sample:
                output["pixel_values"] = self._process_image(sample[key])
                break
        else:
            # No image found, create dummy
            output["pixel_values"] = torch.randn(3, *self.image_size)

        # State
        for key in ["observation.state", "state", "proprio"]:
            if key in sample:
                state = sample[key]
                output["state"] = torch.tensor(state, dtype=torch.float32) if not isinstance(state, torch.Tensor) else state
                break

        # Action (with action chunking)
        if "action" in sample:
            action = sample["action"]
            output["action"] = torch.tensor(action, dtype=torch.float32) if not isinstance(action, torch.Tensor) else action

            # Get future actions for action chunking
            if self.chunk_size > 1:
                future_actions = [output["action"]]
                for i in range(1, self.chunk_size):
                    if idx + i < len(self.hf_dataset):
                        next_sample = self.hf_dataset[idx + i]
                        # Check same episode
                        if next_sample.get("episode_index") == sample.get("episode_index"):
                            next_action = next_sample["action"]
                            future_actions.append(
                                torch.tensor(next_action, dtype=torch.float32)
                                if not isinstance(next_action, torch.Tensor) else next_action
                            )
                        else:
                            future_actions.append(future_actions[-1])  # Repeat last
                    else:
                        future_actions.append(future_actions[-1])
                output["action_chunk"] = torch.stack(future_actions)

        # Episode info
        output["episode_index"] = sample.get("episode_index", 0)
        output["frame_index"] = sample.get("frame_index", idx)

        return output


class PushTDataset(LeRobotDataset):
    """
    Push-T Dataset from LeRobot.

    A simple 2D pushing task where the robot pushes a T-shaped block.
    Good for initial testing and debugging.

    - Action dim: 2 (x, y velocity)
    - Image: Top-down view (96x96 or 224x224)
    - ~25k samples

    Usage:
        dataset = PushTDataset(split="train")
        sample = dataset[0]
        print(sample["pixel_values"].shape)  # (3, 224, 224)
        print(sample["action"].shape)  # (2,) or (chunk_size, 2)
    """

    def __init__(self, split: str = "train", **kwargs):
        kwargs.setdefault("chunk_size", 10)
        super().__init__(
            repo_id="lerobot/pusht",
            split=split,
            **kwargs,
        )


class AlohaDataset(LeRobotDataset):
    """
    ALOHA Bimanual Manipulation Dataset.

    Dual-arm manipulation tasks from the ALOHA robot.
    Available tasks:
    - insertion: Insert peg into hole
    - transfer_cube: Transfer cube between arms

    - Action dim: 14 (7 per arm: 6 joint positions + gripper)
    - Image: Front camera view (480x640)

    Usage:
        dataset = AlohaDataset(task="insertion", split="train")
    """

    TASKS = {
        "insertion": "lerobot/aloha_sim_insertion_human",
        "insertion_scripted": "lerobot/aloha_sim_insertion_scripted",
        "transfer_cube": "lerobot/aloha_sim_transfer_cube_human",
        "transfer_cube_scripted": "lerobot/aloha_sim_transfer_cube_scripted",
    }

    def __init__(
        self,
        task: str = "insertion",
        split: str = "train",
        **kwargs,
    ):
        if task not in self.TASKS:
            raise ValueError(f"Unknown task: {task}. Available: {list(self.TASKS.keys())}")

        kwargs.setdefault("chunk_size", 100)  # ALOHA uses 100-step chunks
        super().__init__(
            repo_id=self.TASKS[task],
            split=split,
            **kwargs,
        )


class XArmDataset(LeRobotDataset):
    """
    xArm Robot Manipulation Dataset.

    Manipulation tasks with the UFactory xArm robot.
    Available tasks:
    - lift: Lift object
    - push: Push object

    - Action dim: 4 (x, y, z, gripper) or 7 (full)
    - Image: Wrist and/or external camera

    Usage:
        dataset = XArmDataset(task="lift", split="train")
    """

    TASKS = {
        "lift": "lerobot/xarm_lift_medium",
        "push": "lerobot/xarm_push_medium",
    }

    def __init__(
        self,
        task: str = "lift",
        split: str = "train",
        **kwargs,
    ):
        if task not in self.TASKS:
            raise ValueError(f"Unknown task: {task}. Available: {list(self.TASKS.keys())}")

        super().__init__(
            repo_id=self.TASKS[task],
            split=split,
            **kwargs,
        )


def collate_lerobot(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for LeRobot datasets."""
    output = {}

    # Stack tensors
    for key in ["pixel_values", "state", "action", "action_chunk"]:
        if key in batch[0]:
            output[key] = torch.stack([b[key] for b in batch])

    # Keep episode/frame info
    for key in ["episode_index", "frame_index"]:
        if key in batch[0]:
            output[key] = torch.tensor([b[key] for b in batch])

    return output


def create_lerobot_dataloader(
    dataset: LeRobotDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader for LeRobot dataset.

    Args:
        dataset: LeRobotDataset instance
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_lerobot,
        pin_memory=pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("LeRobot Dataset Test")
    print("=" * 60)

    # Test PushT
    print("\n1. Testing PushT Dataset:")
    try:
        dataset = PushTDataset(split="train", chunk_size=10)
        print(f"   Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Image shape: {sample['pixel_values'].shape}")
        print(f"   Action shape: {sample['action'].shape}")

        # Test dataloader
        dataloader = create_lerobot_dataloader(dataset, batch_size=4)
        batch = next(iter(dataloader))
        print(f"   Batch pixel_values: {batch['pixel_values'].shape}")
        print(f"   Batch action: {batch['action'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test ALOHA
    print("\n2. Testing ALOHA Dataset:")
    try:
        dataset = AlohaDataset(task="insertion", split="train")
        print(f"   Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"   Image shape: {sample['pixel_values'].shape}")
        print(f"   Action shape: {sample['action'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("LeRobot dataset test complete!")
    print("=" * 60)
