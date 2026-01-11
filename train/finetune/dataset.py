"""
Robot Manipulation Dataset

Unified dataset class for various robot manipulation datasets:
- LeRobot format datasets
- Open X-Embodiment datasets
- Custom datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List
from PIL import Image
import numpy as np


class RobotDataset(Dataset):
    """
    Generic robot manipulation dataset.

    Handles various dataset formats:
    - LeRobot (pusht, aloha, xarm, etc.)
    - Open X-Embodiment (Bridge, RT-1)
    - Custom formats

    Returns standardized format:
    - pixel_values: (C, H, W) image tensor
    - input_ids: (seq_len,) tokenized instruction
    - attention_mask: (seq_len,) attention mask
    - action: (action_dim,) action tensor
    """

    def __init__(
        self,
        dataset_name: str,
        image_processor,
        tokenizer,
        split: str = "train",
        max_samples: Optional[int] = None,
        max_text_length: int = 128,
        default_instruction: str = "Perform the manipulation task.",
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.default_instruction = default_instruction
        self.dataset_name = dataset_name

        # Try to load dataset
        self._load_dataset(dataset_name, split, max_samples)

    def _load_dataset(self, dataset_name: str, split: str, max_samples: Optional[int]):
        """Load dataset from HuggingFace or create dummy."""
        try:
            from datasets import load_dataset

            print(f"Loading dataset: {dataset_name}")
            self.dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

            if max_samples and len(self.dataset) > max_samples:
                self.dataset = self.dataset.select(range(max_samples))

            print(f"Loaded {len(self.dataset)} samples")
            self.is_dummy = False

        except Exception as e:
            print(f"Could not load dataset: {e}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_dataset(max_samples or 100)
            self.is_dummy = True

    def _create_dummy_dataset(self, num_samples: int):
        """Create dummy dataset for testing."""
        instructions = [
            "pick up the red block",
            "move the arm to the left",
            "place the object on the table",
            "push the cube forward",
            "grasp the handle",
            "rotate the object",
            "stack the blocks",
            "open the drawer",
        ]

        self.dummy_data = []
        for i in range(num_samples):
            self.dummy_data.append({
                "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "instruction": instructions[i % len(instructions)],
                "action": np.random.randn(7).astype(np.float32),
            })
        self.dataset = self.dummy_data

    def __len__(self):
        return len(self.dataset)

    def _get_image(self, item: Dict) -> Image.Image:
        """Extract and convert image from dataset item."""
        # Try different image keys
        image = None
        for key in ["image", "observation.image", "pixel_values", "rgb"]:
            if key in item:
                image = item[key]
                break
            # Handle nested observation dict
            if "observation" in item and isinstance(item["observation"], dict):
                if "image" in item["observation"]:
                    image = item["observation"]["image"]
                    break
                if "rgb" in item["observation"]:
                    image = item["observation"]["rgb"]
                    break

        if image is None:
            # Create random image for testing
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = Image.fromarray(image.numpy().astype(np.uint8))

        return image.convert("RGB")

    def _get_instruction(self, item: Dict) -> str:
        """Extract instruction from dataset item."""
        for key in ["instruction", "language_instruction", "task", "text", "prompt"]:
            if key in item and item[key]:
                return str(item[key])

        return self.default_instruction

    def _get_action(self, item: Dict) -> np.ndarray:
        """Extract action from dataset item."""
        action = None
        for key in ["action", "actions"]:
            if key in item:
                action = item[key]
                break

        if action is None:
            action = np.zeros(7, dtype=np.float32)

        if isinstance(action, torch.Tensor):
            action = action.numpy()
        elif isinstance(action, list):
            action = np.array(action)

        return action.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        # Get components
        image = self._get_image(item)
        instruction = self._get_instruction(item)
        action = self._get_action(item)

        # Process image
        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        # Tokenize instruction
        text_inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "action": torch.tensor(action, dtype=torch.float32),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "action": torch.stack([x["action"] for x in batch]),
    }


def create_robot_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for robot dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


class MultiEpisodeDataset(Dataset):
    """
    Dataset that handles multi-step episodes.

    Useful for action chunking and temporal models.
    """

    def __init__(
        self,
        dataset_name: str,
        image_processor,
        tokenizer,
        episode_length: int = 50,
        chunk_size: int = 10,
        **kwargs,
    ):
        self.base_dataset = RobotDataset(
            dataset_name, image_processor, tokenizer, **kwargs
        )
        self.episode_length = episode_length
        self.chunk_size = chunk_size

        # Group by episode
        self._group_episodes()

    def _group_episodes(self):
        """Group dataset samples into episodes."""
        # Simple grouping - assume consecutive samples are from same episode
        self.episodes = []
        current_episode = []

        for i in range(len(self.base_dataset)):
            current_episode.append(i)

            if len(current_episode) >= self.episode_length:
                self.episodes.append(current_episode)
                current_episode = []

        if current_episode:
            self.episodes.append(current_episode)

    def __len__(self):
        return len(self.episodes) * (self.episode_length - self.chunk_size + 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx = idx // (self.episode_length - self.chunk_size + 1)
        step_idx = idx % (self.episode_length - self.chunk_size + 1)

        episode = self.episodes[episode_idx]

        # Get base sample
        base_sample = self.base_dataset[episode[step_idx]]

        # Get action chunk
        action_chunk = []
        for i in range(self.chunk_size):
            if step_idx + i < len(episode):
                sample = self.base_dataset[episode[step_idx + i]]
                action_chunk.append(sample["action"])
            else:
                action_chunk.append(torch.zeros_like(base_sample["action"]))

        action_chunk = torch.stack(action_chunk)

        return {
            **base_sample,
            "action_chunk": action_chunk,
        }


if __name__ == "__main__":
    print("Robot Dataset Module")
    print("Supports: LeRobot, Open X-Embodiment, Custom formats")
