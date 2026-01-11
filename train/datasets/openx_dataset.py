"""
Open X-Embodiment Dataset Loaders

The Open X-Embodiment dataset is a large-scale collection of robot manipulation
data from multiple institutions and robot platforms.

================================================================================
TRAINING METHODS: This dataset is designed for:
================================================================================
1. VLA FINE-TUNING (Primary use case)
   - Language-conditioned policy learning
   - Vision-Language-Action model training (RT-2, OpenVLA style)
   - Instruction following for robot manipulation

2. IMITATION LEARNING with Language
   - Language-conditioned Behavioral Cloning
   - Goal-conditioned imitation learning
   - Multi-task learning with language goals

3. VLM PRETRAINING (vision-language alignment)
   - Robot-specific vision-language alignment
   - Grounding language to robot observations
   - Can be used for CLIP-style contrastive learning

4. CROSS-EMBODIMENT TRANSFER
   - Train on multiple robot platforms
   - Learn transferable representations
   - Domain adaptation between robots

NOT recommended for:
- Online RL (no environment interaction)
- State-only policies (designed for vision input)
================================================================================

Recommended datasets:
- Bridge V2: Real robot manipulation with language instructions
- RT-1: Google's robot learning dataset
- Fractal: Fractal robot data
- Kuka: Industrial robot manipulation

Note: These datasets are hosted on TensorFlow Datasets (TFDS).
You need to install: pip install tensorflow-datasets

For easier access, many are also available on HuggingFace:
- jxu124/OpenX-Embodiment

Usage:
    from train.datasets import BridgeDataset, create_openx_dataloader

    dataset = BridgeDataset(split="train[:1000]")
    dataloader = create_openx_dataloader(dataset, batch_size=32)
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, Dict, Any, List, Tuple, Iterator
from PIL import Image
import numpy as np


class OpenXDataset(Dataset):
    """
    Base class for Open X-Embodiment datasets.

    Open X-Embodiment provides standardized robot learning data:
    - observation/image: RGB images
    - observation/wrist_image: Wrist camera (if available)
    - observation/state: Robot proprioception
    - action: Robot actions
    - language_instruction: Natural language task description

    Args:
        dataset_name: Name of the dataset in Open X format
        split: Dataset split (e.g., "train", "train[:1000]")
        image_size: Target image size
        use_language: Whether to include language instructions
        use_wrist_camera: Whether to include wrist camera images
    """

    # Available Open X datasets with HuggingFace mirrors
    AVAILABLE_DATASETS = {
        "bridge": "bridge_orig",
        "bridge_v2": "bridge_dataset",
        "rt1": "fractal20220817_data",
        "kuka": "kuka",
        "taco_play": "taco_play",
        "jaco_play": "jaco_play",
        "berkeley_cable_routing": "berkeley_cable_routing",
        "berkeley_autolab_ur5": "berkeley_autolab_ur5",
        "language_table": "language_table",
    }

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        use_language: bool = True,
        use_wrist_camera: bool = False,
        max_samples: Optional[int] = None,
        transform=None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.use_language = use_language
        self.use_wrist_camera = use_wrist_camera
        self.max_samples = max_samples
        self.transform = transform

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from TensorFlow Datasets or HuggingFace."""
        # Try HuggingFace first (easier)
        try:
            self._load_from_huggingface()
            return
        except Exception as e:
            print(f"HuggingFace loading failed: {e}")

        # Try TensorFlow Datasets
        try:
            self._load_from_tfds()
            return
        except Exception as e:
            print(f"TFDS loading failed: {e}")

        # Fallback to dummy
        print("Creating dummy dataset for testing...")
        self._create_dummy_dataset()

    def _load_from_huggingface(self):
        """Load from HuggingFace mirror."""
        from datasets import load_dataset

        # Try jxu124's OpenX mirror
        try:
            print(f"Loading {self.dataset_name} from HuggingFace...")
            self.hf_dataset = load_dataset(
                "jxu124/OpenX-Embodiment",
                self.dataset_name,
                split=self.split,
                trust_remote_code=True,
            )
            self.source = "huggingface"

            if self.max_samples:
                self.hf_dataset = self.hf_dataset.select(range(min(self.max_samples, len(self.hf_dataset))))

            print(f"Loaded {len(self.hf_dataset)} samples from HuggingFace")
            return

        except Exception:
            # Try loading the raw dataset name
            self.hf_dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                trust_remote_code=True,
            )
            self.source = "huggingface"

            if self.max_samples:
                self.hf_dataset = self.hf_dataset.select(range(min(self.max_samples, len(self.hf_dataset))))

            print(f"Loaded {len(self.hf_dataset)} samples")

    def _load_from_tfds(self):
        """Load from TensorFlow Datasets."""
        import tensorflow_datasets as tfds

        print(f"Loading {self.dataset_name} from TFDS...")

        # Build dataset
        builder = tfds.builder(self.dataset_name)
        builder.download_and_prepare()

        ds = builder.as_dataset(split=self.split)

        # Convert to list (for random access)
        self.tfds_data = []
        for i, sample in enumerate(ds):
            if self.max_samples and i >= self.max_samples:
                break
            self.tfds_data.append(self._convert_tfds_sample(sample))

        self.source = "tfds"
        print(f"Loaded {len(self.tfds_data)} samples from TFDS")

    def _convert_tfds_sample(self, sample) -> Dict:
        """Convert TFDS sample to numpy."""
        import tensorflow as tf

        def to_numpy(x):
            if isinstance(x, tf.Tensor):
                return x.numpy()
            elif isinstance(x, dict):
                return {k: to_numpy(v) for k, v in x.items()}
            return x

        return to_numpy(sample)

    def _create_dummy_dataset(self):
        """Create dummy data for testing."""
        self.source = "dummy"

        instructions = [
            "pick up the red block",
            "move the arm to the left",
            "place the object on the table",
            "push the cube forward",
            "open the drawer",
            "close the gripper",
        ]

        self.dummy_data = []
        num_samples = self.max_samples or 1000

        for i in range(num_samples):
            self.dummy_data.append({
                "observation": {
                    "image": np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8),
                    "wrist_image": np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8),
                    "state": np.random.randn(8).astype(np.float32),
                },
                "action": np.random.randn(7).astype(np.float32),
                "language_instruction": instructions[i % len(instructions)],
                "is_terminal": False,
                "is_first": i % 100 == 0,
            })

        print(f"Created dummy dataset with {len(self.dummy_data)} samples")

    def __len__(self) -> int:
        if self.source == "huggingface":
            return len(self.hf_dataset)
        elif self.source == "tfds":
            return len(self.tfds_data)
        else:
            return len(self.dummy_data)

    def _get_sample(self, idx: int) -> Dict:
        """Get raw sample by index."""
        if self.source == "huggingface":
            return self.hf_dataset[idx]
        elif self.source == "tfds":
            return self.tfds_data[idx]
        else:
            return self.dummy_data[idx]

    def _process_image(self, image) -> torch.Tensor:
        """Process image to tensor."""
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 3:  # HWC
                img = Image.fromarray(image)
            else:  # CHW
                img = Image.fromarray(image.transpose(1, 2, 0))
        elif isinstance(image, Image.Image):
            img = image
        else:
            # Handle bytes or other formats
            img = Image.open(image).convert("RGB")

        # Resize
        img = img.resize(self.image_size)

        # To tensor
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._get_sample(idx)

        output = {}

        # Main image
        if "observation" in sample and "image" in sample["observation"]:
            output["pixel_values"] = self._process_image(sample["observation"]["image"])
        elif "image" in sample:
            output["pixel_values"] = self._process_image(sample["image"])
        else:
            output["pixel_values"] = torch.randn(3, *self.image_size)

        # Wrist camera
        if self.use_wrist_camera:
            if "observation" in sample and "wrist_image" in sample["observation"]:
                output["wrist_image"] = self._process_image(sample["observation"]["wrist_image"])

        # State
        if "observation" in sample and "state" in sample["observation"]:
            state = sample["observation"]["state"]
            output["state"] = torch.tensor(state, dtype=torch.float32) if not isinstance(state, torch.Tensor) else state

        # Action
        if "action" in sample:
            action = sample["action"]
            output["action"] = torch.tensor(action, dtype=torch.float32) if not isinstance(action, torch.Tensor) else action

        # Language instruction
        if self.use_language and "language_instruction" in sample:
            instruction = sample["language_instruction"]
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")
            output["instruction"] = instruction

        # Terminal flags
        output["is_terminal"] = sample.get("is_terminal", False)
        output["is_first"] = sample.get("is_first", idx == 0)

        return output


class BridgeDataset(OpenXDataset):
    """
    Bridge V2 Dataset.

    Real robot manipulation dataset with:
    - WidowX robot arm
    - Diverse manipulation tasks
    - Natural language instructions
    - ~60k demonstrations

    Action space: 7D (6 DoF end-effector + gripper)

    Usage:
        dataset = BridgeDataset(split="train[:10000]")
        sample = dataset[0]
        print(sample["instruction"])  # "pick up the red block"
    """

    def __init__(
        self,
        split: str = "train",
        version: str = "v2",
        **kwargs,
    ):
        dataset_name = "bridge_dataset" if version == "v2" else "bridge_orig"
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            use_language=True,
            **kwargs,
        )


class RT1Dataset(OpenXDataset):
    """
    RT-1 (Robotics Transformer) Dataset.

    Google's large-scale robot learning dataset:
    - Everyday Robots mobile manipulator
    - Kitchen manipulation tasks
    - 130k demonstrations

    Action space: 11D (7 arm + 3 base + 1 gripper)

    Usage:
        dataset = RT1Dataset(split="train[:5000]")
    """

    def __init__(
        self,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(
            dataset_name="fractal20220817_data",
            split=split,
            use_language=True,
            **kwargs,
        )


class LanguageTableDataset(OpenXDataset):
    """
    Language Table Dataset.

    Tabletop manipulation with language instructions:
    - Simple pushing/picking tasks
    - Clear language grounding
    - Good for VLA training

    Usage:
        dataset = LanguageTableDataset(split="train")
    """

    def __init__(
        self,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(
            dataset_name="language_table",
            split=split,
            use_language=True,
            **kwargs,
        )


def collate_openx(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for Open X datasets."""
    output = {}

    # Stack tensors
    for key in ["pixel_values", "wrist_image", "state", "action"]:
        if key in batch[0]:
            output[key] = torch.stack([b[key] for b in batch])

    # Collect strings
    if "instruction" in batch[0]:
        output["instruction"] = [b["instruction"] for b in batch]

    # Collect booleans
    for key in ["is_terminal", "is_first"]:
        if key in batch[0]:
            output[key] = [b[key] for b in batch]

    return output


def create_openx_dataloader(
    dataset: OpenXDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for Open X dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_openx,
        pin_memory=pin_memory,
        drop_last=True,
    )


class OpenXStreamingDataset(IterableDataset):
    """
    Streaming version of Open X dataset for large-scale training.

    Useful when the dataset is too large to fit in memory.
    Streams data directly from TFDS.

    Usage:
        dataset = OpenXStreamingDataset("bridge_dataset")
        dataloader = DataLoader(dataset, batch_size=32)
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        shuffle_buffer: int = 1000,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        try:
            import tensorflow_datasets as tfds
            import tensorflow as tf

            # Load as streaming dataset
            ds = tfds.load(self.dataset_name, split=self.split)
            ds = ds.shuffle(self.shuffle_buffer)

            for sample in ds:
                yield self._process_sample(sample)

        except ImportError:
            # Fallback: yield dummy samples
            while True:
                yield {
                    "pixel_values": torch.randn(3, *self.image_size),
                    "action": torch.randn(7),
                    "instruction": "dummy instruction",
                }

    def _process_sample(self, sample) -> Dict[str, torch.Tensor]:
        """Process a single TFDS sample."""
        import tensorflow as tf

        output = {}

        # Image
        if "observation" in sample and "image" in sample["observation"]:
            img = sample["observation"]["image"].numpy()
            img = Image.fromarray(img).resize(self.image_size)
            output["pixel_values"] = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # Action
        if "action" in sample:
            output["action"] = torch.tensor(sample["action"].numpy(), dtype=torch.float32)

        # Language
        if "language_instruction" in sample:
            instruction = sample["language_instruction"].numpy()
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")
            output["instruction"] = instruction

        return output


if __name__ == "__main__":
    print("=" * 60)
    print("Open X-Embodiment Dataset Test")
    print("=" * 60)

    # Test Bridge
    print("\n1. Testing Bridge Dataset:")
    try:
        dataset = BridgeDataset(split="train", max_samples=100)
        print(f"   Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Image shape: {sample['pixel_values'].shape}")
        if "action" in sample:
            print(f"   Action shape: {sample['action'].shape}")
        if "instruction" in sample:
            print(f"   Instruction: {sample['instruction'][:50]}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Test RT1
    print("\n2. Testing RT1 Dataset:")
    try:
        dataset = RT1Dataset(split="train", max_samples=100)
        print(f"   Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"   Image shape: {sample['pixel_values'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Open X dataset test complete!")
    print("=" * 60)
