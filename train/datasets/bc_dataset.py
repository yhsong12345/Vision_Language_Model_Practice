"""
Behavior Cloning Dataset

Dataset specifically designed for Behavioral Cloning (BC) training.
Uses D4RL as the primary data source with filtering for high-quality demonstrations.

================================================================================
TRAINING METHODS: This dataset is designed for:
================================================================================
1. BEHAVIORAL CLONING (Primary use case)
   - Standard BC: Direct state-action supervised learning
   - Filtered BC: Train on top-k% highest return trajectories
   - Weighted BC: Weight samples by trajectory return

2. OFFLINE IMITATION LEARNING
   - No reward signal needed (unlike offline RL)
   - Focus on mimicking expert behavior
   - Works with suboptimal demonstrations via filtering

3. VLA PRE-TRAINING
   - Use as warm-start before RL fine-tuning
   - Transfer to new tasks via fine-tuning

NOT recommended for:
- Online RL (no environment interaction)
- Reward learning (rewards not used in BC)
================================================================================

Recommended D4RL datasets for BC:
- hopper-expert-v2: Expert demonstrations (best quality)
- walker2d-expert-v2: Expert demonstrations
- halfcheetah-expert-v2: Expert demonstrations
- hopper-medium-expert-v2: Mix of medium and expert
- kitchen-complete-v0: Full task completion demos

Installation:
    pip install d4rl gymnasium

Usage:
    from train.datasets import BCDataset, create_bc_dataloader

    # Standard BC on expert data
    dataset = BCDataset(env_name="hopper-expert-v2")

    # Filtered BC on top 10% trajectories
    dataset = BCDataset(
        env_name="hopper-medium-v2",
        filter_top_k=0.1,
    )

    dataloader = create_bc_dataloader(dataset, batch_size=256)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple
import numpy as np


class BCDataset(Dataset):
    """
    Behavior Cloning Dataset based on D4RL.

    Provides state-action pairs for supervised imitation learning.
    Supports filtering to use only high-quality demonstrations.

    Args:
        env_name: D4RL environment name (e.g., "hopper-expert-v2")
        filter_top_k: If set, only use top k fraction of trajectories by return
        normalize_observations: Whether to normalize observations
        normalize_actions: Whether to normalize actions to [-1, 1]
        augment: Whether to apply data augmentation (noise injection)
        augment_std: Standard deviation for Gaussian noise augmentation
    """

    # Recommended environments for BC (expert/high-quality data)
    RECOMMENDED_ENVS = {
        # Expert demonstrations (highest quality)
        "expert": [
            "hopper-expert-v2",
            "walker2d-expert-v2",
            "halfcheetah-expert-v2",
            "ant-expert-v2",
        ],
        # Medium-expert mix (good balance)
        "medium_expert": [
            "hopper-medium-expert-v2",
            "walker2d-medium-expert-v2",
            "halfcheetah-medium-expert-v2",
            "ant-medium-expert-v2",
        ],
        # Kitchen tasks
        "kitchen": [
            "kitchen-complete-v0",
            "kitchen-partial-v0",
        ],
        # Adroit dexterous manipulation
        "adroit": [
            "pen-expert-v1",
            "hammer-expert-v1",
            "door-expert-v1",
            "relocate-expert-v1",
        ],
    }

    def __init__(
        self,
        env_name: str = "hopper-expert-v2",
        filter_top_k: Optional[float] = None,
        normalize_observations: bool = True,
        normalize_actions: bool = True,
        augment: bool = False,
        augment_std: float = 0.01,
    ):
        super().__init__()

        self.env_name = env_name
        self.filter_top_k = filter_top_k
        self.normalize_observations = normalize_observations
        self.normalize_actions = normalize_actions
        self.augment = augment
        self.augment_std = augment_std

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load D4RL dataset and prepare for BC."""
        try:
            import d4rl
            import gymnasium as gym

            print(f"Loading D4RL dataset for BC: {self.env_name}")

            # Create environment
            env = gym.make(self.env_name)

            # Get dataset
            dataset = d4rl.qlearning_dataset(env)

            # Extract data
            observations = dataset["observations"]
            actions = dataset["actions"]
            rewards = dataset["rewards"]
            terminals = dataset["terminals"]
            timeouts = dataset.get("timeouts", np.zeros_like(terminals))

            # Store environment info
            self.observation_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high

            env.close()

            # Filter by trajectory return if specified
            if self.filter_top_k is not None:
                observations, actions = self._filter_top_trajectories(
                    observations, actions, rewards, terminals, timeouts
                )

            # Compute normalization statistics
            if self.normalize_observations:
                self.obs_mean = observations.mean(axis=0)
                self.obs_std = observations.std(axis=0) + 1e-8
            else:
                self.obs_mean = np.zeros(self.observation_dim)
                self.obs_std = np.ones(self.observation_dim)

            if self.normalize_actions:
                # Normalize to [-1, 1]
                self.action_mean = (self.action_high + self.action_low) / 2
                self.action_scale = (self.action_high - self.action_low) / 2
            else:
                self.action_mean = np.zeros(self.action_dim)
                self.action_scale = np.ones(self.action_dim)

            self.observations = observations.astype(np.float32)
            self.actions = actions.astype(np.float32)

            print(f"Loaded {len(self.observations)} samples for BC")
            print(f"Observation dim: {self.observation_dim}")
            print(f"Action dim: {self.action_dim}")

        except ImportError:
            print("D4RL or gymnasium not installed. Creating dummy dataset.")
            print("Install with: pip install d4rl gymnasium")
            self._create_dummy_dataset()

        except Exception as e:
            print(f"Could not load D4RL: {e}")
            self._create_dummy_dataset()

    def _filter_top_trajectories(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        timeouts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter to keep only top-k trajectories by return."""
        print(f"Filtering top {self.filter_top_k * 100:.0f}% trajectories...")

        # Find trajectory boundaries
        episode_ends = np.where(terminals | timeouts)[0]
        episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])

        # Compute trajectory returns
        trajectory_returns = []
        trajectory_indices = []

        for start, end in zip(episode_starts, episode_ends):
            traj_return = rewards[start:end + 1].sum()
            trajectory_returns.append(traj_return)
            trajectory_indices.append((start, end + 1))

        trajectory_returns = np.array(trajectory_returns)

        # Get top-k threshold
        threshold = np.percentile(trajectory_returns, (1 - self.filter_top_k) * 100)

        # Filter trajectories
        filtered_obs = []
        filtered_actions = []

        for (start, end), ret in zip(trajectory_indices, trajectory_returns):
            if ret >= threshold:
                filtered_obs.append(observations[start:end])
                filtered_actions.append(actions[start:end])

        filtered_obs = np.concatenate(filtered_obs, axis=0)
        filtered_actions = np.concatenate(filtered_actions, axis=0)

        print(f"Filtered from {len(observations)} to {len(filtered_obs)} samples")
        print(f"Return threshold: {threshold:.2f}")

        return filtered_obs, filtered_actions

    def _create_dummy_dataset(self):
        """Create dummy dataset for testing."""
        num_samples = 10000
        self.observation_dim = 11  # hopper
        self.action_dim = 3

        self.observations = np.random.randn(num_samples, self.observation_dim).astype(np.float32)
        self.actions = np.random.uniform(-1, 1, (num_samples, self.action_dim)).astype(np.float32)

        self.obs_mean = np.zeros(self.observation_dim)
        self.obs_std = np.ones(self.observation_dim)
        self.action_mean = np.zeros(self.action_dim)
        self.action_scale = np.ones(self.action_dim)
        self.action_low = -np.ones(self.action_dim)
        self.action_high = np.ones(self.action_dim)

        print(f"Created dummy BC dataset with {num_samples} samples")

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a state-action pair for BC training."""
        obs = self.observations[idx].copy()
        action = self.actions[idx].copy()

        # Normalize observation
        obs = (obs - self.obs_mean) / self.obs_std

        # Normalize action
        action = (action - self.action_mean) / self.action_scale

        # Apply augmentation (noise injection)
        if self.augment:
            obs = obs + np.random.randn(*obs.shape).astype(np.float32) * self.augment_std

        return {
            "observations": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(action, dtype=torch.float32),
        }

    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        """Get normalization parameters for inference."""
        return {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "action_mean": self.action_mean,
            "action_scale": self.action_scale,
        }

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action back to original scale."""
        return action * self.action_scale + self.action_mean


class FilteredBCDataset(BCDataset):
    """
    BC Dataset with automatic filtering for high-quality demonstrations.

    Convenience wrapper that defaults to filtering top 10% trajectories.
    Use this when working with medium or mixed-quality datasets.

    Usage:
        # Automatically filters to top 10% trajectories
        dataset = FilteredBCDataset(env_name="hopper-medium-v2")
    """

    def __init__(
        self,
        env_name: str = "hopper-medium-v2",
        filter_top_k: float = 0.1,  # Default: top 10%
        **kwargs,
    ):
        super().__init__(
            env_name=env_name,
            filter_top_k=filter_top_k,
            **kwargs,
        )


class WeightedBCDataset(BCDataset):
    """
    BC Dataset with trajectory-based sample weighting.

    Samples from higher-return trajectories more frequently.
    Useful when you want to use all data but prioritize better demonstrations.

    Usage:
        dataset = WeightedBCDataset(env_name="hopper-medium-v2")
        # Samples are weighted by trajectory return
    """

    def __init__(
        self,
        env_name: str = "hopper-medium-v2",
        temperature: float = 1.0,  # Higher = more uniform, lower = more peaked
        **kwargs,
    ):
        self.temperature = temperature
        self._weights = None
        super().__init__(env_name=env_name, **kwargs)

    def _load_dataset(self):
        """Load dataset and compute sample weights."""
        try:
            import d4rl
            import gymnasium as gym

            print(f"Loading weighted BC dataset: {self.env_name}")

            env = gym.make(self.env_name)
            dataset = d4rl.qlearning_dataset(env)

            observations = dataset["observations"]
            actions = dataset["actions"]
            rewards = dataset["rewards"]
            terminals = dataset["terminals"]
            timeouts = dataset.get("timeouts", np.zeros_like(terminals))

            self.observation_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high

            env.close()

            # Compute weights based on trajectory return
            self._weights = self._compute_weights(rewards, terminals, timeouts)

            # Normalization
            if self.normalize_observations:
                self.obs_mean = observations.mean(axis=0)
                self.obs_std = observations.std(axis=0) + 1e-8
            else:
                self.obs_mean = np.zeros(self.observation_dim)
                self.obs_std = np.ones(self.observation_dim)

            if self.normalize_actions:
                self.action_mean = (self.action_high + self.action_low) / 2
                self.action_scale = (self.action_high - self.action_low) / 2
            else:
                self.action_mean = np.zeros(self.action_dim)
                self.action_scale = np.ones(self.action_dim)

            self.observations = observations.astype(np.float32)
            self.actions = actions.astype(np.float32)

            print(f"Loaded {len(self.observations)} weighted samples")

        except ImportError:
            print("D4RL not installed. Creating dummy dataset.")
            self._create_dummy_dataset()
            self._weights = np.ones(len(self.observations))

        except Exception as e:
            print(f"Could not load D4RL: {e}")
            self._create_dummy_dataset()
            self._weights = np.ones(len(self.observations))

    def _compute_weights(
        self,
        rewards: np.ndarray,
        terminals: np.ndarray,
        timeouts: np.ndarray,
    ) -> np.ndarray:
        """Compute per-sample weights based on trajectory return."""
        episode_ends = np.where(terminals | timeouts)[0]
        episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])

        weights = np.zeros(len(rewards))

        for start, end in zip(episode_starts, episode_ends):
            traj_return = rewards[start:end + 1].sum()
            weights[start:end + 1] = traj_return

        # Normalize and apply temperature
        weights = weights - weights.min()
        weights = weights / (weights.max() + 1e-8)
        weights = np.exp(weights / self.temperature)
        weights = weights / weights.sum()

        return weights

    def get_sampler(self):
        """Get weighted sampler for DataLoader."""
        from torch.utils.data import WeightedRandomSampler

        return WeightedRandomSampler(
            weights=self._weights,
            num_samples=len(self._weights),
            replacement=True,
        )


def collate_bc(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for BC datasets."""
    return {
        "observations": torch.stack([b["observations"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
    }


def create_bc_dataloader(
    dataset: BCDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampling: bool = False,
) -> DataLoader:
    """
    Create DataLoader for BC dataset.

    Args:
        dataset: BCDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if use_weighted_sampling=True)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        use_weighted_sampling: Use weighted sampling for WeightedBCDataset

    Returns:
        DataLoader instance
    """
    sampler = None

    if use_weighted_sampling and isinstance(dataset, WeightedBCDataset):
        sampler = dataset.get_sampler()
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_bc,
        pin_memory=pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Behavior Cloning Dataset Test")
    print("=" * 60)

    # Test standard BC dataset
    print("\n1. Testing Standard BC Dataset:")
    try:
        dataset = BCDataset(env_name="hopper-expert-v2")
        print(f"   Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"   Observation shape: {sample['observations'].shape}")
        print(f"   Action shape: {sample['actions'].shape}")

        # Test dataloader
        dataloader = create_bc_dataloader(dataset, batch_size=256)
        batch = next(iter(dataloader))
        print(f"   Batch observations: {batch['observations'].shape}")
        print(f"   Batch actions: {batch['actions'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test filtered BC dataset
    print("\n2. Testing Filtered BC Dataset (top 10%):")
    try:
        dataset = FilteredBCDataset(env_name="hopper-medium-v2")
        print(f"   Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test weighted BC dataset
    print("\n3. Testing Weighted BC Dataset:")
    try:
        dataset = WeightedBCDataset(env_name="hopper-medium-v2")
        print(f"   Dataset size: {len(dataset)}")
        dataloader = create_bc_dataloader(
            dataset, batch_size=256, use_weighted_sampling=True
        )
        batch = next(iter(dataloader))
        print(f"   Weighted batch shape: {batch['observations'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("BC Dataset test complete!")
    print("=" * 60)
