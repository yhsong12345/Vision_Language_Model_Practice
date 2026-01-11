"""
Reinforcement Learning Dataset Loaders

Datasets for offline RL training and evaluation.
These can be used for pre-training policies or for RL fine-tuning.

================================================================================
TRAINING METHODS: This dataset is designed for:
================================================================================
1. OFFLINE REINFORCEMENT LEARNING (Primary use case)
   - Conservative Q-Learning (CQL): Penalize OOD actions
   - Implicit Q-Learning (IQL): Expectile regression
   - TD3+BC: TD3 with behavioral cloning regularization
   - AWAC: Advantage-weighted actor-critic

2. SEQUENCE MODELING FOR RL
   - Decision Transformer: Transformer for trajectory modeling
   - Trajectory Transformer: Full sequence prediction
   - GATO-style multi-task sequence modeling

3. IMITATION LEARNING BASELINES
   - Behavioral Cloning (BC): Supervised action prediction
   - Filtered BC: BC on high-reward trajectories
   - 10% BC: BC on top 10% demonstrations

4. RL FINE-TUNING (with ReplayBuffer)
   - PPO: On-policy fine-tuning from offline initialization
   - SAC: Off-policy fine-tuning with replay buffer
   - GRPO: Group relative policy optimization

5. REWARD LEARNING
   - Inverse RL: Learn reward from demonstrations
   - Preference learning: RLHF-style reward modeling
   - Reward shaping from offline data

USAGE BY TRAINER:
- RLTrainer (SAC, TD3): Use transition mode (trajectory_mode=False)
- PPOTrainer: Use with ReplayBuffer for online collection
- Decision Transformer: Use trajectory mode (trajectory_mode=True)
- BehavioralCloning: Works with both modes
================================================================================

Recommended datasets:
- D4RL: Standard offline RL benchmarks (MuJoCo, Antmaze, Kitchen)
- RoboMimic: Robot manipulation demonstrations
- MetaWorld: Multi-task robot learning
- Minari: Modern offline RL dataset format

Installation:
    # D4RL
    pip install d4rl

    # Minari (recommended for new projects)
    pip install minari

    # RoboMimic
    pip install robomimic

Usage:
    from train.datasets import D4RLDataset, create_rl_dataloader

    dataset = D4RLDataset(env_name="hopper-medium-v2")
    dataloader = create_rl_dataloader(dataset, batch_size=256)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np


class RLDataset(Dataset):
    """
    Base class for RL datasets.

    Standard format:
    - observations: State observations
    - actions: Actions taken
    - rewards: Rewards received
    - next_observations: Next states
    - terminals: Episode termination flags
    - timeouts: Truncation flags (optional)

    Supports both:
    - Transition-level sampling (for SAC, TD3)
    - Trajectory-level sampling (for BC, sequence models)

    Args:
        transitions: Dict of numpy arrays or path to dataset
        trajectory_mode: If True, return full trajectories
        trajectory_length: Fixed trajectory length (if trajectory_mode)
        normalize_observations: Whether to normalize observations
        normalize_rewards: Whether to normalize rewards
    """

    def __init__(
        self,
        transitions: Optional[Dict[str, np.ndarray]] = None,
        trajectory_mode: bool = False,
        trajectory_length: int = 100,
        normalize_observations: bool = True,
        normalize_rewards: bool = False,
    ):
        super().__init__()

        self.trajectory_mode = trajectory_mode
        self.trajectory_length = trajectory_length
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards

        if transitions is not None:
            self._load_transitions(transitions)

    def _load_transitions(self, transitions: Dict[str, np.ndarray]):
        """Load transition data."""
        self.observations = transitions["observations"]
        self.actions = transitions["actions"]
        self.rewards = transitions["rewards"]
        self.next_observations = transitions.get(
            "next_observations",
            np.roll(self.observations, -1, axis=0)
        )
        self.terminals = transitions.get("terminals", np.zeros(len(self.observations)))
        self.timeouts = transitions.get("timeouts", np.zeros(len(self.observations)))

        # Compute statistics for normalization
        if self.normalize_observations:
            self.obs_mean = self.observations.mean(axis=0)
            self.obs_std = self.observations.std(axis=0) + 1e-8
        else:
            self.obs_mean = 0
            self.obs_std = 1

        if self.normalize_rewards:
            self.reward_mean = self.rewards.mean()
            self.reward_std = self.rewards.std() + 1e-8
        else:
            self.reward_mean = 0
            self.reward_std = 1

        # Build trajectory indices if needed
        if self.trajectory_mode:
            self._build_trajectory_indices()

        print(f"Loaded {len(self.observations)} transitions")
        print(f"Observation dim: {self.observations.shape[1]}")
        print(f"Action dim: {self.actions.shape[1]}")

    def _build_trajectory_indices(self):
        """Build indices for trajectory sampling."""
        self.trajectory_starts = [0]

        for i in range(len(self.terminals)):
            if self.terminals[i] or self.timeouts[i]:
                self.trajectory_starts.append(i + 1)

        # Remove last if it's at the end
        if self.trajectory_starts[-1] >= len(self.observations):
            self.trajectory_starts = self.trajectory_starts[:-1]

        print(f"Found {len(self.trajectory_starts)} trajectories")

    def __len__(self) -> int:
        if self.trajectory_mode:
            return len(self.trajectory_starts)
        return len(self.observations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.trajectory_mode:
            return self._get_trajectory(idx)
        return self._get_transition(idx)

    def _get_transition(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single transition."""
        obs = (self.observations[idx] - self.obs_mean) / self.obs_std
        next_obs = (self.next_observations[idx] - self.obs_mean) / self.obs_std
        reward = (self.rewards[idx] - self.reward_mean) / self.reward_std

        return {
            "observations": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(self.actions[idx], dtype=torch.float32),
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "next_observations": torch.tensor(next_obs, dtype=torch.float32),
            "terminals": torch.tensor(self.terminals[idx], dtype=torch.float32),
        }

    def _get_trajectory(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a trajectory segment."""
        start = self.trajectory_starts[idx]

        # Find trajectory end
        if idx + 1 < len(self.trajectory_starts):
            end = self.trajectory_starts[idx + 1]
        else:
            end = len(self.observations)

        # Get trajectory segment
        length = min(end - start, self.trajectory_length)

        obs = self.observations[start:start + length]
        actions = self.actions[start:start + length]
        rewards = self.rewards[start:start + length]

        # Normalize
        obs = (obs - self.obs_mean) / self.obs_std
        rewards = (rewards - self.reward_mean) / self.reward_std

        # Pad if necessary
        if length < self.trajectory_length:
            pad_length = self.trajectory_length - length
            obs = np.pad(obs, ((0, pad_length), (0, 0)))
            actions = np.pad(actions, ((0, pad_length), (0, 0)))
            rewards = np.pad(rewards, (0, pad_length))
            mask = np.concatenate([np.ones(length), np.zeros(pad_length)])
        else:
            mask = np.ones(self.trajectory_length)

        return {
            "observations": torch.tensor(obs, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }


class D4RLDataset(RLDataset):
    """
    D4RL Offline RL Dataset.

    D4RL provides standardized offline RL benchmarks:
    - MuJoCo locomotion: hopper, walker2d, halfcheetah
    - Antmaze: Navigation tasks
    - Kitchen: Multi-task manipulation
    - Adroit: Dexterous manipulation

    Dataset types:
    - random: Random policy
    - medium: Partially trained policy
    - medium-replay: Replay buffer of medium policy
    - medium-expert: Mix of medium and expert
    - expert: Expert policy

    Usage:
        dataset = D4RLDataset(
            env_name="hopper-medium-v2",
            trajectory_mode=False,  # For SAC/TD3
        )

        # For sequence models (Decision Transformer)
        dataset = D4RLDataset(
            env_name="hopper-medium-v2",
            trajectory_mode=True,
            trajectory_length=20,
        )
    """

    ENVIRONMENTS = {
        # MuJoCo locomotion
        "hopper": ["random", "medium", "medium-replay", "medium-expert", "expert"],
        "walker2d": ["random", "medium", "medium-replay", "medium-expert", "expert"],
        "halfcheetah": ["random", "medium", "medium-replay", "medium-expert", "expert"],
        "ant": ["random", "medium", "medium-replay", "medium-expert", "expert"],
        # Antmaze
        "antmaze": ["umaze", "umaze-diverse", "medium-play", "medium-diverse", "large-play", "large-diverse"],
        # Kitchen
        "kitchen": ["complete", "partial", "mixed"],
        # Adroit
        "pen": ["human", "cloned", "expert"],
        "hammer": ["human", "cloned", "expert"],
        "door": ["human", "cloned", "expert"],
        "relocate": ["human", "cloned", "expert"],
    }

    def __init__(
        self,
        env_name: str = "hopper-medium-v2",
        trajectory_mode: bool = False,
        trajectory_length: int = 100,
        normalize_observations: bool = True,
        **kwargs,
    ):
        self.env_name = env_name
        super().__init__(
            trajectory_mode=trajectory_mode,
            trajectory_length=trajectory_length,
            normalize_observations=normalize_observations,
            **kwargs,
        )
        self._load_d4rl()

    def _load_d4rl(self):
        """Load D4RL dataset."""
        try:
            import d4rl
            import gym

            print(f"Loading D4RL dataset: {self.env_name}")

            # Create environment
            env = gym.make(self.env_name)

            # Get dataset
            dataset = d4rl.qlearning_dataset(env)

            self._load_transitions(dataset)

            # Store environment info
            self.observation_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]
            self.action_range = (
                env.action_space.low.min(),
                env.action_space.high.max()
            )

            env.close()

        except ImportError:
            print("D4RL not installed. Creating dummy dataset.")
            print("Install with: pip install d4rl")
            self._create_dummy_dataset()

        except Exception as e:
            print(f"Could not load D4RL: {e}")
            self._create_dummy_dataset()

    def _create_dummy_dataset(self):
        """Create dummy dataset for testing."""
        num_samples = 10000
        obs_dim = 11  # hopper
        action_dim = 3

        self.observation_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = (-1.0, 1.0)

        transitions = {
            "observations": np.random.randn(num_samples, obs_dim).astype(np.float32),
            "actions": np.random.randn(num_samples, action_dim).astype(np.float32),
            "rewards": np.random.randn(num_samples).astype(np.float32),
            "terminals": (np.random.rand(num_samples) < 0.01).astype(np.float32),
            "timeouts": (np.random.rand(num_samples) < 0.01).astype(np.float32),
        }

        self._load_transitions(transitions)
        print(f"Created dummy D4RL dataset with {num_samples} samples")


class MinariDataset(RLDataset):
    """
    Minari Dataset (Modern D4RL replacement).

    Minari is the successor to D4RL with better:
    - Dataset management
    - Version control
    - Reproducibility

    Available datasets on HuggingFace: farama-foundation/minari

    Usage:
        dataset = MinariDataset("pointmaze-medium-v1")
    """

    def __init__(
        self,
        dataset_name: str,
        trajectory_mode: bool = False,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        super().__init__(trajectory_mode=trajectory_mode, **kwargs)
        self._load_minari()

    def _load_minari(self):
        """Load Minari dataset."""
        try:
            import minari

            print(f"Loading Minari dataset: {self.dataset_name}")

            # Download if needed
            dataset = minari.load_dataset(self.dataset_name)

            # Convert to transitions format
            observations = []
            actions = []
            rewards = []
            terminals = []

            for episode in dataset:
                observations.append(episode.observations[:-1])
                actions.append(episode.actions)
                rewards.append(episode.rewards)
                terms = np.zeros(len(episode.actions))
                terms[-1] = 1.0
                terminals.append(terms)

            transitions = {
                "observations": np.concatenate(observations),
                "actions": np.concatenate(actions),
                "rewards": np.concatenate(rewards),
                "terminals": np.concatenate(terminals),
            }

            self._load_transitions(transitions)

        except ImportError:
            print("Minari not installed. Install with: pip install minari")
            self._create_dummy_dataset()

        except Exception as e:
            print(f"Could not load Minari dataset: {e}")
            self._create_dummy_dataset()

    def _create_dummy_dataset(self):
        """Create dummy dataset."""
        D4RLDataset._create_dummy_dataset(self)


class RoboMimicDataset(RLDataset):
    """
    RoboMimic Dataset for Robot Manipulation.

    RoboMimic provides high-quality robot manipulation demonstrations:
    - Lift: Lift a cube
    - Can: Pick and place can
    - Square: Insert square peg
    - Transport: Two-arm transport task

    Data types:
    - ph: Proficient human demonstrations
    - mh: Multi-human (varying skill)
    - mg: Machine-generated

    Usage:
        dataset = RoboMimicDataset(
            task="lift",
            data_type="ph",
            hdf5_path="/path/to/data.hdf5",
        )
    """

    TASKS = ["lift", "can", "square", "transport", "tool_hang"]

    def __init__(
        self,
        task: str = "lift",
        data_type: str = "ph",
        hdf5_path: Optional[str] = None,
        trajectory_mode: bool = True,
        trajectory_length: int = 50,
        use_images: bool = False,
        image_size: Tuple[int, int] = (84, 84),
        **kwargs,
    ):
        self.task = task
        self.data_type = data_type
        self.hdf5_path = hdf5_path
        self.use_images = use_images
        self.image_size = image_size

        super().__init__(
            trajectory_mode=trajectory_mode,
            trajectory_length=trajectory_length,
            **kwargs,
        )
        self._load_robomimic()

    def _load_robomimic(self):
        """Load RoboMimic dataset."""
        if self.hdf5_path is None:
            print("No HDF5 path provided. Creating dummy dataset.")
            self._create_dummy_dataset()
            return

        try:
            import h5py

            print(f"Loading RoboMimic: {self.task} ({self.data_type})")

            with h5py.File(self.hdf5_path, "r") as f:
                # Get all demonstrations
                demos = list(f["data"].keys())

                observations = []
                actions = []
                rewards = []
                terminals = []

                for demo_key in demos:
                    demo = f["data"][demo_key]

                    # Get observations
                    if self.use_images:
                        obs = demo["obs/agentview_image"][:]
                    else:
                        obs = demo["obs/robot0_eef_pos"][:]
                        if "robot0_eef_quat" in demo["obs"]:
                            quat = demo["obs/robot0_eef_quat"][:]
                            obs = np.concatenate([obs, quat], axis=1)

                    observations.append(obs)
                    actions.append(demo["actions"][:])
                    rewards.append(demo["rewards"][:])

                    terms = np.zeros(len(demo["actions"]))
                    terms[-1] = 1.0
                    terminals.append(terms)

            transitions = {
                "observations": np.concatenate(observations),
                "actions": np.concatenate(actions),
                "rewards": np.concatenate(rewards),
                "terminals": np.concatenate(terminals),
            }

            self._load_transitions(transitions)

        except ImportError:
            print("h5py not installed. Install with: pip install h5py")
            self._create_dummy_dataset()

        except Exception as e:
            print(f"Could not load RoboMimic: {e}")
            self._create_dummy_dataset()

    def _create_dummy_dataset(self):
        """Create dummy dataset."""
        num_samples = 5000
        obs_dim = 7 if not self.use_images else (3, *self.image_size)
        action_dim = 7

        if self.use_images:
            observations = np.random.randint(0, 255, (num_samples, *obs_dim), dtype=np.uint8)
        else:
            observations = np.random.randn(num_samples, obs_dim).astype(np.float32)

        transitions = {
            "observations": observations,
            "actions": np.random.randn(num_samples, action_dim).astype(np.float32),
            "rewards": np.random.randn(num_samples).astype(np.float32),
            "terminals": (np.random.rand(num_samples) < 0.02).astype(np.float32),
        }

        self._load_transitions(transitions)


class ReplayBuffer(RLDataset):
    """
    Online Replay Buffer for RL training.

    Supports:
    - Adding new transitions during training
    - Priority sampling (for PER)
    - N-step returns

    Usage:
        buffer = ReplayBuffer(
            observation_dim=11,
            action_dim=3,
            capacity=1000000,
        )

        # Add transitions
        buffer.add(obs, action, reward, next_obs, done)

        # Sample batch
        batch = buffer.sample(batch_size=256)
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int = 1000000,
        prioritized: bool = False,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta

        # Pre-allocate arrays
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)

        if prioritized:
            self.priorities = np.ones(capacity, dtype=np.float32)

        self.size = 0
        self.pointer = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminal: bool,
        priority: Optional[float] = None,
    ):
        """Add a transition to the buffer."""
        self.observations[self.pointer] = observation
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_observations[self.pointer] = next_observation
        self.terminals[self.pointer] = float(terminal)

        if self.prioritized:
            self.priorities[self.pointer] = priority or self.priorities[:self.size].max() if self.size > 0 else 1.0

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        if self.prioritized:
            # Priority sampling
            probs = self.priorities[:self.size] ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

            # Importance sampling weights
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.tensor(weights, dtype=torch.float32)
        else:
            indices = np.random.randint(0, self.size, batch_size)
            weights = None

        batch = {
            "observations": torch.tensor(self.observations[indices], dtype=torch.float32),
            "actions": torch.tensor(self.actions[indices], dtype=torch.float32),
            "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32),
            "next_observations": torch.tensor(self.next_observations[indices], dtype=torch.float32),
            "terminals": torch.tensor(self.terminals[indices], dtype=torch.float32),
        }

        if weights is not None:
            batch["weights"] = weights
            batch["indices"] = indices

        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for prioritized replay."""
        if self.prioritized:
            self.priorities[indices] = priorities + 1e-6

    def __len__(self) -> int:
        return self.size


def collate_rl(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for RL datasets."""
    output = {}

    for key in batch[0].keys():
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)
        else:
            output[key] = values

    return output


def create_rl_dataloader(
    dataset: RLDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for RL dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_rl,
        pin_memory=pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("RL Dataset Test")
    print("=" * 60)

    # Test D4RL
    print("\n1. Testing D4RL Dataset:")
    try:
        dataset = D4RLDataset(
            env_name="hopper-medium-v2",
            trajectory_mode=False,
        )
        print(f"   Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Observation shape: {sample['observations'].shape}")
        print(f"   Action shape: {sample['actions'].shape}")

        # Test dataloader
        dataloader = create_rl_dataloader(dataset, batch_size=256)
        batch = next(iter(dataloader))
        print(f"   Batch observations: {batch['observations'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test trajectory mode
    print("\n2. Testing D4RL (Trajectory Mode):")
    try:
        dataset = D4RLDataset(
            env_name="hopper-medium-v2",
            trajectory_mode=True,
            trajectory_length=20,
        )
        sample = dataset[0]
        print(f"   Trajectory observations: {sample['observations'].shape}")
        print(f"   Trajectory actions: {sample['actions'].shape}")
        print(f"   Mask shape: {sample['mask'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test replay buffer
    print("\n3. Testing Replay Buffer:")
    try:
        buffer = ReplayBuffer(observation_dim=11, action_dim=3, capacity=10000)

        # Add some transitions
        for _ in range(1000):
            buffer.add(
                observation=np.random.randn(11),
                action=np.random.randn(3),
                reward=np.random.randn(),
                next_observation=np.random.randn(11),
                terminal=np.random.rand() < 0.01,
            )

        print(f"   Buffer size: {len(buffer)}")
        batch = buffer.sample(256)
        print(f"   Sampled batch observations: {batch['observations'].shape}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("RL dataset test complete!")
    print("=" * 60)
