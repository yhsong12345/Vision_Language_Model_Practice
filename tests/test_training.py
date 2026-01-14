"""
Training Tests

Tests for training components and pipelines.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBuffers:
    """Tests for experience buffers."""

    def test_base_buffer_interface(self):
        """Test base buffer interface."""
        from train.utils import BaseBuffer

        # BaseBuffer should be abstract
        with pytest.raises(TypeError):
            BaseBuffer(buffer_size=100, obs_dim=4, action_dim=2)

    def test_rollout_buffer_add_and_sample(self):
        """Test rollout buffer add and get operations."""
        from train.utils import RolloutBuffer

        buffer = RolloutBuffer(
            buffer_size=64,
            obs_dim=4,
            action_dim=2,
            device="cpu",
        )

        # Add transitions
        for _ in range(32):
            buffer.add(
                obs=torch.randn(4),
                action=torch.randn(2),
                reward=torch.tensor([1.0]),
                done=torch.tensor([0.0]),
                value=torch.randn(1),
                log_prob=torch.randn(1),
            )

        assert len(buffer) == 32

        # Get all data
        data = buffer.get()
        assert data["obs"].shape[0] == 32

    def test_replay_buffer_circular(self):
        """Test replay buffer circular behavior."""
        from train.utils import ReplayBuffer

        buffer = ReplayBuffer(
            buffer_size=10,
            obs_dim=2,
            action_dim=1,
            device="cpu",
        )

        # Add more than buffer size
        for i in range(15):
            buffer.add(
                obs=torch.tensor([float(i), float(i)]),
                action=torch.tensor([float(i)]),
                reward=torch.tensor([1.0]),
                next_obs=torch.tensor([float(i + 1), float(i + 1)]),
                done=torch.tensor([0.0]),
            )

        # Should only have buffer_size entries
        assert len(buffer) == 10

        # Sample should work
        batch = buffer.sample(5)
        assert batch["obs"].shape == (5, 2)

    def test_offline_buffer_load(self):
        """Test offline buffer loading."""
        from train.utils import OfflineBuffer

        # Create mock data
        data = {
            "observations": np.random.randn(100, 4).astype(np.float32),
            "actions": np.random.randn(100, 2).astype(np.float32),
            "rewards": np.random.randn(100, 1).astype(np.float32),
            "next_observations": np.random.randn(100, 4).astype(np.float32),
            "dones": np.zeros((100, 1), dtype=np.float32),
        }

        buffer = OfflineBuffer(
            buffer_size=100,
            obs_dim=4,
            action_dim=2,
            device="cpu",
        )

        # Load data
        buffer.load_from_dict(data)
        assert len(buffer) == 100

        # Sample
        batch = buffer.sample(32)
        assert batch["obs"].shape == (32, 4)


class TestPolicies:
    """Tests for policy networks."""

    def test_mlp_policy_forward(self):
        """Test MLP policy forward pass."""
        from train.utils import MLPPolicy

        policy = MLPPolicy(
            obs_dim=4,
            action_dim=2,
            hidden_dims=[64, 64],
        )

        obs = torch.randn(8, 4)
        action = policy(obs)

        assert action.shape == (8, 2)

    def test_gaussian_policy_sample(self):
        """Test Gaussian policy sampling."""
        from train.utils import GaussianMLPPolicy

        policy = GaussianMLPPolicy(
            obs_dim=4,
            action_dim=2,
            hidden_dims=[64, 64],
        )

        obs = torch.randn(8, 4)

        # Get mean and log_std
        mean, log_std = policy(obs)
        assert mean.shape == (8, 2)
        assert log_std.shape == (8, 2)

        # Sample action
        action, log_prob = policy.sample(obs)
        assert action.shape == (8, 2)
        assert log_prob.shape == (8,)

    def test_actor_critic(self):
        """Test actor-critic network."""
        from train.utils import ActorCritic

        ac = ActorCritic(
            obs_dim=4,
            action_dim=2,
            hidden_dims=[64, 64],
        )

        obs = torch.randn(8, 4)

        # Get action and value
        action, log_prob, value = ac.act(obs)
        assert action.shape == (8, 2)
        assert value.shape == (8,)


class TestEvaluation:
    """Tests for evaluation utilities."""

    def test_compute_metrics(self):
        """Test metric computation."""
        from train.utils import compute_metrics

        predictions = torch.randn(100, 7)
        targets = torch.randn(100, 7)

        metrics = compute_metrics(predictions, targets)

        assert "mse" in metrics
        assert "mae" in metrics
        assert metrics["mse"] >= 0


class TestLogging:
    """Tests for logging utilities."""

    def test_metrics_tracker_window(self):
        """Test metrics tracker with window."""
        from train.utils import MetricsTracker

        tracker = MetricsTracker(window_size=10)

        # Add 20 values
        for i in range(20):
            tracker.add("loss", float(i))

        # Mean should be over last 10 values
        mean = tracker.get_mean("loss")
        expected = np.mean(range(10, 20))
        assert abs(mean - expected) < 1e-6

    def test_training_logger(self):
        """Test training logger."""
        from train.utils import TrainingLogger
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(
                log_dir=tmpdir,
                use_tensorboard=False,
                use_wandb=False,
            )

            # Log some metrics
            logger.log({"loss": 0.5, "accuracy": 0.9}, step=1)
            logger.log({"loss": 0.4, "accuracy": 0.95}, step=2)

            # Should not raise


class TestTrainingLoop:
    """Tests for training loops."""

    def test_bc_training_step(self):
        """Test behavioral cloning training step."""
        # Simple model
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Mock batch
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 5)

        # Forward
        predicted = model(obs)
        loss = criterion(predicted, actions)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_ppo_loss_computation(self):
        """Test PPO loss computation."""
        batch_size = 32
        action_dim = 4

        # Mock data
        old_log_probs = torch.randn(batch_size)
        new_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        clip_range = 0.2

        # Compute ratio
        ratio = (new_log_probs - old_log_probs).exp()

        # Clipped objective
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        assert pg_loss.dim() == 0  # Scalar


class TestDatasets:
    """Tests for dataset utilities."""

    def test_mock_dataset(self):
        """Test creating a mock dataset."""
        from torch.utils.data import Dataset, DataLoader

        class MockDataset(Dataset):
            def __init__(self, size=100):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    "image": torch.randn(3, 224, 224),
                    "action": torch.randn(7),
                    "instruction": "test instruction",
                }

        dataset = MockDataset(100)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        batch = next(iter(dataloader))
        assert batch["image"].shape == (8, 3, 224, 224)
        assert batch["action"].shape == (8, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
