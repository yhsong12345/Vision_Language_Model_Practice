"""
Model Tests

Tests for VLA model components.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelUtils:
    """Tests for model utilities."""

    def test_get_device_auto(self):
        """Test automatic device detection."""
        from model.utils import get_device

        device = get_device("auto")
        assert isinstance(device, torch.device)

    def test_get_device_cpu(self):
        """Test explicit CPU device."""
        from model.utils import get_device

        device = get_device("cpu")
        assert device.type == "cpu"

    def test_count_parameters(self):
        """Test parameter counting."""
        from model.utils import count_parameters, count_trainable_parameters

        model = nn.Linear(10, 5)
        total = count_parameters(model)
        trainable = count_trainable_parameters(model)

        assert total == 55  # 10*5 + 5
        assert trainable == 55

    def test_freeze_module(self):
        """Test module freezing."""
        from model.utils import freeze_module, count_trainable_parameters

        model = nn.Linear(10, 5)
        freeze_module(model)

        assert count_trainable_parameters(model) == 0

    def test_unfreeze_module(self):
        """Test module unfreezing."""
        from model.utils import freeze_module, unfreeze_module, count_trainable_parameters

        model = nn.Linear(10, 5)
        freeze_module(model)
        unfreeze_module(model)

        assert count_trainable_parameters(model) == 55


class TestActionHeads:
    """Tests for action head components."""

    @pytest.fixture
    def batch_features(self):
        """Create sample features."""
        return torch.randn(4, 256)

    def test_mlp_action_head_forward(self, batch_features):
        """Test MLP action head forward pass."""
        from model.action_head import MLPActionHead

        head = MLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
            chunk_size=1,
        )

        output = head(batch_features)

        assert output["predicted_actions"].shape == (4, 1, 7)

    def test_mlp_action_head_with_chunking(self, batch_features):
        """Test MLP action head with action chunking."""
        from model.action_head import MLPActionHead

        head = MLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
            chunk_size=16,
        )

        output = head(batch_features)

        assert output["predicted_actions"].shape == (4, 16, 7)

    def test_gaussian_mlp_action_head(self, batch_features):
        """Test Gaussian MLP action head."""
        from model.action_head import GaussianMLPActionHead

        head = GaussianMLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
        )

        output = head(batch_features)

        assert "mean" in output
        assert "log_std" in output
        assert output["mean"].shape == (4, 7)

    def test_action_head_loss(self, batch_features):
        """Test action head loss computation."""
        from model.action_head import MLPActionHead

        head = MLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
        )

        target_actions = torch.randn(4, 1, 7)
        output = head(batch_features, actions=target_actions)

        assert "loss" in output
        assert output["loss"].dim() == 0  # Scalar


class TestVisionComponents:
    """Tests for vision components."""

    @pytest.fixture
    def sample_images(self):
        """Create sample images."""
        return torch.randn(2, 3, 224, 224)

    def test_vision_projector_mlp(self, sample_images):
        """Test MLP vision projector."""
        from model.vlm import VisionProjector

        projector = VisionProjector(
            input_dim=768,
            output_dim=1536,
            projector_type="mlp",
        )

        # Mock vision features
        features = torch.randn(2, 196, 768)
        output = projector(features)

        assert output.shape == (2, 196, 1536)


class TestTrainingUtils:
    """Tests for training utilities."""

    def test_metrics_tracker(self):
        """Test metrics tracking."""
        from train.utils import MetricsTracker

        tracker = MetricsTracker()

        # Add values
        tracker.add("loss", 0.5, step=1)
        tracker.add("loss", 0.4, step=2)
        tracker.add("loss", 0.3, step=3)

        # Check mean
        mean = tracker.get_mean("loss")
        assert abs(mean - 0.4) < 1e-6

    def test_rollout_buffer(self):
        """Test rollout buffer."""
        from train.utils import RolloutBuffer

        buffer = RolloutBuffer(
            buffer_size=100,
            obs_dim=4,
            action_dim=2,
            device="cpu",
        )

        # Add samples
        for _ in range(50):
            buffer.add(
                obs=torch.randn(4),
                action=torch.randn(2),
                reward=torch.randn(1),
                done=torch.zeros(1),
                value=torch.randn(1),
                log_prob=torch.randn(1),
            )

        assert len(buffer) == 50

    def test_replay_buffer(self):
        """Test replay buffer."""
        from train.utils import ReplayBuffer

        buffer = ReplayBuffer(
            buffer_size=1000,
            obs_dim=4,
            action_dim=2,
            device="cpu",
        )

        # Add samples
        for _ in range(100):
            buffer.add(
                obs=torch.randn(4),
                action=torch.randn(2),
                reward=torch.randn(1),
                next_obs=torch.randn(4),
                done=torch.zeros(1),
            )

        # Sample batch
        batch = buffer.sample(32)

        assert batch["obs"].shape == (32, 4)
        assert batch["action"].shape == (32, 2)


class TestIntegration:
    """Integration tests."""

    def test_simple_forward_pass(self):
        """Test simple VLA-style forward pass."""
        # Simple encoder-decoder model
        class SimpleVLA(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                )
                self.head = nn.Linear(64, 7)

            def forward(self, x):
                return self.head(self.encoder(x))

        model = SimpleVLA()
        image = torch.randn(1, 3, 224, 224)

        action = model(image)
        assert action.shape == (1, 7)

    def test_training_step(self):
        """Test a simple training step."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Forward
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        output = model(x)
        loss = criterion(output, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


# Smoke tests for optional components
class TestOptionalComponents:
    """Tests that may skip if dependencies are missing."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_inference(self):
        """Test CUDA inference."""
        model = nn.Linear(10, 5).cuda()
        x = torch.randn(4, 10).cuda()

        output = model(x)
        assert output.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
