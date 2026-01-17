"""
Integration Tests for Training Loop

Tests end-to-end training workflows with mock data.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TestSimpleTrainingLoop:
    """Tests for basic training loop functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        num_samples = 100
        inputs = torch.randn(num_samples, 10)
        targets = torch.randn(num_samples, 7)
        return TensorDataset(inputs, targets)

    def test_single_epoch_training(self, simple_model, mock_dataset):
        """Test training for a single epoch."""
        dataloader = DataLoader(mock_dataset, batch_size=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        simple_model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = simple_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        assert avg_loss > 0
        assert not torch.isnan(torch.tensor(avg_loss))

    def test_loss_decreases(self, simple_model, mock_dataset):
        """Test that loss decreases over training."""
        dataloader = DataLoader(mock_dataset, batch_size=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        initial_loss = None
        final_loss = None

        for epoch in range(10):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = simple_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            if epoch == 0:
                initial_loss = avg_loss
            if epoch == 9:
                final_loss = avg_loss

        assert final_loss < initial_loss

    def test_gradient_clipping(self, simple_model, mock_dataset):
        """Test training with gradient clipping."""
        dataloader = DataLoader(mock_dataset, batch_size=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        max_grad_norm = 1.0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = simple_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Clip gradients
            total_norm = torch.nn.utils.clip_grad_norm_(
                simple_model.parameters(), max_grad_norm
            )

            optimizer.step()

            # Check that gradient norm is clipped
            assert total_norm <= max_grad_norm * 1.01  # Small tolerance

    def test_mixed_precision_training(self, simple_model, mock_dataset, device):
        """Test training with mixed precision (if CUDA available)."""
        if device.type != "cuda":
            pytest.skip("Mixed precision requires CUDA")

        simple_model = simple_model.to(device)
        dataloader = DataLoader(mock_dataset, batch_size=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = simple_model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            assert not torch.isnan(loss)


class TestCheckpointing:
    """Tests for model checkpointing."""

    @pytest.fixture
    def trained_model(self, simple_model, mock_dataset):
        """Train a model for a few steps."""
        dataloader = DataLoader(mock_dataset, batch_size=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = simple_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        return simple_model

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        num_samples = 100
        inputs = torch.randn(num_samples, 10)
        targets = torch.randn(num_samples, 7)
        return TensorDataset(inputs, targets)

    def test_save_and_load_state_dict(self, trained_model, tmp_path):
        """Test saving and loading model state dict."""
        checkpoint_path = tmp_path / "model.pt"

        # Save
        torch.save(trained_model.state_dict(), checkpoint_path)

        # Create new model and load
        new_model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Check weights match
        for p1, p2 in zip(trained_model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_save_full_checkpoint(self, trained_model, tmp_path):
        """Test saving full training checkpoint."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        optimizer = torch.optim.Adam(trained_model.parameters())

        checkpoint = {
            "epoch": 5,
            "model_state_dict": trained_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": 0.5,
        }

        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded = torch.load(checkpoint_path, weights_only=False)

        assert loaded["epoch"] == 5
        assert loaded["loss"] == 0.5
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded


class TestEvaluation:
    """Tests for model evaluation."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        num_samples = 100
        inputs = torch.randn(num_samples, 10)
        targets = torch.randn(num_samples, 7)
        return TensorDataset(inputs, targets)

    def test_eval_mode_no_gradients(self, simple_model, mock_dataset):
        """Test that eval mode doesn't compute gradients."""
        dataloader = DataLoader(mock_dataset, batch_size=16)

        simple_model.eval()

        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = simple_model(inputs)

                # Check no gradients are tracked
                assert not outputs.requires_grad

    def test_eval_deterministic(self, simple_model, mock_dataset):
        """Test that eval mode produces deterministic outputs."""
        dataloader = DataLoader(mock_dataset, batch_size=16, shuffle=False)

        simple_model.eval()

        outputs1 = []
        outputs2 = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                outputs1.append(simple_model(inputs))

            for inputs, _ in dataloader:
                outputs2.append(simple_model(inputs))

        for o1, o2 in zip(outputs1, outputs2):
            assert torch.allclose(o1, o2)
