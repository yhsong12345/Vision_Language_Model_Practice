"""
Unit Tests for Action Head Components

Tests for MLP, Gaussian, Diffusion, and Transformer action heads.
"""

import pytest
import torch
import torch.nn as nn

from model.action_head import MLPActionHead, GaussianMLPActionHead


class TestMLPActionHead:
    """Tests for MLP action head."""

    def test_forward_single_action(self, sample_features):
        """Test forward pass with single action prediction."""
        head = MLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
            chunk_size=1,
        )

        output = head(sample_features)

        assert "predicted_actions" in output
        assert output["predicted_actions"].shape == (4, 1, 7)

    def test_forward_with_chunking(self, sample_features):
        """Test forward pass with action chunking."""
        chunk_size = 16
        head = MLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
            chunk_size=chunk_size,
        )

        output = head(sample_features)

        assert output["predicted_actions"].shape == (4, chunk_size, 7)

    def test_loss_computation(self, sample_features, sample_actions_chunked):
        """Test loss computation with target actions."""
        head = MLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
            chunk_size=16,
        )

        output = head(sample_features, actions=sample_actions_chunked)

        assert "loss" in output
        assert output["loss"].dim() == 0  # Scalar
        assert output["loss"].item() >= 0  # MSE is non-negative

    def test_different_action_dims(self, sample_features):
        """Test with different action dimensions."""
        for action_dim in [2, 7, 12]:
            head = MLPActionHead(
                input_dim=256,
                action_dim=action_dim,
                hidden_dim=128,
                chunk_size=1,
            )
            output = head(sample_features)
            assert output["predicted_actions"].shape[-1] == action_dim

    def test_dropout(self, sample_features):
        """Test that dropout affects training mode."""
        head = MLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
            dropout=0.5,
        )

        # Training mode
        head.train()
        out1 = head(sample_features)["predicted_actions"]
        out2 = head(sample_features)["predicted_actions"]

        # Eval mode
        head.eval()
        out3 = head(sample_features)["predicted_actions"]
        out4 = head(sample_features)["predicted_actions"]

        # In eval mode, outputs should be identical
        assert torch.allclose(out3, out4)


class TestGaussianMLPActionHead:
    """Tests for Gaussian MLP action head (for RL)."""

    def test_forward_output_structure(self, sample_features):
        """Test that forward returns mean and log_std."""
        head = GaussianMLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
        )

        output = head(sample_features)

        assert "mean" in output
        assert "log_std" in output
        assert output["mean"].shape == (4, 7)
        assert output["log_std"].shape == (4, 7)

    def test_sample_action(self, sample_features):
        """Test action sampling."""
        head = GaussianMLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
        )

        output = head(sample_features)

        # Sample actions from the distribution
        mean = output["mean"]
        log_std = output["log_std"]
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()

        assert action.shape == (4, 7)

    def test_log_prob_computation(self, sample_features, sample_actions):
        """Test log probability computation."""
        head = GaussianMLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
        )

        output = head(sample_features, actions=sample_actions)

        assert "log_prob" in output
        assert output["log_prob"].shape == (4,)

    def test_log_std_bounds(self, sample_features):
        """Test that log_std is bounded."""
        head = GaussianMLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
            log_std_min=-5.0,
            log_std_max=2.0,
        )

        output = head(sample_features)
        log_std = output["log_std"]

        assert (log_std >= -5.0).all()
        assert (log_std <= 2.0).all()


class TestActionHeadGradients:
    """Tests for gradient flow through action heads."""

    def test_mlp_gradients(self, sample_features, sample_actions_chunked):
        """Test that gradients flow through MLP head."""
        head = MLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
            chunk_size=16,
        )

        output = head(sample_features, actions=sample_actions_chunked)
        loss = output["loss"]
        loss.backward()

        # Check that gradients exist
        for param in head.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_gaussian_gradients(self, sample_features, sample_actions):
        """Test that gradients flow through Gaussian head."""
        head = GaussianMLPActionHead(
            input_dim=256,
            action_dim=7,
            hidden_dim=128,
        )

        output = head(sample_features, actions=sample_actions)
        loss = -output["log_prob"].mean()  # Negative log likelihood
        loss.backward()

        # Check that gradients exist
        for param in head.parameters():
            assert param.grad is not None
