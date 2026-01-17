"""
Unit Tests for Utility Functions

Tests for device utilities, parameter counting, and other utilities.
"""

import pytest
import torch
import torch.nn as nn

from model.utils import (
    get_device,
    count_parameters,
    count_trainable_parameters,
    freeze_module,
    unfreeze_module,
)


class TestDeviceUtils:
    """Tests for device utilities."""

    def test_get_device_auto(self):
        """Test automatic device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)

    def test_get_device_cpu(self):
        """Test explicit CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_cuda_fallback(self):
        """Test CUDA fallback to CPU when not available."""
        device = get_device("cuda")
        # Should either be cuda or cpu depending on availability
        assert device.type in ["cuda", "cpu"]

    def test_get_device_mps(self):
        """Test MPS device handling."""
        device = get_device("mps")
        # Should either be mps or cpu depending on availability
        assert device.type in ["mps", "cpu"]


class TestParameterCounting:
    """Tests for parameter counting utilities."""

    def test_count_linear_parameters(self):
        """Test counting parameters of a linear layer."""
        model = nn.Linear(10, 5)
        total = count_parameters(model)

        # 10 * 5 weights + 5 biases = 55
        assert total == 55

    def test_count_conv_parameters(self):
        """Test counting parameters of a conv layer."""
        model = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        total = count_parameters(model)

        # 3 * 64 * 3 * 3 weights + 64 biases = 1792
        assert total == 1792

    def test_count_nested_parameters(self):
        """Test counting parameters of nested modules."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        total = count_parameters(model)

        # (10 * 20 + 20) + (20 * 5 + 5) = 220 + 105 = 325
        assert total == 325

    def test_count_trainable_all_trainable(self):
        """Test counting trainable parameters when all are trainable."""
        model = nn.Linear(10, 5)
        trainable = count_trainable_parameters(model)
        total = count_parameters(model)

        assert trainable == total

    def test_count_trainable_with_frozen(self):
        """Test counting trainable parameters with frozen layers."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        trainable = count_trainable_parameters(model)
        total = count_parameters(model)

        # Only second layer should be trainable: 20 * 5 + 5 = 105
        assert trainable == 105
        assert trainable < total


class TestModuleFreezing:
    """Tests for module freezing/unfreezing."""

    def test_freeze_module(self):
        """Test freezing a module."""
        model = nn.Linear(10, 5)

        freeze_module(model)

        for param in model.parameters():
            assert param.requires_grad is False

    def test_unfreeze_module(self):
        """Test unfreezing a module."""
        model = nn.Linear(10, 5)

        freeze_module(model)
        unfreeze_module(model)

        for param in model.parameters():
            assert param.requires_grad is True

    def test_freeze_nested_module(self):
        """Test freezing nested modules."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        freeze_module(model)

        trainable = count_trainable_parameters(model)
        assert trainable == 0

    def test_freeze_does_not_affect_bn_stats(self):
        """Test that freezing doesn't affect batch norm running stats."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Linear(20, 5),
        )

        freeze_module(model)

        # BatchNorm should still track running stats unless in eval mode
        bn = model[1]
        assert bn.track_running_stats is True


class TestTensorOperations:
    """Tests for tensor-related utilities."""

    def test_gradients_frozen_module(self):
        """Test that frozen modules don't compute gradients."""
        model = nn.Linear(10, 5)
        freeze_module(model)

        x = torch.randn(4, 10, requires_grad=True)
        y = model(x)
        loss = y.sum()

        loss.backward()

        # Input should have gradient
        assert x.grad is not None

        # Model parameters should not have gradient
        for param in model.parameters():
            assert param.grad is None
