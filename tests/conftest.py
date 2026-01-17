"""
Pytest Configuration and Shared Fixtures

This module provides shared fixtures and configuration for all tests.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from dataclasses import dataclass


# ============================================
# Device Fixtures
# ============================================

@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for consistent testing."""
    return torch.device("cpu")


# ============================================
# Tensor Fixtures
# ============================================

@pytest.fixture
def batch_size() -> int:
    """Standard batch size for testing."""
    return 4


@pytest.fixture
def sample_images(batch_size: int) -> torch.Tensor:
    """Create sample image batch [B, C, H, W]."""
    return torch.randn(batch_size, 3, 224, 224)


@pytest.fixture
def sample_features(batch_size: int) -> torch.Tensor:
    """Create sample feature batch [B, D]."""
    return torch.randn(batch_size, 256)


@pytest.fixture
def sample_sequence_features(batch_size: int) -> torch.Tensor:
    """Create sample sequence features [B, T, D]."""
    return torch.randn(batch_size, 64, 256)


@pytest.fixture
def sample_actions(batch_size: int) -> torch.Tensor:
    """Create sample action batch [B, action_dim]."""
    return torch.randn(batch_size, 7)


@pytest.fixture
def sample_actions_chunked(batch_size: int) -> torch.Tensor:
    """Create sample chunked action batch [B, chunk_size, action_dim]."""
    return torch.randn(batch_size, 16, 7)


# ============================================
# Model Fixtures
# ============================================

@pytest.fixture
def simple_mlp() -> nn.Module:
    """Create a simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 7),
    )


@pytest.fixture
def simple_conv() -> nn.Module:
    """Create a simple CNN for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 7),
    )


# ============================================
# Data Fixtures
# ============================================

@pytest.fixture
def expert_demonstrations() -> Tuple[np.ndarray, np.ndarray]:
    """Create sample expert demonstration data."""
    num_samples = 100
    state_dim = 10
    action_dim = 7

    states = np.random.randn(num_samples, state_dim).astype(np.float32)
    actions = np.random.randn(num_samples, action_dim).astype(np.float32)

    return states, actions


@pytest.fixture
def trajectory_data() -> dict:
    """Create sample trajectory data."""
    num_steps = 50

    return {
        "observations": np.random.randn(num_steps, 3, 224, 224).astype(np.float32),
        "actions": np.random.randn(num_steps, 7).astype(np.float32),
        "rewards": np.random.randn(num_steps).astype(np.float32),
        "dones": np.zeros(num_steps, dtype=bool),
    }


# ============================================
# Configuration Fixtures
# ============================================

@pytest.fixture
def vla_config_dict() -> dict:
    """Return a valid VLA configuration dictionary."""
    return {
        "vision_model_name": "google/siglip-base-patch16-224",
        "llm_model_name": "Qwen/Qwen2-1.5B-Instruct",
        "action_dim": 7,
        "hidden_dim": 256,
        "num_vision_tokens": 32,
        "action_chunk_size": 1,
        "dropout": 0.1,
        "freeze_vision": True,
        "freeze_llm": True,
    }


# ============================================
# Markers
# ============================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "cuda: marks tests that require CUDA")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "requires_model: marks tests that need pretrained models")


# ============================================
# Skip Conditions
# ============================================

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Multiple GPUs not available"
)
