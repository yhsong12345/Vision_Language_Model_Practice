"""
Core - VLA Framework Core Utilities

A comprehensive framework for training Vision-Language-Action models
for robotics and autonomous systems.

This module provides core utilities including:
- Custom exceptions for error handling
- Component registry for dynamic instantiation
- Logging configuration
- Device utilities
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from core.exceptions import (
    VLAError,
    ConfigurationError,
    CheckpointError,
    ModelError,
    TrainingError,
    DatasetError,
    ExportError,
    InferenceError,
)

from core.registry import (
    Registry,
    MODEL_REGISTRY,
    VISION_ENCODER_REGISTRY,
    ACTION_HEAD_REGISTRY,
    TRAINER_REGISTRY,
    DATASET_REGISTRY,
    register_model,
    register_vision_encoder,
    register_action_head,
    register_trainer,
    register_dataset,
    list_available_models,
    list_available_vision_encoders,
    list_available_action_heads,
    create_model,
    create_vision_encoder,
    create_action_head,
)

from core.device_utils import (
    get_device,
    move_to_device,
    get_device_info,
    print_device_info,
)

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "VLAError",
    "ConfigurationError",
    "CheckpointError",
    "ModelError",
    "TrainingError",
    "DatasetError",
    "ExportError",
    "InferenceError",
    # Registry
    "Registry",
    "MODEL_REGISTRY",
    "VISION_ENCODER_REGISTRY",
    "ACTION_HEAD_REGISTRY",
    "TRAINER_REGISTRY",
    "DATASET_REGISTRY",
    "register_model",
    "register_vision_encoder",
    "register_action_head",
    "register_trainer",
    "register_dataset",
    "list_available_models",
    "list_available_vision_encoders",
    "list_available_action_heads",
    "create_model",
    "create_vision_encoder",
    "create_action_head",
    # Device utils
    "get_device",
    "move_to_device",
    "get_device_info",
    "print_device_info",
]
