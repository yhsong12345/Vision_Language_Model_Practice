"""
Training Utilities

Shared utilities for VLA training:
- buffers: Replay and rollout buffers for RL
- policies: Common policy network architectures
- evaluation: Standardized evaluation functions
- logging: Training metrics and checkpointing
- device_utils: Device management utilities
- sim2real: Sim-to-real transfer utilities
"""

from .buffers import (
    BaseBuffer,
    RolloutBuffer,
    ReplayBuffer,
    OfflineBuffer,
)

from .policies import (
    MLPPolicy,
    GaussianMLPPolicy,
    ActorCritic,
)

from .evaluation import (
    evaluate_policy,
    evaluate_in_env,
    compute_metrics,
)

from .logging import (
    TrainingLogger,
    MetricsTracker,
    ExperimentLogger,
    ExperimentConfig,
    BestModelTracker,
    create_experiment_logger,
)

from .device_utils import (
    get_device,
    move_to_device,
)

from .sim2real import (
    DomainRandomization,
    DomainRandomizationConfig,
    SensorNoiseAugmentation,
    SensorNoiseConfig,
    ActionSpaceMapper,
    CameraCalibration,
    create_sim2real_augmentation,
)

__all__ = [
    # Buffers
    "BaseBuffer",
    "RolloutBuffer",
    "ReplayBuffer",
    "OfflineBuffer",
    # Policies
    "MLPPolicy",
    "GaussianMLPPolicy",
    "ActorCritic",
    # Evaluation
    "evaluate_policy",
    "evaluate_in_env",
    "compute_metrics",
    # Logging
    "TrainingLogger",
    "MetricsTracker",
    "ExperimentLogger",
    "ExperimentConfig",
    "BestModelTracker",
    "create_experiment_logger",
    # Device
    "get_device",
    "move_to_device",
    # Sim2Real
    "DomainRandomization",
    "DomainRandomizationConfig",
    "SensorNoiseAugmentation",
    "SensorNoiseConfig",
    "ActionSpaceMapper",
    "CameraCalibration",
    "create_sim2real_augmentation",
]
