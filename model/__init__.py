"""
Model Package

This package contains all model components for VLA (Vision-Language-Action) models:

Submodules:
- vlm/: Vision-Language Model components (vision encoders, projectors)
- vla/: Complete VLA model implementations
- action_head/: Action prediction heads (MLP, Diffusion, Transformer)
- sensor/: Sensor encoders (LiDAR, Radar, IMU)
- fusion/: Multi-modal sensor fusion modules
"""

# VLM components
from .vlm import (
    VisionEncoder,
    VisionEncoderConfig,
    VisionProjector,
    AttentionPoolingProjector,
    PerceiverProjector,
    create_projector,
)

# VLA models
from .vla import (
    VLAModel,
    create_vla_model,
    MultiSensorVLA,
    OpenVLAWrapper,
    SmolVLAWrapper,
)

# Action heads
from .action_head import (
    MLPActionHead,
    GaussianMLPActionHead,
    DiffusionActionHead,
    TransformerActionHead,
    GPTActionHead,
)

# Sensor encoders
from .sensor import (
    PointCloudEncoder,
    PointNetEncoder,
    PointTransformerEncoder,
    RadarEncoder,
    RangeDopplerEncoder,
    IMUEncoder,
    TemporalIMUEncoder,
)

# Fusion modules
from .fusion import (
    SensorFusion,
    CrossModalFusion,
    HierarchicalFusion,
    GatedFusion,
)

__all__ = [
    # VLM
    "VisionEncoder",
    "VisionEncoderConfig",
    "VisionProjector",
    "AttentionPoolingProjector",
    "PerceiverProjector",
    "create_projector",
    # VLA
    "VLAModel",
    "create_vla_model",
    "MultiSensorVLA",
    "OpenVLAWrapper",
    "SmolVLAWrapper",
    # Action heads
    "MLPActionHead",
    "GaussianMLPActionHead",
    "DiffusionActionHead",
    "TransformerActionHead",
    "GPTActionHead",
    # Sensors
    "PointCloudEncoder",
    "PointNetEncoder",
    "PointTransformerEncoder",
    "RadarEncoder",
    "RangeDopplerEncoder",
    "IMUEncoder",
    "TemporalIMUEncoder",
    # Fusion
    "SensorFusion",
    "CrossModalFusion",
    "HierarchicalFusion",
    "GatedFusion",
]
