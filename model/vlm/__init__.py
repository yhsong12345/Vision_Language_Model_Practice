"""
VLM (Vision-Language Model) Components

This module contains vision encoders and projectors for VLM/VLA models:
- VisionEncoder: Wrapper for vision encoders (CLIP, SigLIP, DINOv2)
- VisionProjector: MLP projector (LLaVA-style)
- AttentionPoolingProjector: Cross-attention with learnable queries
- PerceiverProjector: Multi-layer Perceiver-style projector
"""

from .vision_encoder import VisionEncoder, VisionEncoderConfig
from .vision_projector import (
    VisionProjector,
    AttentionPoolingProjector,
    PerceiverProjector,
    create_projector,
)

__all__ = [
    "VisionEncoder",
    "VisionEncoderConfig",
    "VisionProjector",
    "AttentionPoolingProjector",
    "PerceiverProjector",
    "create_projector",
]
