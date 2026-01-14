"""
VLM (Vision-Language Model) Components

This module contains vision encoders, projectors, and VLM model for VLM/VLA:
- VLMModel: Complete Vision-Language Model for pretraining
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
from .vlm_model import VLMModel, create_vlm_model

__all__ = [
    "VLMModel",
    "create_vlm_model",
    "VisionEncoder",
    "VisionEncoderConfig",
    "VisionProjector",
    "AttentionPoolingProjector",
    "PerceiverProjector",
    "create_projector",
]
