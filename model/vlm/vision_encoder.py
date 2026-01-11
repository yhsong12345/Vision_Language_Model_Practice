"""
Vision Encoders for VLM/VLA Models

Supports multiple vision encoder architectures from HuggingFace:
- SigLIP (Google)
- CLIP (OpenAI)
- DINOv2 (Meta)
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoImageProcessor,
    SiglipModel,
    SiglipImageProcessor,
)
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class VisionEncoderConfig:
    """Configuration for vision encoder."""
    model_name: str = "google/siglip-base-patch16-224"
    freeze: bool = False
    output_hidden_states: bool = False

    # Supported vision encoders
    SUPPORTED_ENCODERS = {
        "siglip-base": "google/siglip-base-patch16-224",
        "siglip-large": "google/siglip-large-patch16-384",
        "clip-base": "openai/clip-vit-base-patch32",
        "clip-large": "openai/clip-vit-large-patch14",
        "dinov2-base": "facebook/dinov2-base",
        "dinov2-large": "facebook/dinov2-large",
    }

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "VisionEncoderConfig":
        """Create config from encoder name or full HuggingFace model name."""
        model_name = cls.SUPPORTED_ENCODERS.get(name, name)
        return cls(model_name=model_name, **kwargs)


class VisionEncoder(nn.Module):
    """
    Vision Encoder wrapper for multiple architectures.

    Provides a unified interface for different vision encoders:
    - SigLIP: Best for vision-language alignment
    - CLIP: Strong zero-shot capabilities
    - DINOv2: Self-supervised features

    Args:
        config: VisionEncoderConfig or model name string
    """

    def __init__(self, config: VisionEncoderConfig = None, model_name: str = None):
        super().__init__()

        if config is None:
            config = VisionEncoderConfig(model_name=model_name or "google/siglip-base-patch16-224")

        self.config = config
        self.model_name = config.model_name

        # Detect encoder type
        self.encoder_type = self._detect_encoder_type(self.model_name)

        # Load encoder and processor
        print(f"Loading vision encoder: {self.model_name}")
        self._load_encoder()

        # Freeze if specified
        if config.freeze:
            self.freeze()
            print("Vision encoder frozen")

    def _detect_encoder_type(self, model_name: str) -> str:
        """Detect encoder type from model name."""
        model_name_lower = model_name.lower()
        if "siglip" in model_name_lower:
            return "siglip"
        elif "clip" in model_name_lower:
            return "clip"
        elif "dino" in model_name_lower:
            return "dinov2"
        else:
            return "auto"

    def _load_encoder(self):
        """Load the vision encoder and image processor."""
        if self.encoder_type == "siglip":
            self.encoder = SiglipModel.from_pretrained(self.model_name)
            self.image_processor = SiglipImageProcessor.from_pretrained(self.model_name)
            self.hidden_size = self.encoder.config.vision_config.hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(self.model_name)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.hidden_size = self.encoder.config.hidden_size

    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode images.

        Args:
            pixel_values: (batch, channels, height, width)
            return_dict: Whether to return dict or tuple

        Returns:
            Dict with:
                - last_hidden_state: (batch, num_patches, hidden_size)
                - pooled_output: (batch, hidden_size) if available
        """
        if self.encoder_type == "siglip":
            outputs = self.encoder.vision_model(
                pixel_values,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            outputs = self.encoder(
                pixel_values,
                output_hidden_states=self.config.output_hidden_states,
            )

        result = {
            "last_hidden_state": outputs.last_hidden_state,
        }

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            result["pooled_output"] = outputs.pooler_output

        if self.config.output_hidden_states and hasattr(outputs, "hidden_states"):
            result["hidden_states"] = outputs.hidden_states

        return result

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Simple encoding interface returning just the features.

        Args:
            pixel_values: (batch, channels, height, width)

        Returns:
            features: (batch, num_patches, hidden_size)
        """
        outputs = self.forward(pixel_values)
        return outputs["last_hidden_state"]

    def preprocess(self, images) -> torch.Tensor:
        """
        Preprocess images using the image processor.

        Args:
            images: PIL Image(s) or numpy array(s)

        Returns:
            pixel_values: Preprocessed tensor
        """
        processed = self.image_processor(images=images, return_tensors="pt")
        return processed.pixel_values

    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self.hidden_size

    def get_num_patches(self, image_size: int = 224) -> int:
        """Estimate number of output patches for given image size."""
        if self.encoder_type == "siglip":
            patch_size = self.encoder.config.vision_config.patch_size
        else:
            patch_size = getattr(self.encoder.config, "patch_size", 16)

        num_patches = (image_size // patch_size) ** 2
        # Add 1 for CLS token if present
        if self.encoder_type in ["clip", "dinov2"]:
            num_patches += 1
        return num_patches


if __name__ == "__main__":
    print("=" * 60)
    print("Vision Encoder Test")
    print("=" * 60)

    # Test SigLIP encoder
    config = VisionEncoderConfig.from_name("siglip-base")
    encoder = VisionEncoder(config)

    print(f"\nEncoder type: {encoder.encoder_type}")
    print(f"Hidden size: {encoder.hidden_size}")
    print(f"Num patches (224x224): {encoder.get_num_patches(224)}")

    # Test forward pass
    dummy_image = torch.randn(2, 3, 224, 224)
    outputs = encoder(dummy_image)

    print(f"\nOutput shape: {outputs['last_hidden_state'].shape}")
    print("\nVision Encoder test passed!")
