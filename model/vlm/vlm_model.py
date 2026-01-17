"""
Vision-Language Model (VLM) for Pretraining

A modular VLM architecture for pretraining following the LLaVA paradigm:
- Vision Encoder: Extracts visual features from images
- Vision Projector: Maps vision features to LLM embedding space
- Language Model: Processes fused vision + language for next-token prediction

This is used for VLM pretraining (Stage 1 alignment + Stage 2 instruction tuning)
before adding action heads for VLA training.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Optional, Dict, Any

from .vision_encoder import VisionEncoder, VisionEncoderConfig
from .vision_projector import (
    VisionProjector,
    AttentionPoolingProjector,
    PerceiverProjector,
    create_projector,
)

from model.utils import freeze_module, count_parameters, count_trainable_parameters


class VLMModel(nn.Module):
    """
    Vision-Language Model for Pretraining.

    Architecture:
        1. Vision Encoder (SigLIP/CLIP/DINOv2) -> Extract image features
        2. Vision Projector -> Map to LLM embedding space
        3. LLM (Qwen2/LLaMA/Phi) -> Process fused vision + language

    This model is designed for VLM pretraining with language modeling loss.
    For robot action prediction, use VLAModel which adds an action head.

    Supports:
        - Multiple vision encoders
        - Multiple LLM backends
        - Multiple projector types (MLP, Attention, Perceiver)
        - Mixed precision training
    """

    # Supported vision encoders
    VISION_ENCODERS = {
        "siglip-base": "google/siglip-base-patch16-224",
        "siglip-large": "google/siglip-large-patch16-384",
        "clip-base": "openai/clip-vit-base-patch32",
        "clip-large": "openai/clip-vit-large-patch14",
        "dinov2-base": "facebook/dinov2-base",
        "dinov2-large": "facebook/dinov2-large",
    }

    # Supported LLMs
    LLMS = {
        "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
        "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "phi3-small": "microsoft/Phi-3-small-8k-instruct",
    }

    def __init__(
        self,
        vision_model_name: str = "google/siglip-base-patch16-224",
        llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        projector_type: str = "mlp",
        num_vision_tokens: int = 64,
        dropout: float = 0.0,
        freeze_vision: bool = False,
        freeze_llm: bool = False,
        use_flash_attention: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        """
        Initialize VLM model.

        Args:
            vision_model_name: Vision encoder name (HuggingFace model or key)
            llm_model_name: LLM name (HuggingFace model or key)
            projector_type: Type of projector ("mlp", "attention", "perceiver")
            num_vision_tokens: Number of vision tokens (for attention/perceiver projectors)
            dropout: Dropout probability
            freeze_vision: Whether to freeze vision encoder
            freeze_llm: Whether to freeze LLM
            use_flash_attention: Whether to use Flash Attention 2
            use_gradient_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()

        self.projector_type = projector_type
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Load vision encoder
        print(f"Loading vision encoder: {vision_model_name}")
        vision_config = VisionEncoderConfig(
            model_name=vision_model_name,
            freeze=freeze_vision,
        )
        self.vision_encoder = VisionEncoder(vision_config)
        self.image_processor = self.vision_encoder.image_processor
        vision_dim = self.vision_encoder.get_output_dim()

        # Load LLM with causal LM head for language modeling
        print(f"Loading LLM: {llm_model_name}")
        attn_implementation = "flash_attention_2" if use_flash_attention else "sdpa"
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for LLM")

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_dim = self.llm.config.hidden_size

        # Vision projector
        if projector_type == "mlp":
            self.vision_projector = VisionProjector(
                vision_dim=vision_dim,
                llm_dim=llm_dim,
                dropout=dropout,
            )
        elif projector_type == "attention":
            self.vision_projector = AttentionPoolingProjector(
                vision_dim=vision_dim,
                llm_dim=llm_dim,
                num_tokens=num_vision_tokens,
                dropout=dropout,
            )
        elif projector_type == "perceiver":
            self.vision_projector = PerceiverProjector(
                vision_dim=vision_dim,
                llm_dim=llm_dim,
                num_tokens=num_vision_tokens,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        # Freeze LLM if specified
        if freeze_llm:
            freeze_module(self.llm, verbose=True)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images using the vision encoder and projector.

        Args:
            pixel_values: (batch, channels, height, width)

        Returns:
            vision_embeds: (batch, num_tokens, llm_dim)
        """
        vision_features = self.vision_encoder.encode_image(pixel_values)
        projected = self.vision_projector(vision_features)
        return projected

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VLM pretraining.

        Args:
            pixel_values: (batch, channels, height, width)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) - target token IDs for language modeling loss

        Returns:
            Dict with:
                - loss: Language modeling loss (if labels provided)
                - logits: (batch, seq_len, vocab_size)
        """
        batch_size = pixel_values.shape[0]

        # Encode image
        vision_embeds = self.encode_image(pixel_values)
        num_vision_tokens = vision_embeds.shape[1]

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate: [vision] [text]
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # Create combined attention mask
        vision_mask = torch.ones(
            batch_size, num_vision_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Prepare labels if provided (shift for causal LM)
        combined_labels = None
        if labels is not None:
            # Create labels: -100 for vision tokens (ignored in loss), then actual labels
            vision_labels = torch.full(
                (batch_size, num_vision_tokens),
                -100,  # Ignore index for CrossEntropyLoss
                device=labels.device,
                dtype=labels.dtype,
            )
            combined_labels = torch.cat([vision_labels, labels], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            return_dict=True,
        )

        result = {
            "logits": outputs.logits,
        }

        if outputs.loss is not None:
            result["loss"] = outputs.loss

        return result

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate text given image and prompt.

        Args:
            pixel_values: (batch, channels, height, width)
            input_ids: (batch, seq_len) - prompt tokens
            attention_mask: (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            **generate_kwargs: Additional arguments for generate()

        Returns:
            generated_ids: (batch, generated_seq_len)
        """
        batch_size = pixel_values.shape[0]

        # Encode image
        vision_embeds = self.encode_image(pixel_values)
        num_vision_tokens = vision_embeds.shape[1]

        # Get text embeddings for prompt
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate: [vision] [text]
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # Create combined attention mask
        vision_mask = torch.ones(
            batch_size, num_vision_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )

        return outputs

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load a pretrained VLM model."""
        model = cls(**kwargs)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, path: str):
        """Save the VLM model."""
        torch.save(self.state_dict(), path)

    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            "vision_encoder": count_parameters(self.vision_encoder),
            "vision_encoder_trainable": count_trainable_parameters(self.vision_encoder),
            "llm": count_parameters(self.llm),
            "llm_trainable": count_trainable_parameters(self.llm),
            "vision_projector": count_parameters(self.vision_projector),
            "vision_projector_trainable": count_trainable_parameters(self.vision_projector),
            "total": count_parameters(self),
            "total_trainable": count_trainable_parameters(self),
        }


def create_vlm_model(
    vision_encoder: str = "siglip-base",
    llm: str = "qwen2-1.5b",
    projector_type: str = "mlp",
    **kwargs,
) -> VLMModel:
    """
    Factory function to create a VLM model.

    Args:
        vision_encoder: Key from VLMModel.VISION_ENCODERS or full HF name
        llm: Key from VLMModel.LLMS or full HF name
        projector_type: Type of projector ("mlp", "attention", "perceiver")
        **kwargs: Additional arguments for VLMModel

    Returns:
        VLMModel instance
    """
    vision_model = VLMModel.VISION_ENCODERS.get(vision_encoder, vision_encoder)
    llm_model = VLMModel.LLMS.get(llm, llm)

    return VLMModel(
        vision_model_name=vision_model,
        llm_model_name=llm_model,
        projector_type=projector_type,
        **kwargs,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("VLM Model Test")
    print("=" * 60)

    # Create model
    model = create_vlm_model(
        vision_encoder="siglip-base",
        llm="qwen2-1.5b",
        projector_type="mlp",
        freeze_vision=True,
        freeze_llm=False,
    )

    # Print parameter counts
    params = model.get_param_count()
    print("\nParameter Counts:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_text = model.tokenizer(
        ["Describe this image.", "What do you see?"],
        return_tensors="pt",
        padding=True,
    )
    dummy_input_ids = dummy_text.input_ids.to(device)
    dummy_attention_mask = dummy_text.attention_mask.to(device)
    # Labels are same as input_ids for language modeling
    dummy_labels = dummy_input_ids.clone()

    with torch.amp.autocast(device_type=str(device), enabled=torch.cuda.is_available()):
        outputs = model(
            pixel_values=dummy_image,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            labels=dummy_labels,
        )

    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    print("\n" + "=" * 60)
    print("VLM Model test passed!")
    print("=" * 60)
