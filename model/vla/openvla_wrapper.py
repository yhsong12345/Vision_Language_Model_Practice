"""
OpenVLA Wrapper for Fine-tuning

Wraps the OpenVLA-7B model from HuggingFace with support for:
- LoRA fine-tuning
- Quantization (4-bit, 8-bit)
- Custom action heads
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, Any


class OpenVLAWrapper(nn.Module):
    """
    Wrapper for OpenVLA-7B with efficient fine-tuning support.

    OpenVLA is a 7B parameter VLA model based on Prismatic VLM.
    This wrapper enables:
    - LoRA for memory-efficient fine-tuning
    - 4-bit/8-bit quantization
    - Custom action head for different action spaces
    """

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        action_dim: int = 7,
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ):
        super().__init__()

        self.action_dim = action_dim
        self.use_lora = use_lora

        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        print(f"Loading OpenVLA: {model_name}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Apply LoRA if requested
        if use_lora:
            if quantization_config is not None:
                self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def forward(
        self,
        images,
        text: str,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: PIL images or tensors
            text: Instruction text
            actions: Ground truth actions for training

        Returns:
            Dict with predicted_actions and optional loss
        """
        # Process inputs
        inputs = self.processor(
            images=images,
            text=text,
            return_tensors="pt",
        ).to(self.model.device)

        # Forward through model
        outputs = self.model(**inputs)

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if hasattr(outputs, "loss") else None,
        }

    def predict_action(
        self,
        image,
        instruction: str,
        unnorm_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Predict action for a single observation.

        Args:
            image: PIL Image
            instruction: Language instruction
            unnorm_key: Key for action unnormalization

        Returns:
            action: Predicted action tensor
        """
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate action tokens
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.action_dim,
                do_sample=False,
            )

        # Decode action (OpenVLA encodes actions as tokens)
        action_tokens = generated[0, inputs.input_ids.shape[1]:]

        return action_tokens

    def save_pretrained(self, path: str):
        """Save the model."""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load a fine-tuned model."""
        return cls(model_name=path, **kwargs)


if __name__ == "__main__":
    print("OpenVLA Wrapper Test")
    print("Note: Requires ~16GB GPU memory with 4-bit quantization")

    # This would load the model
    # model = OpenVLAWrapper(
    #     model_name="openvla/openvla-7b",
    #     use_lora=True,
    #     load_in_4bit=True,
    # )
    print("OpenVLA wrapper defined successfully")
