"""
SmolVLA Wrapper

Wrapper for SmolVLA-450M - a lightweight VLA model suitable for:
- Consumer hardware training
- Edge deployment
- Quick prototyping
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class SmolVLAWrapper(nn.Module):
    """
    Wrapper for SmolVLA-450M from HuggingFace.

    SmolVLA is a lightweight (450M parameters) VLA model designed for:
    - Training on consumer GPUs
    - Fast inference
    - LeRobot dataset compatibility
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLA-450M",
        action_dim: int = 7,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.model_name = model_name

        # Try to load from lerobot
        try:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
            self.use_lerobot = True
            print(f"Using LeRobot SmolVLA implementation")
        except ImportError:
            self.use_lerobot = False
            print("LeRobot not available, using HuggingFace transformers")

            from transformers import AutoModelForVision2Seq, AutoProcessor
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

    def setup_for_dataset(self, dataset_info: Dict[str, Any]):
        """
        Configure the model for a specific dataset.

        Args:
            dataset_info: Dict with input/output shape information
        """
        if self.use_lerobot:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig

            config = SmolVLAConfig(
                input_shapes=dataset_info.get("input_shapes", {
                    "observation.image": [3, 224, 224],
                }),
                output_shapes=dataset_info.get("output_shapes", {
                    "action": [self.action_dim],
                }),
            )
            self.policy = SmolVLAPolicy(config)
            return self.policy

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dict containing observation and action tensors

        Returns:
            Dict with loss and predicted actions
        """
        if self.use_lerobot:
            return self.policy.forward(batch)
        else:
            return self.model(**batch)

    def predict_action(
        self,
        observation: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Predict action given observation.

        Args:
            observation: Dict with observation tensors

        Returns:
            Predicted action tensor
        """
        if self.use_lerobot:
            with torch.no_grad():
                return self.policy.select_action(observation)
        else:
            with torch.no_grad():
                outputs = self.model(**observation)
                return outputs.logits

    def save_pretrained(self, path: str):
        """Save model weights."""
        if self.use_lerobot:
            torch.save(self.policy.state_dict(), f"{path}/policy.pt")
        else:
            self.model.save_pretrained(path)

    def load_pretrained(self, path: str):
        """Load model weights."""
        if self.use_lerobot:
            self.policy.load_state_dict(torch.load(f"{path}/policy.pt"))
        else:
            from transformers import AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(path)


if __name__ == "__main__":
    print("SmolVLA Wrapper Test")
    wrapper = SmolVLAWrapper()
    print("SmolVLA wrapper created successfully")
