"""
VLA Fine-tuning Module

Implements supervised fine-tuning on robot manipulation datasets:
- Standard fine-tuning (train projector + action head)
- Full fine-tuning (train all parameters)
- LoRA fine-tuning (memory efficient)
"""

from .vla_finetuner import VLAFineTuner
from .dataset import RobotDataset, create_robot_dataloader

__all__ = [
    "VLAFineTuner",
    "RobotDataset",
    "create_robot_dataloader",
]
