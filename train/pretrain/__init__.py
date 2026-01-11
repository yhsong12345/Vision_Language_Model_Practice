"""
VLM Pretraining Module

Implements pretraining stages for Vision-Language Models:
1. Vision-Language Alignment: Train projector to align vision with LLM
2. Visual Instruction Tuning: Fine-tune on multimodal instructions

This follows the LLaVA-style training paradigm.
"""

from .vlm_pretrainer import VLMPretrainer
from .alignment_trainer import AlignmentTrainer
from .instruction_trainer import InstructionTrainer

__all__ = [
    "VLMPretrainer",
    "AlignmentTrainer",
    "InstructionTrainer",
]
