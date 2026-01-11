"""
VLA Training Module

This module contains training schemes for Vision-Language-Action models:

Submodules:
- pretrain: VLM pretraining (vision-language alignment, instruction tuning)
- finetune: Supervised fine-tuning on robot manipulation data
- rl: Reinforcement learning (PPO, SAC, GRPO)
- il: Imitation learning (BC, DAgger, GAIL)
- datasets: Dataset loaders for various training paradigms
"""

from .pretrain import VLMPretrainer
from .finetune import VLAFineTuner
from .rl import RLTrainer, PPOTrainer, SACTrainer, GRPOTrainer
from .il import ILTrainer, BehavioralCloning, DAgger

# Dataset loaders
from .datasets import (
    # LeRobot
    LeRobotDataset,
    PushTDataset,
    AlohaDataset,
    XArmDataset,
    create_lerobot_dataloader,
    # Open X-Embodiment
    OpenXDataset,
    BridgeDataset,
    RT1Dataset,
    create_openx_dataloader,
    # Driving
    DrivingDataset,
    NuScenesDataset,
    CarlaDataset,
    create_driving_dataloader,
    # RL
    RLDataset,
    D4RLDataset,
    RoboMimicDataset,
    create_rl_dataloader,
)

__all__ = [
    # Trainers
    "VLMPretrainer",
    "VLAFineTuner",
    "RLTrainer",
    "PPOTrainer",
    "SACTrainer",
    "GRPOTrainer",
    "ILTrainer",
    "BehavioralCloning",
    "DAgger",
    # LeRobot datasets
    "LeRobotDataset",
    "PushTDataset",
    "AlohaDataset",
    "XArmDataset",
    "create_lerobot_dataloader",
    # Open X datasets
    "OpenXDataset",
    "BridgeDataset",
    "RT1Dataset",
    "create_openx_dataloader",
    # Driving datasets
    "DrivingDataset",
    "NuScenesDataset",
    "CarlaDataset",
    "create_driving_dataloader",
    # RL datasets
    "RLDataset",
    "D4RLDataset",
    "RoboMimicDataset",
    "create_rl_dataloader",
]
