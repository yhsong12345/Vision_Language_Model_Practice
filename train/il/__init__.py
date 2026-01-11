"""
Imitation Learning Module for VLA

Implements imitation learning algorithms:
- BC: Behavioral Cloning (supervised learning on demonstrations)
- DAgger: Dataset Aggregation (interactive imitation learning)
- GAIL: Generative Adversarial Imitation Learning
"""

from .base_trainer import ILTrainer
from .behavioral_cloning import BehavioralCloning
from .dagger import DAgger
from .gail import GAIL

__all__ = [
    "ILTrainer",
    "BehavioralCloning",
    "DAgger",
    "GAIL",
]
