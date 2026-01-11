"""
VLA Evaluation Module

Provides evaluation tools for Vision-Language-Action models:
- Simulation evaluation (gym environments)
- Real robot evaluation
- Benchmark evaluation
- Metrics computation
"""

from .evaluator import VLAEvaluator
from .benchmark import VLABenchmark
from .metrics import (
    compute_success_rate,
    compute_trajectory_metrics,
    compute_action_metrics,
)

__all__ = [
    "VLAEvaluator",
    "VLABenchmark",
    "compute_success_rate",
    "compute_trajectory_metrics",
    "compute_action_metrics",
]
