"""
VLA Evaluation Module

Provides evaluation tools for Vision-Language-Action models:
- Simulation evaluation (gym environments)
- Real robot evaluation
- Benchmark evaluation (CALVIN, RLBench, SIMPLER, LeRobot)
- Metrics computation
- Custom task benchmarks
"""

from .evaluator import VLAEvaluator
from .benchmark import (
    VLABenchmark,
    BenchmarkConfig,
    CustomTaskBenchmark,
    create_autonomous_driving_benchmark,
    create_manipulation_benchmark,
)
from .metrics import (
    compute_success_rate,
    compute_trajectory_metrics,
    compute_action_metrics,
)

__all__ = [
    # Evaluator
    "VLAEvaluator",
    # Benchmarks
    "VLABenchmark",
    "BenchmarkConfig",
    "CustomTaskBenchmark",
    "create_autonomous_driving_benchmark",
    "create_manipulation_benchmark",
    # Metrics
    "compute_success_rate",
    "compute_trajectory_metrics",
    "compute_action_metrics",
]
