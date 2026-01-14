"""
Experiment Management Module

Provides comprehensive experiment tracking and management:
- Experiment configuration and logging
- Metrics tracking with multiple backends (WandB, TensorBoard, CSV)
- Checkpoint management
- Reproducibility tools
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from collections import defaultdict
import csv

# Import shared ExperimentConfig from train.utils.logging
from train.utils.logging import ExperimentConfig as BaseExperimentConfig


class MetricsLogger:
    """
    Unified metrics logger supporting multiple backends.

    Backends:
    - WandB: Weights & Biases
    - TensorBoard: TensorFlow TensorBoard
    - CSV: Simple CSV logging
    """

    def __init__(
        self,
        run_dir: str,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        project: str = "vla-training",
        name: str = "experiment",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        log_freq: int = 100,
    ):
        self.run_dir = run_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv

        self.backends = {}
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.step = 0

        os.makedirs(run_dir, exist_ok=True)

        # Initialize backends
        if use_wandb:
            self._init_wandb(project, name, config, tags, notes)
        if use_tensorboard:
            self._init_tensorboard()
        if use_csv:
            self._init_csv()

    def _init_wandb(self, project, name, config, tags, notes):
        try:
            import wandb
            wandb.init(
                project=project,
                name=name,
                config=config or {},
                tags=tags or [],
                notes=notes,
                dir=self.run_dir,
            )
            self.backends["wandb"] = wandb
            print("WandB initialized")
        except ImportError:
            print("WandB not available")

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.run_dir, "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            self.backends["tensorboard"] = SummaryWriter(log_dir)
            print(f"TensorBoard logging to {log_dir}")
        except ImportError:
            print("TensorBoard not available")

    def _init_csv(self):
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self.csv_file = None
        self.csv_writer = None
        self.backends["csv"] = True
        print(f"CSV logging to {self.csv_path}")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all backends."""
        if step is not None:
            self.step = step
        else:
            self.step += 1

        metrics["timestamp"] = time.time()

        for k, v in metrics.items():
            self.history[k].append(v)

        if "wandb" in self.backends:
            self.backends["wandb"].log(metrics, step=self.step)

        if "tensorboard" in self.backends:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.backends["tensorboard"].add_scalar(k, v, self.step)

        if "csv" in self.backends:
            self._log_csv(metrics)

    def _log_csv(self, metrics: Dict[str, float]):
        metrics["step"] = self.step
        if self.csv_writer is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(metrics.keys()))
            self.csv_writer.writeheader()
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()

    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        """Log image to backends."""
        if step is not None:
            self.step = step

        if "wandb" in self.backends:
            import wandb
            self.backends["wandb"].log({tag: wandb.Image(image)}, step=self.step)

        if "tensorboard" in self.backends:
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)
            self.backends["tensorboard"].add_image(tag, image, self.step)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of logged metrics."""
        summary = {}
        for k, v in self.history.items():
            if len(v) > 0 and isinstance(v[0], (int, float)):
                summary[f"{k}_mean"] = np.mean(v)
                summary[f"{k}_std"] = np.std(v)
                summary[f"{k}_min"] = np.min(v)
                summary[f"{k}_max"] = np.max(v)
        return summary

    def close(self):
        """Close all backends."""
        if "wandb" in self.backends:
            self.backends["wandb"].finish()
        if "tensorboard" in self.backends:
            self.backends["tensorboard"].close()
        if self.csv_file:
            self.csv_file.close()


class CheckpointManager:
    """Manages model checkpoints with versioning and cleanup."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Dict[str, Any]] = []

        os.makedirs(checkpoint_dir, exist_ok=True)
        self._load_checkpoint_index()

    def _load_checkpoint_index(self):
        index_path = os.path.join(self.checkpoint_dir, "checkpoints.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.checkpoints = json.load(f)

    def _save_checkpoint_index(self):
        index_path = os.path.join(self.checkpoint_dir, "checkpoints.json")
        with open(index_path, "w") as f:
            json.dump(self.checkpoints, f, indent=2)

    def save(
        self,
        state_dict: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint."""
        import torch

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step_{step}_{timestamp}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "step": step,
            "timestamp": timestamp,
            "state_dict": state_dict,
            "metrics": metrics,
        }
        torch.save(checkpoint, path)

        self.checkpoints.append({
            "path": path,
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics,
        })
        self._save_checkpoint_index()

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)

        self._cleanup()
        return path

    def _cleanup(self):
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if os.path.exists(oldest["path"]):
                os.remove(oldest["path"])
        self._save_checkpoint_index()

    def load_latest(self) -> Optional[Dict[str, Any]]:
        import torch
        if not self.checkpoints:
            return None
        return torch.load(self.checkpoints[-1]["path"], map_location="cpu")

    def load_best(self) -> Optional[Dict[str, Any]]:
        import torch
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            return torch.load(best_path, map_location="cpu")
        return None


class ExperimentManager:
    """
    Comprehensive experiment management.

    Combines configuration, logging, checkpointing, and reproducibility.
    """

    def __init__(
        self,
        name: str = "vla_experiment",
        project: str = "vla-training",
        base_dir: str = "./experiments",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        seed: int = 42,
        tags: Optional[List[str]] = None,
        notes: str = "",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.project = project
        self.seed = seed
        self.tags = tags or []
        self.notes = notes

        # Create run directory
        self.run_id = self._generate_run_id()
        self.run_dir = os.path.join(base_dir, project, name, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Save config
        if config:
            self._save_config(config)

        # Initialize components
        self.logger = MetricsLogger(
            run_dir=self.run_dir,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
        )
        self.checkpoint_manager = CheckpointManager(
            os.path.join(self.run_dir, "checkpoints")
        )

        self._set_seeds()
        print(f"Experiment initialized: {self.run_dir}")

    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_str = hashlib.md5(f"{self.name}{timestamp}".encode()).hexdigest()[:6]
        return f"{timestamp}_{hash_str}"

    def _set_seeds(self):
        import random
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def _save_config(self, config: Dict[str, Any]):
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        self.logger.log(metrics, step)

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """Save model checkpoint."""
        return self.checkpoint_manager.save(state_dict, step, metrics, is_best)

    def load_checkpoint(self, mode: str = "latest") -> Optional[Dict[str, Any]]:
        """Load checkpoint. mode: 'latest', 'best', or step number as string."""
        if mode == "latest":
            return self.checkpoint_manager.load_latest()
        elif mode == "best":
            return self.checkpoint_manager.load_best()
        return None

    def finish(self):
        """Finish experiment and cleanup."""
        summary = self.logger.get_summary()
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.close()
        print(f"Experiment finished: {self.run_dir}")


def create_experiment(name: str, project: str = "vla-training", **kwargs) -> ExperimentManager:
    """Convenience function to create an experiment."""
    return ExperimentManager(name=name, project=project, **kwargs)


if __name__ == "__main__":
    manager = ExperimentManager(
        name="test_experiment",
        project="vla-testing",
        use_wandb=False,
        use_tensorboard=True,
    )

    for step in range(100):
        metrics = {
            "loss": np.random.random(),
            "accuracy": np.random.random(),
            "learning_rate": 0.001 * (0.99 ** step),
        }
        manager.log(metrics, step)

        if step % 20 == 0:
            manager.save_checkpoint({"model": "dummy_state"}, step, metrics, is_best=(step == 80))

    manager.finish()
    print("\nExperiment manager test complete!")
