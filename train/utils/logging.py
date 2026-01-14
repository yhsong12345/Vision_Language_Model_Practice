"""
Logging Utilities

Shared logging and metrics tracking for training:
- MetricsTracker: Track and aggregate training metrics
- TrainingLogger: Unified logging interface
- ExperimentLogger: Comprehensive experiment logging with best model saving and W&B
"""

import os
import json
import time
import shutil
from typing import Dict, Any, Optional, List, Union, Callable
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class MetricsTracker:
    """
    Track and aggregate training metrics.

    Supports windowed averaging and logging to file.
    """

    def __init__(
        self,
        window_size: int = 100,
        log_dir: Optional[str] = None,
    ):
        self.window_size = window_size
        self.log_dir = log_dir
        self.metrics = defaultdict(list)
        self.step_metrics = {}
        self.global_step = 0

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def add(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Add single metric value."""
        self.metrics[key].append(value)

        # Keep window size
        if len(self.metrics[key]) > self.window_size:
            self.metrics[key].pop(0)

        if step is not None:
            self.global_step = step

    def add_dict(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Add multiple metrics at once."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.add(key, value, step)

    def get(self, key: str) -> float:
        """Get latest value for metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return 0.0

    def get_mean(self, key: str) -> float:
        """Get windowed mean for metric."""
        if key in self.metrics and self.metrics[key]:
            return float(np.mean(self.metrics[key]))
        return 0.0

    def get_std(self, key: str) -> float:
        """Get windowed std for metric."""
        if key in self.metrics and self.metrics[key]:
            return float(np.std(self.metrics[key]))
        return 0.0

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all tracked metrics."""
        summary = {}
        for key in self.metrics:
            summary[f"{key}_mean"] = self.get_mean(key)
            summary[f"{key}_std"] = self.get_std(key)
            summary[f"{key}_latest"] = self.get(key)
        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.global_step = 0

    def save(self, filename: str = "metrics.json") -> str:
        """Save metrics to JSON file."""
        if self.log_dir is None:
            return ""

        path = os.path.join(self.log_dir, filename)
        data = {
            "global_step": self.global_step,
            "summary": self.get_summary(),
            "raw": {k: list(v) for k, v in self.metrics.items()},
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path


class TrainingLogger:
    """
    Unified training logger.

    Handles console output, file logging, and optional tensorboard/wandb.
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = output_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(output_dir, "logs")

        os.makedirs(self.log_dir, exist_ok=True)

        # Metrics tracker
        self.metrics = MetricsTracker(log_dir=self.log_dir)

        # Timing
        self.start_time = time.time()
        self.step_times = []

        # Log file
        self.log_file = os.path.join(self.log_dir, "training.log")

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                print("TensorBoard not available")

        # Wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or "vla-training",
                    name=self.experiment_name,
                    config=config or {},
                    dir=output_dir,
                )
            except ImportError:
                print("Wandb not available")

        # Save config
        if config:
            self.save_config(config)

    def log(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to all backends."""
        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Track in metrics tracker
        self.metrics.add_dict(metrics, step)

        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)

        # Wandb
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log single scalar value."""
        self.metrics.add(key, value, step)

        if self.writer:
            self.writer.add_scalar(key, value, step)

        if self.wandb_run:
            import wandb
            wandb.log({key: value}, step=step)

    def print(
        self,
        message: str,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Print formatted message to console and log file."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if step is not None:
            elapsed = time.time() - self.start_time
            header = f"[{timestamp}] Step {step} ({elapsed:.0f}s)"
        else:
            header = f"[{timestamp}]"

        if metrics:
            metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            full_message = f"{header} {message} | {metric_str}"
        else:
            full_message = f"{header} {message}"

        print(full_message)

        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(full_message + "\n")

    def print_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Print formatted metrics."""
        elapsed = time.time() - self.start_time
        steps_per_sec = step / max(elapsed, 1)

        lines = [f"Step {step} ({elapsed:.0f}s, {steps_per_sec:.1f} steps/s)"]
        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.4f}")

        message = "\n".join(lines)
        print(message)

        with open(self.log_file, "a") as f:
            f.write(message + "\n\n")

    def save_config(self, config: Dict[str, Any]) -> str:
        """Save configuration to file."""
        path = os.path.join(self.output_dir, "config.json")

        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        return path

    def save_metrics(self) -> str:
        """Save current metrics."""
        return self.metrics.save()

    def close(self) -> None:
        """Close all logging backends."""
        if self.writer:
            self.writer.close()

        if self.wandb_run:
            import wandb
            wandb.finish()

        self.save_metrics()
        self.print("Training completed", metrics=self.metrics.get_summary())


@dataclass
class ExperimentConfig:
    """Configuration for experiment logging."""

    # Experiment info
    experiment_name: str = ""
    project_name: str = "vla-training"
    run_id: str = ""

    # Model info
    model_name: str = ""
    model_type: str = ""
    num_parameters: int = 0

    # Dataset info
    dataset_name: str = ""
    dataset_size: int = 0
    train_samples: int = 0
    val_samples: int = 0

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0

    # Architecture details
    action_dim: int = 7
    action_head_type: str = "mlp"
    vision_encoder: str = ""
    llm_backbone: str = ""
    use_lora: bool = False
    lora_r: int = 16

    # Hardware
    device: str = "cuda"
    num_gpus: int = 1
    mixed_precision: str = "fp32"

    # Additional notes
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


class BestModelTracker:
    """
    Tracks and saves only the best model based on a monitored metric.

    Features:
    - Saves best model with full metadata
    - Supports both minimize (loss) and maximize (accuracy/reward) modes
    - Automatic cleanup of non-best checkpoints
    """

    def __init__(
        self,
        output_dir: str,
        metric_name: str = "val_loss",
        mode: str = "min",  # "min" or "max"
        save_last: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.metric_name = metric_name
        self.mode = mode
        self.save_last = save_last

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1
        self.best_step = -1

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_better(self, value: float) -> bool:
        """Check if value is better than current best."""
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def update(
        self,
        model: nn.Module,
        value: float,
        epoch: int,
        step: int,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update best model if current value is better.

        Returns:
            True if model was saved as new best, False otherwise.
        """
        is_new_best = self.is_better(value)

        if is_new_best:
            self.best_value = value
            self.best_epoch = epoch
            self.best_step = step

            # Save best model
            self._save_checkpoint(
                model, optimizer, scheduler,
                epoch, step, value, extra_info,
                filename="best_model.pt"
            )

        # Optionally save last model (for resume)
        if self.save_last:
            self._save_checkpoint(
                model, optimizer, scheduler,
                epoch, step, value, extra_info,
                filename="last_model.pt"
            )

        return is_new_best

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Any],
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        metric_value: float,
        extra_info: Optional[Dict[str, Any]],
        filename: str,
    ):
        """Save checkpoint with full state."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
            self.metric_name: metric_value,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "timestamp": datetime.now().isoformat(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if extra_info:
            checkpoint["extra_info"] = extra_info

        path = self.output_dir / filename
        torch.save(checkpoint, path)

    def get_best_info(self) -> Dict[str, Any]:
        """Get information about best model."""
        return {
            "metric_name": self.metric_name,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "mode": self.mode,
        }


class ExperimentLogger:
    """
    Comprehensive experiment logger with best model saving and W&B integration.

    Features:
    - Automatic best model tracking and saving
    - Structured experiment logging (hyperparameters, model info, dataset)
    - W&B integration with commonly monitored metrics
    - JSON/YAML log file generation
    - Training curves and metrics history

    Usage:
        logger = ExperimentLogger(
            output_dir="./experiments/exp1",
            config=ExperimentConfig(
                experiment_name="pusht_bc",
                model_name="CustomVLA",
                dataset_name="pusht",
                learning_rate=1e-4,
                ...
            ),
            use_wandb=True,
        )

        for epoch in range(num_epochs):
            # Training loop
            train_loss = train_one_epoch(...)
            val_loss = validate(...)

            # Log metrics
            logger.log_epoch(
                epoch=epoch,
                train_metrics={"loss": train_loss},
                val_metrics={"loss": val_loss},
            )

            # Update best model (saves automatically if better)
            logger.update_best_model(model, val_loss, epoch)

        logger.finish()
    """

    def __init__(
        self,
        output_dir: str,
        config: Optional[ExperimentConfig] = None,
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        resume_run_id: Optional[str] = None,
        log_format: str = "json",  # "json" or "yaml"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Config
        self.config = config or ExperimentConfig()
        if not self.config.experiment_name:
            self.config.experiment_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        if not self.config.run_id:
            self.config.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_format = log_format

        # Best model tracker
        self.best_tracker = BestModelTracker(
            output_dir=str(self.output_dir / "checkpoints"),
            metric_name=monitor_metric,
            mode=monitor_mode,
        )

        # Metrics history
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.epoch_history: List[Dict[str, Any]] = []

        # Timing
        self.start_time = time.time()
        self.epoch_times: List[float] = []

        # Global step counter
        self.global_step = 0

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")

        # Weights & Biases
        self.wandb_run = None
        self.use_wandb = use_wandb
        if use_wandb:
            self._init_wandb(wandb_project, wandb_entity, resume_run_id)

        # Save initial config
        self._save_config()

        # Log start
        self._log_message(f"Experiment started: {self.config.experiment_name}")

    def _init_wandb(
        self,
        project: Optional[str],
        entity: Optional[str],
        resume_run_id: Optional[str],
    ):
        """Initialize Weights & Biases."""
        try:
            import wandb

            wandb_config = self.config.to_dict()

            self.wandb_run = wandb.init(
                project=project or self.config.project_name,
                entity=entity,
                name=self.config.experiment_name,
                config=wandb_config,
                dir=str(self.output_dir),
                resume="allow" if resume_run_id else None,
                id=resume_run_id,
                tags=self.config.tags,
                notes=self.config.notes,
            )

            # Define common metrics for W&B dashboard
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("step/*", step_metric="global_step")

            self._log_message(f"W&B initialized: {wandb.run.url}")

        except ImportError:
            print("Warning: wandb not available. Install with: pip install wandb")
            self.use_wandb = False

    def _save_config(self):
        """Save experiment configuration."""
        config_path = self.output_dir / f"config.{self.log_format}"

        config_dict = {
            "experiment": self.config.to_dict(),
            "monitor": {
                "metric": self.best_tracker.metric_name,
                "mode": self.best_tracker.mode,
            },
            "start_time": datetime.now().isoformat(),
        }

        if self.log_format == "yaml":
            try:
                import yaml
                with open(config_path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            except ImportError:
                # Fallback to JSON
                config_path = self.output_dir / "config.json"
                with open(config_path, "w") as f:
                    json.dump(config_dict, f, indent=2)
        else:
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

    def _log_message(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)

        log_file = self.output_dir / "experiment.log"
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = "step",
    ):
        """
        Log metrics at step level (for fine-grained tracking).

        Common step metrics for W&B:
        - loss, lr, grad_norm, throughput
        """
        self.global_step = step

        # Add to history
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}"
            self.metrics_history[full_key].append(value)

        # TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}/{key}", value, step)

        # W&B
        if self.wandb_run:
            import wandb
            log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
            log_dict["global_step"] = step
            wandb.log(log_dict)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log metrics at epoch level.

        Common epoch metrics for W&B:
        - train/loss, train/accuracy
        - val/loss, val/accuracy, val/reward
        - learning_rate, epoch_time
        """
        epoch_time = time.time() - self.start_time - sum(self.epoch_times)
        self.epoch_times.append(epoch_time)

        epoch_record = {
            "epoch": epoch,
            "epoch_time": epoch_time,
            "total_time": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
        }

        log_dict = {"epoch": epoch}

        # Training metrics
        if train_metrics:
            for key, value in train_metrics.items():
                full_key = f"train/{key}"
                self.metrics_history[full_key].append(value)
                epoch_record[full_key] = value
                log_dict[full_key] = value

        # Validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                full_key = f"val/{key}"
                self.metrics_history[full_key].append(value)
                epoch_record[full_key] = value
                log_dict[full_key] = value

        # Extra metrics (lr, etc.)
        if extra_metrics:
            for key, value in extra_metrics.items():
                self.metrics_history[key].append(value)
                epoch_record[key] = value
                log_dict[key] = value

        # Add timing
        log_dict["epoch_time"] = epoch_time
        log_dict["total_time"] = time.time() - self.start_time

        self.epoch_history.append(epoch_record)

        # TensorBoard
        if self.tb_writer:
            for key, value in log_dict.items():
                if key != "epoch" and isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, epoch)

        # W&B
        if self.wandb_run:
            import wandb
            wandb.log(log_dict)

        # Console log
        self._print_epoch_summary(epoch, train_metrics, val_metrics, epoch_time)

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Optional[Dict[str, float]],
        val_metrics: Optional[Dict[str, float]],
        epoch_time: float,
    ):
        """Print formatted epoch summary."""
        parts = [f"Epoch {epoch}"]

        if train_metrics:
            train_str = " | ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            parts.append(f"Train: {train_str}")

        if val_metrics:
            val_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            parts.append(f"Val: {val_str}")

        parts.append(f"Time: {epoch_time:.1f}s")

        message = " | ".join(parts)
        self._log_message(message)

    def update_best_model(
        self,
        model: nn.Module,
        metric_value: float,
        epoch: int,
        step: Optional[int] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> bool:
        """
        Update best model if current metric is better.

        Returns:
            True if new best model was saved.
        """
        step = step or self.global_step

        extra_info = {
            "config": self.config.to_dict(),
            "metrics_at_save": {
                k: v[-1] if v else None
                for k, v in self.metrics_history.items()
            },
        }

        is_best = self.best_tracker.update(
            model=model,
            value=metric_value,
            epoch=epoch,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_info=extra_info,
        )

        if is_best:
            self._log_message(
                f"New best model! {self.best_tracker.metric_name}={metric_value:.4f} "
                f"(prev best: {self.best_tracker.best_value:.4f})"
            )

            # Log to W&B
            if self.wandb_run:
                import wandb
                wandb.run.summary["best_metric"] = metric_value
                wandb.run.summary["best_epoch"] = epoch

        return is_best

    def log_model_info(self, model: nn.Module):
        """Log model architecture information."""
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.config.num_parameters = num_params

        model_info = {
            "total_parameters": num_params,
            "trainable_parameters": num_trainable,
            "frozen_parameters": num_params - num_trainable,
            "trainable_ratio": num_trainable / max(num_params, 1),
        }

        self._log_message(
            f"Model: {self.config.model_name} | "
            f"Params: {num_params:,} ({num_trainable:,} trainable)"
        )

        # W&B
        if self.wandb_run:
            import wandb
            wandb.config.update(model_info)

        # Save to config
        info_path = self.output_dir / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)

    def save_training_log(self):
        """Save complete training log to file."""
        log_data = {
            "experiment": self.config.to_dict(),
            "best_model": self.best_tracker.get_best_info(),
            "metrics_history": {k: list(v) for k, v in self.metrics_history.items()},
            "epoch_history": self.epoch_history,
            "total_time": time.time() - self.start_time,
            "end_time": datetime.now().isoformat(),
        }

        log_path = self.output_dir / f"training_log.{self.log_format}"

        if self.log_format == "yaml":
            try:
                import yaml
                with open(log_path, "w") as f:
                    yaml.dump(log_data, f, default_flow_style=False)
            except ImportError:
                log_path = self.output_dir / "training_log.json"
                with open(log_path, "w") as f:
                    json.dump(log_data, f, indent=2)
        else:
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2)

        self._log_message(f"Training log saved to: {log_path}")
        return str(log_path)

    def finish(self):
        """Finalize logging and save final state."""
        # Save training log
        self.save_training_log()

        # Log final summary
        total_time = time.time() - self.start_time
        best_info = self.best_tracker.get_best_info()

        summary = (
            f"Training completed in {total_time/60:.1f} min | "
            f"Best {best_info['metric_name']}={best_info['best_value']:.4f} "
            f"at epoch {best_info['best_epoch']}"
        )
        self._log_message(summary)

        # TensorBoard
        if self.tb_writer:
            self.tb_writer.close()

        # W&B
        if self.wandb_run:
            import wandb

            # Log final summary
            wandb.run.summary.update({
                "total_time_min": total_time / 60,
                "best_metric": best_info["best_value"],
                "best_epoch": best_info["best_epoch"],
                "num_epochs": len(self.epoch_history),
            })

            # Upload best model as artifact
            artifact = wandb.Artifact(
                name=f"{self.config.experiment_name}_best_model",
                type="model",
                description=f"Best model with {best_info['metric_name']}={best_info['best_value']:.4f}",
            )
            artifact.add_file(str(self.output_dir / "checkpoints" / "best_model.pt"))
            wandb.log_artifact(artifact)

            wandb.finish()

        return best_info


# Convenience function to create logger from config dict
def create_experiment_logger(
    output_dir: str,
    config_dict: Dict[str, Any],
    use_wandb: bool = False,
    **kwargs,
) -> ExperimentLogger:
    """
    Create ExperimentLogger from a configuration dictionary.

    Args:
        output_dir: Directory to save logs and checkpoints
        config_dict: Dictionary with experiment configuration
        use_wandb: Whether to use W&B logging
        **kwargs: Additional arguments for ExperimentLogger

    Returns:
        Configured ExperimentLogger instance
    """
    config = ExperimentConfig.from_dict(config_dict)
    return ExperimentLogger(
        output_dir=output_dir,
        config=config,
        use_wandb=use_wandb,
        **kwargs,
    )


if __name__ == "__main__":
    print("Testing logging utilities...")

    # Test MetricsTracker
    tracker = MetricsTracker(window_size=10)
    for i in range(20):
        tracker.add("loss", np.random.rand())
        tracker.add("accuracy", 0.5 + 0.5 * np.random.rand())

    print(f"Loss mean: {tracker.get_mean('loss'):.4f}")
    print(f"Accuracy mean: {tracker.get_mean('accuracy'):.4f}")

    # Test TrainingLogger (minimal)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(output_dir=tmpdir, experiment_name="test")
        logger.log({"loss": 0.5, "accuracy": 0.9}, step=100)
        logger.print("Test message", step=100, metrics={"loss": 0.5})
        logger.close()

    print("\nTesting ExperimentLogger...")

    # Test ExperimentLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(
            experiment_name="test_experiment",
            model_name="TestModel",
            dataset_name="test_dataset",
            learning_rate=1e-4,
            batch_size=32,
            num_epochs=10,
        )

        exp_logger = ExperimentLogger(
            output_dir=tmpdir,
            config=config,
            monitor_metric="val_loss",
            monitor_mode="min",
            use_wandb=False,
        )

        # Simulate training
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)

        model = DummyModel()
        exp_logger.log_model_info(model)

        for epoch in range(5):
            train_loss = 1.0 - epoch * 0.15
            val_loss = 1.1 - epoch * 0.18

            exp_logger.log_epoch(
                epoch=epoch,
                train_metrics={"loss": train_loss, "accuracy": 0.5 + epoch * 0.1},
                val_metrics={"loss": val_loss, "accuracy": 0.45 + epoch * 0.12},
                extra_metrics={"learning_rate": 1e-4 * (0.9 ** epoch)},
            )

            exp_logger.update_best_model(model, val_loss, epoch)

        exp_logger.finish()

        # Check files were created
        assert (Path(tmpdir) / "checkpoints" / "best_model.pt").exists()
        assert (Path(tmpdir) / "training_log.json").exists()
        assert (Path(tmpdir) / "config.json").exists()

    print("\nAll logging utilities tests passed!")
