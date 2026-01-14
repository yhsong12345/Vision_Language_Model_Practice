"""
Unified Base Trainer

Abstract base class for all training paradigms (IL, Online RL, Offline RL).
Provides common functionality for training, logging, and checkpointing.
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from train.utils.logging import ExperimentLogger, ExperimentConfig


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    Provides unified interface for:
    - Device management
    - Seed setting
    - Logging and metrics
    - Checkpointing
    - Evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str = "./output",
        device: str = "auto",
        seed: int = 42,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.seed = seed

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        os.makedirs(output_dir, exist_ok=True)

        # Logger (can be set up by subclasses)
        self.logger: Optional[ExperimentLogger] = None
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.experiment_name = experiment_name

    def setup_logger(self, config: ExperimentConfig, monitor_metric: str = "val_loss", monitor_mode: str = "min"):
        """Setup the experiment logger."""
        self.logger = ExperimentLogger(
            output_dir=self.output_dir,
            config=config,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            use_wandb=self.use_wandb,
            wandb_project=self.wandb_project,
        )
        self.logger.log_model_info(self.model)

    @abstractmethod
    def train(self, *args, **kwargs):
        """Run training loop."""
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Evaluate the model."""
        pass

    def save(self, path: Optional[str] = None, **extra):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(self.output_dir, "model.pt")

        checkpoint = {"model_state_dict": self.model.state_dict(), **extra}
        torch.save(checkpoint, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {path}")
        return checkpoint

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """Log metrics to logger."""
        if self.logger:
            self.logger.log_step(step=step, metrics=metrics, prefix=prefix)

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        """Log epoch metrics."""
        if self.logger:
            self.logger.log_epoch(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics)

    def update_best_model(self, metric_value: float, epoch: int, step: int = 0, optimizer=None, scheduler=None):
        """Update best model if metric improved."""
        if self.logger:
            return self.logger.update_best_model(
                model=self.model,
                metric_value=metric_value,
                epoch=epoch,
                step=step,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        return False

    def finish(self):
        """Finalize training."""
        if self.logger:
            self.logger.finish()

    @property
    def num_parameters(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class SupervisedTrainer(BaseTrainer):
    """
    Base class for supervised learning trainers (BC, fine-tuning).

    Provides standard training loop with:
    - DataLoader iteration
    - Loss computation
    - Optimizer step
    - Validation
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

        if self.scheduler:
            self.scheduler.step()

        return {"loss": total_loss / num_batches}

    def train_step(self, batch) -> float:
        """Single training step."""
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    @abstractmethod
    def compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for a batch."""
        pass

    def validate(self, dataloader) -> Dict[str, float]:
        """Validate on dataloader."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        return {"loss": total_loss / num_batches}


class RLTrainer(BaseTrainer):
    """
    Base class for reinforcement learning trainers (PPO, SAC, CQL, IQL).

    Provides common RL functionality:
    - Environment interaction
    - Episode tracking
    - Policy evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        env=None,
        gamma: float = 0.99,
        eval_episodes: int = 10,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.env = env
        self.gamma = gamma
        self.eval_episodes = eval_episodes

        self.episode_rewards = []
        self.episode_lengths = []

    def evaluate(self, num_episodes: Optional[int] = None, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate policy in environment."""
        if self.env is None:
            return {"mean_reward": 0.0, "std_reward": 0.0}

        if num_episodes is None:
            num_episodes = self.eval_episodes

        self.model.eval()
        rewards, lengths = [], []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = self.select_action(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            rewards.append(episode_reward)
            lengths.append(episode_length)

        self.model.train()

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
        }

    @abstractmethod
    def select_action(self, obs, deterministic: bool = False):
        """Select action given observation."""
        pass


__all__ = ["BaseTrainer", "SupervisedTrainer", "RLTrainer"]
