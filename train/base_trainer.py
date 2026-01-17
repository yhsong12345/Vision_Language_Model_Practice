"""
Unified Base Trainer - Abstract base class for all training paradigms.
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from train.utils.logging import ExperimentLogger, ExperimentConfig
from core.device_utils import get_device


class BaseTrainer(ABC):
    """Abstract base class providing: device management, logging, checkpointing, evaluation."""

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
        self.device = get_device(device)
        self.model = model.to(self.device)

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        os.makedirs(output_dir, exist_ok=True)
        self.logger: Optional[ExperimentLogger] = None
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.experiment_name = experiment_name

    def setup_logger(self, config: ExperimentConfig, monitor_metric: str = "val_loss", monitor_mode: str = "min"):
        """Setup experiment logger."""
        self.logger = ExperimentLogger(
            output_dir=self.output_dir, config=config,
            monitor_metric=monitor_metric, monitor_mode=monitor_mode,
            use_wandb=self.use_wandb, wandb_project=self.wandb_project,
        )
        self.logger.log_model_info(self.model)

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        pass

    def save(self, path: Optional[str] = None, **extra):
        path = path or os.path.join(self.output_dir, "model.pt")
        torch.save({"model_state_dict": self.model.state_dict(), **extra}, path)
        print(f"Saved model to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {path}")
        return checkpoint

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        if self.logger:
            self.logger.log_step(step=step, metrics=metrics, prefix=prefix)

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        if self.logger:
            self.logger.log_epoch(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics)

    def update_best_model(self, metric_value: float, epoch: int, step: int = 0, optimizer=None, scheduler=None):
        if self.logger:
            return self.logger.update_best_model(self.model, metric_value, epoch, step, optimizer, scheduler)
        return False

    def finish(self):
        if self.logger:
            self.logger.finish()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class SupervisedTrainer(BaseTrainer):
    """Base class for supervised learning trainers (BC, fine-tuning)."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, scheduler: Optional[Any] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, num_batches = 0, 0
        for batch in dataloader:
            total_loss += self.train_step(batch)
            num_batches += 1
        if self.scheduler:
            self.scheduler.step()
        return {"loss": total_loss / num_batches}

    def train_step(self, batch) -> float:
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    @abstractmethod
    def compute_loss(self, batch) -> torch.Tensor:
        pass

    def validate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_loss, num_batches = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                total_loss += self.compute_loss(batch).item()
                num_batches += 1
        return {"loss": total_loss / num_batches}


class RLTrainer(BaseTrainer):
    """Base class for RL trainers (PPO, SAC, CQL, IQL)."""

    def __init__(self, model: nn.Module, env=None, gamma: float = 0.99, eval_episodes: int = 10, **kwargs):
        super().__init__(model, **kwargs)
        self.env = env
        self.gamma = gamma
        self.eval_episodes = eval_episodes
        self.episode_rewards = []
        self.episode_lengths = []

    def evaluate(self, num_episodes: Optional[int] = None, deterministic: bool = True) -> Dict[str, float]:
        if self.env is None:
            return {"mean_reward": 0.0, "std_reward": 0.0}
        num_episodes = num_episodes or self.eval_episodes
        self.model.eval()
        rewards, lengths = [], []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            ep_reward, ep_length, done = 0, 0, False
            while not done:
                action = self.select_action(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1
            rewards.append(ep_reward)
            lengths.append(ep_length)

        self.model.train()
        return {"mean_reward": np.mean(rewards), "std_reward": np.std(rewards), "mean_length": np.mean(lengths)}

    @abstractmethod
    def select_action(self, obs, deterministic: bool = False):
        pass


__all__ = ["BaseTrainer", "SupervisedTrainer", "RLTrainer"]
