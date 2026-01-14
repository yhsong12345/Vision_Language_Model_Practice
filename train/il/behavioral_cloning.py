"""
Behavioral Cloning (BC)

Simple imitation learning via supervised learning on expert demonstrations.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm

from .base_trainer import ILTrainer, ExpertDataset, PolicyNetwork
from config.training_config import ILConfig
from train.utils.logging import ExperimentLogger, ExperimentConfig


class BehavioralCloning(ILTrainer):
    """
    Behavioral Cloning trainer for standard RL environments.

    Uses supervised learning to map states to actions from expert demonstrations.
    """

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        config: Optional[ILConfig] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        experiment_name: Optional[str] = None,
        **kwargs,
    ):
        config = config or ILConfig.behavioral_cloning()
        super().__init__(env, policy, config.output_dir, **kwargs)

        self.config = config
        self.num_epochs = config.bc_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.val_split = config.bc_validation_split

        self.optimizer = AdamW(self.policy.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=self.learning_rate * 0.01)
        self.criterion = nn.MSELoss() if self.continuous else nn.CrossEntropyLoss()

        # Setup logger
        env_name = env.spec.id if hasattr(env, 'spec') else 'custom'
        self.logger = ExperimentLogger(
            output_dir=config.output_dir,
            config=ExperimentConfig(
                experiment_name=experiment_name or f"bc_{env_name}",
                project_name=wandb_project or "vla-bc-training",
                model_name="PolicyNetwork",
                model_type="MLP",
                dataset_name=env_name,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                optimizer="AdamW",
                scheduler="CosineAnnealingLR",
                action_dim=self.action_dim,
                device=str(self.device),
            ),
            monitor_metric="val_loss",
            monitor_mode="min",
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )
        self.logger.log_model_info(self.policy)

    def train(
        self,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        expert_policy=None,
        num_expert_episodes: int = None,
    ) -> Dict:
        """Train policy using behavioral cloning."""
        print("=" * 60)
        print("Behavioral Cloning Training")
        print("=" * 60)

        # Collect demonstrations if not provided
        if states is None or actions is None:
            if expert_policy is None:
                raise ValueError("Must provide either (states, actions) or expert_policy")
            num_expert_episodes = num_expert_episodes or self.config.num_expert_episodes
            states, actions = self.collect_expert_demonstrations(expert_policy, num_expert_episodes)

        # Create dataset and split
        dataset = ExpertDataset(states, actions)
        train_size = int(len(dataset) * (1 - self.val_split))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Training: {train_size}, Validation: {val_size}")

        train_losses, val_losses = [], []
        global_step = 0

        for epoch in range(self.num_epochs):
            # Training
            self.policy.train()
            epoch_loss, num_batches = 0, 0

            for states_batch, actions_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False):
                states_batch = states_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)

                predicted = self.policy(states_batch)
                loss = self.criterion(predicted, actions_batch if self.continuous else actions_batch.long())

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if global_step % 10 == 0:
                    self.logger.log_step(step=global_step, metrics={"loss": loss.item(), "grad_norm": grad_norm.item(), "lr": self.optimizer.param_groups[0]["lr"]}, prefix="train")

            self.scheduler.step()
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)

            # Validation
            self.policy.eval()
            val_loss = 0
            with torch.no_grad():
                for states_batch, actions_batch in val_loader:
                    states_batch = states_batch.to(self.device)
                    actions_batch = actions_batch.to(self.device)
                    predicted = self.policy(states_batch)
                    val_loss += self.criterion(predicted, actions_batch if self.continuous else actions_batch.long()).item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            self.logger.log_epoch(epoch=epoch, train_metrics={"loss": avg_train_loss}, val_metrics={"loss": avg_val_loss})
            self.logger.update_best_model(self.policy, avg_val_loss, epoch, global_step, self.optimizer, self.scheduler)

        # Final evaluation
        eval_results = self.evaluate()
        print(f"\nFinal: Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

        self.logger.finish()
        return {"train_losses": train_losses, "val_losses": val_losses, "eval_results": eval_results}


class VLABehavioralCloning:
    """Behavioral Cloning for VLA models with image + language inputs."""

    def __init__(
        self,
        model,
        config: Optional[ILConfig] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        experiment_name: Optional[str] = None,
        dataset_name: str = "custom",
    ):
        config = config or ILConfig.behavioral_cloning()
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.bc_epochs, eta_min=config.learning_rate * 0.01)

        os.makedirs(config.output_dir, exist_ok=True)

        self.logger = ExperimentLogger(
            output_dir=config.output_dir,
            config=ExperimentConfig(
                experiment_name=experiment_name or f"vla_bc_{dataset_name}",
                project_name=wandb_project or "vla-bc-training",
                model_name=model.__class__.__name__,
                model_type="VLA",
                dataset_name=dataset_name,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                num_epochs=config.bc_epochs,
                optimizer="AdamW",
                scheduler="CosineAnnealingLR",
                device=str(self.device),
            ),
            monitor_metric="val_loss",
            monitor_mode="min",
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )
        self.logger.log_model_info(self.model)

    def train(self, train_dataloader, val_dataloader=None):
        """Train VLA model using behavioral cloning."""
        print("=" * 60)
        print("VLA Behavioral Cloning")
        print("=" * 60)

        global_step = 0

        for epoch in range(self.config.bc_epochs):
            self.model.train()
            epoch_loss, num_batches = 0, 0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.bc_epochs}"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch.get("action", batch.get("actions")),
                )

                loss = outputs["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if global_step % 10 == 0:
                    self.logger.log_step(step=global_step, metrics={"loss": loss.item(), "grad_norm": grad_norm.item(), "lr": self.optimizer.param_groups[0]["lr"]}, prefix="train")

            self.scheduler.step()
            avg_loss = epoch_loss / num_batches

            val_loss = self._validate(val_dataloader) if val_dataloader else None
            self.logger.log_epoch(epoch=epoch, train_metrics={"loss": avg_loss}, val_metrics={"loss": val_loss} if val_loss else None)

            monitor_value = val_loss if val_loss else avg_loss
            self.logger.update_best_model(self.model, monitor_value, epoch, global_step, self.optimizer, self.scheduler)

        self.logger.finish()

    def _validate(self, val_dataloader) -> float:
        self.model.eval()
        total_loss, num_batches = 0, 0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch.get("action", batch.get("actions")),
                )
                total_loss += outputs["loss"].item()
                num_batches += 1

        return total_loss / num_batches


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Behavioral Cloning Training")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--bc_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./output/bc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import gymnasium as gym
    env = gym.make(args.env)

    config = ILConfig(bc_epochs=args.bc_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, output_dir=args.output_dir)
    trainer = BehavioralCloning(env, config=config)

    # Simple CartPole expert
    def cartpole_expert(state):
        return 1 if state[2] + 0.1 * state[3] > 0 else 0

    trainer.train(expert_policy=cartpole_expert)
    print("\nTraining complete!")
