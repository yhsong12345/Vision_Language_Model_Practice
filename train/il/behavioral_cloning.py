"""
Behavioral Cloning (BC)

The simplest form of imitation learning:
- Collect expert demonstrations
- Train policy via supervised learning
- No environment interaction during training
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

from .base_trainer import ILTrainer, ExpertDataset, PolicyNetwork

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.training_config import ILConfig


class BehavioralCloning(ILTrainer):
    """
    Behavioral Cloning trainer.

    Simple supervised learning approach:
    1. Collect expert demonstrations (state, action) pairs
    2. Train policy to predict actions from states
    3. Uses MSE loss for continuous, CE for discrete actions

    Pros:
    - Simple and easy to implement
    - No environment interaction during training
    - Works well with high-quality demonstrations

    Cons:
    - Distribution shift (covariate shift) problem
    - Requires large amounts of expert data
    - Cannot improve beyond expert performance
    """

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        config: Optional[ILConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = ILConfig.behavioral_cloning()

        super().__init__(env, policy, config.output_dir, **kwargs)

        self.config = config

        # Training params
        self.num_epochs = config.bc_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.val_split = config.bc_validation_split

        # Optimizer
        self.optimizer = Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
        )

        # Loss function
        if self.continuous:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        expert_policy=None,
        num_expert_episodes: int = None,
    ):
        """
        Train policy using behavioral cloning.

        Args:
            states: Expert states (optional if expert_policy provided)
            actions: Expert actions (optional if expert_policy provided)
            expert_policy: Expert policy function for collecting demonstrations
            num_expert_episodes: Number of episodes to collect
        """
        print("=" * 60)
        print("Behavioral Cloning Training")
        print("=" * 60)

        # Collect demonstrations if not provided
        if states is None or actions is None:
            if expert_policy is None:
                raise ValueError("Must provide either (states, actions) or expert_policy")

            if num_expert_episodes is None:
                num_expert_episodes = self.config.num_expert_episodes

            states, actions = self.collect_expert_demonstrations(
                expert_policy, num_expert_episodes
            )

        # Create dataset
        dataset = ExpertDataset(states, actions)

        # Split into train/val
        train_size = int(len(dataset) * (1 - self.val_split))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")

        # Training loop
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            # Training
            self.policy.train()
            epoch_train_loss = 0
            num_batches = 0

            for states_batch, actions_batch in train_loader:
                states_batch = states_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)

                # Forward pass
                predicted_actions = self.policy(states_batch)

                # Compute loss
                if self.continuous:
                    loss = self.criterion(predicted_actions, actions_batch)
                else:
                    loss = self.criterion(predicted_actions, actions_batch.long())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)

            # Validation
            self.policy.eval()
            epoch_val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for states_batch, actions_batch in val_loader:
                    states_batch = states_batch.to(self.device)
                    actions_batch = actions_batch.to(self.device)

                    predicted_actions = self.policy(states_batch)

                    if self.continuous:
                        loss = self.criterion(predicted_actions, actions_batch)
                    else:
                        loss = self.criterion(predicted_actions, actions_batch.long())

                    epoch_val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = epoch_val_loss / num_val_batches
            val_losses.append(avg_val_loss)

            # Logging
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save(os.path.join(self.config.output_dir, "best_policy.pt"))

        # Final evaluation
        print("\nFinal Evaluation:")
        eval_results = self.evaluate()
        print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

        # Save final model
        self.save()

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "eval_results": eval_results,
        }


class VLABehavioralCloning:
    """
    Behavioral Cloning for VLA models.

    Uses the robot manipulation dataset format with:
    - Images
    - Language instructions
    - Actions
    """

    def __init__(
        self,
        model,
        config: Optional[ILConfig] = None,
    ):
        if config is None:
            config = ILConfig.behavioral_cloning()

        self.model = model
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        os.makedirs(config.output_dir, exist_ok=True)

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
    ):
        """
        Train VLA model using behavioral cloning.

        Args:
            train_dataloader: DataLoader with robot manipulation data
            val_dataloader: Optional validation DataLoader
        """
        print("=" * 60)
        print("VLA Behavioral Cloning")
        print("=" * 60)

        best_val_loss = float("inf")

        for epoch in range(self.config.bc_epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.bc_epochs}",
            )

            for batch in progress_bar:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch["action"],
                )

                loss = outputs["loss"]

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

            # Validation
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                print(f"Validation Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save("best_model.pt")

        # Save final model
        self._save("final_model.pt")

    def _validate(self, val_dataloader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch["action"],
                )

                total_loss += outputs["loss"].item()
                num_batches += 1

        return total_loss / num_batches

    def _save(self, filename: str):
        """Save model."""
        path = os.path.join(self.config.output_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Saved model to {path}")


if __name__ == "__main__":
    print("Behavioral Cloning")
    print("Simple supervised imitation learning")
