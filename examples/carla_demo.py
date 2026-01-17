#!/usr/bin/env python3
"""
CARLA Demo - Autonomous Driving VLA

This example demonstrates training and evaluating a VLA model
for autonomous driving in the CARLA simulator.

Requirements:
    1. Install CARLA: https://carla.org/
    2. pip install carla torch transformers

Usage:
    python examples/carla_demo.py --demo
    python examples/carla_demo.py --train --data_path ./carla_data
    python examples/carla_demo.py --eval --checkpoint ./checkpoints/driving_vla.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Framework imports
from model.utils import get_device, count_parameters, count_trainable_parameters
from train.utils import MetricsTracker


@dataclass
class DrivingConfig:
    """Configuration for driving VLA."""
    # Images
    image_size: Tuple[int, int] = (224, 224)
    num_cameras: int = 1  # Can be 1 (front) or 3 (surround)

    # Actions
    action_dim: int = 3  # [steering, throttle, brake]

    # Trajectory
    trajectory_length: int = 10  # Future waypoints

    # Model
    hidden_dim: int = 512
    num_layers: int = 4

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100


class MockCARLADataset(Dataset):
    """
    Mock CARLA dataset for demonstration.

    In production, use CARLADataset from train.datasets
    which loads real driving data from CARLA recordings.

    Real usage:
        from train.datasets import CARLADataset
        dataset = CARLADataset(
            data_root="/path/to/carla_data",
            cameras=["front"],
            use_lidar=True,
        )
    """

    def __init__(
        self,
        num_samples: int = 1000,
        config: DrivingConfig = None,
    ):
        self.config = config or DrivingConfig()
        self.num_samples = num_samples

        print(f"Creating mock CARLA dataset with {num_samples} samples...")

        # Mock data
        H, W = self.config.image_size
        self.images = torch.randn(num_samples, 3, H, W)
        self.actions = torch.zeros(num_samples, self.config.action_dim)
        self.actions[:, 0] = torch.randn(num_samples) * 0.3  # steering
        self.actions[:, 1] = torch.rand(num_samples) * 0.5   # throttle
        self.actions[:, 2] = torch.zeros(num_samples)        # brake

        # Mock trajectories (future waypoints in vehicle frame)
        self.trajectories = torch.zeros(num_samples, self.config.trajectory_length, 2)
        for i in range(self.config.trajectory_length):
            self.trajectories[:, i, 0] = (i + 1) * 2.0  # Forward (x)
            self.trajectories[:, i, 1] = torch.randn(num_samples) * 0.5  # Lateral (y)

        # Speed
        self.speeds = torch.rand(num_samples) * 30  # 0-30 m/s

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "action": self.actions[idx],
            "trajectory": self.trajectories[idx],
            "speed": self.speeds[idx],
            "instruction": "Drive safely following the road",
        }


class DrivingVLASimple(nn.Module):
    """
    Simple Driving VLA model for demonstration.

    For full implementation, use DrivingVLA from model.embodiment
    which includes:
    - Multi-camera fusion
    - BEV (Bird's Eye View) encoding
    - LiDAR/Radar integration
    - Trajectory prediction
    """

    def __init__(self, config: DrivingConfig):
        super().__init__()
        self.config = config

        # Vision encoder (simplified)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
        )

        # Flatten + project
        self.vision_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, config.hidden_dim),
            nn.ReLU(),
        )

        # Speed embedding
        self.speed_embed = nn.Linear(1, config.hidden_dim)

        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Action head (control)
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

        # Trajectory head (waypoints)
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.trajectory_length * 2),
        )

    def forward(
        self,
        image: torch.Tensor,
        speed: torch.Tensor,
        instruction: str = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            image: (B, 3, H, W) camera image
            speed: (B,) current speed in m/s
            instruction: Language instruction (unused in simple model)

        Returns:
            Dictionary with:
            - action: (B, 3) [steering, throttle, brake]
            - trajectory: (B, T, 2) future waypoints [x, y]
        """
        B = image.shape[0]

        # Encode vision
        vision_features = self.vision_encoder(image)
        vision_features = self.vision_proj(vision_features)  # (B, hidden_dim)

        # Encode speed
        speed_features = self.speed_embed(speed.unsqueeze(-1))  # (B, hidden_dim)

        # Combine features
        combined = vision_features + speed_features  # (B, hidden_dim)
        combined = combined.unsqueeze(1)  # (B, 1, hidden_dim)

        # Transformer
        features = self.transformer(combined)  # (B, 1, hidden_dim)
        features = features.squeeze(1)  # (B, hidden_dim)

        # Predict action
        action = self.action_head(features)
        action = torch.tanh(action)  # Normalize to [-1, 1]

        # Scale actions
        action[:, 0] = action[:, 0] * 0.5  # steering: [-0.5, 0.5]
        action[:, 1] = (action[:, 1] + 1) / 2  # throttle: [0, 1]
        action[:, 2] = F.relu(action[:, 2])  # brake: [0, 1]

        # Predict trajectory
        traj_flat = self.trajectory_head(features)
        trajectory = traj_flat.view(B, self.config.trajectory_length, 2)

        return {
            "action": action,
            "trajectory": trajectory,
        }

    @torch.no_grad()
    def get_action(
        self,
        image: torch.Tensor,
        speed: float,
        instruction: str = None,
    ) -> np.ndarray:
        """Get action for deployment."""
        self.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        speed_tensor = torch.tensor([speed], device=image.device)
        outputs = self.forward(image, speed_tensor, instruction)

        return outputs["action"].squeeze(0).cpu().numpy()


class DrivingTrainer:
    """Trainer for Driving VLA."""

    def __init__(
        self,
        model: DrivingVLASimple,
        config: DrivingConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
        )

        self.metrics = MetricsTracker()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_action_loss = 0
        total_traj_loss = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            actions = batch["action"].to(self.device)
            trajectories = batch["trajectory"].to(self.device)
            speeds = batch["speed"].to(self.device)

            # Forward
            outputs = self.model(images, speeds)

            # Compute losses
            action_loss = F.mse_loss(outputs["action"], actions)
            traj_loss = F.mse_loss(outputs["trajectory"], trajectories)
            loss = action_loss + 0.5 * traj_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_action_loss += action_loss.item()
            total_traj_loss += traj_loss.item()

        n_batches = len(dataloader)
        return {
            "action_loss": total_action_loss / n_batches,
            "traj_loss": total_traj_loss / n_batches,
        }

    def train(
        self,
        dataset: Dataset,
        save_dir: str = "./checkpoints",
    ):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("Training Driving VLA")
        print("=" * 60)

        os.makedirs(save_dir, exist_ok=True)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        best_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            metrics = self.train_epoch(dataloader)
            self.scheduler.step()

            total_loss = metrics["action_loss"] + metrics["traj_loss"]
            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Action Loss: {metrics['action_loss']:.4f} | "
                f"Traj Loss: {metrics['traj_loss']:.4f}"
            )

            # Save best
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                    "epoch": epoch,
                }, os.path.join(save_dir, "driving_vla_best.pt"))

        # Save final
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }, os.path.join(save_dir, "driving_vla_final.pt"))

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Loss: {best_loss:.4f}")
        print(f"Saved to: {save_dir}")
        print("=" * 60)


def run_demo():
    """Run a simple demo showing the driving VLA."""
    print("\n" + "=" * 60)
    print("CARLA Driving VLA Demo")
    print("=" * 60)

    config = DrivingConfig()
    device = get_device("auto")

    # Create model
    model = DrivingVLASimple(config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Create sample input
    sample_image = torch.randn(3, 224, 224).to(device)
    current_speed = 15.0  # m/s

    # Get prediction
    action = model.get_action(sample_image, current_speed, "Drive to the destination")

    print(f"\nDemo Prediction:")
    print(f"  Input: Front camera image (224x224)")
    print(f"  Speed: {current_speed:.1f} m/s")
    print(f"  Predicted Action:")
    print(f"    Steering: {action[0]:.4f} (range: -0.5 to 0.5)")
    print(f"    Throttle: {action[1]:.4f} (range: 0 to 1)")
    print(f"    Brake: {action[2]:.4f} (range: 0 to 1)")
    print("=" * 60)

    # Show full output
    print("\nFull Model Output:")
    model.eval()
    with torch.no_grad():
        image = sample_image.unsqueeze(0)
        speed = torch.tensor([current_speed], device=device)
        outputs = model(image, speed)

    print(f"  Action shape: {outputs['action'].shape}")
    print(f"  Trajectory shape: {outputs['trajectory'].shape}")
    print(f"  Trajectory (first 3 waypoints):")
    traj = outputs['trajectory'][0].cpu().numpy()
    for i in range(min(3, len(traj))):
        print(f"    Waypoint {i+1}: x={traj[i, 0]:.2f}m, y={traj[i, 1]:.2f}m")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CARLA Driving VLA Demo")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--eval", action="store_true", help="Evaluate model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--data_path", type=str, default=None, help="Data path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--device", type=str, default="auto", help="Device")

    args = parser.parse_args()

    # Default to demo
    if not (args.demo or args.train or args.eval):
        args.demo = True

    if args.demo:
        run_demo()
        return

    # Setup
    config = DrivingConfig(num_epochs=args.epochs)
    device = get_device(args.device)

    model = DrivingVLASimple(config).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")

    if args.train:
        dataset = MockCARLADataset(num_samples=args.samples, config=config)
        trainer = DrivingTrainer(model, config, device)
        trainer.train(dataset)

    if args.eval:
        print("\n" + "=" * 60)
        print("Evaluating Driving VLA")
        print("=" * 60)

        dataset = MockCARLADataset(num_samples=100, config=config)
        dataloader = DataLoader(dataset, batch_size=32)

        model.eval()
        total_action_error = 0
        total_traj_error = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                actions = batch["action"].to(device)
                trajectories = batch["trajectory"].to(device)
                speeds = batch["speed"].to(device)

                outputs = model(images, speeds)

                total_action_error += F.mse_loss(outputs["action"], actions).item()
                total_traj_error += F.mse_loss(outputs["trajectory"], trajectories).item()

        n_batches = len(dataloader)
        print(f"Action MSE: {total_action_error / n_batches:.4f}")
        print(f"Trajectory MSE: {total_traj_error / n_batches:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
