#!/usr/bin/env python3
"""
PushT Demo - Simple 2D Manipulation

This example demonstrates training a VLA model on the PushT task
from the LeRobot dataset. The task is to push a T-shaped block
to a target location.

Requirements:
    pip install lerobot torch transformers

Usage:
    python examples/pusht_demo.py --train
    python examples/pusht_demo.py --eval --checkpoint ./checkpoints/pusht_vla.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Framework imports
from model import create_vla_model, MLPActionHead
from model.utils import get_device, count_parameters, count_trainable_parameters
from train.utils import MetricsTracker


class MockPushTDataset(Dataset):
    """
    Mock PushT dataset for demonstration.
    In production, use LeRobotDataset from train.datasets.

    Real usage:
        from train.datasets import LeRobotDataset
        dataset = LeRobotDataset(repo_id="lerobot/pusht", split="train")
    """

    def __init__(self, num_samples: int = 1000, image_size: int = 224):
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = 2  # [dx, dy]

        # Generate mock data
        print(f"Creating mock PushT dataset with {num_samples} samples...")
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.actions = torch.randn(num_samples, self.action_dim) * 0.1
        self.instructions = ["Push the T-block to the target"] * num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "action": self.actions[idx],
            "instruction": self.instructions[idx],
        }


def create_model(device: str = "auto"):
    """Create VLA model for PushT task."""
    print("\n" + "=" * 60)
    print("Creating VLA Model for PushT")
    print("=" * 60)

    # For demo, use a simple mock model
    # In production, use create_vla_model()
    model = SimplePushTModel(
        image_size=224,
        action_dim=2,
        hidden_dim=256,
    )

    device = get_device(device)
    model = model.to(device)

    print(f"\nModel Summary:")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"  Device: {device}")

    return model, device


class SimplePushTModel(nn.Module):
    """
    Simple CNN-based model for PushT demonstration.

    For full VLA capability, use:
        model = create_vla_model(
            vision_encoder="siglip-base",
            llm="qwen2-0.5b",
            action_dim=2,
        )
    """

    def __init__(self, image_size: int = 224, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.action_head = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, image: torch.Tensor, instruction: str = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image: (B, 3, H, W) input images
            instruction: Language instruction (unused in simple model)

        Returns:
            actions: (B, action_dim) predicted actions
        """
        features = self.encoder(image)
        actions = self.action_head(features)
        return actions

    def get_action(self, image: torch.Tensor, instruction: str = None) -> torch.Tensor:
        """Get action for single observation."""
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            action = self.forward(image, instruction)
        return action.squeeze(0)


def train(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    save_dir: str = "./checkpoints",
):
    """Train the model with behavioral cloning."""
    print("\n" + "=" * 60)
    print("Training VLA Model on PushT")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # Setup
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    metrics = MetricsTracker()

    # Training loop
    model.train()
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            actions = batch["action"].to(device)

            # Forward pass
            predicted_actions = model(images)
            loss = criterion(predicted_actions, actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        metrics.add("train_loss", avg_loss, epoch)

        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, "pusht_vla_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, save_path)
            print(f"  Saved best model to {save_path}")

    # Save final model
    final_path = os.path.join(save_dir, "pusht_vla_final.pt")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
    }, final_path)

    print("\n" + "=" * 60)
    print(f"Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Saved to: {save_dir}")
    print("=" * 60)

    return model


def evaluate(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_samples: int = 100,
):
    """Evaluate the model."""
    print("\n" + "=" * 60)
    print("Evaluating VLA Model on PushT")
    print("=" * 60)

    model.eval()

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    total_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            actions = batch["action"].to(device)

            predicted_actions = model(images)
            mse = ((predicted_actions - actions) ** 2).mean()
            total_mse += mse.item()
            num_batches += 1

            if num_batches * 32 >= num_samples:
                break

    avg_mse = total_mse / num_batches

    print(f"\nEvaluation Results:")
    print(f"  MSE: {avg_mse:.4f}")
    print(f"  RMSE: {np.sqrt(avg_mse):.4f}")
    print("=" * 60)

    return {"mse": avg_mse, "rmse": np.sqrt(avg_mse)}


def run_demo():
    """Run a simple demo showing the model in action."""
    print("\n" + "=" * 60)
    print("PushT VLA Demo")
    print("=" * 60)

    # Create model
    model, device = create_model()

    # Create sample observation
    sample_image = torch.randn(3, 224, 224).to(device)
    instruction = "Push the T-block to the target"

    # Get action
    action = model.get_action(sample_image, instruction)

    print(f"\nDemo:")
    print(f"  Input: RGB image (224x224)")
    print(f"  Instruction: '{instruction}'")
    print(f"  Output action: [{action[0].item():.4f}, {action[1].item():.4f}]")
    print(f"  (Represents: [dx, dy] movement)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PushT VLA Demo")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")

    args = parser.parse_args()

    # Default to demo if no action specified
    if not (args.train or args.eval or args.demo):
        args.demo = True

    # Create dataset
    dataset = MockPushTDataset(num_samples=args.samples)

    # Create model
    model, device = create_model(args.device)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded successfully!")

    if args.train:
        model = train(
            model=model,
            dataset=dataset,
            device=device,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    if args.eval:
        evaluate(
            model=model,
            dataset=dataset,
            device=device,
        )

    if args.demo:
        run_demo()


if __name__ == "__main__":
    main()
