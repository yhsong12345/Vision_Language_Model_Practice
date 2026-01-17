"""
VLA Fine-tuner

Supervised fine-tuning of VLA models on robot manipulation datasets.
Supports various fine-tuning strategies:
- Projector + Action Head only
- Full fine-tuning
- LoRA fine-tuning

Enhanced monitoring includes:
- Gradient health (norm, clipping frequency)
- Per-dimension action metrics
- Action prediction statistics
- Learning dynamics tracking
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from typing import Optional, Dict, Any, List
from collections import deque
import json
import wandb
import numpy as np

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.training_config import FineTuningConfig
from .dataset import RobotDataset, create_robot_dataloader


class MetricsTracker:
    """
    Tracks and computes training metrics for VLA fine-tuning.

    Monitors:
    - Loss metrics (total, action, language)
    - Gradient health (norm, clipping frequency)
    - Per-dimension action accuracy
    - Action prediction statistics
    - Learning dynamics (smoothed losses, overfitting detection)
    """

    def __init__(self, action_dim: int, window_size: int = 100):
        self.action_dim = action_dim
        self.window_size = window_size

        # Running windows for smoothed metrics
        self.loss_window = deque(maxlen=window_size)
        self.grad_norm_window = deque(maxlen=window_size)

        # Gradient clipping stats
        self.total_steps = 0
        self.clipped_steps = 0

        # Per-dimension tracking
        self.dim_errors = [deque(maxlen=window_size) for _ in range(action_dim)]

        # Action statistics
        self.pred_actions = deque(maxlen=window_size * 32)  # Store more for statistics
        self.gt_actions = deque(maxlen=window_size * 32)

        # Validation tracking for overfitting detection
        self.train_losses = []
        self.val_losses = []

    def update_loss(self, loss: float):
        """Update loss tracking."""
        self.loss_window.append(loss)

    def update_gradient_stats(
        self, model_parameters, max_grad_norm: float, was_clipped: bool = False
    ) -> Dict[str, float]:
        """
        Compute and track gradient statistics.

        Returns dict with gradient metrics.
        """
        self.total_steps += 1

        # Compute gradient norm
        total_norm = 0.0
        for p in model_parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        self.grad_norm_window.append(total_norm)

        # Track clipping
        if was_clipped or total_norm > max_grad_norm:
            self.clipped_steps += 1

        return {
            "grad_norm": total_norm,
            "grad_norm_avg": (
                np.mean(self.grad_norm_window) if self.grad_norm_window else 0
            ),
            "grad_clip_ratio": self.clipped_steps / max(1, self.total_steps),
        }

    def update_action_metrics(
        self, pred_actions: torch.Tensor, gt_actions: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute per-dimension and overall action metrics.

        Args:
            pred_actions: Predicted actions [B, action_dim] or [B, chunk_size, action_dim]
            gt_actions: Ground truth actions [B, action_dim] or [B, chunk_size, action_dim]

        Returns dict with action metrics.
        """
        # Flatten if chunked
        if pred_actions.dim() == 3:
            pred_actions = pred_actions.reshape(-1, pred_actions.size(-1))
            gt_actions = gt_actions.reshape(-1, gt_actions.size(-1))

        pred_np = pred_actions.detach().cpu().numpy()
        gt_np = gt_actions.detach().cpu().numpy()

        # Store for statistics
        for p, g in zip(pred_np, gt_np):
            self.pred_actions.append(p)
            self.gt_actions.append(g)

        # Per-dimension MSE
        dim_mse = {}
        for d in range(min(self.action_dim, pred_actions.size(-1))):
            dim_error = F.mse_loss(pred_actions[:, d], gt_actions[:, d]).item()
            self.dim_errors[d].append(dim_error)
            dim_mse[f"action_mse_dim_{d}"] = dim_error

        # Overall metrics
        mse = F.mse_loss(pred_actions, gt_actions).item()
        mae = F.l1_loss(pred_actions, gt_actions).item()

        # Action prediction statistics
        pred_mean = pred_actions.mean().item()
        pred_std = pred_actions.std().item()
        gt_mean = gt_actions.mean().item()
        gt_std = gt_actions.std().item()

        metrics = {
            "action_mse": mse,
            "action_mae": mae,
            "action_rmse": np.sqrt(mse),
            "pred_action_mean": pred_mean,
            "pred_action_std": pred_std,
            "gt_action_mean": gt_mean,
            "gt_action_std": gt_std,
            "action_mean_diff": abs(pred_mean - gt_mean),
            "action_std_ratio": pred_std / max(gt_std, 1e-6),
            **dim_mse,
        }

        return metrics

    def get_smoothed_loss(self) -> float:
        """Get exponentially smoothed loss."""
        if not self.loss_window:
            return 0.0
        return np.mean(self.loss_window)

    def get_per_dim_summary(self) -> Dict[str, float]:
        """Get summary of per-dimension errors."""
        summary = {}
        for d in range(self.action_dim):
            if self.dim_errors[d]:
                summary[f"action_mse_dim_{d}_avg"] = np.mean(self.dim_errors[d])
        return summary

    def check_overfitting(self, train_loss: float, val_loss: float) -> Dict[str, Any]:
        """
        Check for overfitting by comparing train/val loss trends.

        Returns overfitting indicators.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        gap = val_loss - train_loss
        gap_ratio = gap / max(train_loss, 1e-6)

        # Check if gap is increasing (sign of overfitting)
        is_overfitting = False
        gap_trend = 0.0

        if len(self.val_losses) >= 3:
            recent_gaps = [
                self.val_losses[i] - self.train_losses[i] for i in range(-3, 0)
            ]
            gap_trend = recent_gaps[-1] - recent_gaps[0]
            is_overfitting = gap_trend > 0 and gap_ratio > 0.1

        return {
            "train_val_gap": gap,
            "train_val_gap_ratio": gap_ratio,
            "gap_trend": gap_trend,
            "is_overfitting": is_overfitting,
        }

    def get_action_distribution_stats(self) -> Dict[str, float]:
        """Get action distribution statistics from accumulated predictions."""
        if len(self.pred_actions) < 10:
            return {}

        pred_arr = np.array(list(self.pred_actions))
        gt_arr = np.array(list(self.gt_actions))

        stats = {}
        for d in range(min(self.action_dim, pred_arr.shape[-1])):
            stats[f"pred_dim_{d}_mean"] = pred_arr[:, d].mean()
            stats[f"pred_dim_{d}_std"] = pred_arr[:, d].std()
            stats[f"gt_dim_{d}_mean"] = gt_arr[:, d].mean()
            stats[f"gt_dim_{d}_std"] = gt_arr[:, d].std()

        return stats


class VLAFineTuner:
    """
    Fine-tuner for Vision-Language-Action models.

    Supports:
    - Supervised learning on robot manipulation data
    - Multiple fine-tuning strategies
    - Distributed training with accelerate
    - Mixed precision training
    """

    def __init__(
        self,
        model,
        config: FineTuningConfig,
    ):
        self.model = model
        self.config = config

        os.makedirs(config.output_dir, exist_ok=True)

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Setup model
        self._setup_model()

        # Initialize metrics tracker
        action_dim = getattr(model, "action_dim", 7)
        self.metrics_tracker = MetricsTracker(action_dim=action_dim)

        # Initialize wandb with enhanced config
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config={
                    **vars(config),
                    "action_dim": action_dim,
                    "model_type": type(model).__name__,
                },
            )
            # Define custom wandb metrics for better visualization
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("val/*", step_metric="train/step")
            wandb.define_metric("gradient/*", step_metric="train/step")
            wandb.define_metric("action/*", step_metric="train/step")

    def _setup_model(self):
        """Setup model for fine-tuning based on config."""
        # Freeze vision encoder
        if self.config.freeze_vision:
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False
            print("Vision encoder frozen")

        # Freeze/unfreeze LLM
        if self.config.freeze_llm:
            for param in self.model.llm.parameters():
                param.requires_grad = False
            print("LLM frozen")
        else:
            for param in self.model.llm.parameters():
                param.requires_grad = True

        # Apply LoRA if requested
        if self.config.use_lora:
            self._apply_lora()

        # Vision projector and action head are always trainable
        for param in self.model.vision_projector.parameters():
            param.requires_grad = True
        for param in self.model.action_head.parameters():
            param.requires_grad = True

        # Print parameter summary
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def _apply_lora(self):
        """Apply LoRA to the LLM."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )

            self.model.llm = get_peft_model(self.model.llm, lora_config)
            print("LoRA applied to LLM")
            self.model.llm.print_trainable_parameters()

        except ImportError:
            print("PEFT not installed, skipping LoRA")
            self.config.use_lora = False

    def train(
        self,
        train_dataset,
        val_dataset=None,
    ):
        """
        Run fine-tuning.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
        """
        print("=" * 60)
        print("VLA Fine-tuning")
        print("=" * 60)

        # Create dataloaders
        train_loader = create_robot_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = create_robot_dataloader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs
        if self.config.warmup_steps > 0:
            num_warmup_steps = self.config.warmup_steps
        else:
            num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare for distributed training
        self.model, train_loader, optimizer, scheduler = self.accelerator.prepare(
            self.model, train_loader, optimizer, scheduler
        )
        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)

        # Training loop
        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(
                        pixel_values=batch["pixel_values"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        actions=batch["action"],
                    )
                    loss = outputs["loss"]

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Track gradient stats before clipping
                    grad_metrics = {}
                    if self.accelerator.sync_gradients:
                        # Compute gradient norm before clipping
                        grad_metrics = self.metrics_tracker.update_gradient_stats(
                            trainable_params,
                            self.config.max_grad_norm,
                        )
                        self.accelerator.clip_grad_norm_(
                            trainable_params,
                            self.config.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Update metrics tracker
                loss_value = loss.item()
                self.metrics_tracker.update_loss(loss_value)
                epoch_loss += loss_value
                num_batches += 1
                global_step += 1

                # Compute action metrics if predictions available
                action_metrics = {}
                if "predicted_actions" in outputs and "action" in batch:
                    action_metrics = self.metrics_tracker.update_action_metrics(
                        outputs["predicted_actions"],
                        batch["action"],
                    )

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    smoothed_loss = self.metrics_tracker.get_smoothed_loss()
                    lr = scheduler.get_last_lr()[0]

                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss_value:.4f}",
                            "smooth": f"{smoothed_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "grad": f"{grad_metrics.get('grad_norm', 0):.2f}",
                        }
                    )

                    if self.config.use_wandb and self.accelerator.is_main_process:
                        # Core training metrics
                        log_dict = {
                            "train/loss": loss_value,
                            "train/loss_smoothed": smoothed_loss,
                            "train/loss_avg": avg_loss,
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                            "train/step": global_step,
                        }

                        # Gradient health metrics
                        if grad_metrics:
                            log_dict.update(
                                {
                                    "gradient/norm": grad_metrics["grad_norm"],
                                    "gradient/norm_avg": grad_metrics["grad_norm_avg"],
                                    "gradient/clip_ratio": grad_metrics[
                                        "grad_clip_ratio"
                                    ],
                                }
                            )

                        # Action prediction metrics
                        if action_metrics:
                            log_dict.update(
                                {
                                    "action/mse": action_metrics["action_mse"],
                                    "action/mae": action_metrics["action_mae"],
                                    "action/rmse": action_metrics["action_rmse"],
                                    "action/pred_mean": action_metrics[
                                        "pred_action_mean"
                                    ],
                                    "action/pred_std": action_metrics[
                                        "pred_action_std"
                                    ],
                                    "action/gt_mean": action_metrics["gt_action_mean"],
                                    "action/gt_std": action_metrics["gt_action_std"],
                                    "action/mean_diff": action_metrics[
                                        "action_mean_diff"
                                    ],
                                    "action/std_ratio": action_metrics[
                                        "action_std_ratio"
                                    ],
                                }
                            )
                            # Per-dimension MSE
                            for key, value in action_metrics.items():
                                if key.startswith("action_mse_dim_"):
                                    dim = key.split("_")[-1]
                                    log_dict[f"action/mse_dim_{dim}"] = value

                        wandb.log(log_dict)

                # Evaluation
                if val_loader is not None and global_step % self.config.eval_steps == 0:
                    val_metrics = self._evaluate(val_loader)
                    val_loss = val_metrics["loss"]

                    # Check for overfitting
                    train_loss_current = self.metrics_tracker.get_smoothed_loss()
                    overfit_metrics = self.metrics_tracker.check_overfitting(
                        train_loss_current, val_loss
                    )

                    if self.accelerator.is_main_process:
                        print(
                            f"\nStep {global_step} - Val Loss: {val_loss:.4f} "
                            f"| Action MSE: {val_metrics.get('action_mse', 0):.4f} "
                            f"| Gap: {overfit_metrics['train_val_gap']:.4f}"
                        )

                        if overfit_metrics["is_overfitting"]:
                            print("  WARNING: Potential overfitting detected!")

                        if self.config.use_wandb:
                            val_log = {
                                "val/loss": val_loss,
                                "val/action_mse": val_metrics.get("action_mse", 0),
                                "val/action_mae": val_metrics.get("action_mae", 0),
                                "val/action_rmse": val_metrics.get("action_rmse", 0),
                                "train/step": global_step,
                                # Overfitting detection
                                "overfit/train_val_gap": overfit_metrics[
                                    "train_val_gap"
                                ],
                                "overfit/gap_ratio": overfit_metrics[
                                    "train_val_gap_ratio"
                                ],
                                "overfit/gap_trend": overfit_metrics["gap_trend"],
                                "overfit/is_overfitting": int(
                                    overfit_metrics["is_overfitting"]
                                ),
                            }
                            # Per-dimension validation MSE
                            for key, value in val_metrics.items():
                                if key.startswith("action_mse_dim_"):
                                    dim = key.split("_")[-1]
                                    val_log[f"val/action_mse_dim_{dim}"] = value

                            wandb.log(val_log)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint("best")

                    self.model.train()

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{global_step}")

            # Epoch summary
            avg_epoch_loss = epoch_loss / num_batches
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")

                # Log epoch-level summary
                if self.config.use_wandb:
                    epoch_log = {
                        "epoch/loss": avg_epoch_loss,
                        "epoch/epoch": epoch + 1,
                        "train/step": global_step,
                    }

                    # Add per-dimension summary
                    dim_summary = self.metrics_tracker.get_per_dim_summary()
                    for key, value in dim_summary.items():
                        epoch_log[f"epoch/{key}"] = value

                    # Add action distribution stats
                    dist_stats = self.metrics_tracker.get_action_distribution_stats()
                    for key, value in dist_stats.items():
                        epoch_log[f"epoch/{key}"] = value

                    wandb.log(epoch_log)

        # Save final model
        self._save_checkpoint("final")

        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.finish()

    def _evaluate(self, val_loader) -> Dict[str, float]:
        """
        Evaluate on validation set with comprehensive metrics.

        Returns:
            Dict with loss and action metrics.
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Accumulators for action metrics
        all_pred_actions = []
        all_gt_actions = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch["action"],
                )
                total_loss += outputs["loss"].item()
                num_batches += 1

                # Collect action predictions
                if "predicted_actions" in outputs:
                    all_pred_actions.append(outputs["predicted_actions"].cpu())
                    all_gt_actions.append(batch["action"].cpu())

        avg_loss = total_loss / num_batches
        metrics = {"loss": avg_loss}

        # Compute action metrics if available
        if all_pred_actions:
            pred_actions = torch.cat(all_pred_actions, dim=0)
            gt_actions = torch.cat(all_gt_actions, dim=0)

            # Flatten if chunked
            if pred_actions.dim() == 3:
                pred_actions = pred_actions.reshape(-1, pred_actions.size(-1))
                gt_actions = gt_actions.reshape(-1, gt_actions.size(-1))

            # Overall metrics
            mse = F.mse_loss(pred_actions, gt_actions).item()
            mae = F.l1_loss(pred_actions, gt_actions).item()

            metrics.update(
                {
                    "action_mse": mse,
                    "action_mae": mae,
                    "action_rmse": np.sqrt(mse),
                }
            )

            # Per-dimension MSE
            action_dim = pred_actions.size(-1)
            for d in range(action_dim):
                dim_mse = F.mse_loss(pred_actions[:, d], gt_actions[:, d]).item()
                metrics[f"action_mse_dim_{d}"] = dim_mse

        return metrics

    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.config.output_dir, name)
            os.makedirs(save_path, exist_ok=True)

            unwrapped_model = self.accelerator.unwrap_model(self.model)

            # Save full model or just trainable parts
            if self.config.use_lora:
                # Save LoRA weights
                unwrapped_model.llm.save_pretrained(os.path.join(save_path, "lora"))

                # Save other trainable components
                torch.save(
                    {
                        "vision_projector": unwrapped_model.vision_projector.state_dict(),
                        "action_head": unwrapped_model.action_head.state_dict(),
                    },
                    os.path.join(save_path, "components.pt"),
                )
            else:
                # Save full model
                unwrapped_model.save_pretrained(os.path.join(save_path, "model.pt"))

            # Save config
            with open(os.path.join(save_path, "config.json"), "w") as f:
                config_dict = vars(self.config)
                config_dict = {
                    k: (
                        str(v)
                        if not isinstance(v, (int, float, bool, str, type(None)))
                        else v
                    )
                    for k, v in config_dict.items()
                }
                json.dump(config_dict, f, indent=2)

            print(f"Saved checkpoint to {save_path}")


def finetune_vla(
    model,
    dataset_name: str = "lerobot/pusht",
    output_dir: str = "./finetuned_vla",
    **kwargs,
):
    """
    Convenience function for fine-tuning.

    Args:
        model: VLA model to fine-tune
        dataset_name: Name of the robot dataset
        output_dir: Output directory
        **kwargs: Additional config arguments
    """
    from config.training_config import FineTuningConfig

    config = FineTuningConfig(
        output_dir=output_dir,
        dataset_name=dataset_name,
        **kwargs,
    )

    # Create dataset
    train_dataset = RobotDataset(
        dataset_name=dataset_name,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
    )

    # Create trainer and train
    trainer = VLAFineTuner(model, config)
    trainer.train(train_dataset)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VLA Fine-tuner - Supervised fine-tuning of VLA models on robot manipulation datasets"
    )

    # Model arguments
    parser.add_argument(
        "--vision-model",
        type=str,
        default="google/siglip-base-patch16-224",
        help="Vision encoder model name (default: google/siglip-base-patch16-224)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="LLM model name (default: Qwen/Qwen2-1.5B-Instruct)",
    )
    parser.add_argument(
        "--pretrained-vlm",
        type=str,
        default=None,
        help="Path to pretrained VLM checkpoint (optional)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action dimension (default: 7)",
    )
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=1,
        help="Action chunk size for temporal consistency (default: 1)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="lerobot/pusht",
        help="Robot dataset name (default: lerobot/pusht)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./finetuned_vla",
        help="Output directory for checkpoints (default: ./finetuned_vla)",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm (default: 1.0)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )

    # Freezing options
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        help="Freeze vision encoder",
    )
    parser.add_argument(
        "--freeze-llm",
        action="store_true",
        help="Freeze LLM backbone",
    )

    # LoRA arguments
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for efficient fine-tuning",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha (default: 64)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)",
    )

    # Hardware arguments
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Mixed precision mode (default: bf16)",
    )

    # Logging arguments
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency (default: 10)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Checkpoint save frequency (default: 500)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluation frequency (default: 500)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vla-finetuning",
        help="W&B project name (default: vla-finetuning)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="vla_finetune",
        help="Experiment name (default: vla_finetune)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("VLA Fine-tuner")
    print("=" * 60)
    print(f"Vision Model: {args.vision_model}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Action Dim: {args.action_dim}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Freeze Vision: {args.freeze_vision}")
    print(f"Freeze LLM: {args.freeze_llm}")
    print(f"Use LoRA: {args.use_lora}")
    print("=" * 60)

    # Create VLA model
    from model.vla import VLAModel

    if args.pretrained_vlm:
        print(f"Loading pretrained VLM from {args.pretrained_vlm}")
        model = VLAModel.from_pretrained_vlm(
            args.pretrained_vlm,
            action_dim=args.action_dim,
            action_chunk_size=args.action_chunk_size,
        )
    else:
        model = VLAModel(
            vision_model_name=args.vision_model,
            llm_model_name=args.llm_model,
            action_dim=args.action_dim,
            action_chunk_size=args.action_chunk_size,
            freeze_vision=args.freeze_vision,
            freeze_llm=args.freeze_llm,
        )

    # Run fine-tuning
    finetune_vla(
        model=model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        freeze_vision=args.freeze_vision,
        freeze_llm=args.freeze_llm,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mixed_precision=args.mixed_precision,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        experiment_name=args.experiment_name,
    )
