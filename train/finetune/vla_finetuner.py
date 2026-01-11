"""
VLA Fine-tuner

Supervised fine-tuning of VLA models on robot manipulation datasets.
Supports various fine-tuning strategies:
- Projector + Action Head only
- Full fine-tuning
- LoRA fine-tuning
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from typing import Optional, Dict, Any
import json
import wandb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.training_config import FineTuningConfig
from .dataset import RobotDataset, create_robot_dataloader


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

        # Initialize wandb
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=vars(config),
            )

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

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            trainable_params,
                            self.config.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]

                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr:.2e}",
                    })

                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                            "train/step": global_step,
                        })

                # Evaluation
                if val_loader is not None and global_step % self.config.eval_steps == 0:
                    val_loss = self._evaluate(val_loader)

                    if self.accelerator.is_main_process:
                        print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}")

                        if self.config.use_wandb:
                            wandb.log({
                                "val/loss": val_loss,
                                "val/step": global_step,
                            })

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

        # Save final model
        self._save_checkpoint("final")

        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.finish()

    def _evaluate(self, val_loader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

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

        return total_loss / num_batches

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
                torch.save({
                    "vision_projector": unwrapped_model.vision_projector.state_dict(),
                    "action_head": unwrapped_model.action_head.state_dict(),
                }, os.path.join(save_path, "components.pt"))
            else:
                # Save full model
                unwrapped_model.save_pretrained(os.path.join(save_path, "model.pt"))

            # Save config
            with open(os.path.join(save_path, "config.json"), "w") as f:
                config_dict = vars(self.config)
                config_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                               for k, v in config_dict.items()}
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
    print("VLA Fine-tuner")
    print("Usage: See train/finetune/README.md for examples")
