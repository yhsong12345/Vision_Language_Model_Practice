"""
VLM Pretrainer

Main class for Vision-Language Model pretraining following LLaVA paradigm:
1. Stage 1 (Alignment): Train projector to align vision features with LLM
2. Stage 2 (Instruction): Fine-tune LLM on visual instruction data
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model.vla import VLAModel
from config.training_config import PretrainingConfig


class PretrainingDataset(Dataset):
    """Dataset for VLM pretraining."""

    def __init__(
        self,
        dataset_name: str,
        image_processor,
        tokenizer,
        max_length: int = 2048,
        split: str = "train",
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        try:
            from datasets import load_dataset
            self.dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"Could not load dataset: {e}")
            self.dataset = None

    def __len__(self):
        return len(self.dataset) if self.dataset else 0

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image
        image = item.get("image", item.get("images", None))
        if image is not None:
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt",
            ).pixel_values.squeeze(0)
        else:
            pixel_values = torch.zeros(3, 224, 224)

        # Process text
        conversations = item.get("conversations", [])
        if conversations:
            # Format conversation
            text = ""
            for conv in conversations:
                role = conv.get("from", conv.get("role", "user"))
                content = conv.get("value", conv.get("content", ""))
                text += f"{role}: {content}\n"
        else:
            text = item.get("text", item.get("caption", ""))

        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "labels": text_inputs.input_ids.squeeze(0).clone(),
        }


class VLMPretrainer:
    """
    Vision-Language Model Pretrainer.

    Implements the two-stage pretraining paradigm:
    - Stage 1: Vision-Language Alignment (train projector only)
    - Stage 2: Visual Instruction Tuning (fine-tune LLM)
    """

    def __init__(
        self,
        model: VLAModel,
        config: PretrainingConfig,
    ):
        self.model = model
        self.config = config

        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Move model to device
        self.model = self.accelerator.prepare(self.model)

        # Initialize logging
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=vars(config),
            )

    def train_stage1_alignment(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        """
        Stage 1: Vision-Language Alignment

        Only trains the vision projector while keeping vision encoder
        and LLM frozen. Uses image-caption pairs to align visual
        features with language embeddings.
        """
        print("=" * 60)
        print("Stage 1: Vision-Language Alignment")
        print("=" * 60)

        # Freeze everything except projector
        self._freeze_for_alignment()

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4,
            pin_memory=True,
        )

        # Optimizer (only projector parameters)
        projector_params = [p for p in self.model.vision_projector.parameters() if p.requires_grad]
        optimizer = AdamW(
            projector_params,
            lr=self.config.alignment_lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * self.config.alignment_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare for distributed training
        train_loader, optimizer, scheduler = self.accelerator.prepare(
            train_loader, optimizer, scheduler
        )

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(self.config.alignment_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.alignment_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self._alignment_forward(batch)
                    loss = outputs["loss"]

                    # Backward pass
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            projector_params,
                            self.config.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log({
                            "stage1/loss": loss.item(),
                            "stage1/lr": scheduler.get_last_lr()[0],
                            "stage1/step": global_step,
                        })

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save stage 1 checkpoint
        self._save_checkpoint("stage1_alignment")

    def train_stage2_instruction(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        """
        Stage 2: Visual Instruction Tuning

        Fine-tunes the LLM on visual instruction data while keeping
        the vision encoder frozen. Trains both projector and LLM.
        """
        print("=" * 60)
        print("Stage 2: Visual Instruction Tuning")
        print("=" * 60)

        # Unfreeze LLM for instruction tuning
        self._unfreeze_for_instruction()

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4,
            pin_memory=True,
        )

        # Optimizer (projector + LLM parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=self.config.instruction_lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * self.config.instruction_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare for distributed training
        train_loader, optimizer, scheduler = self.accelerator.prepare(
            train_loader, optimizer, scheduler
        )

        # Training loop
        self.model.train()
        global_step = 0
        best_loss = float("inf")

        for epoch in range(self.config.instruction_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.instruction_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    # Forward pass (language modeling objective)
                    outputs = self._instruction_forward(batch)
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
                global_step += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log({
                            "stage2/loss": loss.item(),
                            "stage2/lr": scheduler.get_last_lr()[0],
                            "stage2/step": global_step,
                        })

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"stage2_step_{global_step}")

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save_checkpoint("stage2_best")

        # Save final checkpoint
        self._save_checkpoint("stage2_final")

    def _alignment_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for alignment training (projector only)."""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Encode image
        vision_embeds = self.model.encode_image(pixel_values)
        num_vision_tokens = vision_embeds.shape[1]
        batch_size = pixel_values.shape[0]

        # Get text embeddings
        text_embeds = self.model.llm.get_input_embeddings()(input_ids)

        # Combine vision and text
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # Create attention mask
        vision_mask = torch.ones(
            batch_size, num_vision_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Create labels (shift for language modeling)
        # Vision tokens should have -100 labels (ignored)
        vision_labels = torch.full(
            (batch_size, num_vision_tokens),
            -100,
            device=labels.device,
            dtype=labels.dtype
        )
        combined_labels = torch.cat([vision_labels, labels], dim=1)

        # Forward through LLM
        outputs = self.model.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

        return {"loss": outputs.loss}

    def _instruction_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for instruction tuning."""
        # Same as alignment forward but with full model training
        return self._alignment_forward(batch)

    def _freeze_for_alignment(self):
        """Freeze model for Stage 1 (alignment)."""
        # Freeze vision encoder
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        # Freeze LLM
        for param in self.model.llm.parameters():
            param.requires_grad = False

        # Keep projector trainable
        for param in self.model.vision_projector.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    def _unfreeze_for_instruction(self):
        """Unfreeze model for Stage 2 (instruction tuning)."""
        # Keep vision encoder frozen
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False

        # Unfreeze LLM
        for param in self.model.llm.parameters():
            param.requires_grad = True

        # Keep projector trainable
        for param in self.model.vision_projector.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.config.output_dir, name)
            os.makedirs(save_path, exist_ok=True)

            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(os.path.join(save_path, "model.pt"))

            # Save config
            with open(os.path.join(save_path, "config.json"), "w") as f:
                json.dump(vars(self.config), f, indent=2, default=str)

            print(f"Saved checkpoint to {save_path}")


def pretrain_vlm(
    vision_model: str = "google/siglip-base-patch16-224",
    llm_model: str = "Qwen/Qwen2-1.5B-Instruct",
    alignment_dataset: str = "liuhaotian/LLaVA-Pretrain",
    instruction_dataset: str = "liuhaotian/LLaVA-Instruct-150K",
    output_dir: str = "./pretrained_vlm",
    **kwargs,
):
    """
    Convenience function to run full VLM pretraining.

    Args:
        vision_model: Vision encoder name
        llm_model: LLM name
        alignment_dataset: Dataset for Stage 1
        instruction_dataset: Dataset for Stage 2
        output_dir: Output directory
        **kwargs: Additional config arguments
    """
    # Create model
    model = VLAModel(
        vision_model_name=vision_model,
        llm_model_name=llm_model,
        action_dim=7,
    )

    # Create config
    config = PretrainingConfig(
        output_dir=output_dir,
        dataset_name=alignment_dataset,
        instruction_dataset=instruction_dataset,
        **kwargs,
    )

    # Create trainer
    trainer = VLMPretrainer(model, config)

    # Create datasets
    alignment_data = PretrainingDataset(
        alignment_dataset,
        model.image_processor,
        model.tokenizer,
    )
    instruction_data = PretrainingDataset(
        instruction_dataset,
        model.image_processor,
        model.tokenizer,
    )

    # Run training
    trainer.train_stage1_alignment(alignment_data)
    trainer.train_stage2_instruction(instruction_data)

    return model


if __name__ == "__main__":
    print("VLM Pretrainer")
    print("Usage: python vlm_pretrainer.py")
    print("\nThis module implements LLaVA-style VLM pretraining:")
    print("  Stage 1: Vision-Language Alignment (projector only)")
    print("  Stage 2: Visual Instruction Tuning (projector + LLM)")
