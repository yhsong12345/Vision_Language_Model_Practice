#!/usr/bin/env python3
"""
VLA Training Framework CLI

Unified command-line interface for training, evaluation, and deployment.

Usage:
    python run.py train --config configs/pusht_bc.yaml
    python run.py train --preset pusht-bc
    python run.py eval --checkpoint ./checkpoints/model.pt
    python run.py infer --image robot.jpg --instruction "Pick up cube"
    python run.py export --checkpoint model.pt --format onnx
"""

import argparse
import sys
import os
from pathlib import Path


def train_command(args):
    """Train a VLA model."""
    print("\n" + "=" * 60)
    print("VLA Training")
    print("=" * 60)

    from config.hydra_config import (
        ExperimentConfig,
        get_preset,
        create_config_from_args,
    )

    # Load config
    if args.preset:
        print(f"Using preset: {args.preset}")
        preset = get_preset(args.preset)
        config = ExperimentConfig()
        # Apply preset
        for section, values in preset.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    elif args.config:
        print(f"Loading config: {args.config}")
        from config.hydra_config import load_config, HYDRA_AVAILABLE
        if HYDRA_AVAILABLE:
            from omegaconf import OmegaConf
            config = OmegaConf.load(args.config)
        else:
            config = ExperimentConfig()
    else:
        config = create_config_from_args(args)

    print(f"\nConfiguration:")
    print(f"  Model: {config.model.vision_encoder} + {config.model.llm}")
    print(f"  Action dim: {config.model.action_dim}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.max_epochs}")

    # Run training
    if args.dry_run:
        print("\n[Dry run - not starting training]")
        return

    # Import and run trainer
    print("\nStarting training...")

    # Run example demo
    from examples.pusht_demo import train as pusht_train, create_model, MockPushTDataset
    model, device = create_model(args.device)
    dataset = MockPushTDataset(num_samples=args.samples)
    pusht_train(model, dataset, device, num_epochs=args.epochs)


def eval_command(args):
    """Evaluate a trained model."""
    print("\n" + "=" * 60)
    print("VLA Evaluation")
    print("=" * 60)

    if not args.checkpoint:
        print("Error: --checkpoint required for evaluation")
        return

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num episodes: {args.episodes}")

    # Run evaluation demo
    from examples.pusht_demo import evaluate, create_model, MockPushTDataset
    model, device = create_model(args.device)
    dataset = MockPushTDataset(num_samples=100)

    if os.path.exists(args.checkpoint):
        import torch
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded checkpoint")

    evaluate(model, dataset, device)


def infer_command(args):
    """Run inference."""
    print("\n" + "=" * 60)
    print("VLA Inference")
    print("=" * 60)

    from infer import VLAInferenceEngine, InferenceConfig

    config = InferenceConfig(
        model_path=args.checkpoint,
        device=args.device,
        precision=args.precision,
    )

    engine = VLAInferenceEngine(config)

    if args.image:
        result = engine.predict(args.image, args.instruction)
        print(f"\nInstruction: {args.instruction}")
        print(f"Action: {result['action']}")
        print(f"Time: {result['inference_time_ms']:.2f} ms")

    if args.video:
        result = engine.process_video(args.video, args.instruction, args.output)
        print(f"\nProcessed {result['num_frames']} frames")
        print(f"FPS: {result['achievable_fps']:.1f}")

    if args.benchmark:
        engine.benchmark()


def export_command(args):
    """Export model to deployment format."""
    print("\n" + "=" * 60)
    print("Model Export")
    print("=" * 60)

    from model.utils import export_model, ExportConfig
    import torch
    import torch.nn as nn

    if not args.checkpoint:
        print("No checkpoint provided, creating demo model...")

        class DemoModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 7)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).flatten(1)
                return self.fc(x)

        model = DemoModel()
    else:
        # Load model from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # Would need to instantiate correct model architecture
        print("Loading model from checkpoint...")

    config = ExportConfig(
        output_dir=args.output_dir,
        model_name=args.name,
    )

    formats = args.format.split(",")
    exported = export_model(model, formats=formats, config=config)

    print(f"\nExported formats: {list(exported.keys())}")


def list_command(args):
    """List available presets and configs."""
    from config.hydra_config import list_presets

    print("\n" + "=" * 60)
    print("Available Presets")
    print("=" * 60)

    for preset in list_presets():
        print(f"  - {preset}")

    print("\nUsage: python run.py train --preset <name>")


def demo_command(args):
    """Run demos."""
    print("\n" + "=" * 60)
    print("VLA Demos")
    print("=" * 60)

    if args.demo == "pusht":
        from examples.pusht_demo import run_demo
        run_demo()
    elif args.demo == "mujoco":
        from examples.mujoco_demo import main as mujoco_main
        sys.argv = ["mujoco_demo.py", "--env", args.env, "--train"]
        mujoco_main()
    elif args.demo == "carla":
        from examples.carla_demo import run_demo
        run_demo()
    elif args.demo == "inference":
        from examples.inference_demo import run_demo
        run_demo()
    else:
        print(f"Unknown demo: {args.demo}")
        print("Available: pusht, mujoco, carla, inference")


def main():
    parser = argparse.ArgumentParser(
        description="VLA Training Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a VLA model")
    train_parser.add_argument("--config", type=str, help="Path to config file")
    train_parser.add_argument("--preset", type=str, help="Use a preset config")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--device", type=str, default="auto", help="Device")
    train_parser.add_argument("--samples", type=int, default=1000, help="Training samples")
    train_parser.add_argument("--dry-run", action="store_true", help="Dry run")
    train_parser.set_defaults(func=train_command)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Num episodes")
    eval_parser.add_argument("--device", type=str, default="auto", help="Device")
    eval_parser.set_defaults(func=eval_command)

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    infer_parser.add_argument("--image", type=str, help="Input image")
    infer_parser.add_argument("--video", type=str, help="Input video")
    infer_parser.add_argument("--instruction", type=str, default="Execute task",
                               help="Instruction")
    infer_parser.add_argument("--output", type=str, help="Output path")
    infer_parser.add_argument("--device", type=str, default="auto", help="Device")
    infer_parser.add_argument("--precision", type=str, default="fp32",
                               choices=["fp32", "fp16", "bf16"])
    infer_parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    infer_parser.set_defaults(func=infer_command)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    export_parser.add_argument("--format", type=str, default="torchscript",
                                help="Export format (onnx,torchscript,quantized)")
    export_parser.add_argument("--output-dir", type=str, default="./exported",
                                help="Output directory")
    export_parser.add_argument("--name", type=str, default="vla_model",
                                help="Model name")
    export_parser.set_defaults(func=export_command)

    # List command
    list_parser = subparsers.add_parser("list", help="List presets")
    list_parser.set_defaults(func=list_command)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demos")
    demo_parser.add_argument("demo", type=str, nargs="?", default="pusht",
                              help="Demo name (pusht, mujoco, carla, inference)")
    demo_parser.add_argument("--env", type=str, default="CartPole-v1",
                              help="Environment for mujoco demo")
    demo_parser.set_defaults(func=demo_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
