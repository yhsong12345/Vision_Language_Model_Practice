"""
Unified Offline RL Training Entry Point

Supports multiple offline RL algorithms via --algo flag:
- cql: Conservative Q-Learning
- iql: Implicit Q-Learning
- td3bc: TD3 with Behavioral Cloning
- dt: Decision Transformer

Usage:
    python train/offline_rl/train.py --algo cql --config config.yaml
    python train/offline_rl/train.py --algo iql --dataset hopper-medium-v2
    python train/offline_rl/train.py --algo dt --num_epochs 100
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train.offline_rl import (
    CQLTrainer,
    IQLTrainer,
    TD3BCTrainer,
    DecisionTransformerTrainer,
)
from train.offline_rl.base_trainer import (
    OfflineRLConfig,
    OfflineReplayBuffer,
    create_dummy_dataset,
    load_d4rl_dataset,
)


# Algorithm registry
TRAINERS = {
    "cql": CQLTrainer,
    "iql": IQLTrainer,
    "td3bc": TD3BCTrainer,
    "td3_bc": TD3BCTrainer,  # alias
    "dt": DecisionTransformerTrainer,
    "decision_transformer": DecisionTransformerTrainer,  # alias
}

# Default configurations per algorithm
ALGO_DEFAULTS = {
    "cql": {
        "cql_alpha": 5.0,
        "cql_num_samples": 10,
        "use_automatic_alpha": True,
    },
    "iql": {
        "expectile": 0.7,
        "temperature": 3.0,
    },
    "td3bc": {
        "alpha": 2.5,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
    },
    "dt": {
        "context_length": 20,
        "n_layer": 3,
        "n_head": 1,
        "n_embd": 128,
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Offline RL Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required: Algorithm selection
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=list(TRAINERS.keys()),
        help="Offline RL algorithm to use",
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="hopper-medium-v2", help="D4RL dataset name")
    parser.add_argument("--obs_dim", type=int, default=11, help="Observation dimension (for dummy data)")
    parser.add_argument("--action_dim", type=int, default=3, help="Action dimension (for dummy data)")

    # Training
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--discount_gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")

    # CQL specific
    parser.add_argument("--cql_alpha", type=float, default=5.0, help="CQL alpha (conservatism)")
    parser.add_argument("--cql_num_samples", type=int, default=10, help="CQL OOD samples")

    # IQL specific
    parser.add_argument("--expectile", type=float, default=0.7, help="IQL expectile")
    parser.add_argument("--temperature", type=float, default=3.0, help="IQL temperature")

    # TD3+BC specific
    parser.add_argument("--alpha", type=float, default=2.5, help="TD3+BC BC weight")

    # Decision Transformer specific
    parser.add_argument("--context_length", type=int, default=20, help="DT context length")
    parser.add_argument("--n_layer", type=int, default=3, help="DT transformer layers")
    parser.add_argument("--n_head", type=int, default=1, help="DT attention heads")

    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not specified)")
    parser.add_argument("--eval_freq", type=int, default=10, help="Evaluation frequency (epochs)")
    parser.add_argument("--save_freq", type=int, default=50, help="Checkpoint save frequency (epochs)")
    parser.add_argument("--log_freq", type=int, default=1, help="Logging frequency")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--use_dummy_data", action="store_true", help="Use dummy data for testing")

    return parser.parse_args()


def create_trainer(algo: str, obs_dim: int, action_dim: int, config: OfflineRLConfig, args):
    """Create trainer instance based on algorithm selection."""
    trainer_cls = TRAINERS[algo]

    # Build algorithm-specific kwargs
    kwargs = {"obs_dim": obs_dim, "action_dim": action_dim, "config": config}

    if algo == "cql":
        kwargs.update({
            "cql_alpha": args.cql_alpha,
            "cql_num_samples": args.cql_num_samples,
        })
    elif algo == "iql":
        kwargs.update({
            "expectile": args.expectile,
            "temperature": args.temperature,
        })
    elif algo in ["td3bc", "td3_bc"]:
        kwargs.update({
            "alpha": args.alpha,
        })
    elif algo in ["dt", "decision_transformer"]:
        kwargs.update({
            "context_length": args.context_length,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
        })

    return trainer_cls(**kwargs)


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"./output/offline_rl/{args.algo}"

    print("=" * 60)
    print(f"Unified Offline RL Training - {args.algo.upper()}")
    print("=" * 60)
    print(f"Algorithm: {args.algo}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create config
    config = OfflineRLConfig(
        dataset_name=args.dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        discount_gamma=args.discount_gamma,
        tau=args.tau,
        output_dir=args.output_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        seed=args.seed,
    )

    # Load dataset
    if args.use_dummy_data:
        print("Using dummy dataset for testing...")
        dataset = create_dummy_dataset(obs_dim=args.obs_dim, action_dim=args.action_dim)
        obs_dim, action_dim = args.obs_dim, args.action_dim
    else:
        try:
            print(f"Loading D4RL dataset: {args.dataset}")
            dataset = load_d4rl_dataset(args.dataset)
            obs_dim = dataset["observations"].shape[1]
            action_dim = dataset["actions"].shape[1]
        except Exception as e:
            print(f"Failed to load D4RL dataset: {e}")
            print("Falling back to dummy dataset...")
            dataset = create_dummy_dataset(obs_dim=args.obs_dim, action_dim=args.action_dim)
            obs_dim, action_dim = args.obs_dim, args.action_dim

    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Dataset size: {len(dataset['observations'])}")

    # Create buffer and load dataset
    buffer = OfflineReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    buffer.load_dataset(dataset)

    # Create trainer
    trainer = create_trainer(args.algo, obs_dim, action_dim, config, args)

    # Train
    trainer.train(buffer)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
