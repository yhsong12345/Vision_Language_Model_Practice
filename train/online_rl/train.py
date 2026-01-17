"""
Unified Online RL Training Entry Point

Supports multiple online RL algorithms via --algo flag:
- ppo: Proximal Policy Optimization
- sac: Soft Actor-Critic
- grpo: Group Relative Policy Optimization

Usage:
    python train/online_rl/train.py --algo ppo --env CartPole-v1
    python train/online_rl/train.py --algo sac --env HalfCheetah-v4
    python train/online_rl/train.py --algo grpo --total_timesteps 500000
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train.online_rl import PPOTrainer, SACTrainer, GRPOTrainer


# Algorithm registry
TRAINERS = {
    "ppo": PPOTrainer,
    "sac": SACTrainer,
    "grpo": GRPOTrainer,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Online RL Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required: Algorithm selection
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=list(TRAINERS.keys()),
        help="Online RL algorithm to use",
    )

    # Environment
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    # Training
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--discount_gamma", type=float, default=0.99, help="Discount factor")

    # PPO specific
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Steps per rollout (PPO)")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--ppo_clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95, help="GAE lambda (PPO)")
    parser.add_argument("--ppo_value_coef", type=float, default=0.5, help="Value loss coefficient (PPO)")
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.01, help="Entropy coefficient (PPO)")

    # SAC specific
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Replay buffer size (SAC)")
    parser.add_argument("--learning_starts", type=int, default=10000, help="Learning starts (SAC)")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient (SAC)")
    parser.add_argument("--auto_entropy", action="store_true", help="Auto-tune entropy (SAC)")

    # GRPO specific
    parser.add_argument("--group_size", type=int, default=8, help="Group size (GRPO)")
    parser.add_argument("--kl_coef", type=float, default=0.1, help="KL coefficient (GRPO)")

    # Logging & Evaluation
    parser.add_argument("--eval_freq", type=int, default=5000, help="Evaluation frequency (timesteps)")
    parser.add_argument("--save_freq", type=int, default=10000, help="Checkpoint save frequency")
    parser.add_argument("--log_freq", type=int, default=1000, help="Logging frequency")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Evaluation episodes")

    # W&B logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="vla-online-rl", help="W&B project name")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")

    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")

    return parser.parse_args()


def create_trainer(algo: str, env, args):
    """Create trainer instance based on algorithm selection."""
    trainer_cls = TRAINERS[algo]

    # Import config
    from config.training_config import RLConfig

    if algo == "ppo":
        config = RLConfig(
            algorithm="ppo",
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            ppo_epochs=args.ppo_epochs,
            ppo_clip_range=args.ppo_clip_range,
            ppo_gae_lambda=args.ppo_gae_lambda,
            ppo_value_coef=args.ppo_value_coef,
            ppo_entropy_coef=args.ppo_entropy_coef,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            discount_gamma=args.discount_gamma,
            output_dir=args.output_dir,
        )
        return trainer_cls(
            env=env,
            config=config,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            experiment_name=args.experiment_name,
        )

    elif algo == "sac":
        config = RLConfig(
            algorithm="sac",
            total_timesteps=args.total_timesteps,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            tau=args.tau,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            discount_gamma=args.discount_gamma,
            output_dir=args.output_dir,
        )
        return trainer_cls(
            env=env,
            config=config,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            experiment_name=args.experiment_name,
        )

    elif algo == "grpo":
        config = RLConfig(
            algorithm="grpo",
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            discount_gamma=args.discount_gamma,
            output_dir=args.output_dir,
        )
        return trainer_cls(
            env=env,
            config=config,
            group_size=args.group_size,
            kl_coef=args.kl_coef,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            experiment_name=args.experiment_name,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"./output/online_rl/{args.algo}"

    print("=" * 60)
    print(f"Unified Online RL Training - {args.algo.upper()}")
    print("=" * 60)
    print(f"Algorithm: {args.algo}")
    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create environment
    try:
        import gymnasium as gym
        env = gym.make(args.env)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    except Exception as e:
        print(f"Failed to create environment '{args.env}': {e}")
        print("Please install gymnasium and required dependencies.")
        sys.exit(1)

    # Create trainer
    trainer = create_trainer(args.algo, env, args)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load(args.resume)

    # Train
    trainer.train()

    # Cleanup
    env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
