"""
Unified Imitation Learning Training Entry Point

Supports multiple IL algorithms via --algo flag:
- bc: Behavioral Cloning
- dagger: Dataset Aggregation
- gail: Generative Adversarial Imitation Learning

Usage:
    python train/il/train.py --algo bc --env CartPole-v1
    python train/il/train.py --algo dagger --env HalfCheetah-v4
    python train/il/train.py --algo gail --num_expert_episodes 100
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train.il import BehavioralCloning, DAgger, GAIL


# Algorithm registry
TRAINERS = {
    "bc": BehavioralCloning,
    "behavioral_cloning": BehavioralCloning,  # alias
    "dagger": DAgger,
    "gail": GAIL,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Imitation Learning Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required: Algorithm selection
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=list(TRAINERS.keys()),
        help="Imitation learning algorithm to use",
    )

    # Environment
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment name")

    # Expert data
    parser.add_argument("--expert_data", type=str, default=None, help="Path to expert demonstrations (.npz)")
    parser.add_argument("--num_expert_episodes", type=int, default=100, help="Number of expert episodes to collect")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Maximum steps per episode")

    # Training - Common
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    # BC specific
    parser.add_argument("--bc_epochs", type=int, default=100, help="BC training epochs")
    parser.add_argument("--bc_validation_split", type=float, default=0.1, help="Validation split ratio")

    # DAgger specific
    parser.add_argument("--dagger_iterations", type=int, default=10, help="DAgger iterations")
    parser.add_argument("--dagger_episodes_per_iter", type=int, default=20, help="Episodes per DAgger iteration")
    parser.add_argument("--dagger_bc_epochs", type=int, default=50, help="BC epochs per DAgger iteration")
    parser.add_argument("--dagger_beta_start", type=float, default=1.0, help="Initial expert intervention probability")
    parser.add_argument("--dagger_beta_decay", type=float, default=0.9, help="Beta decay rate")

    # GAIL specific
    parser.add_argument("--gail_total_timesteps", type=int, default=500000, help="GAIL total timesteps")
    parser.add_argument("--gail_disc_updates", type=int, default=5, help="Discriminator updates per iteration")
    parser.add_argument("--gail_gen_updates", type=int, default=10, help="Generator (policy) updates per iteration")

    # Logging & Evaluation
    parser.add_argument("--eval_freq", type=int, default=10, help="Evaluation frequency")
    parser.add_argument("--eval_episodes", type=int, default=20, help="Evaluation episodes")

    # W&B logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="vla-il-training", help="W&B project name")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")

    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")

    return parser.parse_args()


def load_expert_data(path: str):
    """Load expert demonstrations from file."""
    data = np.load(path)
    return data["states"], data["actions"]


def create_simple_expert_policy(env):
    """Create a simple rule-based expert policy for testing."""
    env_name = env.spec.id if hasattr(env, "spec") and env.spec else ""

    if "CartPole" in env_name:
        # Simple cart-pole balancing heuristic
        def expert(state):
            return 1 if state[2] + 0.1 * state[3] > 0 else 0
        return expert
    elif "MountainCar" in env_name:
        # Simple mountain car heuristic
        def expert(state):
            return 2 if state[1] > 0 else 0
        return expert
    else:
        # Random policy as fallback
        def expert(state):
            return env.action_space.sample()
        return expert


def create_trainer(algo: str, env, args):
    """Create trainer instance based on algorithm selection."""
    trainer_cls = TRAINERS[algo]

    # Import config
    from config.training_config import ILConfig

    if algo in ["bc", "behavioral_cloning"]:
        config = ILConfig(
            bc_epochs=args.bc_epochs,
            bc_validation_split=args.bc_validation_split,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_expert_episodes=args.num_expert_episodes,
            output_dir=args.output_dir,
        )
        return trainer_cls(
            env=env,
            config=config,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            experiment_name=args.experiment_name,
        )

    elif algo == "dagger":
        config = ILConfig(
            dagger_iterations=args.dagger_iterations,
            dagger_episodes_per_iter=args.dagger_episodes_per_iter,
            dagger_bc_epochs=args.dagger_bc_epochs,
            dagger_beta_start=args.dagger_beta_start,
            dagger_beta_decay=args.dagger_beta_decay,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_expert_episodes=args.num_expert_episodes,
            output_dir=args.output_dir,
        )
        return trainer_cls(
            env=env,
            config=config,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            experiment_name=args.experiment_name,
        )

    elif algo == "gail":
        config = ILConfig(
            gail_total_timesteps=args.gail_total_timesteps,
            gail_discriminator_updates=args.gail_disc_updates,
            gail_generator_updates=args.gail_gen_updates,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_expert_episodes=args.num_expert_episodes,
            output_dir=args.output_dir,
        )
        return trainer_cls(
            env=env,
            config=config,
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
        args.output_dir = f"./output/il/{args.algo}"

    print("=" * 60)
    print(f"Unified Imitation Learning Training - {args.algo.upper()}")
    print("=" * 60)
    print(f"Algorithm: {args.algo}")
    print(f"Environment: {args.env}")
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

    # Load or collect expert data
    if args.expert_data:
        print(f"Loading expert data from: {args.expert_data}")
        states, actions = load_expert_data(args.expert_data)
        trainer.train(states=states, actions=actions)
    else:
        print(f"Collecting {args.num_expert_episodes} expert episodes...")
        expert_policy = create_simple_expert_policy(env)
        trainer.train(expert_policy=expert_policy, num_expert_episodes=args.num_expert_episodes)

    # Final evaluation
    eval_results = trainer.evaluate(num_episodes=args.eval_episodes)
    print(f"\nFinal Evaluation:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
    print(f"  Mean Length: {eval_results['mean_length']:.1f}")

    # Cleanup
    env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
