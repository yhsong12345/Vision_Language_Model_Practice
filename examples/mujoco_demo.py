#!/usr/bin/env python3
"""
MuJoCo Demo - Online RL Training

This example demonstrates training a VLA-style policy on MuJoCo
environments using PPO (Proximal Policy Optimization).

Requirements:
    pip install gymnasium mujoco torch

Usage:
    python examples/mujoco_demo.py --env CartPole-v1 --train
    python examples/mujoco_demo.py --env Pendulum-v1 --train --timesteps 100000
    python examples/mujoco_demo.py --env HalfCheetah-v4 --eval --checkpoint ./checkpoints/ppo.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

try:
    import gymnasium as gym
except ImportError:
    print("Please install gymnasium: pip install gymnasium")
    sys.exit(1)

# Framework imports
from model.utils import get_device, count_parameters
from train.utils import MetricsTracker, RolloutBuffer


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    For VLA integration, the actor would use the VLA model
    conditioned on visual observations and language instructions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = True,
    ):
        super().__init__()

        self.continuous = continuous
        self.action_dim = action_dim

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action distribution and value."""
        features = self.features(obs)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            return mean, std
        else:
            logits = self.actor(features)
            return logits, None

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value."""
        features = self.features(obs)
        return self.critic(features)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value."""
        features = self.features(obs)
        value = self.critic(features)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits = self.actor(features)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for inference."""
        if self.continuous:
            mean, std = self.forward(obs)
            if deterministic:
                return mean
            dist = Normal(mean, std)
            return dist.sample()
        else:
            logits, _ = self.forward(obs)
            if deterministic:
                return logits.argmax(-1)
            dist = Categorical(logits=logits)
            return dist.sample()


class PPOTrainer:
    """
    Proximal Policy Optimization trainer.

    This is a simplified version for demonstration.
    For full implementation, see train/online_rl/ppo_trainer.py
    """

    def __init__(
        self,
        env: gym.Env,
        model: ActorCritic,
        device: torch.device,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.env = env
        self.model = model
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.metrics = MetricsTracker()

    def collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """Collect experience from environment."""
        obs_list, action_list, reward_list = [], [], []
        value_list, log_prob_list, done_list = [], [], []

        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        for _ in range(self.n_steps):
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(obs.unsqueeze(0))
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                value = value.squeeze(0)

            # Store
            obs_list.append(obs)
            action_list.append(action)
            value_list.append(value)
            log_prob_list.append(log_prob)

            # Step environment
            if self.model.continuous:
                next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            else:
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())

            done = terminated or truncated
            reward_list.append(torch.tensor(reward, device=self.device))
            done_list.append(torch.tensor(done, device=self.device))

            if done:
                obs, _ = self.env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            else:
                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        # Compute last value for bootstrapping
        with torch.no_grad():
            last_value = self.model.get_value(obs.unsqueeze(0)).squeeze()

        # Stack tensors
        observations = torch.stack(obs_list)
        actions = torch.stack(action_list)
        rewards = torch.stack(reward_list)
        values = torch.stack(value_list)
        log_probs = torch.stack(log_prob_list)
        dones = torch.stack(done_list).float()

        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        return {
            "observations": observations,
            "actions": actions,
            "log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def train_step(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training iteration."""
        observations = rollout_data["observations"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["log_probs"]
        advantages = rollout_data["advantages"]
        returns = rollout_data["returns"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Mini-batch training
        indices = np.arange(len(observations))
        total_loss = 0
        total_pg_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, len(observations), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get new values
                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    batch_obs, batch_actions
                )

                # Policy loss (PPO clip)
                ratio = (new_log_probs - batch_old_log_probs).exp()
                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = self.n_epochs * (len(observations) // self.batch_size)
        return {
            "loss": total_loss / n_updates,
            "pg_loss": total_pg_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        save_dir: str = "./checkpoints",
    ):
        """Train the policy."""
        print("\n" + "=" * 60)
        print("PPO Training")
        print("=" * 60)
        print(f"Environment: {self.env.spec.id if hasattr(self.env, 'spec') else 'Custom'}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Steps per rollout: {self.n_steps}")
        print("=" * 60 + "\n")

        os.makedirs(save_dir, exist_ok=True)

        n_iterations = total_timesteps // self.n_steps
        episode_rewards = []
        best_reward = float("-inf")

        for iteration in range(n_iterations):
            # Collect rollouts
            rollout_data = self.collect_rollouts()

            # Train
            train_metrics = self.train_step(rollout_data)

            # Evaluate periodically
            if (iteration + 1) % log_interval == 0:
                eval_reward = self.evaluate(n_episodes=5)
                episode_rewards.append(eval_reward)

                print(
                    f"Iteration {iteration + 1}/{n_iterations} | "
                    f"Reward: {eval_reward:.2f} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Entropy: {train_metrics['entropy']:.4f}"
                )

                # Save best
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    save_path = os.path.join(save_dir, "ppo_best.pt")
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "iteration": iteration,
                        "reward": best_reward,
                    }, save_path)

        # Save final
        final_path = os.path.join(save_dir, "ppo_final.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "iteration": n_iterations,
        }, final_path)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Reward: {best_reward:.2f}")
        print(f"Saved to: {save_dir}")
        print("=" * 60)

    def evaluate(self, n_episodes: int = 10) -> float:
        """Evaluate the policy."""
        self.model.eval()
        episode_rewards = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            episode_reward = 0

            while True:
                action = self.model.get_action(obs.unsqueeze(0), deterministic=True)
                action = action.squeeze(0)

                if self.model.continuous:
                    next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                else:
                    next_obs, reward, terminated, truncated, _ = self.env.step(action.item())

                episode_reward += reward

                if terminated or truncated:
                    break

                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

            episode_rewards.append(episode_reward)

        self.model.train()
        return np.mean(episode_rewards)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo PPO Demo")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")

    args = parser.parse_args()

    # Default to train if no action specified
    if not (args.train or args.eval):
        args.train = True

    # Create environment
    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True

    print(f"\nEnvironment: {args.env}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Continuous: {continuous}")

    # Create model
    device = get_device(args.device)
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        continuous=continuous,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded!")

    # Create trainer
    trainer = PPOTrainer(env=env, model=model, device=device)

    if args.train:
        trainer.train(total_timesteps=args.timesteps)

    if args.eval:
        print("\n" + "=" * 60)
        print("Evaluating Policy")
        print("=" * 60)
        reward = trainer.evaluate(n_episodes=10)
        print(f"Mean Reward: {reward:.2f}")
        print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
