"""
GAIL (Generative Adversarial Imitation Learning)

Learns reward function and policy simultaneously:
- Discriminator distinguishes expert vs. policy trajectories
- Policy optimizes against learned reward
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

from .base_trainer import ILTrainer, ExpertDataset, PolicyNetwork

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.training_config import ILConfig


class Discriminator(nn.Module):
    """
    Discriminator network for GAIL.

    Classifies state-action pairs as expert or policy generated.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of (state, action) being from expert.

        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)

        Returns:
            prob: (batch, 1) probability of being expert
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get GAIL reward for (state, action).

        reward = -log(1 - D(s,a))

        This encourages policy to be classified as expert.
        """
        with torch.no_grad():
            d = self.forward(state, action)
            reward = -torch.log(1 - d + 1e-8)
        return reward


class ActorCriticGAIL(nn.Module):
    """Actor-Critic network for GAIL policy optimization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = True,
    ):
        super().__init__()

        self.continuous = continuous

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor (policy)
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        features = self.features(state)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            value = self.critic(features)
            return mean, std, value
        else:
            logits = self.actor(features)
            value = self.critic(features)
            return logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            if self.continuous:
                mean, std, _ = self.forward(state)
                if deterministic:
                    return mean.squeeze(0)
                dist = torch.distributions.Normal(mean, std)
                return dist.sample().squeeze(0)
            else:
                logits, _ = self.forward(state)
                if deterministic:
                    return torch.argmax(logits, dim=-1).squeeze(0)
                probs = F.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).squeeze(-1).squeeze(0)

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits, value = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return value.squeeze(-1), log_prob, entropy


class GAIL(ILTrainer):
    """
    GAIL (Generative Adversarial Imitation Learning) trainer.

    Alternates between:
    1. Update discriminator to distinguish expert vs. policy
    2. Update policy using PPO with discriminator as reward

    Advantages over BC:
    - Can achieve better than expert performance
    - No need for action labels (can work with state-only demos)
    - Learns reward function that generalizes

    Disadvantages:
    - Requires environment interaction
    - More complex training
    - Can be unstable
    """

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        config: Optional[ILConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = ILConfig.gail()

        # Create actor-critic policy
        state_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'shape'):
            action_dim = env.action_space.shape[0]
            continuous = True
        else:
            action_dim = env.action_space.n
            continuous = False

        if policy is None:
            policy = ActorCriticGAIL(
                state_dim, action_dim, continuous=continuous
            )

        super().__init__(env, policy, config.output_dir, **kwargs)

        self.config = config

        # Create discriminator
        self.discriminator = Discriminator(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.gail_disc_hidden_dim,
        ).to(self.device)

        # Optimizers
        self.policy_optimizer = Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
        )
        self.disc_optimizer = Adam(
            self.discriminator.parameters(),
            lr=config.gail_disc_lr,
        )

        # PPO params
        self.ppo_epochs = 4
        self.clip_range = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.gae_lambda = 0.95

        # GAIL params
        self.disc_updates = config.gail_disc_updates
        self.reward_scale = config.gail_reward_scale

    def train(
        self,
        expert_states: Optional[np.ndarray] = None,
        expert_actions: Optional[np.ndarray] = None,
        expert_policy=None,
        num_expert_episodes: int = 50,
        total_timesteps: int = 100000,
        rollout_steps: int = 2048,
    ):
        """
        Run GAIL training.

        Args:
            expert_states: Expert states
            expert_actions: Expert actions
            expert_policy: Expert policy for collecting demos
            num_expert_episodes: Episodes to collect if using expert_policy
            total_timesteps: Total training timesteps
            rollout_steps: Steps per rollout
        """
        print("=" * 60)
        print("GAIL Training")
        print("=" * 60)

        # Collect expert demonstrations if not provided
        if expert_states is None or expert_actions is None:
            if expert_policy is None:
                raise ValueError("Must provide expert data or policy")

            expert_states, expert_actions = self.collect_expert_demonstrations(
                expert_policy, num_expert_episodes
            )

        # Create expert dataloader
        expert_dataset = ExpertDataset(expert_states, expert_actions)
        expert_loader = DataLoader(
            expert_dataset,
            batch_size=64,
            shuffle=True,
        )

        # Training loop
        timestep = 0
        best_reward = float("-inf")

        progress_bar = tqdm(total=total_timesteps, desc="Training")

        while timestep < total_timesteps:
            # Collect rollout
            rollout = self._collect_rollout(rollout_steps)
            timestep += len(rollout["states"])

            # Update discriminator
            disc_loss = self._update_discriminator(
                rollout, expert_loader
            )

            # Compute GAIL rewards
            gail_rewards = self._compute_gail_rewards(rollout)

            # Update policy with PPO
            policy_metrics = self._update_policy(rollout, gail_rewards)

            # Logging
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(list(self.episode_rewards))

                progress_bar.set_postfix({
                    "reward": f"{mean_reward:.1f}",
                    "disc_loss": f"{disc_loss:.3f}",
                })

                if mean_reward > best_reward:
                    best_reward = mean_reward
                    self.save(os.path.join(self.config.output_dir, "best_policy.pt"))

            progress_bar.update(len(rollout["states"]))

        progress_bar.close()

        # Final evaluation
        print("\nFinal Evaluation:")
        eval_results = self.evaluate()
        print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

        self.save()
        return eval_results

    def _collect_rollout(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """Collect rollout data."""
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        state, _ = self.env.reset()

        for _ in range(num_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                if self.continuous:
                    mean, std, value = self.policy(state_tensor.unsqueeze(0))
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)
                else:
                    logits, value = self.policy(state_tensor.unsqueeze(0))
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

            action_np = action.squeeze(0).cpu().numpy()

            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            states.append(state)
            actions.append(action_np)
            log_probs.append(log_prob.cpu())
            values.append(value.squeeze().cpu())
            rewards.append(reward)
            dones.append(done)

            if done:
                self.episode_rewards.append(reward)
                state, _ = self.env.reset()
            else:
                state = next_state

        return {
            "states": torch.tensor(np.array(states), dtype=torch.float32),
            "actions": torch.tensor(np.array(actions), dtype=torch.float32),
            "log_probs": torch.stack(log_probs),
            "values": torch.stack(values),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
        }

    def _update_discriminator(
        self,
        rollout: Dict[str, torch.Tensor],
        expert_loader: DataLoader,
    ) -> float:
        """Update discriminator."""
        total_loss = 0

        for _ in range(self.disc_updates):
            # Get expert batch
            try:
                expert_states, expert_actions = next(iter(expert_loader))
            except StopIteration:
                expert_loader = DataLoader(expert_loader.dataset, batch_size=64, shuffle=True)
                expert_states, expert_actions = next(iter(expert_loader))

            expert_states = expert_states.to(self.device)
            expert_actions = expert_actions.to(self.device)

            # Sample policy batch
            batch_size = len(expert_states)
            indices = np.random.choice(len(rollout["states"]), batch_size)
            policy_states = rollout["states"][indices].to(self.device)
            policy_actions = rollout["actions"][indices].to(self.device)

            # Expert prediction (should be 1)
            expert_pred = self.discriminator(expert_states, expert_actions)
            expert_loss = F.binary_cross_entropy(
                expert_pred,
                torch.ones_like(expert_pred),
            )

            # Policy prediction (should be 0)
            policy_pred = self.discriminator(policy_states, policy_actions)
            policy_loss = F.binary_cross_entropy(
                policy_pred,
                torch.zeros_like(policy_pred),
            )

            loss = expert_loss + policy_loss

            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.disc_updates

    def _compute_gail_rewards(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute GAIL rewards using discriminator."""
        states = rollout["states"].to(self.device)
        actions = rollout["actions"].to(self.device)

        rewards = self.discriminator.get_reward(states, actions)
        return rewards.squeeze(-1) * self.reward_scale

    def _update_policy(
        self,
        rollout: Dict[str, torch.Tensor],
        gail_rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy using PPO with GAIL rewards."""
        states = rollout["states"].to(self.device)
        actions = rollout["actions"].to(self.device)
        old_log_probs = rollout["log_probs"].to(self.device)
        old_values = rollout["values"].to(self.device)
        dones = rollout["dones"].to(self.device)

        # Compute advantages using GAE
        advantages = torch.zeros_like(gail_rewards)
        returns = torch.zeros_like(gail_rewards)
        last_gae = 0

        for t in reversed(range(len(gail_rewards))):
            if t == len(gail_rewards) - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = old_values[t + 1]
                next_non_terminal = 1 - dones[t]

            delta = gail_rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + old_values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.ppo_epochs):
            values, log_probs, entropy = self.policy.evaluate_actions(states, actions)

            ratio = torch.exp(log_probs - old_log_probs.squeeze())

            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.policy_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }


if __name__ == "__main__":
    print("GAIL Trainer")
    print("Generative Adversarial Imitation Learning")
