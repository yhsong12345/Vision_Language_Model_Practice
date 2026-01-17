"""
GAIL (Generative Adversarial Imitation Learning) - Learns reward and policy via adversarial training.
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

from .base_trainer import ILTrainer, ExpertDataset
from config.training_config import ILConfig


class Discriminator(nn.Module):
    """Discriminator classifying state-action pairs as expert or policy generated."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=-1))

    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return -torch.log(1 - self.forward(state, action) + 1e-8)


class ActorCriticGAIL(nn.Module):
    """Actor-Critic network for GAIL policy optimization."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, continuous: bool = True):
        super().__init__()
        self.continuous = continuous
        self.features = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        features = self.features(state)
        if self.continuous:
            return self.actor_mean(features), self.actor_log_std.exp(), self.critic(features)
        return self.actor(features), self.critic(features)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            if self.continuous:
                mean, std, _ = self.forward(state)
                return mean.squeeze(0) if deterministic else torch.distributions.Normal(mean, std).sample().squeeze(0)
            else:
                logits, _ = self.forward(state)
                return torch.argmax(logits, dim=-1).squeeze(0) if deterministic else torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1).squeeze(0)

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            return value.squeeze(-1), dist.log_prob(action).sum(-1), dist.entropy().sum(-1)
        else:
            logits, value = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)
            return value.squeeze(-1), dist.log_prob(action), dist.entropy()


class GAIL(ILTrainer):
    """GAIL trainer: alternates discriminator updates with PPO policy updates."""

    def __init__(self, env, policy: Optional[nn.Module] = None, config: Optional[ILConfig] = None, **kwargs):
        config = config or ILConfig.gail()
        state_dim = env.observation_space.shape[0]
        continuous = hasattr(env.action_space, 'shape')
        action_dim = env.action_space.shape[0] if continuous else env.action_space.n

        if policy is None:
            policy = ActorCriticGAIL(state_dim, action_dim, continuous=continuous)

        super().__init__(env, policy, config.output_dir, **kwargs)
        self.config = config

        self.discriminator = Discriminator(state_dim, action_dim, config.gail_disc_hidden_dim).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.learning_rate)
        self.disc_optimizer = Adam(self.discriminator.parameters(), lr=config.gail_disc_lr)

        self.ppo_epochs, self.clip_range = 4, 0.2
        self.value_coef, self.entropy_coef = 0.5, 0.01
        self.gamma, self.gae_lambda = 0.99, 0.95
        self.disc_updates, self.reward_scale = config.gail_disc_updates, config.gail_reward_scale
        self.episode_rewards = []

    def train(self, expert_states: Optional[np.ndarray] = None, expert_actions: Optional[np.ndarray] = None,
              expert_policy=None, num_expert_episodes: int = 50, total_timesteps: int = 100000, rollout_steps: int = 2048):
        print(f"{'=' * 60}\nGAIL Training\n{'=' * 60}")

        if expert_states is None or expert_actions is None:
            if expert_policy is None:
                raise ValueError("Must provide expert data or policy")
            expert_states, expert_actions = self.collect_expert_demonstrations(expert_policy, num_expert_episodes)

        expert_loader = DataLoader(ExpertDataset(expert_states, expert_actions), batch_size=64, shuffle=True)
        timestep, best_reward = 0, float("-inf")
        progress_bar = tqdm(total=total_timesteps, desc="Training")

        while timestep < total_timesteps:
            rollout = self._collect_rollout(rollout_steps)
            timestep += len(rollout["states"])

            disc_loss = self._update_discriminator(rollout, expert_loader)
            gail_rewards = self.discriminator.get_reward(rollout["states"].to(self.device), rollout["actions"].to(self.device)).squeeze(-1) * self.reward_scale
            self._update_policy(rollout, gail_rewards)

            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards)
                progress_bar.set_postfix({"reward": f"{mean_reward:.1f}", "disc_loss": f"{disc_loss:.3f}"})
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    self.save(os.path.join(self.config.output_dir, "best_policy.pt"))
            progress_bar.update(len(rollout["states"]))

        progress_bar.close()
        eval_results = self.evaluate()
        print(f"\nFinal: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        self.save()
        return eval_results

    def _collect_rollout(self, num_steps: int) -> Dict[str, torch.Tensor]:
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
        state, _ = self.env.reset()

        for _ in range(num_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                if self.continuous:
                    mean, std, value = self.policy(state_tensor.unsqueeze(0))
                    dist = torch.distributions.Normal(mean, std)
                else:
                    logits, value = self.policy(state_tensor.unsqueeze(0))
                    dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1) if self.continuous else dist.log_prob(action)

            action_np = action.squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            states.append(state); actions.append(action_np); log_probs.append(log_prob.cpu())
            values.append(value.squeeze().cpu()); rewards.append(reward); dones.append(done)

            if done:
                self.episode_rewards.append(reward)
                state, _ = self.env.reset()
            else:
                state = next_state

        return {"states": torch.tensor(np.array(states), dtype=torch.float32), "actions": torch.tensor(np.array(actions), dtype=torch.float32),
                "log_probs": torch.stack(log_probs), "values": torch.stack(values), "rewards": torch.tensor(rewards, dtype=torch.float32), "dones": torch.tensor(dones, dtype=torch.float32)}

    def _update_discriminator(self, rollout: Dict[str, torch.Tensor], expert_loader: DataLoader) -> float:
        total_loss = 0
        for _ in range(self.disc_updates):
            try:
                expert_states, expert_actions = next(iter(expert_loader))
            except StopIteration:
                expert_loader = DataLoader(expert_loader.dataset, batch_size=64, shuffle=True)
                expert_states, expert_actions = next(iter(expert_loader))

            expert_states, expert_actions = expert_states.to(self.device), expert_actions.to(self.device)
            batch_size = len(expert_states)
            indices = np.random.choice(len(rollout["states"]), batch_size)
            policy_states, policy_actions = rollout["states"][indices].to(self.device), rollout["actions"][indices].to(self.device)

            expert_loss = F.binary_cross_entropy(self.discriminator(expert_states, expert_actions), torch.ones(batch_size, 1, device=self.device))
            policy_loss = F.binary_cross_entropy(self.discriminator(policy_states, policy_actions), torch.zeros(batch_size, 1, device=self.device))
            loss = expert_loss + policy_loss

            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()
            total_loss += loss.item()
        return total_loss / self.disc_updates

    def _update_policy(self, rollout: Dict[str, torch.Tensor], gail_rewards: torch.Tensor):
        states, actions = rollout["states"].to(self.device), rollout["actions"].to(self.device)
        old_log_probs, old_values, dones = rollout["log_probs"].to(self.device), rollout["values"].to(self.device), rollout["dones"].to(self.device)

        # GAE
        advantages, returns, last_gae = torch.zeros_like(gail_rewards), torch.zeros_like(gail_rewards), 0
        for t in reversed(range(len(gail_rewards))):
            next_value = 0 if t == len(gail_rewards) - 1 else old_values[t + 1]
            next_non_terminal = 0 if t == len(gail_rewards) - 1 else 1 - dones[t]
            delta = gail_rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t], returns[t] = last_gae, last_gae + old_values[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            values, log_probs, entropy = self.policy.evaluate_actions(states, actions)
            ratio = torch.exp(log_probs - old_log_probs.squeeze())
            surr1, surr2 = ratio * advantages, torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            loss = -torch.min(surr1, surr2).mean() + self.value_coef * F.mse_loss(values, returns) - self.entropy_coef * entropy.mean()

            self.policy_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()


class VLAGAIL:
    """GAIL for VLA models with image + language inputs."""

    def __init__(self, model, config: Optional[ILConfig] = None):
        from core.device_utils import get_device

        config = config or ILConfig.gail()
        self.model, self.config = model, config
        self.device = get_device("auto")
        self.model = self.model.to(self.device)
        self.action_dim = 7

        self.discriminator = nn.Sequential(
            nn.Linear(256 + self.action_dim, config.gail_disc_hidden_dim), nn.ReLU(),
            nn.Linear(config.gail_disc_hidden_dim, config.gail_disc_hidden_dim), nn.ReLU(),
            nn.Linear(config.gail_disc_hidden_dim, 1), nn.Sigmoid(),
        ).to(self.device)

        self.feature_extractor = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(3 * 16, 256), nn.ReLU()).to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.policy_optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        self.disc_optimizer = torch.optim.Adam(list(self.discriminator.parameters()) + list(self.feature_extractor.parameters()), lr=config.gail_disc_lr)
        self.disc_updates, self.reward_scale = config.gail_disc_updates, config.gail_reward_scale
        os.makedirs(config.output_dir, exist_ok=True)

    def train(self, expert_dataloader, total_steps: int = 10000):
        print(f"{'=' * 60}\nVLA GAIL Training\n{'=' * 60}")
        step, best_reward = 0, float("-inf")
        progress_bar = tqdm(total=total_steps, desc="Training")

        while step < total_steps:
            for batch in expert_dataloader:
                if step >= total_steps:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}

                self.model.eval()
                with torch.no_grad():
                    policy_actions = self.model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])["predicted_actions"]

                self.model.train()
                # Update discriminator
                for _ in range(self.disc_updates):
                    features = self.feature_extractor(batch["pixel_values"])
                    expert_loss = F.binary_cross_entropy(self.discriminator(torch.cat([features, batch["action"]], dim=-1)), torch.ones(len(features), 1, device=self.device))
                    policy_loss = F.binary_cross_entropy(self.discriminator(torch.cat([features.detach(), policy_actions.detach()], dim=-1)), torch.zeros(len(features), 1, device=self.device))
                    disc_loss = expert_loss + policy_loss
                    self.disc_optimizer.zero_grad()
                    disc_loss.backward()
                    self.disc_optimizer.step()

                features = self.feature_extractor(batch["pixel_values"])
                rewards = -torch.log(1 - self.discriminator(torch.cat([features, policy_actions], dim=-1)) + 1e-8) * self.reward_scale
                bc_loss = self.model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], actions=batch["action"])["loss"]
                loss = bc_loss + 0.1 * (-rewards.mean())

                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.policy_optimizer.step()

                step += 1
                progress_bar.update(1)
                if step % 100 == 0:
                    mean_reward = rewards.mean().item()
                    progress_bar.set_postfix({"disc_loss": f"{disc_loss.item():.3f}", "reward": f"{mean_reward:.3f}"})
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        self.save(os.path.join(self.config.output_dir, "best_model.pt"))

        progress_bar.close()
        self.save(os.path.join(self.config.output_dir, "final_model.pt"))

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(), "discriminator": self.discriminator.state_dict(), "feature_extractor": self.feature_extractor.state_dict()}, path)
        print(f"Saved model to {path}")


def create_simple_expert(env_name: str):
    if env_name == "CartPole-v1":
        return lambda s: 1 if s[2] + 0.1 * s[3] > 0 else 0
    raise ValueError(f"No simple expert for {env_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GAIL Training")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--expert_data", type=str, default=None)
    parser.add_argument("--num_expert_episodes", type=int, default=50)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gail_disc_hidden_dim", type=int, default=256)
    parser.add_argument("--gail_disc_updates", type=int, default=5)
    parser.add_argument("--gail_disc_lr", type=float, default=3e-4)
    parser.add_argument("--gail_reward_scale", type=float, default=1.0)
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="./output/gail")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = ILConfig(gail_disc_hidden_dim=args.gail_disc_hidden_dim, gail_disc_updates=args.gail_disc_updates,
                      gail_disc_lr=args.gail_disc_lr, gail_reward_scale=args.gail_reward_scale,
                      learning_rate=args.learning_rate, batch_size=args.batch_size, output_dir=args.output_dir)

    if args.model_path:
        from model.vla import VLAModel
        from model.vla.vla_model import VLAConfig
        from train.finetune.dataset import RobotDataset

        model = VLAModel(VLAConfig())
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
        trainer = VLAGAIL(model, config=config)
        try:
            trainer.train(DataLoader(RobotDataset(dataset_name="lerobot/pusht", max_samples=1000), batch_size=args.batch_size, shuffle=True), total_steps=args.total_timesteps)
        except Exception as e:
            print(f"Error: {e}")
    else:
        import gymnasium as gym
        trainer = GAIL(gym.make(args.env), config=config)
        if args.resume:
            trainer.load(args.resume)
        if args.expert_data and os.path.exists(args.expert_data):
            data = np.load(args.expert_data)
            trainer.train(expert_states=data["states"], expert_actions=data["actions"], total_timesteps=args.total_timesteps)
        else:
            trainer.train(expert_policy=create_simple_expert(args.env), num_expert_episodes=args.num_expert_episodes, total_timesteps=args.total_timesteps)

    print("\nTraining complete!")
