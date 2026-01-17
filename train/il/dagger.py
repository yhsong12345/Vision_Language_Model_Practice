"""
DAgger (Dataset Aggregation) - Interactive imitation learning addressing distribution shift.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
import numpy as np

from .base_trainer import ILTrainer, ExpertDataset
from config.training_config import ILConfig


class DAgger(ILTrainer):
    """DAgger trainer: iteratively collects data under learned policy with expert corrections."""

    def __init__(self, env, expert_policy, policy: Optional[nn.Module] = None, config: Optional[ILConfig] = None, **kwargs):
        config = config or ILConfig.dagger()
        super().__init__(env, policy, config.output_dir, **kwargs)

        self.config, self.expert_policy = config, expert_policy
        self.num_iterations = config.dagger_iterations
        self.episodes_per_iter = config.dagger_episodes_per_iter
        self.beta_schedule = config.dagger_beta_schedule
        self.initial_beta = config.dagger_initial_beta
        self.bc_epochs = config.bc_epochs
        self.batch_size = config.batch_size

        self.states_buffer, self.actions_buffer = [], []
        self.optimizer = Adam(self.policy.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss() if self.continuous else nn.CrossEntropyLoss()

    def get_beta(self, iteration: int) -> float:
        if self.beta_schedule == "linear":
            return max(0, self.initial_beta * (1 - iteration / self.num_iterations))
        elif self.beta_schedule == "exponential":
            return self.initial_beta * (0.9 ** iteration)
        return self.initial_beta

    def collect_with_dagger(self, num_episodes: int, beta: float, max_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        states, actions, episode_rewards = [], [], []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            ep_reward = 0
            for _ in range(max_steps):
                expert_action = self.expert_policy(state)
                if np.random.random() < beta:
                    action = expert_action
                else:
                    action = self.policy.get_action(torch.tensor(state, dtype=torch.float32).to(self.device), deterministic=False)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()

                states.append(state)
                actions.append(expert_action)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
                state = next_state
            episode_rewards.append(ep_reward)

        print(f"Collected {len(states)} transitions (β={beta:.2f}), mean reward: {np.mean(episode_rewards):.2f}")
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32 if self.continuous else np.int64)

    def train_bc_epoch(self, dataloader: DataLoader) -> float:
        self.policy.train()
        total_loss, num_batches = 0, 0
        for states_batch, actions_batch in dataloader:
            states_batch, actions_batch = states_batch.to(self.device), actions_batch.to(self.device)
            loss = self.criterion(self.policy(states_batch), actions_batch if self.continuous else actions_batch.long())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches

    def train(self, initial_states: Optional[np.ndarray] = None, initial_actions: Optional[np.ndarray] = None, num_initial_episodes: int = 20):
        print(f"{'=' * 60}\nDAgger Training\n{'=' * 60}")
        print(f"Iterations: {self.num_iterations}, Episodes/iter: {self.episodes_per_iter}, Beta: {self.beta_schedule}")

        if initial_states is None or initial_actions is None:
            initial_states, initial_actions = self.collect_expert_demonstrations(self.expert_policy, num_initial_episodes)

        self.states_buffer, self.actions_buffer = [initial_states], [initial_actions]
        iteration_results = []

        for iteration in range(self.num_iterations):
            beta = self.get_beta(iteration)
            all_states, all_actions = np.concatenate(self.states_buffer), np.concatenate(self.actions_buffer)
            print(f"\n{'=' * 40}\nIteration {iteration + 1}/{self.num_iterations} | β={beta:.3f} | Dataset: {len(all_states)}\n{'=' * 40}")

            dataloader = DataLoader(ExpertDataset(all_states, all_actions), batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.bc_epochs):
                loss = self.train_bc_epoch(dataloader)
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{self.bc_epochs}, Loss: {loss:.4f}")

            eval_results = self.evaluate()
            print(f"Eval: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

            if iteration < self.num_iterations - 1:
                new_states, new_actions = self.collect_with_dagger(self.episodes_per_iter, beta)
                self.states_buffer.append(new_states)
                self.actions_buffer.append(new_actions)

            iteration_results.append({"iteration": iteration + 1, "beta": beta, "dataset_size": len(all_states), **eval_results})

        final_eval = self.evaluate(num_episodes=50)
        print(f"\n{'=' * 60}\nFinal: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}\n{'=' * 60}")
        self.save()
        return {"iterations": iteration_results, "final_eval": final_eval}


def simple_expert_policy(env_name: str = "CartPole-v1"):
    if env_name == "CartPole-v1":
        return lambda s: 1 if s[2] + 0.1 * s[3] > 0 else 0
    raise ValueError(f"No expert policy for {env_name}")


class VLADAgger:
    """DAgger for VLA models with image + language inputs."""

    def __init__(self, model, expert_fn, config: Optional[ILConfig] = None):
        from core.device_utils import get_device

        config = config or ILConfig.dagger()
        self.model, self.expert_fn, self.config = model, expert_fn, config
        self.device = get_device("auto")
        self.model = self.model.to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)

        self.num_iterations = config.dagger_iterations
        self.beta_schedule = config.dagger_beta_schedule
        self.initial_beta = config.dagger_initial_beta
        self.data_buffer = []
        os.makedirs(config.output_dir, exist_ok=True)

    def get_beta(self, iteration: int) -> float:
        if self.beta_schedule == "linear":
            return max(0, self.initial_beta * (1 - iteration / self.num_iterations))
        elif self.beta_schedule == "exponential":
            return self.initial_beta * (0.9 ** iteration)
        return self.initial_beta

    def collect_data(self, dataloader, beta: float, num_samples: int = 100) -> List[Dict]:
        self.model.eval()
        collected = []
        with torch.no_grad():
            for batch in dataloader:
                if len(collected) >= num_samples:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                expert_action = self.expert_fn(batch)
                for i in range(len(batch["pixel_values"])):
                    collected.append({
                        "pixel_values": batch["pixel_values"][i].cpu(),
                        "input_ids": batch["input_ids"][i].cpu(),
                        "attention_mask": batch["attention_mask"][i].cpu(),
                        "action": expert_action[i].cpu() if torch.is_tensor(expert_action) else torch.tensor(expert_action[i]),
                    })
        return collected

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss, num_batches = 0, 0
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"], actions=batch["action"])["loss"]
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def train(self, initial_dataloader, num_epochs_per_iter: int = 10):
        print(f"{'=' * 60}\nVLA DAgger Training\n{'=' * 60}")

        for batch in initial_dataloader:
            for i in range(len(batch["pixel_values"])):
                self.data_buffer.append({k: v[i].cpu() for k, v in batch.items()})

        for iteration in range(self.num_iterations):
            beta = self.get_beta(iteration)
            print(f"\nIteration {iteration + 1}/{self.num_iterations} | Beta: {beta:.3f} | Buffer: {len(self.data_buffer)}")

            class BufferDataset:
                def __init__(self, buffer): self.buffer = buffer
                def __len__(self): return len(self.buffer)
                def __getitem__(self, idx): return self.buffer[idx]

            buffer_loader = DataLoader(BufferDataset(self.data_buffer), batch_size=self.config.batch_size, shuffle=True,
                                       collate_fn=lambda x: {k: torch.stack([d[k] for d in x]) for k in x[0].keys()})

            for epoch in range(num_epochs_per_iter):
                loss = self.train_epoch(buffer_loader)
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1}/{num_epochs_per_iter}, Loss: {loss:.4f}")

            if iteration < self.num_iterations - 1:
                self.data_buffer.extend(self.collect_data(initial_dataloader, beta))

        torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, "final_model.pt"))
        print(f"Saved model to {self.config.output_dir}/final_model.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DAgger Training")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dagger_iterations", type=int, default=10)
    parser.add_argument("--dagger_episodes_per_iter", type=int, default=20)
    parser.add_argument("--dagger_beta_schedule", type=str, default="linear", choices=["constant", "linear", "exponential"])
    parser.add_argument("--dagger_initial_beta", type=float, default=1.0)
    parser.add_argument("--bc_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./output/dagger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--expert_data", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = ILConfig(dagger_iterations=args.dagger_iterations, dagger_episodes_per_iter=args.dagger_episodes_per_iter,
                      dagger_beta_schedule=args.dagger_beta_schedule, dagger_initial_beta=args.dagger_initial_beta,
                      bc_epochs=args.bc_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, output_dir=args.output_dir)

    if args.model_path:
        from model.vla import VLAModel
        from model.vla.vla_model import VLAConfig
        from train.finetune.dataset import RobotDataset

        model = VLAModel(VLAConfig())
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
        trainer = VLADAgger(model, lambda b: b.get("action", torch.zeros(len(b["pixel_values"]), 7)), config=config)
        try:
            trainer.train(DataLoader(RobotDataset(dataset_name="lerobot/pusht", max_samples=1000), batch_size=args.batch_size, shuffle=True))
        except Exception as e:
            print(f"Error: {e}")
    else:
        import gymnasium as gym
        trainer = DAgger(env=gym.make(args.env), expert_policy=simple_expert_policy(args.env), config=config)
        if args.resume:
            trainer.load(args.resume)
        if args.expert_data and os.path.exists(args.expert_data):
            data = np.load(args.expert_data)
            trainer.train(initial_states=data["states"], initial_actions=data["actions"])
        else:
            trainer.train()

    print("\nTraining complete!")
