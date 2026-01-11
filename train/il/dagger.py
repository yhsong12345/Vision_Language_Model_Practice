"""
DAgger (Dataset Aggregation)

Interactive imitation learning that addresses distribution shift:
1. Train initial policy via BC
2. Execute learned policy in environment
3. Query expert for correct actions
4. Aggregate new data with existing dataset
5. Repeat
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
import numpy as np
from tqdm import tqdm

from .base_trainer import ILTrainer, ExpertDataset, PolicyNetwork

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.training_config import ILConfig


class DAgger(ILTrainer):
    """
    DAgger (Dataset Aggregation) trainer.

    Addresses the distribution shift problem in behavioral cloning
    by iteratively collecting data under the learned policy and
    querying an expert for corrections.

    Algorithm:
    1. Initialize dataset D with expert demonstrations
    2. For iteration i = 1 to N:
       a. Train policy π_i on D
       b. Execute π_i in environment
       c. Query expert for correct actions
       d. Aggregate: D = D ∪ new_data
    3. Return final policy

    The β (beta) parameter controls the mix of expert and learned policy
    during data collection.
    """

    def __init__(
        self,
        env,
        expert_policy,
        policy: Optional[nn.Module] = None,
        config: Optional[ILConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = ILConfig.dagger()

        super().__init__(env, policy, config.output_dir, **kwargs)

        self.config = config
        self.expert_policy = expert_policy

        # DAgger specific params
        self.num_iterations = config.dagger_iterations
        self.episodes_per_iter = config.dagger_episodes_per_iter
        self.beta_schedule = config.dagger_beta_schedule
        self.initial_beta = config.dagger_initial_beta
        self.bc_epochs = config.bc_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate

        # Aggregated dataset
        self.states_buffer = []
        self.actions_buffer = []

        # Optimizer
        self.optimizer = Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
        )

        # Loss function
        if self.continuous:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def get_beta(self, iteration: int) -> float:
        """
        Get β value for current iteration.

        β = probability of using expert policy.
        """
        if self.beta_schedule == "constant":
            return self.initial_beta

        elif self.beta_schedule == "linear":
            # Linear decay from initial_beta to 0
            return max(0, self.initial_beta * (1 - iteration / self.num_iterations))

        elif self.beta_schedule == "exponential":
            # Exponential decay
            decay_rate = 0.9
            return self.initial_beta * (decay_rate ** iteration)

        else:
            return self.initial_beta

    def collect_with_dagger(
        self,
        num_episodes: int,
        beta: float,
        max_steps: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect data using β-mixture of expert and learned policy.

        Args:
            num_episodes: Number of episodes to collect
            beta: Probability of using expert action
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (states, expert_actions)
        """
        states = []
        actions = []
        episode_rewards = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Get expert action (always needed for labeling)
                expert_action = self.expert_policy(state)

                # Decide which action to execute
                if np.random.random() < beta:
                    action = expert_action
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                    action = self.policy.get_action(state_tensor, deterministic=False)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()

                # Store state and EXPERT action (key difference from BC)
                states.append(state)
                actions.append(expert_action)

                # Execute action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

                state = next_state

            episode_rewards.append(episode_reward)

        print(f"Collected {len(states)} transitions (β={beta:.2f}), "
              f"mean reward: {np.mean(episode_rewards):.2f}")

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32 if self.continuous else np.int64),
        )

    def train_bc_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch of behavioral cloning."""
        self.policy.train()
        total_loss = 0
        num_batches = 0

        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(self.device)
            actions_batch = actions_batch.to(self.device)

            predicted_actions = self.policy(states_batch)

            if self.continuous:
                loss = self.criterion(predicted_actions, actions_batch)
            else:
                loss = self.criterion(predicted_actions, actions_batch.long())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        initial_states: Optional[np.ndarray] = None,
        initial_actions: Optional[np.ndarray] = None,
        num_initial_episodes: int = 20,
    ):
        """
        Run DAgger training.

        Args:
            initial_states: Initial expert states (optional)
            initial_actions: Initial expert actions (optional)
            num_initial_episodes: Episodes for initial BC if no data provided
        """
        print("=" * 60)
        print("DAgger Training")
        print("=" * 60)
        print(f"Iterations: {self.num_iterations}")
        print(f"Episodes per iteration: {self.episodes_per_iter}")
        print(f"Beta schedule: {self.beta_schedule}")

        # Collect initial demonstrations if not provided
        if initial_states is None or initial_actions is None:
            print("\nCollecting initial expert demonstrations...")
            initial_states, initial_actions = self.collect_expert_demonstrations(
                self.expert_policy, num_initial_episodes
            )

        # Initialize buffer with initial data
        self.states_buffer = [initial_states]
        self.actions_buffer = [initial_actions]

        # DAgger iterations
        iteration_results = []

        for iteration in range(self.num_iterations):
            print(f"\n{'=' * 40}")
            print(f"DAgger Iteration {iteration + 1}/{self.num_iterations}")
            print(f"{'=' * 40}")

            # Get current beta
            beta = self.get_beta(iteration)
            print(f"β = {beta:.3f}")

            # Create dataset from aggregated data
            all_states = np.concatenate(self.states_buffer)
            all_actions = np.concatenate(self.actions_buffer)

            print(f"Dataset size: {len(all_states)}")

            dataset = ExpertDataset(all_states, all_actions)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )

            # Train policy via BC
            print("Training policy...")
            for epoch in range(self.bc_epochs):
                loss = self.train_bc_epoch(dataloader)

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{self.bc_epochs}, Loss: {loss:.4f}")

            # Evaluate current policy
            eval_results = self.evaluate()
            print(f"Evaluation: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

            # Collect new data (if not last iteration)
            if iteration < self.num_iterations - 1:
                new_states, new_actions = self.collect_with_dagger(
                    num_episodes=self.episodes_per_iter,
                    beta=beta,
                )

                # Aggregate data
                self.states_buffer.append(new_states)
                self.actions_buffer.append(new_actions)

            iteration_results.append({
                "iteration": iteration + 1,
                "beta": beta,
                "dataset_size": len(all_states),
                "mean_reward": eval_results["mean_reward"],
                "std_reward": eval_results["std_reward"],
            })

        # Final evaluation
        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)

        final_eval = self.evaluate(num_episodes=50)
        print(f"Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")

        # Save final model
        self.save()

        return {
            "iterations": iteration_results,
            "final_eval": final_eval,
        }


def simple_expert_policy(env_name: str = "CartPole-v1"):
    """
    Create a simple expert policy for testing.
    """
    if env_name == "CartPole-v1":
        def policy(state):
            pole_angle = state[2]
            pole_velocity = state[3]

            if pole_angle + 0.1 * pole_velocity > 0:
                return 1
            else:
                return 0
        return policy

    else:
        raise ValueError(f"No expert policy for {env_name}")


if __name__ == "__main__":
    print("DAgger Trainer")
    print("Interactive imitation learning with dataset aggregation")

    # Quick test
    try:
        import gymnasium as gym

        env = gym.make("CartPole-v1")
        expert = simple_expert_policy("CartPole-v1")

        trainer = DAgger(
            env=env,
            expert_policy=expert,
            config=ILConfig(
                dagger_iterations=3,
                dagger_episodes_per_iter=10,
                bc_epochs=20,
            ),
        )

        print("DAgger trainer created successfully")

    except ImportError:
        print("Gymnasium not installed")
