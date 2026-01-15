# Training VLA on Simulation Benchmarks

This comprehensive guide covers the complete training process for Vision-Language-Action models on standard simulation benchmarks, including D4RL, MetaWorld, RoboMimic, and MuJoCo environments.

## Table of Contents

1. [Overview](#overview)
2. [Supported Benchmarks](#supported-benchmarks)
3. [D4RL Offline RL](#d4rl-offline-rl)
4. [MetaWorld Multi-Task](#metaworld-multi-task)
5. [RoboMimic Manipulation](#robomimic-manipulation)
6. [MuJoCo Locomotion](#mujoco-locomotion)
7. [LIBERO Language-Conditioned](#libero-language-conditioned)
8. [Calvin Long-Horizon](#calvin-long-horizon)
9. [Simulation-to-Real Transfer](#simulation-to-real-transfer)
10. [Best Practices](#best-practices)
11. [Evaluation Protocols](#evaluation-protocols)
12. [Benchmark Results](#benchmark-results)

---

## Overview

### Simulation Benchmarks Pipeline

```
+=======================================================================================+
|                        SIMULATION BENCHMARKS TRAINING PIPELINE                         |
+=======================================================================================+
|                                                                                        |
|  BENCHMARK SELECTION                                                                   |
|  +-----------------------------------------------------------------------------------+ |
|  |  D4RL  |  MetaWorld  |  RoboMimic  |  MuJoCo  |  LIBERO  |  Calvin               | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  DATA TYPE                                                                             |
|  +-----------------------------------------------------------------------------------+ |
|  |  Offline Dataset  |  Online Environment  |  Mixed (Offline + Online)              | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  TRAINING METHOD                                                                       |
|  +-----------------------------------------------------------------------------------+ |
|  |  Behavioral Cloning  |  Offline RL (IQL/CQL)  |  Online RL (PPO/SAC)  | Hybrid   | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  EVALUATION                                                                            |
|  +-----------------------------------------------------------------------------------+ |
|  |  Success Rate  |  Normalized Score  |  Episode Return  |  Task Completion        | |
|  +-----------------------------------------------------------------------------------+ |
|                                                                                        |
+=======================================================================================+
```

### Benchmark Comparison

| Benchmark | Tasks | Data Type | Observation | Action | Key Features |
|-----------|-------|-----------|-------------|--------|--------------|
| **D4RL** | 12 | Offline | State | Continuous | Standardized offline RL |
| **MetaWorld** | 50 | Online | Image/State | Continuous | Multi-task manipulation |
| **RoboMimic** | 5 | Offline | Image | Continuous | Human demonstrations |
| **MuJoCo** | 8 | Online | State | Continuous | Locomotion/control |
| **LIBERO** | 130 | Offline | Image | Continuous | Language-conditioned |
| **Calvin** | 34 | Online | Image | Continuous | Long-horizon |

---

## Supported Benchmarks

### Installation

```bash
# Core simulation environments
pip install gymnasium mujoco

# D4RL
pip install d4rl

# MetaWorld
pip install metaworld

# RoboMimic
pip install robomimic

# LIBERO
pip install libero

# Calvin
pip install calvin-env
```

### Environment Setup

```python
from train.benchmarks import BenchmarkEnvironment

# Create benchmark environment
env = BenchmarkEnvironment(
    benchmark="metaworld",
    task="pick-place-v2",
    observation_type="image",  # image, state, both
    action_type="continuous",
    render_mode="rgb_array",
)

# Standard gym interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

---

## D4RL Offline RL

### Dataset Loading

```python
import d4rl
import gymnasium as gym
from train.datasets import D4RLDataset

# Load D4RL dataset
env = gym.make("hopper-medium-v2")
dataset = d4rl.qlearning_dataset(env)

# Convert to PyTorch dataset
class D4RLDataset(torch.utils.data.Dataset):
    """D4RL dataset wrapper."""

    def __init__(
        self,
        env_name: str,
        normalize: bool = True,
        max_episode_length: int = 1000,
    ):
        env = gym.make(env_name)
        self.dataset = d4rl.qlearning_dataset(env)

        self.observations = torch.tensor(self.dataset["observations"], dtype=torch.float32)
        self.actions = torch.tensor(self.dataset["actions"], dtype=torch.float32)
        self.rewards = torch.tensor(self.dataset["rewards"], dtype=torch.float32)
        self.next_observations = torch.tensor(self.dataset["next_observations"], dtype=torch.float32)
        self.terminals = torch.tensor(self.dataset["terminals"], dtype=torch.float32)

        if normalize:
            self._normalize()

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            "observation": self.observations[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_observation": self.next_observations[idx],
            "terminal": self.terminals[idx],
        }

    def _normalize(self):
        """Normalize observations and actions."""
        self.obs_mean = self.observations.mean(dim=0)
        self.obs_std = self.observations.std(dim=0) + 1e-6
        self.observations = (self.observations - self.obs_mean) / self.obs_std
        self.next_observations = (self.next_observations - self.obs_mean) / self.obs_std

        self.action_mean = self.actions.mean(dim=0)
        self.action_std = self.actions.std(dim=0) + 1e-6
        self.actions = (self.actions - self.action_mean) / self.action_std
```

### IQL Training (Recommended)

```python
from train.offline_rl import IQLTrainer
from config import OfflineRLConfig

class IQLTrainer:
    """
    Implicit Q-Learning (IQL) for offline RL.

    Reference: "Offline Reinforcement Learning with Implicit Q-Learning"
               (Kostrikov et al., 2021)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: OfflineRLConfig,
    ):
        self.config = config

        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim=256)
        self.critic_1 = QNetwork(state_dim, action_dim, hidden_dim=256)
        self.critic_2 = QNetwork(state_dim, action_dim, hidden_dim=256)
        self.value = ValueNetwork(state_dim, hidden_dim=256)

        # Target networks
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=config.learning_rate,
        )
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.learning_rate)

    def train(
        self,
        dataset: D4RLDataset,
        num_epochs: int = 1000,
    ):
        """Train IQL on D4RL dataset."""
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        for epoch in range(num_epochs):
            for batch in dataloader:
                obs = batch["observation"].cuda()
                action = batch["action"].cuda()
                reward = batch["reward"].cuda()
                next_obs = batch["next_observation"].cuda()
                terminal = batch["terminal"].cuda()

                # Update value network
                value_loss = self._update_value(obs, action)

                # Update critic networks
                critic_loss = self._update_critics(obs, action, reward, next_obs, terminal)

                # Update actor
                actor_loss = self._update_actor(obs, action)

                # Update target networks
                self._soft_update_targets()

            if epoch % 100 == 0:
                eval_return = self._evaluate()
                print(f"Epoch {epoch}: Return = {eval_return:.2f}")

    def _update_value(self, obs: torch.Tensor, action: torch.Tensor) -> float:
        """Update value network with expectile regression."""
        with torch.no_grad():
            q1 = self.target_critic_1(obs, action)
            q2 = self.target_critic_2(obs, action)
            q = torch.min(q1, q2)

        v = self.value(obs)

        # Expectile loss
        diff = q - v
        weight = torch.where(
            diff > 0,
            self.config.expectile,
            1 - self.config.expectile,
        )
        value_loss = (weight * (diff ** 2)).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return value_loss.item()

    def _update_critics(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminal: torch.Tensor,
    ) -> float:
        """Update critic networks with TD learning."""
        with torch.no_grad():
            next_v = self.value(next_obs)
            target_q = reward + (1 - terminal) * self.config.gamma * next_v

        q1 = self.critic_1(obs, action)
        q2 = self.critic_2(obs, action)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, obs: torch.Tensor, action: torch.Tensor) -> float:
        """Update actor with advantage-weighted regression."""
        with torch.no_grad():
            v = self.value(obs)
            q1 = self.critic_1(obs, action)
            q2 = self.critic_2(obs, action)
            q = torch.min(q1, q2)

            advantage = q - v
            exp_advantage = torch.exp(advantage * self.config.temperature)
            exp_advantage = torch.clamp(exp_advantage, max=100.0)

        # Log probability of actions
        dist = self.actor(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Weighted BC loss
        actor_loss = -(exp_advantage * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()


# Training configuration
d4rl_config = OfflineRLConfig(
    algorithm="iql",
    expectile=0.7,
    temperature=3.0,
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
)

# Train on D4RL benchmarks
datasets = {
    "hopper-medium-v2": D4RLDataset("hopper-medium-v2"),
    "hopper-medium-expert-v2": D4RLDataset("hopper-medium-expert-v2"),
    "walker2d-medium-expert-v2": D4RLDataset("walker2d-medium-expert-v2"),
}

for name, dataset in datasets.items():
    print(f"Training IQL on {name}")
    trainer = IQLTrainer(
        state_dim=dataset.observations.shape[1],
        action_dim=dataset.actions.shape[1],
        config=d4rl_config,
    )
    trainer.train(dataset)
```

### CQL Training

```python
from train.offline_rl import CQLTrainer

class CQLTrainer:
    """
    Conservative Q-Learning (CQL) for offline RL.

    Reference: "Conservative Q-Learning for Offline Reinforcement Learning"
               (Kumar et al., 2020)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: OfflineRLConfig,
    ):
        self.config = config

        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim)
        self.critic_1 = QNetwork(state_dim, action_dim)
        self.critic_2 = QNetwork(state_dim, action_dim)

        # CQL alpha (conservative penalty weight)
        self.log_alpha_cql = torch.tensor(np.log(config.cql_alpha), requires_grad=True)

    def _compute_cql_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CQL conservative loss."""
        B = obs.shape[0]

        # Sample random actions
        random_actions = torch.rand(B, self.config.num_random_actions, self.config.action_dim) * 2 - 1
        random_actions = random_actions.to(obs.device)

        # Sample policy actions
        policy_dist = self.actor(obs)
        policy_actions = policy_dist.sample((self.config.num_random_actions,)).permute(1, 0, 2)

        # Compute Q-values for random and policy actions
        obs_repeat = obs.unsqueeze(1).expand(-1, self.config.num_random_actions, -1)

        q1_random = self.critic_1(obs_repeat, random_actions)
        q2_random = self.critic_2(obs_repeat, random_actions)
        q1_policy = self.critic_1(obs_repeat, policy_actions)
        q2_policy = self.critic_2(obs_repeat, policy_actions)

        # Logsumexp for conservative penalty
        cat_q1 = torch.cat([q1_random, q1_policy], dim=1)
        cat_q2 = torch.cat([q2_random, q2_policy], dim=1)

        cql_penalty_1 = torch.logsumexp(cat_q1 / self.config.cql_temperature, dim=1)
        cql_penalty_2 = torch.logsumexp(cat_q2 / self.config.cql_temperature, dim=1)

        # Q-value on dataset actions
        q1_data = self.critic_1(obs, action)
        q2_data = self.critic_2(obs, action)

        # CQL loss
        cql_loss = (
            self.log_alpha_cql.exp() * (cql_penalty_1.mean() + cql_penalty_2.mean() - q1_data.mean() - q2_data.mean())
        )

        return cql_loss
```

---

## MetaWorld Multi-Task

### Multi-Task Training

```python
import metaworld
from train.benchmarks import MetaWorldDataset

class MetaWorldTrainer:
    """
    Train VLA on MetaWorld's 50 manipulation tasks.
    """

    def __init__(
        self,
        model: ManipulationVLA,
        config: MultiTaskConfig,
    ):
        self.model = model
        self.config = config

        # Create MetaWorld benchmark
        self.benchmark = metaworld.ML45()  # 45 training tasks

    def create_multi_task_dataset(
        self,
        demos_per_task: int = 100,
    ) -> MetaWorldDataset:
        """Create dataset from all tasks."""
        all_demos = []

        for task_name in self.benchmark.train_tasks:
            env = self.benchmark.train_classes[task_name]()
            task = random.choice([t for t in self.benchmark.train_tasks if t.env_name == task_name])
            env.set_task(task)

            # Collect demonstrations
            demos = self._collect_demonstrations(env, demos_per_task)

            for demo in demos:
                demo["task_name"] = task_name
                demo["task_description"] = self._get_task_description(task_name)

            all_demos.extend(demos)

        return MetaWorldDataset(all_demos)

    def train_language_conditioned(
        self,
        dataset: MetaWorldDataset,
        num_epochs: int = 200,
    ):
        """Train with language conditioning for task specification."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                output = self.model(
                    images=batch["image"].cuda(),
                    instruction=batch["task_description"],
                )

                loss = F.mse_loss(output["action"], batch["action"].cuda())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Evaluate on held-out tasks
            if epoch % 10 == 0:
                success_rate = self._evaluate_multi_task()
                print(f"Epoch {epoch}: Success Rate = {success_rate:.2%}")

    def _get_task_description(self, task_name: str) -> str:
        """Get natural language description for task."""
        descriptions = {
            "reach-v2": "Reach to the target position",
            "push-v2": "Push the object to the goal",
            "pick-place-v2": "Pick up the object and place it at the target",
            "door-open-v2": "Open the door",
            "drawer-open-v2": "Open the drawer",
            "drawer-close-v2": "Close the drawer",
            "button-press-v2": "Press the button",
            "peg-insert-side-v2": "Insert the peg into the hole from the side",
            "window-open-v2": "Open the window",
            "window-close-v2": "Close the window",
            # ... more tasks
        }
        return descriptions.get(task_name, f"Perform the {task_name} task")

    def _evaluate_multi_task(self) -> float:
        """Evaluate on held-out tasks."""
        total_success = 0
        total_episodes = 0

        for task_name in self.benchmark.test_tasks[:5]:  # Sample test tasks
            env = self.benchmark.test_classes[task_name]()
            task = random.choice([t for t in self.benchmark.test_tasks if t.env_name == task_name])
            env.set_task(task)

            for episode in range(10):
                obs, info = env.reset()
                done = False
                step = 0

                while not done and step < 500:
                    # Get image observation
                    image = self._get_image(env)
                    instruction = self._get_task_description(task_name)

                    with torch.no_grad():
                        output = self.model(
                            images=image.unsqueeze(0).cuda(),
                            instruction=instruction,
                        )

                    action = output["action"][0].cpu().numpy()
                    obs, reward, done, truncated, info = env.step(action)
                    step += 1

                if info.get("success"):
                    total_success += 1
                total_episodes += 1

        return total_success / total_episodes


# Multi-task dataset
dataset = MetaWorldDataset(
    tasks="all",  # All 50 tasks
    demos_per_task=100,
)

# Language-conditioned model
model = create_vla_model(
    vision_encoder="siglip-base",
    llm="qwen2-1.5b",
    action_dim=4,  # End-effector control
    use_language=True,
)

# Multi-task training
config = ILConfig(
    learning_rate=1e-4,
    batch_size=128,
    num_epochs=200,
    task_sampling="uniform",
)

trainer = MetaWorldTrainer(model, config)
trainer.train_language_conditioned(dataset, num_epochs=200)
```

---

## RoboMimic Manipulation

### RoboMimic Training

```python
import robomimic
from robomimic.utils.dataset import SequenceDataset

class RoboMimicTrainer:
    """
    Train on RoboMimic human demonstration datasets.

    Tasks:
    - Lift: Pick up cube
    - Can: Pick and place can
    - Square: Assemble square
    - Transport: Bimanual transport
    - Tool Hang: Hang tool on rack
    """

    def __init__(
        self,
        model: ManipulationVLA,
        config: RoboMimicConfig,
    ):
        self.model = model
        self.config = config

    def load_dataset(
        self,
        task: str = "lift",
        dataset_type: str = "ph",  # ph (proficient human), mh (multi-human)
    ) -> torch.utils.data.Dataset:
        """Load RoboMimic dataset."""
        dataset_path = f"datasets/{task}/{dataset_type}/low_dim.hdf5"

        dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            dataset_keys=["actions"],
            seq_length=10,  # History length
            frame_stack=1,
            pad_seq_length=True,
            pad_frame_stack=True,
        )

        return dataset

    def train_with_action_chunking(
        self,
        dataset,
        chunk_size: int = 16,
        num_epochs: int = 200,
    ):
        """
        Train with action chunking for temporal consistency.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                # Get observation history
                obs_history = batch["obs"]  # Dict of observation keys

                # Get action chunk
                action_chunk = batch["actions"][:, :chunk_size]

                output = self.model(
                    observations=obs_history,
                    chunk_size=chunk_size,
                )

                # MSE loss on action chunk
                loss = F.mse_loss(output["action"], action_chunk)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 10 == 0:
                success_rate = self._evaluate()
                print(f"Epoch {epoch}: Success Rate = {success_rate:.2%}")


# RoboMimic training recipes
ROBOMIMIC_LIFT_CONFIG = RoboMimicConfig(
    task="lift",
    dataset_type="ph",
    learning_rate=1e-4,
    batch_size=16,
    chunk_size=10,
    num_epochs=200,
)

ROBOMIMIC_CAN_CONFIG = RoboMimicConfig(
    task="can",
    dataset_type="ph",
    learning_rate=1e-4,
    batch_size=16,
    chunk_size=16,
    num_epochs=300,
)

ROBOMIMIC_SQUARE_CONFIG = RoboMimicConfig(
    task="square",
    dataset_type="mh",  # Multi-human for harder task
    learning_rate=5e-5,
    batch_size=8,
    chunk_size=20,
    num_epochs=500,
)
```

---

## MuJoCo Locomotion

### MuJoCo Training

```python
import gymnasium as gym
from train.online_rl import PPOTrainer, SACTrainer

class MuJoCoTrainer:
    """
    Train on MuJoCo continuous control tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        env_name: str,
        config: OnlineRLConfig,
    ):
        self.model = model
        self.env = gym.make(env_name)
        self.config = config

    def train_ppo(
        self,
        total_timesteps: int = 1_000_000,
    ):
        """Train with PPO."""
        trainer = PPOTrainer(
            model=self.model,
            env=self.env,
            learning_rate=self.config.learning_rate,
            clip_range=self.config.ppo_clip_range,
            entropy_coef=self.config.entropy_coef,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
        )

        trainer.learn(total_timesteps)

    def train_sac(
        self,
        total_timesteps: int = 1_000_000,
    ):
        """Train with SAC."""
        trainer = SACTrainer(
            model=self.model,
            env=self.env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
        )

        trainer.learn(total_timesteps)


# MuJoCo task configurations
MUJOCO_CONFIGS = {
    "Ant-v4": OnlineRLConfig(
        algorithm="ppo",
        learning_rate=3e-4,
        ppo_clip_range=0.2,
        entropy_coef=0.0,
        n_steps=2048,
        batch_size=64,
        total_timesteps=3_000_000,
    ),
    "HalfCheetah-v4": OnlineRLConfig(
        algorithm="sac",
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        total_timesteps=1_000_000,
    ),
    "Humanoid-v4": OnlineRLConfig(
        algorithm="ppo",
        learning_rate=5e-5,
        ppo_clip_range=0.2,
        entropy_coef=0.001,
        n_steps=2048,
        batch_size=512,
        total_timesteps=10_000_000,
    ),
}
```

---

## LIBERO Language-Conditioned

### LIBERO Training

```python
from libero.libero import benchmark

class LIBEROTrainer:
    """
    Train on LIBERO language-conditioned manipulation benchmark.

    130 tasks across 10 task suites with natural language instructions.
    """

    def __init__(
        self,
        model: ManipulationVLA,
        config: LIBEROConfig,
    ):
        self.model = model
        self.config = config

        # Load LIBERO benchmark
        self.benchmark = benchmark.get_benchmark_dict()

    def load_task_suite(
        self,
        suite_name: str = "libero_spatial",
    ):
        """Load a LIBERO task suite."""
        suite = self.benchmark[suite_name]

        datasets = []
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            demo_file = task.demo_file

            dataset = LiberoDataset(
                demo_file=demo_file,
                task_description=task.language,
            )
            datasets.append(dataset)

        return torch.utils.data.ConcatDataset(datasets)

    def train_language_conditioned(
        self,
        suite_name: str = "libero_spatial",
        num_epochs: int = 100,
    ):
        """Train with language conditioning."""
        dataset = self.load_task_suite(suite_name)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataloader:
                output = self.model(
                    images=batch["image"].cuda(),
                    instruction=batch["instruction"],
                )

                loss = F.mse_loss(output["action"], batch["action"].cuda())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 10 == 0:
                success = self._evaluate(suite_name)
                print(f"Epoch {epoch}: {suite_name} Success = {success:.2%}")

    def _evaluate(self, suite_name: str, num_episodes: int = 50) -> float:
        """Evaluate on LIBERO suite."""
        suite = self.benchmark[suite_name]
        total_success = 0

        for task_id in range(min(suite.n_tasks, 10)):
            task = suite.get_task(task_id)
            env = task.make_env()

            for ep in range(num_episodes // suite.n_tasks):
                obs = env.reset()
                done = False
                step = 0

                while not done and step < 300:
                    image = self._process_obs(obs)

                    with torch.no_grad():
                        output = self.model(
                            images=image.unsqueeze(0).cuda(),
                            instruction=task.language,
                        )

                    action = output["action"][0].cpu().numpy()
                    obs, reward, done, info = env.step(action)
                    step += 1

                if info.get("success"):
                    total_success += 1

        return total_success / num_episodes


# LIBERO task suites
LIBERO_SUITES = [
    "libero_spatial",    # Spatial reasoning
    "libero_object",     # Object manipulation
    "libero_goal",       # Goal-conditioned
    "libero_10",         # 10 diverse tasks
    "libero_90",         # 90 training tasks
]
```

---

## Calvin Long-Horizon

### Calvin Training

```python
from calvin_env.envs.play_table_env import PlayTableSimEnv

class CalvinTrainer:
    """
    Train on CALVIN long-horizon manipulation benchmark.

    34 tasks, evaluated on 1000-step rollouts with chained instructions.
    """

    def __init__(
        self,
        model: ManipulationVLA,
        config: CalvinConfig,
    ):
        self.model = model
        self.config = config

    def train_multi_task(
        self,
        data_path: str,
        num_epochs: int = 100,
    ):
        """Train on Calvin multi-task dataset."""
        dataset = CalvinDataset(
            data_path=data_path,
            split="training",
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                output = self.model(
                    images=batch["rgb_static"].cuda(),
                    gripper_images=batch["rgb_gripper"].cuda(),
                    instruction=batch["language_annotation"],
                )

                loss = F.mse_loss(output["action"], batch["action"].cuda())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 10 == 0:
                avg_len = self._evaluate_chain()
                print(f"Epoch {epoch}: Avg Chain Length = {avg_len:.2f}")

    def _evaluate_chain(self, num_sequences: int = 100) -> float:
        """Evaluate chained task completion."""
        env = PlayTableSimEnv(
            observation_space_keys=["rgb_static", "rgb_gripper"],
        )

        total_completed = 0
        max_tasks = 5

        for seq in range(num_sequences):
            obs = env.reset()
            tasks_completed = 0

            for task_idx in range(max_tasks):
                instruction = self._sample_instruction()
                success = self._execute_task(env, obs, instruction)

                if success:
                    tasks_completed += 1
                    obs = env.get_obs()
                else:
                    break

            total_completed += tasks_completed

        return total_completed / num_sequences

    def _execute_task(
        self,
        env,
        obs: Dict,
        instruction: str,
        max_steps: int = 360,
    ) -> bool:
        """Execute single task in environment."""
        for step in range(max_steps):
            image = self._process_obs(obs)

            with torch.no_grad():
                output = self.model(
                    images=image.unsqueeze(0).cuda(),
                    instruction=instruction,
                )

            action = output["action"][0].cpu().numpy()
            obs, reward, done, info = env.step(action)

            if info.get("success"):
                return True

        return False
```

---

## Simulation-to-Real Transfer

### Domain Randomization

```python
class DomainRandomization:
    """
    Domain randomization for sim-to-real transfer.
    """

    def __init__(
        self,
        config: DomainRandomizationConfig,
    ):
        self.config = config

    def randomize_visual(self, image: torch.Tensor) -> torch.Tensor:
        """Randomize visual appearance."""
        # Color jitter
        if self.config.use_color_jitter:
            image = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            )(image)

        # Random noise
        if self.config.use_noise:
            noise = torch.randn_like(image) * 0.1
            image = image + noise

        # Cutout augmentation
        if self.config.use_cutout:
            image = self._random_cutout(image)

        return torch.clamp(image, 0, 1)

    def randomize_dynamics(self, env):
        """Randomize physics parameters."""
        if hasattr(env, 'model'):
            model = env.model

            # Randomize friction
            model.geom_friction *= np.random.uniform(0.5, 1.5, model.geom_friction.shape)

            # Randomize mass
            model.body_mass *= np.random.uniform(0.8, 1.2, model.body_mass.shape)

            # Randomize damping
            model.dof_damping *= np.random.uniform(0.5, 2.0, model.dof_damping.shape)


class SimToRealTrainer:
    """Train with sim-to-real transfer techniques."""

    def train_with_domain_randomization(
        self,
        model: nn.Module,
        env,
        num_episodes: int = 10000,
    ):
        """Train with extensive domain randomization."""
        domain_rand = DomainRandomization(self.config)

        for episode in range(num_episodes):
            # Randomize environment
            domain_rand.randomize_dynamics(env)

            obs = env.reset()
            done = False

            while not done:
                # Randomize visual observation
                image = domain_rand.randomize_visual(obs["image"])

                action = model.get_action(image)
                obs, reward, done, info = env.step(action)

                # Store for training
                self._store_transition(image, action, reward)

            # Train on collected data
            self._train_step()
```

---

## Best Practices

### Training Recommendations

```python
# General best practices for simulation benchmarks

BENCHMARK_BEST_PRACTICES = {
    "d4rl": {
        "algorithm": "iql",  # Most stable for offline
        "expectile": 0.7,    # 0.7-0.9 range
        "temperature": 3.0,
        "normalize": True,
        "batch_size": 256,
    },
    "metaworld": {
        "algorithm": "bc_then_rl",  # BC warm-start
        "multi_task": True,
        "language_conditioning": True,
        "action_repeat": 2,
        "success_threshold": 0.8,
    },
    "robomimic": {
        "algorithm": "bc_rnn",  # Use history
        "chunk_size": 10,
        "gmm_modes": 5,
        "use_proprio": True,
    },
    "mujoco": {
        "algorithm": "ppo",  # For locomotion
        "entropy_coef": 0.0,  # No entropy for continuous
        "normalize_advantage": True,
        "gae_lambda": 0.95,
    },
    "libero": {
        "algorithm": "bc",
        "language_encoder": "t5-base",
        "image_aug": True,
        "action_chunk": 16,
    },
    "calvin": {
        "algorithm": "bc",
        "multi_view": True,  # Static + gripper cam
        "chain_evaluation": True,
        "max_chain_length": 5,
    },
}
```

---

## Evaluation Protocols

### Standardized Evaluation

```python
class BenchmarkEvaluator:
    """Standardized evaluation across benchmarks."""

    def evaluate_d4rl(
        self,
        model: nn.Module,
        env_name: str,
        num_episodes: int = 100,
    ) -> Dict[str, float]:
        """Evaluate on D4RL with normalized score."""
        env = gym.make(env_name)
        returns = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_return = 0
            done = False

            while not done:
                action = model.get_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward

            returns.append(episode_return)

        # Normalize score
        ref_min = d4rl.infos.REF_MIN_SCORE[env_name]
        ref_max = d4rl.infos.REF_MAX_SCORE[env_name]

        normalized = 100 * (np.mean(returns) - ref_min) / (ref_max - ref_min)

        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "normalized_score": normalized,
        }

    def evaluate_metaworld(
        self,
        model: nn.Module,
        task_name: str,
        num_episodes: int = 50,
    ) -> Dict[str, float]:
        """Evaluate MetaWorld task."""
        benchmark = metaworld.ML1(task_name)
        env = benchmark.train_classes[task_name]()

        successes = []
        for _ in range(num_episodes):
            task = random.choice([t for t in benchmark.train_tasks if t.env_name == task_name])
            env.set_task(task)

            obs, _ = env.reset()
            success = False

            for _ in range(500):
                action = model.get_action(obs)
                obs, reward, done, truncated, info = env.step(action)

                if info.get("success"):
                    success = True
                    break

            successes.append(success)

        return {
            "success_rate": np.mean(successes),
        }
```

---

## Benchmark Results

### Expected Performance

```
+====================================================================================+
|                          SIMULATION BENCHMARK RESULTS                               |
+====================================================================================+
|                                                                                     |
| D4RL (IQL):                                                                         |
| Dataset                    | Score  | VLA Baseline | State-of-Art                  |
| --------------------------|--------|--------------|-------------------------------|
| hopper-medium-v2           | 66.3   | 68.2         | 71.5 (CQL)                   |
| hopper-medium-expert-v2    | 91.5   | 93.1         | 95.8 (IQL)                   |
| walker2d-medium-v2         | 78.3   | 80.5         | 83.7 (TD3+BC)                |
| walker2d-medium-expert-v2  | 109.6  | 111.2        | 113.0 (IQL)                  |
|                                                                                     |
| MetaWorld (Multi-Task):                                                             |
| Task Suite                 | Success Rate | VLA      | Baseline (BC)               |
| --------------------------|--------------|----------|------------------------------|
| ML1 (single task)          | 85-95%       | 92%      | 78%                          |
| ML10 (10 tasks)            | 70-85%       | 82%      | 65%                          |
| ML45 (45 tasks)            | 55-70%       | 68%      | 52%                          |
|                                                                                     |
| LIBERO:                                                                             |
| Suite                      | Success Rate | Avg Steps | Language Accuracy           |
| --------------------------|--------------|-----------|------------------------------|
| LIBERO-Spatial             | 78.5%        | 145       | 85.2%                        |
| LIBERO-Object              | 72.3%        | 168       | 82.1%                        |
| LIBERO-Goal                | 65.8%        | 192       | 78.4%                        |
| LIBERO-10                  | 68.4%        | 175       | 80.5%                        |
|                                                                                     |
| Calvin:                                                                             |
| Metric                     | VLA    | HULC   | RT-1                           |
| --------------------------|--------|--------|--------------------------------|
| Avg Chain Length (D→D)     | 2.8    | 2.6    | 2.3                            |
| Task Success (single)      | 82.5%  | 78.3%  | 75.1%                          |
| Long-horizon (5 tasks)     | 38.2%  | 32.5%  | 28.4%                          |
|                                                                                     |
+====================================================================================+
```

---

## Summary

This guide covered simulation benchmark training:

1. **D4RL**: Offline RL with IQL/CQL
2. **MetaWorld**: Multi-task manipulation
3. **RoboMimic**: Human demonstration learning
4. **MuJoCo**: Locomotion with PPO/SAC
5. **LIBERO**: Language-conditioned manipulation
6. **Calvin**: Long-horizon chained tasks

**Key recommendations:**
- Use IQL for D4RL offline RL (most stable)
- Language conditioning improves multi-task generalization
- Action chunking helps with temporal consistency
- Domain randomization enables sim-to-real transfer
- Evaluate with standard protocols for fair comparison

---

## Related Documents

- [Training Pipeline Overview](training_pipeline_overview.md)
- [Training Recipes](training_recipes.md)
- [Robot Manipulation Training](training_robot_manipulation.md)
- [Real Robot Deployment](training_real_robot_deployment.md)
- [Architecture Guide](architecture.md)
