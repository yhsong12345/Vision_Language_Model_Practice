"""
VLA Evaluator

Comprehensive evaluation for Vision-Language-Action models in:
- Simulation environments
- Real robot settings
- Various benchmarks
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from dataclasses import dataclass
import json
from tqdm import tqdm
from collections import defaultdict

from .metrics import (
    compute_success_rate,
    compute_trajectory_metrics,
    compute_action_metrics,
)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    num_episodes: int = 100
    max_steps: int = 500
    seed: int = 42
    deterministic: bool = True
    save_trajectories: bool = False
    save_videos: bool = False
    output_dir: str = "./eval_results"


class VLAEvaluator:
    """
    Evaluator for Vision-Language-Action models.

    Supports:
    - Multiple evaluation environments
    - Various success criteria
    - Trajectory recording
    - Video recording
    - Comprehensive metrics
    """

    def __init__(
        self,
        model,
        config: Optional[EvalConfig] = None,
    ):
        self.model = model
        self.config = config or EvalConfig()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Set seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def evaluate_gym(
        self,
        env,
        num_episodes: int = None,
        instructions: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a Gym environment.

        Args:
            env: Gymnasium environment
            num_episodes: Number of evaluation episodes
            instructions: List of language instructions

        Returns:
            Evaluation results dictionary
        """
        if num_episodes is None:
            num_episodes = self.config.num_episodes

        if instructions is None:
            instructions = ["Perform the task."]

        results = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_successes": [],
            "trajectories": [],
        }

        for ep in tqdm(range(num_episodes), desc="Evaluating"):
            state, info = env.reset()
            instruction = instructions[ep % len(instructions)]

            episode_reward = 0
            episode_length = 0
            trajectory = []
            done = False

            while not done and episode_length < self.config.max_steps:
                # Get action from model
                action = self._get_action(state, instruction)

                # Environment step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Record trajectory
                if self.config.save_trajectories:
                    trajectory.append({
                        "state": state.tolist() if isinstance(state, np.ndarray) else state,
                        "action": action.tolist() if isinstance(action, np.ndarray) else action,
                        "reward": float(reward),
                    })

                episode_reward += reward
                episode_length += 1
                state = next_state

            results["episode_rewards"].append(episode_reward)
            results["episode_lengths"].append(episode_length)
            results["episode_successes"].append(info.get("success", reward > 0))

            if self.config.save_trajectories:
                results["trajectories"].append(trajectory)

        # Compute summary statistics
        results["summary"] = {
            "mean_reward": np.mean(results["episode_rewards"]),
            "std_reward": np.std(results["episode_rewards"]),
            "mean_length": np.mean(results["episode_lengths"]),
            "success_rate": np.mean(results["episode_successes"]),
            "num_episodes": num_episodes,
        }

        return results

    def evaluate_robot_dataset(
        self,
        dataloader,
        compute_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a robot manipulation dataset.

        Args:
            dataloader: DataLoader with robot data
            compute_metrics: Whether to compute detailed metrics

        Returns:
            Evaluation results
        """
        all_predicted = []
        all_ground_truth = []
        total_loss = 0
        num_samples = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch["action"],
                )

            predicted = outputs["predicted_actions"].cpu().numpy()
            ground_truth = batch["action"].cpu().numpy()

            all_predicted.append(predicted)
            all_ground_truth.append(ground_truth)
            total_loss += outputs["loss"].item() * len(predicted)
            num_samples += len(predicted)

        all_predicted = np.concatenate(all_predicted)
        all_ground_truth = np.concatenate(all_ground_truth)

        results = {
            "mse_loss": total_loss / num_samples,
            "num_samples": num_samples,
        }

        if compute_metrics:
            action_metrics = compute_action_metrics(all_predicted, all_ground_truth)
            results.update(action_metrics)

        return results

    def evaluate_lerobot(
        self,
        dataset_name: str = "lerobot/pusht",
        num_episodes: int = 50,
    ) -> Dict[str, Any]:
        """
        Evaluate on LeRobot environments.

        Args:
            dataset_name: LeRobot dataset/environment name
            num_episodes: Number of episodes

        Returns:
            Evaluation results
        """
        try:
            from lerobot.common.envs.factory import make_env

            env = make_env(dataset_name)
            return self.evaluate_gym(env, num_episodes)

        except ImportError:
            print("LeRobot not installed")
            return {"error": "LeRobot not installed"}

    def evaluate_simulated_robot(
        self,
        robot_type: str = "panda",
        tasks: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate on simulated robot environments.

        Args:
            robot_type: Type of robot (panda, ur5, xarm, etc.)
            tasks: List of tasks to evaluate

        Returns:
            Evaluation results
        """
        try:
            # Try MuJoCo environments
            import gymnasium as gym

            if tasks is None:
                tasks = [f"{robot_type}_reach", f"{robot_type}_push"]

            all_results = {}

            for task in tasks:
                try:
                    env = gym.make(task)
                    results = self.evaluate_gym(env, self.config.num_episodes // len(tasks))
                    all_results[task] = results["summary"]
                except Exception as e:
                    all_results[task] = {"error": str(e)}

            return all_results

        except ImportError:
            return {"error": "Gymnasium robotics not installed"}

    def _get_action(
        self,
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """Get action from model given state and instruction."""
        # Handle different model interfaces
        if hasattr(self.model, 'predict_action'):
            # VLA model with predict_action method
            action = self.model.predict_action(
                image=state,  # Assumes state is image or can be converted
                instruction=instruction,
                device=self.device,
            )
        elif hasattr(self.model, 'get_action'):
            # Simple policy network
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            action = self.model.get_action(state_tensor, deterministic=self.config.deterministic)
        else:
            # Default: forward pass
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.model(state_tensor)
            action = action.squeeze(0)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        return action

    def save_results(self, results: Dict[str, Any], filename: str = "eval_results.json"):
        """Save evaluation results to file."""
        path = os.path.join(self.config.output_dir, filename)

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        results = convert_numpy(results)

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {path}")

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        env,
        num_episodes: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same environment.

        Args:
            models: Dictionary of model name -> model
            env: Evaluation environment
            num_episodes: Episodes per model

        Returns:
            Comparison results
        """
        comparison = {}

        for name, model in models.items():
            print(f"\nEvaluating: {name}")
            self.model = model.to(self.device)
            self.model.eval()

            results = self.evaluate_gym(env, num_episodes)
            comparison[name] = results["summary"]

        # Print comparison table
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        print(f"{'Model':<20} {'Mean Reward':<15} {'Success Rate':<15}")
        print("-" * 60)

        for name, metrics in comparison.items():
            print(f"{name:<20} {metrics['mean_reward']:<15.2f} {metrics['success_rate']:<15.2%}")

        return comparison


class RealRobotEvaluator:
    """
    Evaluator for real robot deployment.

    Provides:
    - Safety checks
    - Human supervision interface
    - Logging and recording
    """

    def __init__(
        self,
        model,
        robot_interface,
        safety_config: Optional[Dict] = None,
    ):
        self.model = model
        self.robot = robot_interface
        self.safety_config = safety_config or self._default_safety_config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def _default_safety_config(self) -> Dict:
        """Default safety configuration."""
        return {
            "max_velocity": 0.5,  # m/s
            "max_acceleration": 1.0,  # m/s^2
            "workspace_limits": {
                "x": [-0.5, 0.5],
                "y": [-0.5, 0.5],
                "z": [0.0, 0.5],
            },
            "force_limit": 50.0,  # N
            "require_confirmation": True,
        }

    def check_action_safety(self, action: np.ndarray) -> Tuple[bool, str]:
        """
        Check if an action is safe to execute.

        Returns:
            Tuple of (is_safe, reason)
        """
        # Check velocity limits
        if hasattr(self.robot, 'current_velocity'):
            velocity = np.linalg.norm(action[:3])  # Assume first 3 are position/velocity
            if velocity > self.safety_config["max_velocity"]:
                return False, f"Velocity {velocity:.2f} exceeds limit"

        # Check workspace limits
        if hasattr(self.robot, 'current_position'):
            limits = self.safety_config["workspace_limits"]
            pos = self.robot.current_position
            for i, axis in enumerate(["x", "y", "z"]):
                if not limits[axis][0] <= pos[i] <= limits[axis][1]:
                    return False, f"Position outside workspace limits"

        return True, "Safe"

    def run_episode(
        self,
        instruction: str,
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Run a single evaluation episode on real robot.

        Args:
            instruction: Language instruction
            max_steps: Maximum steps

        Returns:
            Episode results
        """
        trajectory = []
        episode_reward = 0

        # Get initial observation
        obs = self.robot.get_observation()

        for step in range(max_steps):
            # Get action from model
            action = self._get_action(obs, instruction)

            # Safety check
            is_safe, reason = self.check_action_safety(action)
            if not is_safe:
                print(f"Unsafe action detected: {reason}")
                if self.safety_config["require_confirmation"]:
                    response = input("Override and execute? (y/n): ")
                    if response.lower() != 'y':
                        break

            # Execute action
            obs, reward, done, info = self.robot.step(action)

            trajectory.append({
                "observation": obs,
                "action": action.tolist(),
                "reward": reward,
            })

            episode_reward += reward

            if done:
                break

        return {
            "trajectory": trajectory,
            "total_reward": episode_reward,
            "num_steps": len(trajectory),
            "success": info.get("success", False),
        }

    def _get_action(self, obs: Dict, instruction: str) -> np.ndarray:
        """Get action from model."""
        with torch.no_grad():
            action = self.model.predict_action(
                image=obs.get("image"),
                instruction=instruction,
                device=self.device,
            )

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        return action


if __name__ == "__main__":
    print("VLA Evaluator")
    print("Comprehensive evaluation for Vision-Language-Action models")
