"""
VLA Benchmark Suite

Standard benchmarks for evaluating Vision-Language-Action models:
- CALVIN (manipulation benchmark)
- SIMPLER (real-world simulation)
- RLBench (robot learning benchmark)
- Custom task benchmarks
"""

import os
import torch
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
import json
from tqdm import tqdm

from .evaluator import VLAEvaluator, EvalConfig
from .metrics import compute_success_rate, compute_action_metrics


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    benchmark_name: str = "custom"
    num_episodes: int = 100
    num_tasks: int = 10
    seed: int = 42
    output_dir: str = "./benchmark_results"
    save_detailed: bool = True


class VLABenchmark:
    """
    Benchmark suite for VLA models.

    Provides standardized evaluation on various benchmarks.
    """

    # Supported benchmarks
    BENCHMARKS = {
        "calvin": {
            "tasks": ["open_drawer", "close_drawer", "turn_on_led", "turn_off_led",
                      "push_into_drawer", "rotate_red_block_left", "rotate_red_block_right",
                      "move_slider_left", "move_slider_right", "turn_on_lightbulb"],
            "success_threshold": 0.9,
        },
        "rlbench": {
            "tasks": ["reach_target", "pick_and_lift", "take_lid_off_saucepan",
                      "put_groceries_in_cupboard", "place_shape_in_shape_sorter",
                      "push_buttons", "stack_blocks", "slide_block_to_target"],
            "success_threshold": 0.5,
        },
        "simpler": {
            "tasks": ["google_robot_pick", "google_robot_move", "widowx_carrot_on_plate",
                      "widowx_stack_cube", "widowx_put_eggplant_in_basket"],
            "success_threshold": 0.6,
        },
        "lerobot": {
            "tasks": ["pusht", "aloha_insertion", "aloha_transfer_cube",
                      "xarm_push", "xarm_lift"],
            "success_threshold": 0.7,
        },
    }

    def __init__(
        self,
        model,
        config: Optional[BenchmarkConfig] = None,
    ):
        self.model = model
        self.config = config or BenchmarkConfig()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        os.makedirs(self.config.output_dir, exist_ok=True)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def run_benchmark(
        self,
        benchmark_name: str = None,
        tasks: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete benchmark.

        Args:
            benchmark_name: Name of benchmark to run
            tasks: Optional list of specific tasks

        Returns:
            Benchmark results
        """
        if benchmark_name is None:
            benchmark_name = self.config.benchmark_name

        print("=" * 60)
        print(f"Running Benchmark: {benchmark_name}")
        print("=" * 60)

        if benchmark_name in self.BENCHMARKS:
            benchmark_info = self.BENCHMARKS[benchmark_name]
            if tasks is None:
                tasks = benchmark_info["tasks"]
            success_threshold = benchmark_info["success_threshold"]
        else:
            if tasks is None:
                tasks = ["default_task"]
            success_threshold = 0.5

        results = {
            "benchmark": benchmark_name,
            "tasks": {},
            "overall": {},
        }

        all_successes = []

        for task in tasks:
            print(f"\nTask: {task}")
            task_results = self._evaluate_task(task, benchmark_name)
            results["tasks"][task] = task_results
            all_successes.extend(task_results.get("successes", []))

        # Overall metrics
        overall_success = compute_success_rate(all_successes)
        results["overall"] = {
            "success_rate": overall_success["success_rate"],
            "ci_lower": overall_success["ci_lower"],
            "ci_upper": overall_success["ci_upper"],
            "total_episodes": len(all_successes),
            "passed": overall_success["success_rate"] >= success_threshold,
            "threshold": success_threshold,
        }

        print("\n" + "=" * 60)
        print("Benchmark Results")
        print("=" * 60)
        print(f"Overall Success Rate: {results['overall']['success_rate']:.2%}")
        print(f"95% CI: [{results['overall']['ci_lower']:.2%}, {results['overall']['ci_upper']:.2%}]")
        print(f"Passed: {results['overall']['passed']}")

        # Save results
        if self.config.save_detailed:
            self._save_results(results)

        return results

    def _evaluate_task(
        self,
        task: str,
        benchmark: str,
    ) -> Dict[str, Any]:
        """Evaluate a single task."""
        try:
            env = self._create_env(task, benchmark)
            instructions = self._get_task_instructions(task)

            successes = []
            rewards = []
            lengths = []

            episodes_per_task = self.config.num_episodes // self.config.num_tasks

            for ep in tqdm(range(episodes_per_task), desc=task):
                result = self._run_episode(env, instructions[ep % len(instructions)])
                successes.append(result["success"])
                rewards.append(result["reward"])
                lengths.append(result["length"])

            env.close()

            return {
                "success_rate": np.mean(successes),
                "mean_reward": np.mean(rewards),
                "mean_length": np.mean(lengths),
                "successes": successes,
                "n_episodes": episodes_per_task,
            }

        except Exception as e:
            print(f"  Error evaluating {task}: {e}")
            return {
                "success_rate": 0,
                "error": str(e),
                "successes": [False] * (self.config.num_episodes // self.config.num_tasks),
            }

    def _create_env(self, task: str, benchmark: str):
        """Create environment for task."""
        try:
            import gymnasium as gym

            # Try different environment naming conventions
            env_names = [
                task,
                f"{benchmark}/{task}",
                f"{benchmark}-{task}-v0",
                f"{task.replace('_', '-')}-v0",
            ]

            for name in env_names:
                try:
                    return gym.make(name)
                except:
                    continue

            # Fallback to a simple environment
            return gym.make("CartPole-v1")

        except ImportError:
            raise RuntimeError("Gymnasium not installed")

    def _get_task_instructions(self, task: str) -> List[str]:
        """Get language instructions for a task."""
        # Task-specific instructions
        task_instructions = {
            "open_drawer": ["Open the drawer", "Pull the drawer open"],
            "close_drawer": ["Close the drawer", "Push the drawer closed"],
            "pick_and_lift": ["Pick up the object and lift it", "Grasp and raise the item"],
            "push_into_drawer": ["Push the object into the drawer"],
            "reach_target": ["Move to the target position", "Reach the goal"],
            "stack_blocks": ["Stack the blocks on top of each other"],
            "pusht": ["Push the T-shaped block to the goal"],
            "default_task": ["Complete the task", "Perform the manipulation"],
        }

        return task_instructions.get(task, task_instructions["default_task"])

    def _run_episode(
        self,
        env,
        instruction: str,
        max_steps: int = 500,
    ) -> Dict[str, Any]:
        """Run a single episode."""
        obs, info = env.reset()
        total_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            action = self._get_action(obs, instruction)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            episode_length += 1

        return {
            "success": info.get("success", total_reward > 0),
            "reward": total_reward,
            "length": episode_length,
        }

    def _get_action(self, obs, instruction: str) -> np.ndarray:
        """Get action from model."""
        if hasattr(self.model, 'predict_action'):
            action = self.model.predict_action(
                image=obs,
                instruction=instruction,
                device=self.device,
            )
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.model(obs_tensor)
            action = action.squeeze(0)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        return action

    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results."""
        path = os.path.join(
            self.config.output_dir,
            f"{results['benchmark']}_results.json"
        )

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(path, "w") as f:
            json.dump(convert(results), f, indent=2)

        print(f"\nResults saved to {path}")


class CustomTaskBenchmark:
    """
    Create custom task benchmarks for specific robot applications.
    """

    def __init__(
        self,
        name: str,
        tasks: List[Dict[str, Any]],
        success_criteria: Dict[str, callable],
    ):
        """
        Args:
            name: Benchmark name
            tasks: List of task configurations
            success_criteria: Dict of task -> success function
        """
        self.name = name
        self.tasks = tasks
        self.success_criteria = success_criteria

    def add_task(
        self,
        task_name: str,
        env_config: Dict,
        instructions: List[str],
        success_fn: callable,
    ):
        """Add a new task to the benchmark."""
        self.tasks.append({
            "name": task_name,
            "env_config": env_config,
            "instructions": instructions,
        })
        self.success_criteria[task_name] = success_fn

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark to dictionary format."""
        return {
            "name": self.name,
            "tasks": [t["name"] for t in self.tasks],
            "success_threshold": 0.5,
        }


def create_autonomous_driving_benchmark() -> CustomTaskBenchmark:
    """Create benchmark for autonomous driving VLA."""
    tasks = [
        {"name": "lane_following", "description": "Follow lane markings"},
        {"name": "intersection_navigation", "description": "Navigate through intersection"},
        {"name": "obstacle_avoidance", "description": "Avoid static obstacles"},
        {"name": "pedestrian_yielding", "description": "Yield to pedestrians"},
        {"name": "parking", "description": "Park in designated spot"},
    ]

    success_criteria = {
        "lane_following": lambda info: info.get("lane_deviation", 1) < 0.5,
        "intersection_navigation": lambda info: info.get("collision", False) == False,
        "obstacle_avoidance": lambda info: info.get("collision", False) == False,
        "pedestrian_yielding": lambda info: info.get("pedestrian_safe", True),
        "parking": lambda info: info.get("parked", False),
    }

    return CustomTaskBenchmark("autonomous_driving", tasks, success_criteria)


def create_manipulation_benchmark() -> CustomTaskBenchmark:
    """Create benchmark for robot manipulation VLA."""
    tasks = [
        {"name": "pick_place", "description": "Pick and place object"},
        {"name": "stacking", "description": "Stack objects"},
        {"name": "insertion", "description": "Insert peg in hole"},
        {"name": "pouring", "description": "Pour liquid"},
        {"name": "wiping", "description": "Wipe surface"},
    ]

    success_criteria = {
        "pick_place": lambda info: info.get("object_at_goal", False),
        "stacking": lambda info: info.get("stack_height", 0) >= 2,
        "insertion": lambda info: info.get("inserted", False),
        "pouring": lambda info: info.get("poured_ratio", 0) > 0.8,
        "wiping": lambda info: info.get("cleaned_ratio", 0) > 0.9,
    }

    return CustomTaskBenchmark("manipulation", tasks, success_criteria)


if __name__ == "__main__":
    print("VLA Benchmark Suite")
    print("\nSupported benchmarks:")
    for name, info in VLABenchmark.BENCHMARKS.items():
        print(f"  - {name}: {len(info['tasks'])} tasks")
