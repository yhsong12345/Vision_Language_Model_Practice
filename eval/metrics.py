"""
Evaluation Metrics for VLA Models

Comprehensive metrics for evaluating Vision-Language-Action models:
- Success rate metrics
- Trajectory quality metrics
- Action prediction metrics
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import warnings


def compute_success_rate(
    successes: List[bool],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute success rate with confidence interval.

    Args:
        successes: List of success booleans
        confidence: Confidence level for interval

    Returns:
        Dictionary with success rate and confidence interval
    """
    n = len(successes)
    if n == 0:
        return {"success_rate": 0, "ci_lower": 0, "ci_upper": 0, "n": 0}

    success_rate = np.mean(successes)

    # Wilson score interval for binomial proportion
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    denominator = 1 + z**2 / n
    center = (success_rate + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((success_rate * (1 - success_rate) + z**2 / (4 * n)) / n) / denominator

    return {
        "success_rate": success_rate,
        "ci_lower": max(0, center - margin),
        "ci_upper": min(1, center + margin),
        "n": n,
    }


def compute_trajectory_metrics(
    trajectories: List[List[Dict]],
    goal_positions: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute trajectory quality metrics.

    Args:
        trajectories: List of trajectories, each is list of (state, action) dicts
        goal_positions: Optional goal positions for goal-reaching tasks

    Returns:
        Dictionary of trajectory metrics
    """
    metrics = {
        "mean_length": 0,
        "mean_smoothness": 0,
        "mean_efficiency": 0,
    }

    if not trajectories:
        return metrics

    lengths = []
    smoothness_scores = []
    efficiency_scores = []

    for i, traj in enumerate(trajectories):
        if not traj:
            continue

        # Trajectory length
        lengths.append(len(traj))

        # Extract actions
        actions = np.array([step.get("action", [0] * 7) for step in traj])

        # Smoothness: Average jerk (derivative of acceleration)
        if len(actions) >= 3:
            velocities = np.diff(actions, axis=0)
            accelerations = np.diff(velocities, axis=0)

            if len(accelerations) > 0:
                jerk = np.mean(np.abs(np.diff(accelerations, axis=0)))
                smoothness = 1.0 / (1.0 + jerk)
                smoothness_scores.append(smoothness)

        # Efficiency: Path length ratio (straight line / actual path)
        if goal_positions is not None and i < len(goal_positions):
            positions = np.array([step.get("state", [0, 0, 0])[:3] for step in traj])

            if len(positions) >= 2:
                actual_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                direct_length = np.linalg.norm(positions[-1] - positions[0])

                if actual_length > 0:
                    efficiency = direct_length / actual_length
                    efficiency_scores.append(min(1.0, efficiency))

    metrics["mean_length"] = np.mean(lengths) if lengths else 0
    metrics["std_length"] = np.std(lengths) if lengths else 0
    metrics["mean_smoothness"] = np.mean(smoothness_scores) if smoothness_scores else 0
    metrics["mean_efficiency"] = np.mean(efficiency_scores) if efficiency_scores else 0

    return metrics


def compute_action_metrics(
    predicted_actions: np.ndarray,
    ground_truth_actions: np.ndarray,
    action_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute action prediction metrics.

    Args:
        predicted_actions: (N, action_dim) predicted actions
        ground_truth_actions: (N, action_dim) ground truth actions
        action_names: Optional names for each action dimension

    Returns:
        Dictionary of action metrics
    """
    if len(predicted_actions) == 0:
        return {"error": "No predictions"}

    # Ensure same shape
    assert predicted_actions.shape == ground_truth_actions.shape, \
        f"Shape mismatch: {predicted_actions.shape} vs {ground_truth_actions.shape}"

    n_samples, action_dim = predicted_actions.shape

    if action_names is None:
        action_names = [f"action_{i}" for i in range(action_dim)]

    # Overall metrics
    mse = np.mean((predicted_actions - ground_truth_actions) ** 2)
    mae = np.mean(np.abs(predicted_actions - ground_truth_actions))
    rmse = np.sqrt(mse)

    # Per-dimension metrics
    per_dim_mse = np.mean((predicted_actions - ground_truth_actions) ** 2, axis=0)
    per_dim_mae = np.mean(np.abs(predicted_actions - ground_truth_actions), axis=0)

    # Correlation per dimension
    per_dim_corr = []
    for i in range(action_dim):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = stats.pearsonr(predicted_actions[:, i], ground_truth_actions[:, i])
            per_dim_corr.append(corr if not np.isnan(corr) else 0)

    # R-squared (coefficient of determination)
    ss_res = np.sum((predicted_actions - ground_truth_actions) ** 2)
    ss_tot = np.sum((ground_truth_actions - np.mean(ground_truth_actions, axis=0)) ** 2)
    r_squared = 1 - (ss_res / (ss_tot + 1e-8))

    # Action magnitude error (for continuous actions)
    pred_magnitude = np.linalg.norm(predicted_actions, axis=1)
    gt_magnitude = np.linalg.norm(ground_truth_actions, axis=1)
    magnitude_error = np.mean(np.abs(pred_magnitude - gt_magnitude))

    # Direction error (cosine similarity)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dot_product = np.sum(predicted_actions * ground_truth_actions, axis=1)
        norms_product = pred_magnitude * gt_magnitude + 1e-8
        cosine_sim = np.mean(dot_product / norms_product)

    metrics = {
        # Overall
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r_squared": float(r_squared),
        "magnitude_error": float(magnitude_error),
        "cosine_similarity": float(cosine_sim),
        "n_samples": n_samples,

        # Per-dimension
        "per_dimension": {
            action_names[i]: {
                "mse": float(per_dim_mse[i]),
                "mae": float(per_dim_mae[i]),
                "correlation": float(per_dim_corr[i]),
            }
            for i in range(action_dim)
        },
    }

    return metrics


def compute_temporal_consistency(
    action_sequence: np.ndarray,
) -> Dict[str, float]:
    """
    Compute temporal consistency metrics for action sequences.

    Args:
        action_sequence: (T, action_dim) sequence of actions

    Returns:
        Temporal consistency metrics
    """
    if len(action_sequence) < 2:
        return {"consistency": 1.0}

    # Action velocity (first derivative)
    velocities = np.diff(action_sequence, axis=0)
    mean_velocity = np.mean(np.abs(velocities))

    # Action acceleration (second derivative)
    if len(action_sequence) >= 3:
        accelerations = np.diff(velocities, axis=0)
        mean_acceleration = np.mean(np.abs(accelerations))
    else:
        mean_acceleration = 0

    # Jerk (third derivative)
    if len(action_sequence) >= 4:
        jerks = np.diff(accelerations, axis=0)
        mean_jerk = np.mean(np.abs(jerks))
    else:
        mean_jerk = 0

    # Consistency score (lower jerk = higher consistency)
    consistency = 1.0 / (1.0 + mean_jerk)

    return {
        "mean_velocity": float(mean_velocity),
        "mean_acceleration": float(mean_acceleration),
        "mean_jerk": float(mean_jerk),
        "consistency": float(consistency),
    }


def compute_goal_reaching_metrics(
    final_positions: List[np.ndarray],
    goal_positions: List[np.ndarray],
    success_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Compute goal-reaching metrics.

    Args:
        final_positions: List of final positions reached
        goal_positions: List of goal positions
        success_threshold: Distance threshold for success

    Returns:
        Goal reaching metrics
    """
    if len(final_positions) != len(goal_positions):
        return {"error": "Mismatched list lengths"}

    distances = []
    successes = []

    for final, goal in zip(final_positions, goal_positions):
        dist = np.linalg.norm(np.array(final) - np.array(goal))
        distances.append(dist)
        successes.append(dist < success_threshold)

    return {
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "success_rate": float(np.mean(successes)),
        "n_episodes": len(final_positions),
        "success_threshold": success_threshold,
    }


def compute_safety_metrics(
    trajectories: List[List[Dict]],
    workspace_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    velocity_limit: float = 1.0,
    acceleration_limit: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute safety metrics for robot trajectories.

    Args:
        trajectories: List of trajectories
        workspace_limits: Dict with axis limits, e.g., {"x": (-1, 1), "y": (-1, 1)}
        velocity_limit: Maximum allowed velocity
        acceleration_limit: Maximum allowed acceleration

    Returns:
        Safety metrics
    """
    if workspace_limits is None:
        workspace_limits = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.0, 1.0),
        }

    violations = {
        "workspace": 0,
        "velocity": 0,
        "acceleration": 0,
    }
    total_steps = 0

    for traj in trajectories:
        positions = []
        for step in traj:
            state = step.get("state", step.get("observation", [0, 0, 0]))
            if isinstance(state, dict):
                state = state.get("position", [0, 0, 0])
            positions.append(np.array(state[:3]))
            total_steps += 1

        positions = np.array(positions)

        # Check workspace violations
        for i, axis in enumerate(["x", "y", "z"]):
            if axis in workspace_limits:
                low, high = workspace_limits[axis]
                if np.any(positions[:, i] < low) or np.any(positions[:, i] > high):
                    violations["workspace"] += 1

        # Check velocity violations
        if len(positions) >= 2:
            velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            violations["velocity"] += np.sum(velocities > velocity_limit)

        # Check acceleration violations
        if len(positions) >= 3:
            velocities = np.diff(positions, axis=0)
            accelerations = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
            violations["acceleration"] += np.sum(accelerations > acceleration_limit)

    return {
        "workspace_violations": violations["workspace"],
        "velocity_violations": violations["velocity"],
        "acceleration_violations": violations["acceleration"],
        "total_violations": sum(violations.values()),
        "violation_rate": sum(violations.values()) / max(1, total_steps),
        "is_safe": sum(violations.values()) == 0,
    }


def aggregate_metrics(
    all_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate metrics from multiple evaluation runs.

    Args:
        all_metrics: List of metric dictionaries

    Returns:
        Aggregated metrics with mean and std
    """
    if not all_metrics:
        return {}

    aggregated = {}

    # Get all numeric keys
    numeric_keys = set()
    for m in all_metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                numeric_keys.add(k)

    for key in numeric_keys:
        values = [m.get(key, 0) for m in all_metrics if key in m]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))

    aggregated["n_runs"] = len(all_metrics)

    return aggregated


if __name__ == "__main__":
    print("VLA Metrics Module")
    print("Comprehensive metrics for evaluation")

    # Test with dummy data
    pred = np.random.randn(100, 7)
    gt = np.random.randn(100, 7) * 0.5 + pred * 0.5

    metrics = compute_action_metrics(pred, gt)
    print("\nAction Metrics:")
    for k, v in metrics.items():
        if k != "per_dimension":
            print(f"  {k}: {v}")
