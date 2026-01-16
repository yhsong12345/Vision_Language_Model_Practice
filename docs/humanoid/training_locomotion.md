# Humanoid Locomotion Training

This document covers the complete training process for bipedal locomotion in humanoid robots, including standing, walking, running, and terrain adaptation.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Locomotion Training Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Standing & Balance                                                 │
│  ├── Static balance                                                         │
│  ├── Push recovery                                                          │
│  └── Weight shifting                                                        │
│                                                                              │
│  Phase 2: Basic Walking                                                      │
│  ├── Slow walking (0.3 m/s)                                                 │
│  ├── Normal walking (0.5-1.0 m/s)                                           │
│  └── Fast walking (1.0-1.5 m/s)                                             │
│                                                                              │
│  Phase 3: Advanced Locomotion                                                │
│  ├── Running                                                                │
│  ├── Turning (in-place, while walking)                                      │
│  ├── Lateral walking                                                        │
│  └── Backward walking                                                       │
│                                                                              │
│  Phase 4: Terrain Adaptation                                                 │
│  ├── Slopes (up/down)                                                       │
│  ├── Stairs (up/down)                                                       │
│  ├── Rough terrain                                                          │
│  └── Stepping stones                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Standing & Balance

### Training Configuration

```python
from train.online_rl import PPOTrainer
from config.humanoid_config import LocomotionConfig

config = LocomotionConfig(
    # Environment
    env_name="humanoid-stand-v0",
    num_envs=4096,               # Parallel environments (Isaac Gym)

    # Observation
    observe_joint_pos=True,
    observe_joint_vel=True,
    observe_imu=True,
    observe_foot_contact=True,

    # Action
    action_type="position",      # Joint position targets
    action_scale=0.5,            # Action scaling

    # Training
    algorithm="ppo",
    learning_rate=3e-4,
    num_steps=1_000_000,
    batch_size=64,
    n_epochs=10,

    # Reward
    reward_weights={
        "upright": 1.0,
        "survival": 0.5,
        "smooth_motion": 0.1,
        "feet_on_ground": 0.3,
    },
)
```

### Standing Reward Function

```python
def compute_standing_reward(
    root_height: torch.Tensor,
    root_orientation: torch.Tensor,
    joint_velocities: torch.Tensor,
    foot_contacts: torch.Tensor,
    target_height: float = 1.0,
) -> torch.Tensor:
    """
    Reward function for standing balance.

    Args:
        root_height: (num_envs,) COM height
        root_orientation: (num_envs, 4) quaternion
        joint_velocities: (num_envs, num_joints)
        foot_contacts: (num_envs, 4) binary contact flags
    """
    # Upright reward (penalize tilt)
    up_vec = torch.tensor([0, 0, 1], device=root_orientation.device)
    torso_up = quat_rotate(root_orientation, up_vec)
    upright_reward = torch.sum(torso_up * up_vec, dim=-1)

    # Height reward
    height_reward = torch.exp(-torch.abs(root_height - target_height))

    # Smooth motion reward (minimize joint velocities)
    smooth_reward = -torch.sum(joint_velocities ** 2, dim=-1) * 0.01

    # Feet on ground reward
    feet_reward = torch.sum(foot_contacts.float(), dim=-1) / 4.0

    total_reward = (
        upright_reward +
        0.5 * height_reward +
        smooth_reward +
        0.3 * feet_reward
    )

    return total_reward
```

### Training Command

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-stand-v0 \
    --num-envs 4096 \
    --num-steps 1000000 \
    --learning-rate 3e-4 \
    --output-dir ./output/humanoid/locomotion/standing
```

---

## Phase 2: Basic Walking

### Curriculum Learning for Walking

```python
class WalkingCurriculum:
    """
    Curriculum learning for walking speeds.

    Progressively increases target velocity as policy improves.
    """

    def __init__(self):
        self.stages = [
            {"velocity": 0.3, "success_threshold": 0.8},
            {"velocity": 0.5, "success_threshold": 0.8},
            {"velocity": 0.8, "success_threshold": 0.8},
            {"velocity": 1.0, "success_threshold": 0.7},
            {"velocity": 1.5, "success_threshold": 0.6},
        ]
        self.current_stage = 0

    def get_target_velocity(self) -> float:
        return self.stages[self.current_stage]["velocity"]

    def update(self, success_rate: float):
        threshold = self.stages[self.current_stage]["success_threshold"]
        if success_rate >= threshold and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"Advancing to stage {self.current_stage}: "
                  f"velocity={self.get_target_velocity()} m/s")
```

### Walking Reward Function

```python
def compute_walking_reward(
    root_velocity: torch.Tensor,
    root_height: torch.Tensor,
    root_orientation: torch.Tensor,
    joint_torques: torch.Tensor,
    joint_velocities: torch.Tensor,
    foot_contacts: torch.Tensor,
    target_velocity: float,
) -> Dict[str, torch.Tensor]:
    """
    Reward function for walking locomotion.
    """
    rewards = {}

    # Forward velocity reward (main objective)
    forward_vel = root_velocity[:, 0]  # x velocity
    velocity_error = torch.abs(forward_vel - target_velocity)
    rewards["velocity"] = torch.exp(-velocity_error)

    # Lateral velocity penalty (minimize sideways drift)
    lateral_vel = root_velocity[:, 1]  # y velocity
    rewards["lateral_penalty"] = -torch.abs(lateral_vel) * 0.5

    # Upright bonus
    up_vec = torch.tensor([0, 0, 1], device=root_orientation.device)
    torso_up = quat_rotate(root_orientation, up_vec)
    rewards["upright"] = torch.sum(torso_up * up_vec, dim=-1)

    # Height maintenance
    rewards["height"] = torch.exp(-torch.abs(root_height - 1.0))

    # Energy efficiency (minimize torques)
    rewards["energy"] = -torch.sum(joint_torques ** 2, dim=-1) * 0.001

    # Smooth motion (minimize acceleration/jerk)
    vel_diff = joint_velocities - self.prev_velocities
    rewards["smooth"] = -torch.sum(vel_diff ** 2, dim=-1) * 0.01

    # Gait symmetry reward
    left_leg_vel = joint_velocities[:, 12:18]   # Left leg joints
    right_leg_vel = joint_velocities[:, 18:24]  # Right leg joints
    symmetry = torch.sum((left_leg_vel + right_leg_vel) ** 2, dim=-1)
    rewards["symmetry"] = torch.exp(-symmetry * 0.1)

    # Foot clearance (proper swing phase)
    foot_height = self._get_foot_heights()
    rewards["clearance"] = torch.where(
        ~foot_contacts[:, :2].any(dim=-1),  # During swing
        foot_height.clamp(min=0.02, max=0.1),  # Reward proper clearance
        torch.zeros_like(foot_height[:, 0])
    )

    # Total reward
    total = (
        1.0 * rewards["velocity"] +
        0.5 * rewards["upright"] +
        0.3 * rewards["height"] +
        rewards["energy"] +
        rewards["smooth"] +
        0.3 * rewards["symmetry"] +
        0.2 * rewards["clearance"] +
        rewards["lateral_penalty"]
    )

    rewards["total"] = total
    return rewards
```

### Training Commands

```bash
# Initialize from standing policy
python train/online_rl/ppo_trainer.py \
    --env humanoid-walk-v0 \
    --pretrained ./output/humanoid/locomotion/standing/best.pt \
    --curriculum \
    --target-velocities 0.3,0.5,0.8,1.0,1.5 \
    --num-steps 5000000 \
    --output-dir ./output/humanoid/locomotion/walking
```

---

## Phase 3: Advanced Locomotion

### Running

```python
running_config = LocomotionConfig(
    env_name="humanoid-run-v0",
    target_velocity=3.0,          # m/s
    allow_flight_phase=True,      # Both feet can leave ground
    reward_weights={
        "velocity": 1.5,
        "upright": 0.3,
        "energy": -0.005,
        "smooth": -0.01,
    },
)
```

### Turning

```python
turning_config = LocomotionConfig(
    env_name="humanoid-turn-v0",
    target_yaw_velocity=1.0,      # rad/s
    reward_weights={
        "yaw_velocity": 1.0,
        "upright": 0.5,
        "stability": 0.3,
    },
)
```

### Omnidirectional Walking

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-omni-v0 \
    --pretrained ./output/humanoid/locomotion/walking/best.pt \
    --command-dim 3 \
    --command-range "vx:[-1.5,1.5],vy:[-0.5,0.5],vyaw:[-1.0,1.0]" \
    --num-steps 10000000 \
    --output-dir ./output/humanoid/locomotion/omnidirectional
```

---

## Phase 4: Terrain Adaptation

### Domain Randomization

```python
class TerrainRandomization:
    """
    Randomize terrain parameters for robust locomotion.
    """

    def __init__(self):
        self.params = {
            # Ground properties
            "friction": (0.5, 1.5),
            "restitution": (0.0, 0.3),

            # Terrain geometry
            "slope_angle": (-20, 20),      # degrees
            "roughness": (0.0, 0.05),      # meters
            "step_height": (0.0, 0.2),     # meters

            # External disturbances
            "push_force": (0, 100),        # Newtons
            "push_interval": (50, 200),    # timesteps
        }

    def sample(self) -> Dict[str, float]:
        return {
            key: np.random.uniform(*bounds)
            for key, bounds in self.params.items()
        }
```

### Terrain Types

| Terrain | Description | Configuration |
|---------|-------------|---------------|
| Flat | Standard flat ground | `slope=0, roughness=0` |
| Slopes Up | Uphill walking | `slope=10-20 deg` |
| Slopes Down | Downhill walking | `slope=-10 to -20 deg` |
| Stairs Up | Stair climbing | `step_height=0.17m, step_depth=0.28m` |
| Stairs Down | Stair descent | `step_height=-0.17m` |
| Rough | Uneven terrain | `roughness=0.02-0.05m` |
| Stepping Stones | Discrete footholds | `spacing=0.4m, size=0.3m` |

### Terrain-Adaptive Training

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-terrain-v0 \
    --pretrained ./output/humanoid/locomotion/omnidirectional/best.pt \
    --domain-randomization \
    --terrain-curriculum \
    --terrain-types flat,slopes,stairs,rough \
    --num-steps 20000000 \
    --num-envs 8192 \
    --output-dir ./output/humanoid/locomotion/terrain
```

### Heightmap Perception

```python
class TerrainPerceptionModule(nn.Module):
    """
    Process terrain heightmap for adaptive locomotion.
    """

    def __init__(
        self,
        heightmap_size: Tuple[int, int] = (20, 20),
        resolution: float = 0.05,  # meters per cell
        output_dim: int = 64,
    ):
        super().__init__()
        self.heightmap_size = heightmap_size
        self.resolution = resolution

        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, output_dim),
        )

    def forward(self, heightmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heightmap: (batch, H, W) terrain height relative to robot

        Returns:
            terrain_features: (batch, output_dim)
        """
        x = heightmap.unsqueeze(1)  # Add channel dim
        return self.encoder(x)
```

---

## Simulation Environments

### MuJoCo Setup

```python
import mujoco

class MuJoCoLocomotionEnv:
    """MuJoCo-based locomotion environment."""

    def __init__(
        self,
        robot_xml: str = "humanoid.xml",
        terrain: str = "flat",
        control_freq: float = 100.0,
    ):
        self.model = mujoco.MjModel.from_xml_path(robot_xml)
        self.data = mujoco.MjData(self.model)
        self.control_freq = control_freq

        # Get joint indices
        self.leg_joints = self._get_leg_joint_indices()

    def step(self, action: np.ndarray) -> Tuple:
        # Apply PD control
        target_pos = action[:self.num_joints]
        self.data.ctrl[:] = self._pd_control(target_pos)

        # Step simulation
        for _ in range(int(240 / self.control_freq)):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_termination()

        return obs, reward, done, {}

    def _get_observation(self) -> Dict:
        return {
            "joint_pos": self.data.qpos[7:].copy(),
            "joint_vel": self.data.qvel[6:].copy(),
            "root_pos": self.data.qpos[:3].copy(),
            "root_quat": self.data.qpos[3:7].copy(),
            "root_vel": self.data.qvel[:3].copy(),
            "root_angvel": self.data.qvel[3:6].copy(),
            "foot_contact": self._get_foot_contacts(),
        }
```

### Isaac Gym Setup (GPU-Accelerated)

```python
from isaacgym import gymapi, gymtorch

class IsaacGymLocomotionEnv:
    """GPU-accelerated locomotion environment."""

    def __init__(
        self,
        num_envs: int = 4096,
        device: str = "cuda",
    ):
        self.gym = gymapi.acquire_gym()
        self.num_envs = num_envs
        self.device = device

        # Create simulation
        self._create_sim()
        self._create_envs()
        self._allocate_buffers()

    def step(self, actions: torch.Tensor) -> Tuple:
        """Vectorized step for all environments."""
        # Apply actions
        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(actions)
        )

        # Step physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Compute observations and rewards (all on GPU)
        obs = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()

        # Auto-reset done environments
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._reset_envs(reset_ids)

        return obs, rewards, dones, {}
```

---

## Training with Reference Motion (AMP)

### Adversarial Motion Priors

```python
class AMPLocomotionTrainer:
    """
    Train locomotion with Adversarial Motion Priors.

    Combines task reward with style reward from motion discriminator.
    """

    def __init__(
        self,
        policy: LocomotionPolicy,
        env: LocomotionEnv,
        reference_motions: MoCapDataset,
        task_reward_weight: float = 0.5,
        style_reward_weight: float = 0.5,
    ):
        self.policy = policy
        self.env = env
        self.reference_motions = reference_motions

        # Motion discriminator
        self.discriminator = MotionDiscriminator(
            obs_dim=policy.obs_dim,
            hidden_dim=512,
        )

        self.task_weight = task_reward_weight
        self.style_weight = style_reward_weight

    def compute_style_reward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute style reward from discriminator.

        Reward = -log(1 - D(s)) where D predicts "fake"
        """
        with torch.no_grad():
            disc_pred = self.discriminator(observations)
        style_reward = -torch.log(1 - disc_pred + 1e-6)
        return style_reward

    def update_discriminator(
        self,
        policy_obs: torch.Tensor,
        expert_obs: torch.Tensor,
    ):
        """Update discriminator to distinguish policy from expert."""
        # Expert should be classified as 1, policy as 0
        expert_pred = self.discriminator(expert_obs)
        policy_pred = self.discriminator(policy_obs)

        expert_loss = F.binary_cross_entropy(
            expert_pred,
            torch.ones_like(expert_pred)
        )
        policy_loss = F.binary_cross_entropy(
            policy_pred,
            torch.zeros_like(policy_pred)
        )

        disc_loss = expert_loss + policy_loss

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        return disc_loss.item()

    def train_step(self):
        # Collect rollouts
        rollouts = self.collect_rollouts()

        # Compute task reward
        task_rewards = rollouts["rewards"]

        # Compute style reward
        style_rewards = self.compute_style_reward(rollouts["observations"])

        # Combined reward
        total_rewards = (
            self.task_weight * task_rewards +
            self.style_weight * style_rewards
        )

        # Update policy with PPO
        self.update_policy(rollouts, total_rewards)

        # Update discriminator
        expert_batch = self.reference_motions.sample(len(rollouts["observations"]))
        disc_loss = self.update_discriminator(
            rollouts["observations"],
            expert_batch["observations"]
        )

        return {
            "task_reward": task_rewards.mean().item(),
            "style_reward": style_rewards.mean().item(),
            "disc_loss": disc_loss,
        }
```

### Training Command

```bash
python train/online_rl/amp_trainer.py \
    --env humanoid-walk-v0 \
    --reference-motions ./data/mocap/walking \
    --task-reward-weight 0.5 \
    --style-reward-weight 0.5 \
    --num-steps 10000000 \
    --output-dir ./output/humanoid/locomotion/amp
```

---

## Datasets for Locomotion

| Dataset | Public Source | Description | Use Case |
|---------|---------------|-------------|----------|
| CMU MoCap | [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/) | 2600+ human motion sequences | Reference motion for AMP |
| AMASS | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) | 40+ hours unified MoCap | Motion primitive learning |
| D4RL Humanoid | [imone/D4RL](https://huggingface.co/datasets/imone/D4RL) | Humanoid locomotion trajectories | Offline RL pretraining |
| Isaac Gym Demos | Built-in | Simulation demonstrations | Online RL training |

---

## Best Practices

### Training Tips

1. **Start with standing**: Always train stable standing before walking
2. **Use curriculum**: Gradually increase velocity targets
3. **Regularize torques**: Penalize high torques for energy efficiency
4. **Include phase variable**: Add gait phase (sin/cos) to observations
5. **Domain randomization**: Randomize physics for sim-to-real transfer

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Robot falls immediately | Poorly initialized policy | Start from standing policy |
| Asymmetric gait | Unbalanced reward | Add symmetry reward |
| High energy consumption | No torque penalty | Add energy penalty term |
| Poor sim-to-real | Overfitting to simulation | Domain randomization |
| Jerky motion | No smoothness term | Add action smoothness penalty |

---

## Next Steps

- [Manipulation Training](training_manipulation.md) - Arm control and grasping
- [Whole-Body Control](training_whole_body.md) - Loco-manipulation
- [Deployment](deployment.md) - Real robot deployment
