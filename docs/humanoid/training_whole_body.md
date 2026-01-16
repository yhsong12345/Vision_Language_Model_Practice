# Humanoid Whole-Body Control Training

This document covers training for integrated whole-body control, combining locomotion and manipulation (loco-manipulation) with balance maintenance.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Whole-Body Control Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Language Instruction                                                        │
│  "Walk to the table and pick up the cup"                                    │
│          │                                                                   │
│          ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                     Task Decomposition                           │        │
│  │  1. Navigate to table  2. Reach for cup  3. Grasp cup  4. Lift  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│          │                                                                   │
│          ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                   Whole-Body Controller                          │        │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │        │
│  │  │ Locomotion  │  │Manipulation │  │  Balance    │             │        │
│  │  │   Policy    │  │   Policy    │  │ Controller  │             │        │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │        │
│  │         │                │                │                     │        │
│  │         └────────────────┼────────────────┘                     │        │
│  │                          │                                      │        │
│  │                   Task Prioritization                           │        │
│  │              (Balance > Safety > Task)                          │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│          │                                                                   │
│          ▼                                                                   │
│  32 DoF Joint Commands                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## WholeBodyController Architecture

### Core Implementation

```python
from model.embodiment.humanoid import WholeBodyController, HumanoidConfig

class WholeBodyController(nn.Module):
    """
    Whole-body controller that coordinates locomotion and manipulation.
    """

    def __init__(self, config: HumanoidConfig):
        super().__init__()
        self.config = config

        # Locomotion policy (legs + torso)
        self.locomotion = LocomotionPolicy(config)

        # Manipulation policy (arms + hands)
        self.manipulation = ManipulationPolicy(config)

        # Task coordinator
        self.coordinator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Balance controller
        self.balance_net = nn.Sequential(
            nn.Linear(config.hidden_dim + 9, config.hidden_dim // 2),  # + IMU
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.action_dim),
        )

        # Task priority weights (learned)
        self.priority_net = nn.Sequential(
            nn.Linear(config.hidden_dim, 3),  # locomotion, manipulation, balance
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        velocity_command: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        imu_data: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
        language_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = joint_positions.shape[0]

        # Get locomotion action
        loco_output = self.locomotion(
            joint_positions, joint_velocities, joint_torques,
            velocity_command, imu_data, foot_contacts,
        )

        # Get manipulation action (if visual input available)
        if image is not None:
            arm_positions = joint_positions[:, :14]  # First 14 are arm joints
            arm_velocities = joint_velocities[:, :14]
            manip_output = self.manipulation(
                image, arm_positions, arm_velocities,
                target_poses, language_features,
            )
            manip_features = manip_output["features"]
        else:
            manip_output = None
            manip_features = torch.zeros(
                batch_size, self.config.hidden_dim,
                device=joint_positions.device
            )

        # Coordinate tasks
        combined_features = torch.cat([
            loco_output["features"],
            manip_features
        ], dim=-1)
        coordinated_features = self.coordinator(combined_features)

        # Compute task priorities
        priorities = self.priority_net(coordinated_features)

        # Balance correction
        if imu_data is not None:
            balance_input = torch.cat([coordinated_features, imu_data], dim=-1)
        else:
            balance_input = F.pad(coordinated_features, (0, 9))
        balance_correction = self.balance_net(balance_input)

        # Combine actions with priorities
        loco_action = loco_output["action_mean"]

        if manip_output is not None:
            manip_full = torch.zeros_like(loco_action)
            manip_full[:, :14] = manip_output["arm_actions"]
        else:
            manip_full = torch.zeros_like(loco_action)

        # Weighted combination
        final_action = (
            priorities[:, 0:1] * loco_action +
            priorities[:, 1:2] * manip_full +
            priorities[:, 2:3] * balance_correction
        )

        return {
            "action": final_action,
            "locomotion_action": loco_action,
            "manipulation_action": manip_full,
            "balance_correction": balance_correction,
            "priorities": priorities,
            "gripper_actions": manip_output["gripper_actions"] if manip_output else None,
        }
```

---

## Hierarchical Control

### Task-Skill-Motion-Control Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Level 1: Task Level (VLM)                                                   │
│  ├── Input: Language instruction + Scene observation                        │
│  ├── Output: Task parameters (goal position, object, action type)           │
│  └── Frequency: ~1 Hz                                                       │
│                                                                              │
│  Level 2: Skill Level                                                        │
│  ├── Input: Task parameters + Current state                                 │
│  ├── Output: Skill selection + Skill parameters                             │
│  └── Frequency: ~10 Hz                                                      │
│                                                                              │
│  Level 3: Motion Level                                                       │
│  ├── Input: Skill embedding + Initial state                                 │
│  ├── Output: Joint trajectory (action chunk)                                │
│  └── Frequency: ~50 Hz                                                      │
│                                                                              │
│  Level 4: Control Level (PD)                                                 │
│  ├── Input: Joint trajectory                                                │
│  ├── Output: Torque commands                                                │
│  └── Frequency: ~1000 Hz                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hierarchical Controller Implementation

```python
class HierarchicalWholeBodyController(nn.Module):
    """
    Hierarchical control for whole-body humanoid tasks.
    """

    def __init__(self, config: HumanoidConfig):
        super().__init__()
        self.config = config

        # Level 1: Task Planner (VLM-based)
        self.task_planner = TaskPlanner(
            vlm_dim=config.llm_hidden_dim,
            hidden_dim=config.hidden_dim,
            num_tasks=20,
        )

        # Level 2: Skill Selector
        self.skill_selector = SkillSelector(
            input_dim=config.hidden_dim,
            num_skills=16,
            skill_param_dim=16,
        )

        # Level 3: Motion Generator
        self.motion_generator = MotionGenerator(
            skill_dim=64 + 16,  # skill embedding + parameters
            num_joints=config.num_joints,
            horizon=50,  # frames
        )

        # Level 4: PD Controller
        self.pd_controller = PDController(
            num_joints=config.num_joints,
            kp=torch.ones(config.num_joints) * 100.0,
            kd=torch.ones(config.num_joints) * 10.0,
        )

    def forward(
        self,
        image: torch.Tensor,
        proprioception: Dict[str, torch.Tensor],
        language_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Level 1: Plan task
        task_output = self.task_planner(
            image=image,
            language_features=language_features,
        )

        # Level 2: Select skill
        skill_probs, skill_params = self.skill_selector(
            task_features=task_output["task_features"],
            state=proprioception["joint_pos"],
        )

        # Get skill embedding
        skill_id = skill_probs.argmax(dim=-1)
        skill_embedding = self.skill_selector.get_embedding(skill_id)

        # Level 3: Generate motion trajectory
        trajectory = self.motion_generator(
            skill_embedding=skill_embedding,
            skill_params=skill_params,
            initial_state=proprioception["joint_pos"],
        )

        return {
            "trajectory": trajectory,
            "skill_probs": skill_probs,
            "skill_params": skill_params,
            "task_features": task_output["task_features"],
        }

    def get_torque(
        self,
        target_pos: torch.Tensor,
        current_pos: torch.Tensor,
        current_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Level 4: PD control to torque."""
        return self.pd_controller(target_pos, current_pos, current_vel)
```

### Training Hierarchical Controller

```python
class HierarchicalTrainer:
    """
    Train hierarchical whole-body controller level by level.
    """

    def __init__(self, model: HierarchicalWholeBodyController):
        self.model = model

    def train_task_planner(
        self,
        dataset: TaskPlanningDataset,
        num_epochs: int = 50,
    ):
        """Train task planner to decompose instructions."""
        optimizer = torch.optim.Adam(
            self.model.task_planner.parameters(),
            lr=1e-4,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                output = self.model.task_planner(
                    image=batch["images"],
                    language_features=batch["language_features"],
                )

                # Classification loss over task types
                loss = F.cross_entropy(
                    output["task_logits"],
                    batch["task_labels"],
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Task planner epoch {epoch}: loss={loss.item():.4f}")

    def train_skill_selector(
        self,
        dataset: SkillDataset,
        num_epochs: int = 100,
    ):
        """Train skill selection from task + state."""
        optimizer = torch.optim.Adam(
            self.model.skill_selector.parameters(),
            lr=1e-4,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                skill_probs, skill_params = self.model.skill_selector(
                    task_features=batch["task_features"],
                    state=batch["state"],
                )

                # Classification loss
                cls_loss = F.cross_entropy(skill_probs, batch["skill_label"])

                # Parameter regression loss
                param_loss = F.mse_loss(skill_params, batch["skill_params"])

                loss = cls_loss + 0.5 * param_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Skill selector epoch {epoch}: loss={loss.item():.4f}")

    def train_motion_generator(
        self,
        dataset: MotionDataset,
        num_epochs: int = 100,
    ):
        """Train motion generation from skills."""
        optimizer = torch.optim.Adam(
            self.model.motion_generator.parameters(),
            lr=1e-4,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                trajectory = self.model.motion_generator(
                    skill_embedding=batch["skill_embedding"],
                    skill_params=batch["skill_params"],
                    initial_state=batch["initial_joints"],
                )

                # Trajectory loss
                traj_loss = F.mse_loss(trajectory, batch["expert_trajectory"])

                # Smoothness loss
                vel = trajectory[:, 1:] - trajectory[:, :-1]
                smooth_loss = torch.norm(vel[:, 1:] - vel[:, :-1], dim=-1).mean()

                loss = traj_loss + 0.1 * smooth_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Motion generator epoch {epoch}: loss={loss.item():.4f}")

    def train_end_to_end(
        self,
        env: HumanoidEnv,
        num_steps: int = 10_000_000,
    ):
        """Fine-tune entire hierarchy with RL."""
        params = (
            list(self.model.task_planner.parameters()) +
            list(self.model.skill_selector.parameters()) +
            list(self.model.motion_generator.parameters())
        )

        trainer = PPOTrainer(
            policy=self.model,
            env=env,
            learning_rate=1e-5,  # Small LR for fine-tuning
        )

        trainer.learn(num_steps)
```

---

## Loco-Manipulation Training

### Walking While Carrying

```python
class LocoManipulationTrainer:
    """
    Train combined locomotion and manipulation.
    """

    def __init__(
        self,
        locomotion_policy_path: str,
        manipulation_policy_path: str,
    ):
        self.model = HumanoidVLA()

        # Initialize from pretrained policies
        self.model.controller.locomotion.load_state_dict(
            torch.load(locomotion_policy_path)["model"]
        )
        self.model.controller.manipulation.load_state_dict(
            torch.load(manipulation_policy_path)["model"]
        )

    def train_carry_object(
        self,
        env: LocoManipEnv,
        num_steps: int = 5_000_000,
    ):
        """Train to walk while carrying objects."""
        # Reward components
        rewards = {
            "locomotion": 0.5,       # Forward progress
            "object_stability": 0.3, # Keep object stable
            "task_completion": 0.2,  # Reach goal
        }

        # Curriculum: increasing object weight
        weights = [0.5, 1.0, 2.0, 5.0]  # kg

        for weight in weights:
            print(f"Training with {weight}kg object")
            env.set_object_weight(weight)

            trainer = PPOTrainer(
                model=self.model,
                env=env,
                learning_rate=3e-4,
            )

            trainer.learn(num_steps // len(weights))

    def train_mobile_manipulation(
        self,
        env: MobileManipEnv,
        num_steps: int = 10_000_000,
    ):
        """Train mobile manipulation tasks."""
        reward_fn = MobileManipReward(
            approach_weight=0.3,
            manipulation_weight=0.5,
            retreat_weight=0.2,
        )

        env.set_reward_fn(reward_fn)

        trainer = PPOTrainer(
            model=self.model,
            env=env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)
```

### Mobile Manipulation Reward

```python
class MobileManipReward:
    """
    Reward function for mobile manipulation.
    """

    def __init__(
        self,
        approach_weight: float = 0.3,
        manipulation_weight: float = 0.5,
        retreat_weight: float = 0.2,
    ):
        self.weights = {
            "approach": approach_weight,
            "manipulation": manipulation_weight,
            "retreat": retreat_weight,
        }

    def __call__(
        self,
        robot_position: torch.Tensor,
        object_position: torch.Tensor,
        target_position: torch.Tensor,
        ee_position: torch.Tensor,
        object_grasped: torch.Tensor,
        task_phase: str,
    ) -> Dict[str, torch.Tensor]:
        rewards = {}

        if task_phase == "approach":
            # Reward for approaching the object
            robot_to_object = torch.norm(
                robot_position[:, :2] - object_position[:, :2],
                dim=-1
            )
            rewards["approach"] = torch.exp(-robot_to_object) * self.weights["approach"]

        elif task_phase == "manipulation":
            # Reward for successful manipulation
            ee_to_object = torch.norm(ee_position - object_position, dim=-1)
            rewards["manipulation"] = (
                torch.exp(-ee_to_object) +
                object_grasped.float() * 5.0
            ) * self.weights["manipulation"]

        elif task_phase == "retreat":
            # Reward for returning to target
            robot_to_target = torch.norm(
                robot_position[:, :2] - target_position[:, :2],
                dim=-1
            )
            rewards["retreat"] = (
                object_grasped.float() *
                torch.exp(-robot_to_target)
            ) * self.weights["retreat"]

        rewards["total"] = sum(rewards.values())
        return rewards
```

---

## Balance Control

### Balance Controller

```python
class BalanceController(nn.Module):
    """
    Balance controller for humanoid stability.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_joints: int = 32,
    ):
        super().__init__()

        # Input: IMU + foot contacts + COM velocity
        self.encoder = nn.Sequential(
            nn.Linear(9 + 4 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output: joint corrections
        self.correction_head = nn.Linear(hidden_dim, num_joints)

    def forward(
        self,
        imu_data: torch.Tensor,
        foot_contacts: torch.Tensor,
        com_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute balance correction.

        Args:
            imu_data: (batch, 9) orientation + angular vel + acceleration
            foot_contacts: (batch, 4) binary contact flags
            com_velocity: (batch, 3) center of mass velocity

        Returns:
            correction: (batch, num_joints) joint position corrections
        """
        x = torch.cat([imu_data, foot_contacts, com_velocity], dim=-1)
        features = self.encoder(x)
        correction = self.correction_head(features)
        return correction


class ZMPBalanceController:
    """
    Zero Moment Point (ZMP) based balance controller.
    """

    def __init__(
        self,
        support_polygon: np.ndarray,
        com_height: float = 0.9,
    ):
        self.support_polygon = support_polygon
        self.com_height = com_height

    def compute_zmp(
        self,
        com_position: np.ndarray,
        com_acceleration: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Zero Moment Point from COM dynamics.

        ZMP_x = COM_x - COM_z * a_x / g
        ZMP_y = COM_y - COM_z * a_y / g
        """
        g = 9.81
        zmp_x = com_position[0] - com_position[2] * com_acceleration[0] / g
        zmp_y = com_position[1] - com_position[2] * com_acceleration[1] / g
        return np.array([zmp_x, zmp_y])

    def is_stable(self, zmp: np.ndarray) -> bool:
        """Check if ZMP is within support polygon."""
        from shapely.geometry import Point, Polygon
        return Polygon(self.support_polygon).contains(Point(zmp))

    def compute_recovery_action(
        self,
        current_zmp: np.ndarray,
        target_zmp: np.ndarray = None,
    ) -> np.ndarray:
        """Compute hip adjustment to recover balance."""
        if target_zmp is None:
            target_zmp = self.support_polygon.mean(axis=0)

        zmp_error = target_zmp - current_zmp
        hip_adjustment = zmp_error * 0.1  # Proportional control
        return hip_adjustment
```

---

## Task Priority Control

### Priority-Based Whole-Body IK

```python
class TaskPriorityController:
    """
    Task priority based whole-body inverse kinematics.

    Priority order:
    1. Balance (highest)
    2. Safety constraints
    3. Task execution (lowest)
    """

    def __init__(
        self,
        robot_model,
        num_joints: int = 32,
    ):
        self.robot = robot_model
        self.num_joints = num_joints

    def solve(
        self,
        balance_task: Dict,
        safety_task: Dict,
        execution_task: Dict,
    ) -> np.ndarray:
        """
        Solve for joint velocities with task priorities.

        Uses null-space projection to satisfy lower priority tasks
        without interfering with higher priority ones.
        """
        # Initialize
        q_dot = np.zeros(self.num_joints)
        null_space = np.eye(self.num_joints)

        # Priority 1: Balance task
        J_balance = self._get_balance_jacobian()
        x_dot_balance = balance_task["velocity"]

        q_dot_balance = np.linalg.pinv(J_balance) @ x_dot_balance
        q_dot += q_dot_balance

        # Update null space
        null_space = null_space @ (
            np.eye(self.num_joints) -
            np.linalg.pinv(J_balance) @ J_balance
        )

        # Priority 2: Safety task
        J_safety = self._get_safety_jacobian()
        x_dot_safety = safety_task["velocity"]

        J_safety_proj = J_safety @ null_space
        q_dot_safety = np.linalg.pinv(J_safety_proj) @ (
            x_dot_safety - J_safety @ q_dot
        )
        q_dot += null_space @ q_dot_safety

        # Update null space
        null_space = null_space @ (
            np.eye(self.num_joints) -
            np.linalg.pinv(J_safety_proj) @ J_safety_proj
        )

        # Priority 3: Execution task
        J_exec = self._get_execution_jacobian()
        x_dot_exec = execution_task["velocity"]

        J_exec_proj = J_exec @ null_space
        q_dot_exec = np.linalg.pinv(J_exec_proj) @ (
            x_dot_exec - J_exec @ q_dot
        )
        q_dot += null_space @ q_dot_exec

        return q_dot
```

---

## Training Commands

### Complete Whole-Body Training Pipeline

```bash
# Stage 1: Train locomotion (see training_locomotion.md)
python train/online_rl/ppo_trainer.py \
    --env humanoid-walk-v0 \
    --num-steps 5000000 \
    --output-dir ./output/humanoid/locomotion

# Stage 2: Train manipulation (see training_manipulation.md)
python train/il/behavioral_cloning.py \
    --model humanoid-manipulation \
    --dataset manipulation-demonstrations \
    --output-dir ./output/humanoid/manipulation

# Stage 3: Train whole-body controller
python train/embodiment/train_humanoid_vla.py \
    --pretrained-locomotion ./output/humanoid/locomotion/best.pt \
    --pretrained-manipulation ./output/humanoid/manipulation/best.pt \
    --train-whole-body \
    --output-dir ./output/humanoid/whole-body

# Stage 4: Train loco-manipulation
python train/online_rl/ppo_trainer.py \
    --env humanoid-loco-manip-v0 \
    --pretrained ./output/humanoid/whole-body/best.pt \
    --tasks carry,mobile-manip,reach-while-walk \
    --num-steps 10000000 \
    --output-dir ./output/humanoid/loco-manip

# Stage 5: Train hierarchical controller
python train/embodiment/train_hierarchical.py \
    --pretrained ./output/humanoid/loco-manip/best.pt \
    --train-task-planner \
    --train-skill-selector \
    --train-motion-generator \
    --output-dir ./output/humanoid/hierarchical

# Stage 6: End-to-end fine-tuning
python train/online_rl/ppo_trainer.py \
    --env humanoid-full-v0 \
    --pretrained ./output/humanoid/hierarchical/best.pt \
    --num-steps 20000000 \
    --output-dir ./output/humanoid/final
```

---

## Datasets for Whole-Body Control

| Dataset | Public Source | Description | Use Case |
|---------|---------------|-------------|----------|
| HumanoidBench | [humanoid-bench.github.io](https://humanoid-bench.github.io/) | Loco-manipulation tasks | Whole-body training |
| AMASS | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) | Whole-body human motion | Motion primitives |
| CMU MoCap | [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/) | Diverse activities | Reference motion |
| Isaac Gym | Simulation | Generated trajectories | Online RL |

---

## Best Practices

### Training Tips

1. **Train components separately first**: Locomotion, then manipulation, then integration
2. **Freeze lower levels**: When training higher levels, freeze lower-level policies
3. **Use task curriculum**: Start with simpler tasks before complex loco-manipulation
4. **Balance is critical**: Always maintain balance priority in whole-body control
5. **Smooth transitions**: Reward smooth transitions between locomotion and manipulation

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Falls during manipulation | Balance not prioritized | Increase balance weight |
| Jerky transitions | Sudden mode switches | Add transition smoothing |
| Poor task execution | Conflicting objectives | Use null-space projection |
| Slow convergence | Too many DOF | Hierarchical training |

---

## Next Steps

- [Locomotion Training](training_locomotion.md) - Bipedal locomotion details
- [Manipulation Training](training_manipulation.md) - Arm control details
- [Training Datasets](training_datasets.md) - Dataset information
- [Deployment](deployment.md) - Real robot deployment
