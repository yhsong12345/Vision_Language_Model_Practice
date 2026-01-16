# Humanoid Manipulation Training

This document covers the complete training process for humanoid arm control and manipulation, including reaching, grasping, and bimanual coordination.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Manipulation Training Pipeline                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Reaching                                                           │
│  ├── End-effector position control                                          │
│  ├── Inverse kinematics learning                                            │
│  └── Workspace coverage                                                     │
│                                                                              │
│  Phase 2: Grasping                                                           │
│  ├── Pre-grasp approach                                                     │
│  ├── Grasp execution                                                        │
│  ├── Object lifting                                                         │
│  └── Multi-object grasping                                                  │
│                                                                              │
│  Phase 3: Dexterous Manipulation                                             │
│  ├── In-hand manipulation                                                   │
│  ├── Tool use                                                               │
│  └── Precision tasks                                                        │
│                                                                              │
│  Phase 4: Bimanual Coordination                                              │
│  ├── Two-handed grasping                                                    │
│  ├── Coordinated manipulation                                               │
│  └── Handover tasks                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Arm Configuration

### 7-DoF Arm Joints (per arm)

| Joint | Type | Range (rad) | Description |
|-------|------|-------------|-------------|
| Shoulder Pitch | Revolute | [-π, π] | Forward/backward swing |
| Shoulder Roll | Revolute | [-π/2, π/2] | Side raise |
| Shoulder Yaw | Revolute | [-π, π] | Rotation |
| Elbow | Revolute | [0, 2.62] | Flexion/extension |
| Wrist Yaw | Revolute | [-π, π] | Forearm rotation |
| Wrist Pitch | Revolute | [-π/2, π/2] | Up/down bend |
| Wrist Roll | Revolute | [-π, π] | Hand rotation |

### Manipulation Policy Architecture

```python
from model.embodiment.humanoid import ManipulationPolicy

class ManipulationPolicy(nn.Module):
    """
    Manipulation policy for humanoid arm control.
    """

    def __init__(self, config: HumanoidConfig):
        super().__init__()
        self.arm_joints = 14  # Both arms (7 each)

        # Visual encoder for manipulation
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, config.hidden_dim),
        )

        # Arm proprioception encoder
        self.arm_proprio_encoder = nn.Sequential(
            nn.Linear(self.arm_joints * 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
        )

        # End-effector goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(6 * 2, config.hidden_dim // 2),  # 6D pose for each hand
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
        )

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        # Action heads
        self.arm_action_head = nn.Linear(config.hidden_dim, self.arm_joints)
        self.gripper_head = nn.Linear(config.hidden_dim, 2)

    def forward(
        self,
        image: torch.Tensor,
        arm_joint_positions: torch.Tensor,
        arm_joint_velocities: torch.Tensor,
        target_poses: Optional[torch.Tensor] = None,
        language_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Encode visual input
        visual_features = self.visual_encoder(image)

        # Encode proprioception
        arm_state = torch.cat([arm_joint_positions, arm_joint_velocities], dim=-1)
        arm_features = self.arm_proprio_encoder(arm_state)

        # Encode goal
        if target_poses is not None:
            goal_features = self.goal_encoder(target_poses)
        else:
            goal_features = torch.zeros_like(arm_features)

        # Combine features
        features = torch.cat([visual_features, arm_features, goal_features], dim=-1)

        # Add language conditioning
        if language_features is not None:
            features = features + language_features

        # Generate actions
        policy_features = self.policy(features)
        arm_actions = self.arm_action_head(policy_features)
        gripper_actions = torch.sigmoid(self.gripper_head(policy_features))

        return {
            "arm_actions": arm_actions,
            "gripper_actions": gripper_actions,
            "features": policy_features,
        }
```

---

## Phase 1: Reaching Training

### Behavioral Cloning for Reaching

```python
class ReachingTrainer:
    """
    Train reaching behavior from demonstrations.
    """

    def __init__(
        self,
        model: ManipulationPolicy,
        learning_rate: float = 1e-4,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_ee_error = 0

        for batch in dataloader:
            # Unpack batch
            images = batch["images"].to(self.device)
            arm_pos = batch["arm_joint_positions"].to(self.device)
            arm_vel = batch["arm_joint_velocities"].to(self.device)
            target_pos = batch["target_positions"].to(self.device)
            expert_joints = batch["expert_joint_positions"].to(self.device)

            # Forward pass
            outputs = self.model(
                image=images,
                arm_joint_positions=arm_pos,
                arm_joint_velocities=arm_vel,
                target_poses=target_pos,
            )

            # Joint position loss
            joint_loss = F.mse_loss(outputs["arm_actions"], expert_joints)

            # End-effector position loss (computed via FK)
            predicted_ee = self.forward_kinematics(outputs["arm_actions"])
            ee_loss = F.mse_loss(predicted_ee[:, :3], target_pos[:, :3])

            # Total loss
            loss = joint_loss + 0.5 * ee_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_ee_error += ee_loss.item()

        return {
            "loss": total_loss / len(dataloader),
            "ee_error": total_ee_error / len(dataloader),
        }
```

### Reaching with Reinforcement Learning

```python
def compute_reaching_reward(
    ee_position: torch.Tensor,
    target_position: torch.Tensor,
    joint_velocities: torch.Tensor,
    success_threshold: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """
    Reward function for reaching tasks.
    """
    rewards = {}

    # Distance to target
    distance = torch.norm(ee_position - target_position, dim=-1)
    rewards["distance"] = -distance

    # Success bonus
    success = (distance < success_threshold).float()
    rewards["success"] = success * 10.0

    # Smoothness penalty
    rewards["smooth"] = -torch.sum(joint_velocities ** 2, dim=-1) * 0.01

    # Total
    rewards["total"] = (
        rewards["distance"] +
        rewards["success"] +
        rewards["smooth"]
    )

    return rewards
```

### Training Command

```bash
# Behavioral cloning
python train/il/behavioral_cloning.py \
    --model humanoid-manipulation \
    --dataset reaching-demonstrations \
    --learning-rate 1e-4 \
    --num-epochs 100 \
    --output-dir ./output/humanoid/manipulation/reaching-bc

# RL fine-tuning
python train/online_rl/sac_trainer.py \
    --env humanoid-reach-v0 \
    --pretrained ./output/humanoid/manipulation/reaching-bc/best.pt \
    --num-steps 1000000 \
    --output-dir ./output/humanoid/manipulation/reaching-rl
```

---

## Phase 2: Grasping Training

### Two-Phase Grasping Training

```python
class GraspingTrainer:
    """
    Train grasping in two phases:
    1. Approach phase (BC)
    2. Grasp execution (RL)
    """

    def __init__(self, model: ManipulationPolicy, env: GraspingEnv):
        self.model = model
        self.env = env

    def train_approach_phase(
        self,
        dataset: GraspingDataset,
        num_epochs: int = 50,
    ):
        """Train approach behavior with behavioral cloning."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            for batch in dataset:
                # Filter to pre-grasp phase only
                approach_mask = batch["phase"] == "approach"

                outputs = self.model(
                    image=batch["images"][approach_mask],
                    arm_joint_positions=batch["arm_pos"][approach_mask],
                    arm_joint_velocities=batch["arm_vel"][approach_mask],
                )

                loss = F.mse_loss(
                    outputs["arm_actions"],
                    batch["expert_joints"][approach_mask],
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Approach phase epoch {epoch}: loss={loss.item():.4f}")

    def train_grasp_execution(
        self,
        num_steps: int = 1_000_000,
    ):
        """Train grasp execution with RL."""
        self.env.set_reward_fn(GraspRewardFunction(
            approach_weight=0.1,
            grasp_weight=0.5,
            lift_weight=0.4,
            success_bonus=10.0,
        ))

        trainer = SACTrainer(
            model=self.model,
            env=self.env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
        )

        trainer.learn(num_steps)
```

### Grasping Reward Function

```python
class GraspRewardFunction:
    """
    Reward function for grasping tasks.
    """

    def __init__(
        self,
        approach_weight: float = 0.1,
        grasp_weight: float = 0.5,
        lift_weight: float = 0.4,
        success_bonus: float = 10.0,
    ):
        self.approach_weight = approach_weight
        self.grasp_weight = grasp_weight
        self.lift_weight = lift_weight
        self.success_bonus = success_bonus

    def __call__(
        self,
        ee_position: torch.Tensor,
        object_position: torch.Tensor,
        gripper_state: torch.Tensor,
        object_grasped: torch.Tensor,
        object_lifted: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        rewards = {}

        # Approach reward (distance to object)
        distance = torch.norm(ee_position - object_position, dim=-1)
        rewards["approach"] = torch.exp(-distance * 5) * self.approach_weight

        # Grasp reward (gripper closed on object)
        rewards["grasp"] = object_grasped.float() * self.grasp_weight

        # Lift reward (object height)
        if object_lifted.any():
            lift_height = object_position[:, 2] - 0.1  # Above initial height
            rewards["lift"] = torch.clamp(lift_height, 0, 0.3) * self.lift_weight
        else:
            rewards["lift"] = torch.zeros_like(distance)

        # Success bonus
        success = object_lifted & (lift_height > 0.1)
        rewards["success"] = success.float() * self.success_bonus

        # Total
        rewards["total"] = sum(rewards.values())

        return rewards
```

### Multi-Object Grasping

```bash
python train/online_rl/sac_trainer.py \
    --env humanoid-grasp-multi-v0 \
    --pretrained ./output/humanoid/manipulation/grasping/best.pt \
    --object-types box,cylinder,sphere,mug,bottle \
    --num-steps 5000000 \
    --output-dir ./output/humanoid/manipulation/multi-object
```

---

## Phase 3: Dexterous Manipulation

### Hand Configuration (Optional Dexterous Hand)

| Finger | Joints | DoF |
|--------|--------|-----|
| Thumb | CMC, MCP, IP | 3 |
| Index | MCP, PIP, DIP | 3 |
| Middle | MCP, PIP, DIP | 3 |
| Ring | MCP, PIP, DIP | 3 |
| Pinky | MCP, PIP, DIP | 3 |
| **Total per hand** | | **15** |

### In-Hand Manipulation

```python
class InHandManipulationTrainer:
    """
    Train in-hand manipulation skills.
    """

    def __init__(self, model: DexterousHandPolicy, env: InHandEnv):
        self.model = model
        self.env = env

    def train_object_rotation(
        self,
        num_steps: int = 5_000_000,
    ):
        """Train to rotate object in hand."""
        reward_fn = RotationRewardFunction(
            target_rotation_axis=[0, 0, 1],
            target_rotation_speed=0.5,  # rad/s
        )

        self.env.set_reward_fn(reward_fn)

        trainer = PPOTrainer(
            model=self.model,
            env=self.env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)

    def train_finger_gaiting(
        self,
        num_steps: int = 5_000_000,
    ):
        """Train finger gaiting for object repositioning."""
        reward_fn = FingerGaitingRewardFunction(
            maintain_grasp_weight=0.5,
            reposition_weight=0.5,
        )

        self.env.set_reward_fn(reward_fn)

        trainer = PPOTrainer(
            model=self.model,
            env=self.env,
            learning_rate=3e-4,
        )

        trainer.learn(num_steps)
```

### Tool Use

```bash
python train/online_rl/ppo_trainer.py \
    --env humanoid-tool-use-v0 \
    --pretrained ./output/humanoid/manipulation/grasping/best.pt \
    --tools hammer,screwdriver,wrench \
    --num-steps 10000000 \
    --output-dir ./output/humanoid/manipulation/tool-use
```

---

## Phase 4: Bimanual Coordination

### Bimanual Policy Architecture

```python
class BimanualCoordinator(nn.Module):
    """
    Coordinate left and right arm actions.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_joints_per_arm: int = 7,
    ):
        super().__init__()
        self.num_joints = num_joints_per_arm

        # Individual arm encoders
        self.left_encoder = nn.Sequential(
            nn.Linear(num_joints_per_arm * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.right_encoder = nn.Sequential(
            nn.Linear(num_joints_per_arm * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention for coordination
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )

        # Action heads
        self.left_action_head = nn.Linear(hidden_dim, num_joints_per_arm)
        self.right_action_head = nn.Linear(hidden_dim, num_joints_per_arm)

    def forward(
        self,
        vlm_features: torch.Tensor,
        left_proprio: torch.Tensor,
        right_proprio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode each arm
        left_feat = self.left_encoder(left_proprio)
        right_feat = self.right_encoder(right_proprio)

        # Stack for attention
        arm_features = torch.stack([left_feat, right_feat], dim=1)

        # Cross-attention with VLM features as query
        coordinated, _ = self.cross_attention(
            vlm_features.unsqueeze(1),
            arm_features,
            arm_features,
        )

        # Generate actions
        left_actions = self.left_action_head(coordinated[:, 0] + left_feat)
        right_actions = self.right_action_head(coordinated[:, 0] + right_feat)

        return left_actions, right_actions

    def collision_avoidance_loss(
        self,
        left_actions: torch.Tensor,
        right_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize configurations that could cause arm collision."""
        # Compute end-effector positions
        left_ee = self.forward_kinematics_left(left_actions)
        right_ee = self.forward_kinematics_right(right_actions)

        # Distance between end-effectors
        distance = torch.norm(left_ee - right_ee, dim=-1)

        # Penalty for being too close
        min_distance = 0.1  # 10cm minimum
        collision_penalty = F.relu(min_distance - distance)

        return collision_penalty.mean()
```

### Bimanual Training

```python
class BimanualTrainer:
    """
    Train bimanual manipulation tasks.
    """

    def __init__(
        self,
        model: ManipulationPolicy,
        coordinator: BimanualCoordinator,
        env: BimanualEnv,
    ):
        self.model = model
        self.coordinator = coordinator
        self.env = env

    def train(
        self,
        dataset: BimanualDataset,
        num_epochs: int = 100,
    ):
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(self.coordinator.parameters()),
            lr=1e-4,
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                # Get VLM features
                vlm_features = self.model.encode(
                    images=batch["images"],
                    instruction=batch["instruction"],
                )

                # Coordinate arms
                left_targets, right_targets = self.coordinator(
                    vlm_features,
                    batch["left_proprio"],
                    batch["right_proprio"],
                )

                # Imitation loss
                left_loss = F.mse_loss(left_targets, batch["left_expert"])
                right_loss = F.mse_loss(right_targets, batch["right_expert"])

                # Coordination loss
                coord_loss = self.coordinator.collision_avoidance_loss(
                    left_targets, right_targets
                )

                total_loss = left_loss + right_loss + 0.1 * coord_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: loss={total_loss.item():.4f}")
```

### Training Commands

```bash
# Two-handed grasping
python train/il/behavioral_cloning.py \
    --model humanoid-bimanual \
    --dataset two-handed-grasping \
    --coordination-loss 0.1 \
    --output-dir ./output/humanoid/manipulation/bimanual-grasp

# Coordinated manipulation
python train/online_rl/ppo_trainer.py \
    --env humanoid-bimanual-v0 \
    --pretrained ./output/humanoid/manipulation/bimanual-grasp/best.pt \
    --tasks carry-large-object,pour,fold \
    --num-steps 10000000 \
    --output-dir ./output/humanoid/manipulation/bimanual-rl
```

---

## Visual Servoing

### Eye-Hand Coordination

```python
class VisualServoingController:
    """
    Visual servoing for precise manipulation.
    """

    def __init__(
        self,
        camera_intrinsics: np.ndarray,
        control_gain: float = 0.5,
    ):
        self.K = camera_intrinsics
        self.gain = control_gain

    def compute_jacobian(
        self,
        image_points: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute image Jacobian for visual servoing.

        Args:
            image_points: (batch, 2) pixel coordinates
            depth: (batch,) depth values

        Returns:
            jacobian: (batch, 2, 6) image Jacobian
        """
        u, v = image_points[:, 0], image_points[:, 1]
        Z = depth
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # Interaction matrix
        Lx = torch.stack([
            -fx / Z,
            torch.zeros_like(Z),
            (u - cx) / Z,
            (u - cx) * (v - cy) / fy,
            -(fx**2 + (u - cx)**2) / fx,
            (v - cy),
        ], dim=-1)

        Ly = torch.stack([
            torch.zeros_like(Z),
            -fy / Z,
            (v - cy) / Z,
            (fy**2 + (v - cy)**2) / fy,
            -(u - cx) * (v - cy) / fx,
            -(u - cx),
        ], dim=-1)

        return torch.stack([Lx, Ly], dim=1)

    def compute_velocity(
        self,
        current_image_point: torch.Tensor,
        target_image_point: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute end-effector velocity for visual servoing.
        """
        # Image error
        error = target_image_point - current_image_point

        # Jacobian
        J = self.compute_jacobian(current_image_point, depth)

        # Pseudo-inverse
        J_pinv = torch.linalg.pinv(J)

        # Velocity command
        velocity = -self.gain * J_pinv @ error.unsqueeze(-1)

        return velocity.squeeze(-1)
```

---

## Datasets for Manipulation

| Dataset | Public Source | Description | Use Case |
|---------|---------------|-------------|----------|
| DROID | [cadene/droid](https://huggingface.co/datasets/cadene/droid) | VR teleoperation data | BC pretraining |
| LeRobot ALOHA | [lerobot/aloha_sim](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) | Bimanual manipulation | Bimanual tasks |
| Open X-Embodiment | [jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) | Multi-robot manipulation | Cross-embodiment |
| ContactDB | [contactdb.cc.gatech.edu](https://contactdb.cc.gatech.edu/) | Grasping contact maps | Grasp planning |
| YCB Objects | [ycb-benchmarks.org](http://ycb-benchmarks.org/) | Standard object set | Evaluation |

---

## Best Practices

### Training Tips

1. **Start with BC**: Use behavioral cloning for initial policy
2. **Phase decomposition**: Train approach and execution separately
3. **Visual features**: Use pretrained vision encoders (CLIP, DinoV2)
4. **Object-centric**: Use object-centric representations when possible
5. **Force feedback**: Include force/torque sensing for contact-rich tasks

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Missed grasps | Poor approach trajectory | Train approach phase separately |
| Drops objects | Insufficient grip force | Add grip force reward |
| Arm collision | No coordination | Add collision avoidance loss |
| Slow execution | Over-regularized | Reduce smoothness penalty |
| Poor generalization | Overfitting to objects | Domain randomization |

---

## Next Steps

- [Whole-Body Control](training_whole_body.md) - Loco-manipulation integration
- [Locomotion Training](training_locomotion.md) - Bipedal locomotion
- [Deployment](deployment.md) - Real robot deployment
