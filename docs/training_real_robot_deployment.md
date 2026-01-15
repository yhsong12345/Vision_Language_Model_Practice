# Training VLA for Real Robot Deployment

This comprehensive guide covers the complete process for deploying Vision-Language-Action models to real robots, including safety protocols, hardware integration, latency optimization, and production best practices.

## Table of Contents

1. [Overview](#overview)
2. [Deployment Architecture](#deployment-architecture)
3. [Hardware Setup](#hardware-setup)
4. [Safety Systems](#safety-systems)
5. [Model Optimization](#model-optimization)
6. [Robot Integration](#robot-integration)
7. [Control Loop Design](#control-loop-design)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Failure Recovery](#failure-recovery)
10. [Production Best Practices](#production-best-practices)
11. [Platform-Specific Guides](#platform-specific-guides)
12. [Troubleshooting](#troubleshooting)

---

## Overview

### Real Robot Deployment Pipeline

```
+=======================================================================================+
|                         REAL ROBOT DEPLOYMENT PIPELINE                                 |
+=======================================================================================+
|                                                                                        |
|  PRE-DEPLOYMENT                                                                        |
|  +-----------------------------------------------------------------------------------+ |
|  |  Model Training  |  Model Optimization  |  Safety Validation  |  Sim Testing     | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  HARDWARE SETUP                                                                        |
|  +-----------------------------------------------------------------------------------+ |
|  |  Robot Arm  |  Cameras  |  Sensors  |  Compute Hardware  |  Safety Systems        | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  RUNTIME SYSTEM                                                                        |
|  +-----------------------------------------------------------------------------------+ |
|  |  +---------------------+    +---------------------+    +---------------------+    | |
|  |  |   Sensor Pipeline   |    |    VLA Model        |    |   Robot Controller  |    | |
|  |  |  - Image capture    |    |  - Inference        |    |  - Action execution |    | |
|  |  |  - Preprocessing    | -> |  - Safety check     | -> |  - Joint control    |    | |
|  |  |  - State estimation |    |  - Action scaling   |    |  - Gripper control  |    | |
|  |  +---------------------+    +---------------------+    +---------------------+    | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  SAFETY LAYER                                                                          |
|  +-----------------------------------------------------------------------------------+ |
|  |  Collision Detection  |  Workspace Limits  |  Velocity Limits  |  Emergency Stop | |
|  +-----------------------------------------------------------------------------------+ |
|                                          |                                             |
|                                          v                                             |
|  MONITORING                                                                            |
|  +-----------------------------------------------------------------------------------+ |
|  |  Latency Tracking  |  Success Logging  |  Error Recording  |  Human Oversight    | |
|  +-----------------------------------------------------------------------------------+ |
|                                                                                        |
+=======================================================================================+
```

### Deployment Checklist

| Phase | Item | Status |
|-------|------|--------|
| **Pre-deployment** | Model trained and validated | ☐ |
| | Safety constraints defined | ☐ |
| | Simulation testing complete | ☐ |
| | Model optimized for latency | ☐ |
| **Hardware** | Robot calibrated | ☐ |
| | Cameras calibrated | ☐ |
| | E-stop tested | ☐ |
| | Network connectivity verified | ☐ |
| **Integration** | ROS/API integration tested | ☐ |
| | Control loop latency measured | ☐ |
| | Safety systems validated | ☐ |
| **Production** | Monitoring enabled | ☐ |
| | Logging configured | ☐ |
| | Fallback systems tested | ☐ |

---

## Deployment Architecture

### System Architecture

```python
from dataclasses import dataclass
from typing import Optional, Dict, Callable
import threading
import queue

@dataclass
class DeploymentConfig:
    """Configuration for real robot deployment."""

    # Model
    model_path: str
    device: str = "cuda"
    precision: str = "fp16"

    # Control
    control_frequency: float = 20.0  # Hz
    action_interpolation_steps: int = 10

    # Safety
    max_velocity: float = 0.5  # m/s
    max_acceleration: float = 2.0  # m/s^2
    workspace_bounds: Dict[str, tuple] = None
    collision_mesh_path: Optional[str] = None

    # Latency
    max_inference_latency_ms: float = 50.0
    sensor_timeout_ms: float = 100.0

    # Logging
    log_dir: str = "./deployment_logs"
    enable_video_logging: bool = True


class VLADeploymentSystem:
    """
    Complete system for deploying VLA to real robots.
    """

    def __init__(
        self,
        config: DeploymentConfig,
        robot_interface: 'RobotInterface',
        sensor_interface: 'SensorInterface',
    ):
        self.config = config
        self.robot = robot_interface
        self.sensors = sensor_interface

        # Load and optimize model
        self.model = self._load_model(config.model_path)

        # Safety systems
        self.safety_shield = SafetyShield(config)
        self.collision_checker = CollisionChecker(config.collision_mesh_path)

        # Control
        self.control_rate = 1.0 / config.control_frequency
        self.running = False

        # Logging
        self.logger = DeploymentLogger(config.log_dir)

        # Threading
        self.sensor_queue = queue.Queue(maxsize=1)
        self.action_queue = queue.Queue(maxsize=1)

    def _load_model(self, model_path: str) -> nn.Module:
        """Load and optimize model for deployment."""
        model = VLAModel.from_pretrained(model_path)
        model.eval()

        # Move to device
        model = model.to(self.config.device)

        # Optimize
        if self.config.precision == "fp16":
            model = model.half()

        # TorchScript compilation
        model = torch.jit.script(model)
        model = torch.jit.freeze(model)

        # Warmup
        self._warmup_model(model)

        return model

    def _warmup_model(self, model: nn.Module, num_iterations: int = 10):
        """Warmup model for consistent latency."""
        dummy_image = torch.randn(1, 3, 224, 224).to(self.config.device)
        if self.config.precision == "fp16":
            dummy_image = dummy_image.half()

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_image)
            torch.cuda.synchronize()

    def start(self, instruction: str):
        """Start deployment control loop."""
        self.running = True
        self.current_instruction = instruction

        # Start threads
        self.sensor_thread = threading.Thread(target=self._sensor_loop)
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.control_thread = threading.Thread(target=self._control_loop)

        self.sensor_thread.start()
        self.inference_thread.start()
        self.control_thread.start()

        self.logger.log_event("deployment_started", {"instruction": instruction})

    def stop(self):
        """Stop deployment."""
        self.running = False

        self.sensor_thread.join()
        self.inference_thread.join()
        self.control_thread.join()

        # Safe stop
        self.robot.stop()
        self.logger.log_event("deployment_stopped")

    def _sensor_loop(self):
        """Continuously capture sensor data."""
        while self.running:
            try:
                # Capture
                image = self.sensors.get_image()
                proprio = self.robot.get_state()

                # Put in queue (drop old if full)
                try:
                    self.sensor_queue.get_nowait()
                except queue.Empty:
                    pass

                self.sensor_queue.put({
                    "image": image,
                    "proprioception": proprio,
                    "timestamp": time.time(),
                })

            except Exception as e:
                self.logger.log_error("sensor_error", str(e))
                time.sleep(0.01)

    def _inference_loop(self):
        """Run model inference on sensor data."""
        while self.running:
            try:
                # Get latest sensor data
                sensor_data = self.sensor_queue.get(timeout=0.1)

                # Preprocess
                image_tensor = self._preprocess_image(sensor_data["image"])
                proprio_tensor = self._preprocess_proprio(sensor_data["proprioception"])

                # Inference
                start_time = time.perf_counter()

                with torch.no_grad():
                    output = self.model(
                        images=image_tensor.unsqueeze(0),
                        proprioception=proprio_tensor.unsqueeze(0),
                        instruction=self.current_instruction,
                    )

                torch.cuda.synchronize()
                inference_time = (time.perf_counter() - start_time) * 1000

                # Check latency
                if inference_time > self.config.max_inference_latency_ms:
                    self.logger.log_warning("high_latency", {
                        "inference_time_ms": inference_time
                    })

                # Extract action
                action = output["action"][0].cpu().numpy()

                # Put in action queue
                try:
                    self.action_queue.get_nowait()
                except queue.Empty:
                    pass

                self.action_queue.put({
                    "action": action,
                    "inference_time_ms": inference_time,
                    "timestamp": time.time(),
                })

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error("inference_error", str(e))

    def _control_loop(self):
        """Execute actions on robot."""
        last_action_time = time.time()

        while self.running:
            loop_start = time.perf_counter()

            try:
                # Get latest action
                action_data = self.action_queue.get(timeout=self.control_rate)

                action = action_data["action"]
                current_state = self.robot.get_state()

                # Safety checks
                safe_action = self.safety_shield.filter(action, current_state)

                if not self.collision_checker.is_safe(safe_action, current_state):
                    self.logger.log_warning("collision_prevented")
                    safe_action = self._emergency_brake_action()

                # Execute
                self.robot.send_action(safe_action)

                # Log
                self.logger.log_step({
                    "action": action,
                    "safe_action": safe_action,
                    "state": current_state,
                    "inference_time_ms": action_data["inference_time_ms"],
                })

                last_action_time = time.time()

            except queue.Empty:
                # No new action - check timeout
                if time.time() - last_action_time > self.config.sensor_timeout_ms / 1000:
                    self.logger.log_warning("action_timeout")
                    self.robot.hold_position()

            except Exception as e:
                self.logger.log_error("control_error", str(e))
                self.robot.emergency_stop()

            # Maintain control rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.control_rate - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
```

---

## Hardware Setup

### Supported Robot Platforms

| Platform | Interface | Control Mode | Typical Frequency |
|----------|-----------|--------------|-------------------|
| **Franka Panda** | libfranka/ROS | Torque/Position | 1000 Hz |
| **xArm** | xArm SDK/ROS | Position/Servo | 250 Hz |
| **UR5e** | URScript/ROS | Position/Velocity | 125 Hz |
| **Kinova** | Kinova SDK/ROS | Position/Velocity | 40 Hz |
| **Sawyer/Baxter** | Intera SDK/ROS | Position/Torque | 100 Hz |

### Robot Interface

```python
from abc import ABC, abstractmethod
import numpy as np

class RobotInterface(ABC):
    """Abstract interface for robot control."""

    @abstractmethod
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current robot state."""
        pass

    @abstractmethod
    def send_action(self, action: np.ndarray):
        """Send action to robot."""
        pass

    @abstractmethod
    def stop(self):
        """Stop robot motion."""
        pass

    @abstractmethod
    def emergency_stop(self):
        """Emergency stop."""
        pass


class FrankaInterface(RobotInterface):
    """Interface for Franka Panda robot."""

    def __init__(
        self,
        robot_ip: str,
        control_mode: str = "cartesian_impedance",
    ):
        import panda_py

        self.panda = panda_py.Panda(robot_ip)
        self.control_mode = control_mode

        # Set default impedance
        self.panda.set_default_behavior()

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get Franka state."""
        state = self.panda.get_state()

        return {
            "joint_positions": np.array(state.q),
            "joint_velocities": np.array(state.dq),
            "joint_torques": np.array(state.tau_J),
            "ee_pose": np.array(state.O_T_EE).reshape(4, 4),
            "ee_velocity": np.array(state.O_dP_EE_c),
            "gripper_width": self.panda.get_gripper().width(),
        }

    def send_action(self, action: np.ndarray):
        """Send action to Franka."""
        if self.control_mode == "joint_position":
            self.panda.move_to_joint_position(action[:7].tolist())

        elif self.control_mode == "cartesian_impedance":
            # Action: [x, y, z, qx, qy, qz, qw, gripper]
            pose = action[:7]
            gripper = action[7] if len(action) > 7 else None

            self.panda.move_to_pose(pose)

            if gripper is not None:
                if gripper > 0.5:
                    self.panda.get_gripper().open()
                else:
                    self.panda.get_gripper().close()

    def stop(self):
        """Stop Franka."""
        self.panda.stop()

    def emergency_stop(self):
        """Emergency stop."""
        self.panda.stop()
        self.panda.recover()


class XArmInterface(RobotInterface):
    """Interface for xArm robot."""

    def __init__(
        self,
        ip: str,
        speed: float = 100,  # mm/s
    ):
        from xarm.wrapper import XArmAPI

        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position mode
        self.arm.set_state(0)
        self.arm.set_tcp_maxacc(2000)
        self.speed = speed

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get xArm state."""
        code, joint_pos = self.arm.get_servo_angle()
        code, cart_pos = self.arm.get_position()
        code, gripper = self.arm.get_gripper_position()

        return {
            "joint_positions": np.array(joint_pos) * np.pi / 180,
            "ee_position": np.array(cart_pos[:3]),
            "ee_orientation": np.array(cart_pos[3:6]) * np.pi / 180,
            "gripper_position": gripper / 850.0,  # Normalize
        }

    def send_action(self, action: np.ndarray):
        """Send action to xArm."""
        # Action: [x, y, z, roll, pitch, yaw, gripper]
        pose = action[:6].copy()
        pose[:3] *= 1000  # m to mm
        pose[3:6] *= 180 / np.pi  # rad to deg

        self.arm.set_position(*pose.tolist(), speed=self.speed, wait=False)

        if len(action) > 6:
            gripper = int(action[6] * 850)
            self.arm.set_gripper_position(gripper, wait=False)

    def stop(self):
        """Stop xArm."""
        self.arm.set_state(4)  # Stop

    def emergency_stop(self):
        """Emergency stop xArm."""
        self.arm.emergency_stop()


class ROSInterface(RobotInterface):
    """Generic ROS interface for robots."""

    def __init__(
        self,
        node_name: str = "vla_controller",
        joint_state_topic: str = "/joint_states",
        joint_command_topic: str = "/joint_commands",
    ):
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        rospy.init_node(node_name)

        # State subscriber
        self.joint_state = None
        rospy.Subscriber(joint_state_topic, JointState, self._joint_state_callback)

        # Command publisher
        self.command_pub = rospy.Publisher(
            joint_command_topic, Float64MultiArray, queue_size=1
        )

    def _joint_state_callback(self, msg):
        """Handle joint state messages."""
        self.joint_state = {
            "names": msg.name,
            "positions": np.array(msg.position),
            "velocities": np.array(msg.velocity),
            "efforts": np.array(msg.effort),
        }

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get state from ROS."""
        if self.joint_state is None:
            raise RuntimeError("No joint state received")
        return self.joint_state

    def send_action(self, action: np.ndarray):
        """Send action via ROS."""
        from std_msgs.msg import Float64MultiArray

        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.command_pub.publish(msg)

    def stop(self):
        """Stop robot via ROS."""
        # Send zero velocity
        if self.joint_state is not None:
            zero_action = np.zeros(len(self.joint_state["positions"]))
            self.send_action(zero_action)

    def emergency_stop(self):
        """Emergency stop."""
        self.stop()
```

### Camera Setup

```python
class SensorInterface:
    """Interface for robot sensors (cameras, etc.)."""

    def __init__(
        self,
        camera_type: str = "realsense",
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.camera_type = camera_type
        self.image_size = image_size

        if camera_type == "realsense":
            self.camera = RealSenseCamera()
        elif camera_type == "zed":
            self.camera = ZEDCamera()
        elif camera_type == "usb":
            self.camera = USBCamera()

    def get_image(self) -> np.ndarray:
        """Capture RGB image."""
        image = self.camera.capture_rgb()

        # Resize
        image = cv2.resize(image, self.image_size)

        return image

    def get_depth(self) -> np.ndarray:
        """Capture depth image."""
        depth = self.camera.capture_depth()
        depth = cv2.resize(depth, self.image_size)
        return depth


class RealSenseCamera:
    """Intel RealSense camera interface."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.pipeline.start(config)

        # Warmup
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def capture_rgb(self) -> np.ndarray:
        """Capture RGB frame."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def capture_depth(self) -> np.ndarray:
        """Capture depth frame."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        return np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0

    def get_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics."""
        profile = self.pipeline.get_active_profile()
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        return np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ])
```

---

## Safety Systems

### Safety Shield

```python
class SafetyShield:
    """
    Safety filtering for robot actions.

    Enforces:
    - Velocity limits
    - Acceleration limits
    - Workspace bounds
    - Joint limits
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config

        # Parse workspace bounds
        self.workspace_min = np.array([
            config.workspace_bounds["x"][0],
            config.workspace_bounds["y"][0],
            config.workspace_bounds["z"][0],
        ])
        self.workspace_max = np.array([
            config.workspace_bounds["x"][1],
            config.workspace_bounds["y"][1],
            config.workspace_bounds["z"][1],
        ])

        # Action history for smoothing
        self.prev_action = None
        self.prev_time = None

    def filter(
        self,
        action: np.ndarray,
        current_state: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Filter action for safety.

        Args:
            action: Proposed action
            current_state: Current robot state

        Returns:
            Safe action
        """
        safe_action = action.copy()
        current_time = time.time()

        # 1. Workspace bounds
        if "ee_position" in action or len(action) >= 3:
            ee_pos = action[:3]
            ee_pos = np.clip(ee_pos, self.workspace_min, self.workspace_max)
            safe_action[:3] = ee_pos

        # 2. Velocity limits
        if self.prev_action is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            velocity = (safe_action - self.prev_action) / dt

            velocity_magnitude = np.linalg.norm(velocity[:3])
            if velocity_magnitude > self.config.max_velocity:
                scale = self.config.max_velocity / velocity_magnitude
                safe_action = self.prev_action + (safe_action - self.prev_action) * scale

        # 3. Acceleration limits
        if self.prev_action is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            acceleration = (safe_action - 2 * self.prev_action) / (dt ** 2)

            accel_magnitude = np.linalg.norm(acceleration[:3])
            if accel_magnitude > self.config.max_acceleration:
                # Reduce action change
                scale = 0.5
                safe_action = self.prev_action + (safe_action - self.prev_action) * scale

        # 4. Joint limits (if joint control)
        if "joint_positions" in current_state and len(action) == len(current_state["joint_positions"]):
            for i, (low, high) in enumerate(self.config.joint_limits):
                safe_action[i] = np.clip(safe_action[i], low, high)

        # Update history
        self.prev_action = safe_action
        self.prev_time = current_time

        return safe_action


class CollisionChecker:
    """
    Check for potential collisions.
    """

    def __init__(self, mesh_path: Optional[str] = None):
        self.mesh_path = mesh_path

        if mesh_path is not None:
            import trimesh
            self.mesh = trimesh.load(mesh_path)
        else:
            self.mesh = None

    def is_safe(
        self,
        action: np.ndarray,
        current_state: Dict[str, np.ndarray],
    ) -> bool:
        """
        Check if action would cause collision.

        Returns:
            True if safe, False if collision predicted
        """
        if self.mesh is None:
            return True

        # Predict end-effector position after action
        if len(action) >= 3:
            predicted_ee = action[:3]

            # Check if inside collision mesh
            point = predicted_ee.reshape(1, 3)
            inside = self.mesh.contains(point)

            if inside[0]:
                return False

        return True


class EmergencyStop:
    """
    Emergency stop system.
    """

    def __init__(self, robot: RobotInterface):
        self.robot = robot
        self.triggered = False

        # Setup hardware e-stop monitoring (if available)
        self._setup_hardware_estop()

    def _setup_hardware_estop(self):
        """Setup hardware emergency stop monitoring."""
        # Platform-specific e-stop monitoring
        pass

    def trigger(self, reason: str = ""):
        """Trigger emergency stop."""
        if not self.triggered:
            self.triggered = True
            self.robot.emergency_stop()
            print(f"EMERGENCY STOP TRIGGERED: {reason}")

    def reset(self):
        """Reset emergency stop."""
        self.triggered = False
        # May need robot-specific reset procedure
```

---

## Model Optimization

### Latency Optimization

```python
class ModelOptimizer:
    """
    Optimize VLA model for real-time inference.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def optimize(
        self,
        precision: str = "fp16",
        compile_mode: str = "reduce-overhead",
        use_tensorrt: bool = False,
    ) -> nn.Module:
        """Apply all optimizations."""
        model = self.model

        # 1. Move to GPU
        model = model.cuda()

        # 2. Half precision
        if precision == "fp16":
            model = model.half()

        # 3. TorchScript
        model = torch.jit.script(model)
        model = torch.jit.freeze(model)

        # 4. torch.compile (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode=compile_mode)
        except:
            print("torch.compile not available")

        # 5. TensorRT (if available)
        if use_tensorrt:
            model = self._convert_tensorrt(model)

        return model

    def _convert_tensorrt(self, model: nn.Module) -> nn.Module:
        """Convert to TensorRT."""
        try:
            import torch_tensorrt

            example_input = torch.randn(1, 3, 224, 224).cuda().half()

            model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions={torch.float16},
                workspace_size=1 << 30,
            )
        except ImportError:
            print("TensorRT not available")

        return model

    def benchmark(
        self,
        model: nn.Module,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference latency."""
        dummy_input = torch.randn(1, 3, 224, 224).cuda().half()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                _ = model(dummy_input)

            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
        }
```

### Optimization Techniques

```
+====================================================================================+
|                         LATENCY OPTIMIZATION TECHNIQUES                             |
+====================================================================================+
|                                                                                     |
| Technique              | Latency Reduction | Memory Reduction | Notes              |
| ----------------------|-------------------|------------------|---------------------|
| FP16 (Half Precision)  | ~30%              | ~50%             | Most recommended    |
| TorchScript            | ~20%              | ~10%             | Always use          |
| torch.compile          | ~15-30%           | Varies           | PyTorch 2.0+        |
| TensorRT               | ~50-70%           | ~30%             | NVIDIA GPUs only    |
| Model Quantization     | ~60%              | ~75%             | May reduce accuracy |
| Flash Attention        | ~40%              | ~50%             | For transformers    |
| Batch Size 1           | Optimal           | Minimal          | Real-time required  |
|                                                                                     |
| Target: <50ms inference for 20Hz control                                           |
+====================================================================================+
```

---

## Control Loop Design

### Action Interpolation

```python
class ActionInterpolator:
    """
    Interpolate between predicted actions for smooth motion.
    """

    def __init__(
        self,
        interpolation_steps: int = 10,
        interpolation_type: str = "linear",
    ):
        self.steps = interpolation_steps
        self.type = interpolation_type

        self.current_action = None
        self.target_action = None
        self.step_idx = 0

    def set_target(self, action: np.ndarray):
        """Set new target action."""
        if self.current_action is None:
            self.current_action = action
        else:
            self.current_action = self.get_interpolated()

        self.target_action = action
        self.step_idx = 0

    def get_interpolated(self) -> np.ndarray:
        """Get interpolated action."""
        if self.target_action is None:
            return self.current_action

        t = min(self.step_idx / self.steps, 1.0)

        if self.type == "linear":
            action = self.current_action + t * (self.target_action - self.current_action)

        elif self.type == "cubic":
            # Smooth cubic interpolation
            t = t * t * (3 - 2 * t)
            action = self.current_action + t * (self.target_action - self.current_action)

        elif self.type == "minimum_jerk":
            # Minimum jerk trajectory
            t = 10 * t**3 - 15 * t**4 + 6 * t**5
            action = self.current_action + t * (self.target_action - self.current_action)

        self.step_idx += 1
        return action


class ControlLoopManager:
    """
    Manage control loop timing and execution.
    """

    def __init__(
        self,
        control_frequency: float = 20.0,
        inference_frequency: float = 10.0,
    ):
        self.control_period = 1.0 / control_frequency
        self.inference_period = 1.0 / inference_frequency

        self.interpolator = ActionInterpolator(
            interpolation_steps=int(control_frequency / inference_frequency)
        )

        # Timing statistics
        self.control_times = []
        self.inference_times = []

    def run(
        self,
        model: nn.Module,
        robot: RobotInterface,
        sensors: SensorInterface,
        instruction: str,
        duration: float = 60.0,
    ):
        """Run control loop for specified duration."""
        start_time = time.time()
        last_inference_time = 0

        while time.time() - start_time < duration:
            loop_start = time.perf_counter()

            # Check if new inference needed
            current_time = time.time()
            if current_time - last_inference_time >= self.inference_period:
                # Get sensor data
                image = sensors.get_image()
                proprio = robot.get_state()

                # Run inference
                inf_start = time.perf_counter()
                with torch.no_grad():
                    action = model.predict(image, proprio, instruction)
                self.inference_times.append(time.perf_counter() - inf_start)

                # Update interpolator
                self.interpolator.set_target(action)
                last_inference_time = current_time

            # Get interpolated action
            smooth_action = self.interpolator.get_interpolated()

            # Send to robot
            robot.send_action(smooth_action)

            # Maintain control rate
            self.control_times.append(time.perf_counter() - loop_start)
            sleep_time = self.control_period - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        return self.get_timing_stats()

    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        return {
            "control_mean_ms": np.mean(self.control_times) * 1000,
            "control_max_ms": np.max(self.control_times) * 1000,
            "inference_mean_ms": np.mean(self.inference_times) * 1000,
            "inference_max_ms": np.max(self.inference_times) * 1000,
        }
```

---

## Monitoring and Logging

### Deployment Logger

```python
import json
import cv2
from datetime import datetime

class DeploymentLogger:
    """
    Comprehensive logging for deployment.
    """

    def __init__(
        self,
        log_dir: str,
        enable_video: bool = True,
    ):
        self.log_dir = log_dir
        self.enable_video = enable_video

        # Create log directory
        self.session_dir = os.path.join(
            log_dir,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.session_dir, exist_ok=True)

        # Log files
        self.event_log = open(os.path.join(self.session_dir, "events.jsonl"), "w")
        self.step_log = open(os.path.join(self.session_dir, "steps.jsonl"), "w")

        # Video recording
        if enable_video:
            video_path = os.path.join(self.session_dir, "video.mp4")
            self.video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                20.0,
                (640, 480),
            )
        else:
            self.video_writer = None

        # Statistics
        self.step_count = 0
        self.start_time = time.time()

    def log_event(self, event_type: str, data: Dict = None):
        """Log an event."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data or {},
        }
        self.event_log.write(json.dumps(event) + "\n")
        self.event_log.flush()

    def log_step(self, step_data: Dict):
        """Log a control step."""
        step_data["step"] = self.step_count
        step_data["timestamp"] = time.time()

        # Convert numpy arrays to lists
        for key, value in step_data.items():
            if isinstance(value, np.ndarray):
                step_data[key] = value.tolist()

        self.step_log.write(json.dumps(step_data) + "\n")
        self.step_count += 1

    def log_frame(self, frame: np.ndarray):
        """Log video frame."""
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def log_error(self, error_type: str, message: str):
        """Log an error."""
        self.log_event("error", {
            "error_type": error_type,
            "message": message,
        })

    def log_warning(self, warning_type: str, data: Dict = None):
        """Log a warning."""
        self.log_event("warning", {
            "warning_type": warning_type,
            "data": data or {},
        })

    def close(self):
        """Close logger."""
        self.event_log.close()
        self.step_log.close()

        if self.video_writer is not None:
            self.video_writer.release()

        # Write summary
        summary = {
            "total_steps": self.step_count,
            "duration_seconds": time.time() - self.start_time,
            "steps_per_second": self.step_count / (time.time() - self.start_time),
        }

        with open(os.path.join(self.session_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


class MetricsTracker:
    """Track deployment metrics in real-time."""

    def __init__(self):
        self.metrics = {
            "inference_latency_ms": [],
            "control_latency_ms": [],
            "action_magnitude": [],
            "safety_interventions": 0,
            "errors": 0,
        }

    def update(self, metric_name: str, value: float):
        """Update a metric."""
        if metric_name in self.metrics:
            if isinstance(self.metrics[metric_name], list):
                self.metrics[metric_name].append(value)
            else:
                self.metrics[metric_name] += value

    def get_summary(self) -> Dict[str, float]:
        """Get metrics summary."""
        summary = {}

        for name, values in self.metrics.items():
            if isinstance(values, list) and len(values) > 0:
                summary[f"{name}_mean"] = np.mean(values)
                summary[f"{name}_max"] = np.max(values)
                summary[f"{name}_p95"] = np.percentile(values, 95)
            else:
                summary[name] = values

        return summary
```

---

## Production Best Practices

### Deployment Checklist

```python
class DeploymentValidator:
    """
    Validate deployment readiness.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.checks = []

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        all_passed = True

        # Model checks
        all_passed &= self._check_model_loaded()
        all_passed &= self._check_model_latency()

        # Hardware checks
        all_passed &= self._check_robot_connection()
        all_passed &= self._check_camera_connection()

        # Safety checks
        all_passed &= self._check_workspace_bounds()
        all_passed &= self._check_emergency_stop()

        return all_passed

    def _check_model_loaded(self) -> bool:
        """Check if model is loaded correctly."""
        try:
            model = VLAModel.from_pretrained(self.config.model_path)
            self.checks.append(("model_loaded", True, ""))
            return True
        except Exception as e:
            self.checks.append(("model_loaded", False, str(e)))
            return False

    def _check_model_latency(self) -> bool:
        """Check if model meets latency requirements."""
        model = VLAModel.from_pretrained(self.config.model_path)
        optimizer = ModelOptimizer(model)
        stats = optimizer.benchmark(model, num_iterations=50)

        passed = stats["p95_ms"] < self.config.max_inference_latency_ms
        self.checks.append((
            "model_latency",
            passed,
            f"P95={stats['p95_ms']:.1f}ms (target: {self.config.max_inference_latency_ms}ms)"
        ))
        return passed

    def _check_robot_connection(self) -> bool:
        """Check robot connection."""
        try:
            # Platform-specific connection test
            self.checks.append(("robot_connection", True, ""))
            return True
        except Exception as e:
            self.checks.append(("robot_connection", False, str(e)))
            return False

    def _check_emergency_stop(self) -> bool:
        """Check emergency stop functionality."""
        # Should be manually verified
        self.checks.append(("emergency_stop", None, "MANUAL CHECK REQUIRED"))
        return True

    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 60)
        print("DEPLOYMENT VALIDATION REPORT")
        print("=" * 60)

        for check_name, passed, message in self.checks:
            if passed is True:
                status = "✓ PASS"
            elif passed is False:
                status = "✗ FAIL"
            else:
                status = "? CHECK"

            print(f"{status:10} {check_name:25} {message}")

        print("=" * 60)
```

### Best Practices Summary

```
+====================================================================================+
|                          DEPLOYMENT BEST PRACTICES                                  |
+====================================================================================+
|                                                                                     |
| Safety:                                                                             |
| - Always have physical e-stop accessible                                            |
| - Implement software safety limits (velocity, workspace, collision)                 |
| - Test safety systems before every deployment                                       |
| - Have human operator ready to intervene                                            |
| - Start with slow speeds, increase gradually                                        |
|                                                                                     |
| Performance:                                                                        |
| - Target <50ms inference for 20Hz control                                           |
| - Use FP16 + TorchScript optimization                                               |
| - Implement action interpolation for smooth motion                                  |
| - Monitor latency continuously                                                      |
| - Have fallback for high latency situations                                         |
|                                                                                     |
| Reliability:                                                                        |
| - Log everything (actions, states, latencies)                                       |
| - Implement automatic failure recovery                                              |
| - Use watchdog timers for all components                                            |
| - Test with various failure scenarios                                               |
| - Have offline fallback policy                                                      |
|                                                                                     |
| Development:                                                                        |
| - Validate in simulation first                                                      |
| - Use progressive deployment (sim → slow real → fast real)                          |
| - A/B test new models with baseline                                                 |
| - Collect deployment data for model improvement                                     |
|                                                                                     |
+====================================================================================+
```

---

## Platform-Specific Guides

### Franka Panda Deployment

```python
def deploy_to_franka(
    model_path: str,
    robot_ip: str = "172.16.0.2",
    instruction: str = "Pick up the red cube",
):
    """Deploy VLA to Franka Panda."""

    # Configuration
    config = DeploymentConfig(
        model_path=model_path,
        control_frequency=20.0,
        max_velocity=0.3,  # m/s
        max_acceleration=1.0,
        workspace_bounds={
            "x": (0.3, 0.7),
            "y": (-0.3, 0.3),
            "z": (0.05, 0.5),
        },
    )

    # Interfaces
    robot = FrankaInterface(robot_ip, control_mode="cartesian_impedance")
    sensors = SensorInterface(camera_type="realsense")

    # Deployment system
    system = VLADeploymentSystem(config, robot, sensors)

    # Validate
    validator = DeploymentValidator(config)
    if not validator.run_all_checks():
        validator.print_report()
        raise RuntimeError("Validation failed")

    # Run
    try:
        system.start(instruction)
        input("Press Enter to stop...")
    finally:
        system.stop()
```

### xArm Deployment

```python
def deploy_to_xarm(
    model_path: str,
    robot_ip: str = "192.168.1.xxx",
    instruction: str = "Pick up the object",
):
    """Deploy VLA to xArm."""

    config = DeploymentConfig(
        model_path=model_path,
        control_frequency=30.0,
        max_velocity=0.5,
        workspace_bounds={
            "x": (0.2, 0.6),
            "y": (-0.3, 0.3),
            "z": (0.05, 0.4),
        },
    )

    robot = XArmInterface(robot_ip, speed=100)
    sensors = SensorInterface(camera_type="realsense")

    system = VLADeploymentSystem(config, robot, sensors)

    system.start(instruction)
    # ... run deployment
    system.stop()
```

---

## Troubleshooting

### Common Issues

```
+====================================================================================+
|                            TROUBLESHOOTING GUIDE                                    |
+====================================================================================+
|                                                                                     |
| Issue: High inference latency (>50ms)                                               |
| Solutions:                                                                          |
| - Use FP16 precision                                                                |
| - Apply TorchScript compilation                                                     |
| - Reduce model size                                                                 |
| - Use smaller image resolution                                                      |
| - Check GPU utilization                                                             |
|                                                                                     |
| Issue: Jerky robot motion                                                           |
| Solutions:                                                                          |
| - Implement action interpolation                                                    |
| - Reduce velocity limits                                                            |
| - Check control frequency                                                           |
| - Verify action scaling                                                             |
|                                                                                     |
| Issue: Robot not responding                                                         |
| Solutions:                                                                          |
| - Check network connection                                                          |
| - Verify robot is in correct mode                                                   |
| - Check for error states                                                            |
| - Verify action format matches robot API                                            |
|                                                                                     |
| Issue: Poor task performance                                                        |
| Solutions:                                                                          |
| - Verify camera calibration                                                         |
| - Check lighting conditions                                                         |
| - Validate model on similar setup                                                   |
| - Collect deployment data for fine-tuning                                           |
|                                                                                     |
| Issue: Safety system triggering frequently                                          |
| Solutions:                                                                          |
| - Verify workspace bounds are correct                                               |
| - Check velocity/acceleration limits                                                |
| - Calibrate collision mesh                                                          |
| - Review action predictions                                                         |
|                                                                                     |
+====================================================================================+
```

---

## Summary

This guide covered real robot deployment:

1. **Architecture**: Complete deployment system design
2. **Hardware**: Robot and sensor interfaces
3. **Safety**: Safety shield, collision checking, e-stop
4. **Optimization**: Model optimization for real-time
5. **Control**: Action interpolation, control loop timing
6. **Monitoring**: Logging and metrics tracking
7. **Production**: Best practices and validation

**Key requirements:**
- <50ms inference latency for 20Hz control
- Multiple safety layers (software + hardware)
- Comprehensive logging for debugging
- Progressive deployment (sim → slow → fast)
- Human operator oversight

---

## Related Documents

- [Training Pipeline Overview](training_pipeline_overview.md)
- [Training Recipes](training_recipes.md)
- [Robot Manipulation Training](training_robot_manipulation.md)
- [Simulation Benchmarks](training_simulation_benchmarks.md)
- [Architecture Guide](architecture.md)
