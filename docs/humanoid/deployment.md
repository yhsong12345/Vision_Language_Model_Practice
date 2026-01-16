# Humanoid Deployment Guide

This document covers the deployment of trained VLA models to real humanoid robots, including safety considerations, model optimization, and real-time control.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Deployment Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Trained Model                                                               │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │ Model Export    │     │ Optimization    │     │ Safety          │       │
│  │ (TorchScript/   │────▶│ (Quantization/  │────▶│ Validation      │       │
│  │  ONNX)          │     │  TensorRT)      │     │                 │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                           │                 │
│                                                           ▼                 │
│                                                  ┌─────────────────┐       │
│                                                  │ Real Robot      │       │
│                                                  │ Deployment      │       │
│                                                  └─────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Safety-First Deployment

### Critical Safety Requirements

| Requirement | Description | Implementation |
|-------------|-------------|----------------|
| Joint Limits | Respect joint position limits | Hard clipping before execution |
| Velocity Limits | Limit joint velocities | Velocity scaling |
| Torque Limits | Limit motor torques | Torque saturation |
| Balance Monitoring | Detect instability | ZMP/COM monitoring |
| Emergency Stop | Immediate halt capability | Hardware E-stop + software |
| Collision Avoidance | Prevent self-collision | Real-time collision checking |

### Safety Shield Implementation

```python
from model.safety import SafetyShield, SafetyConfig

class HumanoidSafetyShield:
    """
    Safety shield for humanoid robot deployment.

    Monitors and filters all commands before execution.
    """

    def __init__(
        self,
        joint_limits: Dict[str, Tuple[float, float]],
        max_joint_velocity: float = 5.0,    # rad/s
        max_joint_torque: float = 100.0,    # Nm
        min_com_height: float = 0.3,        # meters
    ):
        self.joint_limits = joint_limits
        self.max_velocity = max_joint_velocity
        self.max_torque = max_joint_torque
        self.min_com_height = min_com_height

        # Balance monitor
        self.balance_monitor = BalanceMonitor()

        # Collision checker
        self.collision_checker = SelfCollisionChecker()

    def check_and_filter(
        self,
        target_positions: np.ndarray,
        current_positions: np.ndarray,
        current_velocities: np.ndarray,
        imu_data: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        Check safety constraints and filter commands.

        Returns:
            filtered_positions: Safe joint targets
            is_safe: Whether original command was safe
        """
        is_safe = True
        filtered = target_positions.copy()

        # 1. Joint position limits
        for i, (name, (lower, upper)) in enumerate(self.joint_limits.items()):
            if filtered[i] < lower:
                filtered[i] = lower
                is_safe = False
            elif filtered[i] > upper:
                filtered[i] = upper
                is_safe = False

        # 2. Velocity limits
        delta = filtered - current_positions
        max_delta = self.max_velocity * dt
        if np.any(np.abs(delta) > max_delta):
            delta = np.clip(delta, -max_delta, max_delta)
            filtered = current_positions + delta
            is_safe = False

        # 3. Acceleration limits
        implied_velocity = delta / dt
        acceleration = (implied_velocity - current_velocities) / dt
        max_accel = 50.0  # rad/s^2
        if np.any(np.abs(acceleration) > max_accel):
            is_safe = False
            # Scale down to respect acceleration
            scale = min(1.0, max_accel / np.max(np.abs(acceleration)))
            filtered = current_positions + delta * scale

        # 4. Self-collision check
        if self.collision_checker.check(filtered):
            is_safe = False
            # Fall back to current position
            filtered = current_positions

        # 5. Balance check
        if not self.balance_monitor.is_stable(imu_data):
            is_safe = False
            # Apply balance correction
            filtered = self._apply_balance_correction(filtered, imu_data)

        return filtered, is_safe

    def _apply_balance_correction(
        self,
        target: np.ndarray,
        imu_data: np.ndarray,
    ) -> np.ndarray:
        """Apply corrective action to maintain balance."""
        # Compute required hip adjustment
        roll, pitch = imu_data[0], imu_data[1]

        # Simple proportional correction
        hip_correction = np.array([
            -pitch * 0.5,  # Lean forward/backward
            -roll * 0.3,   # Lean left/right
        ])

        corrected = target.copy()
        # Apply to hip joints (indices depend on robot)
        corrected[10:12] += hip_correction  # Example hip indices

        return corrected
```

---

## Model Export

### TorchScript Export

```python
from model.utils.export import TorchScriptExporter

exporter = TorchScriptExporter()

# Export with tracing (for models without control flow)
traced_path = exporter.export_traced(
    model=humanoid_vla,
    sample_input=sample_input,
    output_path="./deployed/humanoid_vla_traced.pt",
)

# Export with scripting (for models with control flow)
scripted_path = exporter.export_scripted(
    model=humanoid_vla,
    output_path="./deployed/humanoid_vla_scripted.pt",
)

# Verify export
exporter.verify(
    original_model=humanoid_vla,
    exported_path=traced_path,
    sample_input=sample_input,
)
```

### ONNX Export

```python
from model.utils.export import ONNXExporter

exporter = ONNXExporter()

# Standard export
onnx_path = exporter.export(
    model=humanoid_vla,
    sample_input=sample_input,
    output_path="./deployed/humanoid_vla.onnx",
    input_names=["image", "proprioception", "language_features"],
    output_names=["joint_targets", "gripper_actions"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "proprioception": {0: "batch_size"},
    },
)

# Optimize for inference
optimized_path = exporter.optimize_for_inference(
    onnx_path=onnx_path,
    output_path="./deployed/humanoid_vla_optimized.onnx",
)
```

### TensorRT Optimization (NVIDIA GPUs)

```python
import tensorrt as trt

class TensorRTOptimizer:
    """Optimize ONNX model with TensorRT."""

    def __init__(self, precision: str = "fp16"):
        self.precision = precision
        self.logger = trt.Logger(trt.Logger.WARNING)

    def optimize(
        self,
        onnx_path: str,
        output_path: str,
        max_batch_size: int = 1,
    ) -> str:
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        if self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # Requires calibration dataset

        # Build engine
        engine = builder.build_engine(network, config)

        # Save
        with open(output_path, "wb") as f:
            f.write(engine.serialize())

        return output_path
```

---

## Real-Time Control Loop

### Deployment Architecture

```python
class RealHumanoidDeployment:
    """
    Real robot deployment with safety-critical control.
    """

    def __init__(
        self,
        model_path: str,
        robot_interface: HumanoidRobotInterface,
        control_frequency: float = 50.0,  # Hz
    ):
        # Load optimized model
        self.model = self._load_model(model_path)

        # Robot interface
        self.robot = robot_interface

        # Control parameters
        self.control_freq = control_frequency
        self.dt = 1.0 / control_frequency

        # Safety components
        self.safety_shield = HumanoidSafetyShield(
            joint_limits=robot_interface.get_joint_limits(),
        )

        # State tracking
        self.current_instruction = None
        self.is_running = False

    def _load_model(self, path: str):
        """Load optimized model for inference."""
        if path.endswith(".pt"):
            return torch.jit.load(path).eval().cuda()
        elif path.endswith(".onnx"):
            import onnxruntime as ort
            return ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
        elif path.endswith(".engine"):
            return self._load_tensorrt(path)

    def run(self, instruction: str):
        """Main control loop."""
        self.current_instruction = instruction
        self.is_running = True

        rate = Rate(self.control_freq)

        try:
            while self.is_running:
                # 1. Get sensor data
                sensor_data = self._get_sensor_data()

                # 2. Safety pre-check
                if not self._pre_safety_check(sensor_data):
                    self._trigger_safe_stop()
                    continue

                # 3. Run inference
                action = self._run_inference(sensor_data)

                # 4. Safety filter
                safe_action, is_safe = self.safety_shield.check_and_filter(
                    target_positions=action["joint_targets"],
                    current_positions=sensor_data["joint_pos"],
                    current_velocities=sensor_data["joint_vel"],
                    imu_data=sensor_data["imu"],
                    dt=self.dt,
                )

                if not is_safe:
                    self.logger.warning("Action filtered by safety shield")

                # 5. Execute action
                self.robot.set_joint_positions(safe_action)

                # 6. Handle gripper
                if action["gripper_actions"] is not None:
                    self.robot.set_gripper(action["gripper_actions"])

                rate.sleep()

        except Exception as e:
            self.logger.error(f"Control loop error: {e}")
            self._trigger_emergency_stop()

    def _get_sensor_data(self) -> Dict:
        """Collect all sensor data."""
        return {
            "images": self.robot.get_camera_images(),
            "joint_pos": self.robot.get_joint_positions(),
            "joint_vel": self.robot.get_joint_velocities(),
            "joint_torque": self.robot.get_joint_torques(),
            "imu": self.robot.get_imu_data(),
            "foot_contacts": self.robot.get_foot_contacts(),
        }

    def _pre_safety_check(self, sensor_data: Dict) -> bool:
        """Quick safety check before inference."""
        # Check balance
        imu = sensor_data["imu"]
        roll, pitch = imu[0], imu[1]
        if abs(roll) > 0.5 or abs(pitch) > 0.5:  # ~30 degrees
            self.logger.warning("Excessive tilt detected")
            return False

        # Check foot contacts
        if not any(sensor_data["foot_contacts"]):
            self.logger.warning("No foot contact detected")
            return False

        return True

    @torch.no_grad()
    def _run_inference(self, sensor_data: Dict) -> Dict:
        """Run model inference."""
        # Prepare inputs
        image = torch.tensor(sensor_data["images"]).unsqueeze(0).cuda()
        proprio = self._prepare_proprioception(sensor_data)
        language = self._encode_instruction(self.current_instruction)

        # Run model
        if isinstance(self.model, torch.jit.ScriptModule):
            output = self.model(image, proprio, language)
        else:
            # ONNX Runtime
            output = self.model.run(
                None,
                {
                    "image": image.cpu().numpy(),
                    "proprioception": proprio.cpu().numpy(),
                    "language_features": language.cpu().numpy(),
                }
            )

        return {
            "joint_targets": output[0].cpu().numpy()[0],
            "gripper_actions": output[1].cpu().numpy()[0] if len(output) > 1 else None,
        }

    def _trigger_safe_stop(self):
        """Controlled stop maintaining balance."""
        self.logger.info("Triggering safe stop")
        self.is_running = False

        # Move to safe standing pose
        safe_pose = self.robot.get_standing_pose()
        self.robot.move_to_pose(safe_pose, duration=2.0)

    def _trigger_emergency_stop(self):
        """Immediate emergency stop."""
        self.logger.error("Emergency stop!")
        self.is_running = False

        # Disable motors immediately
        self.robot.emergency_stop()
```

---

## Robot Interface

### Standard Interface

```python
from abc import ABC, abstractmethod

class HumanoidRobotInterface(ABC):
    """Abstract interface for humanoid robots."""

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        pass

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        pass

    @abstractmethod
    def get_joint_torques(self) -> np.ndarray:
        """Get current joint torques."""
        pass

    @abstractmethod
    def get_imu_data(self) -> np.ndarray:
        """Get IMU data (orientation, angular vel, acceleration)."""
        pass

    @abstractmethod
    def get_foot_contacts(self) -> np.ndarray:
        """Get foot contact states."""
        pass

    @abstractmethod
    def get_camera_images(self) -> np.ndarray:
        """Get camera images."""
        pass

    @abstractmethod
    def set_joint_positions(self, positions: np.ndarray):
        """Set target joint positions."""
        pass

    @abstractmethod
    def set_gripper(self, state: np.ndarray):
        """Set gripper state."""
        pass

    @abstractmethod
    def emergency_stop(self):
        """Trigger emergency stop."""
        pass

    @abstractmethod
    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """Get joint position limits."""
        pass

    @abstractmethod
    def get_standing_pose(self) -> np.ndarray:
        """Get safe standing pose."""
        pass
```

### ROS2 Interface Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, Imu
from std_msgs.msg import Float64MultiArray

class ROS2HumanoidInterface(HumanoidRobotInterface, Node):
    """ROS2-based humanoid robot interface."""

    def __init__(self):
        Node.__init__(self, "humanoid_vla_controller")

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_callback,
            10,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            "/imu/data",
            self._imu_callback,
            10,
        )
        self.image_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self._image_callback,
            10,
        )

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            "/joint_commands",
            10,
        )

        # State storage
        self._joint_positions = None
        self._joint_velocities = None
        self._imu_data = None
        self._image = None

    def get_joint_positions(self) -> np.ndarray:
        return self._joint_positions

    def get_joint_velocities(self) -> np.ndarray:
        return self._joint_velocities

    def get_imu_data(self) -> np.ndarray:
        return self._imu_data

    def get_camera_images(self) -> np.ndarray:
        return self._image

    def set_joint_positions(self, positions: np.ndarray):
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self.joint_cmd_pub.publish(msg)
```

---

## Performance Optimization

### Inference Latency Targets

| Component | Target | Typical |
|-----------|--------|---------|
| Image preprocessing | < 5ms | 2-3ms |
| VLM inference | < 15ms | 10-12ms |
| Action head | < 2ms | 1ms |
| Safety check | < 1ms | 0.5ms |
| **Total** | **< 20ms** | **~15ms** |

### Optimization Techniques

```python
class OptimizedInference:
    """Optimized inference pipeline."""

    def __init__(self, model_path: str):
        # Use CUDA graphs for repeated inference
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None

    def warmup(self, sample_input: Dict):
        """Warmup and capture CUDA graph."""
        # Warmup runs
        for _ in range(10):
            self._forward(sample_input)

        # Capture graph
        self.static_input = {k: v.clone() for k, v in sample_input.items()}
        self.cuda_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.cuda_graph):
            self.static_output = self._forward(self.static_input)

    def forward(self, inputs: Dict) -> Dict:
        """Fast inference using CUDA graph."""
        # Copy inputs to static buffers
        for k, v in inputs.items():
            self.static_input[k].copy_(v)

        # Replay graph
        self.cuda_graph.replay()

        return {k: v.clone() for k, v in self.static_output.items()}
```

---

## Monitoring and Logging

### Real-Time Monitoring

```python
class DeploymentMonitor:
    """Monitor deployment health and performance."""

    def __init__(self):
        self.metrics = {
            "inference_time_ms": [],
            "control_loop_time_ms": [],
            "safety_violations": 0,
            "balance_warnings": 0,
        }

    def log_inference_time(self, time_ms: float):
        self.metrics["inference_time_ms"].append(time_ms)

    def log_safety_violation(self):
        self.metrics["safety_violations"] += 1

    def log_balance_warning(self):
        self.metrics["balance_warnings"] += 1

    def get_summary(self) -> Dict:
        return {
            "avg_inference_ms": np.mean(self.metrics["inference_time_ms"]),
            "max_inference_ms": np.max(self.metrics["inference_time_ms"]),
            "safety_violations": self.metrics["safety_violations"],
            "balance_warnings": self.metrics["balance_warnings"],
        }
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Model exported and verified
- [ ] Safety shield tested in simulation
- [ ] Joint limits configured correctly
- [ ] Emergency stop tested
- [ ] Communication latency measured
- [ ] Inference latency under target

### First Deployment

- [ ] Robot in safe environment (e.g., suspended)
- [ ] Human supervisor present
- [ ] E-stop accessible
- [ ] Recording enabled for debugging
- [ ] Start with simple commands only

### Production Deployment

- [ ] Full safety validation complete
- [ ] Monitoring dashboard active
- [ ] Alerting configured
- [ ] Fallback procedures documented
- [ ] Recovery procedures tested

---

## Commands

### Export Model

```bash
# Export to TorchScript
python -c "
from model.utils.export import TorchScriptExporter
from model.embodiment import HumanoidVLA
import torch

model = HumanoidVLA.from_pretrained('./output/humanoid/final')
exporter = TorchScriptExporter()
exporter.export_traced(
    model=model,
    sample_input=torch.randn(1, 3, 224, 224),
    output_path='./deployed/humanoid_vla.pt'
)
"

# Export to ONNX
python -c "
from model.utils.export import ONNXExporter
from model.embodiment import HumanoidVLA

model = HumanoidVLA.from_pretrained('./output/humanoid/final')
exporter = ONNXExporter()
exporter.export(
    model=model,
    sample_input={'image': torch.randn(1, 3, 224, 224)},
    output_path='./deployed/humanoid_vla.onnx'
)
"
```

### Run Deployment

```bash
# Start deployment
python deploy/humanoid_deployment.py \
    --model-path ./deployed/humanoid_vla.pt \
    --robot-interface ros2 \
    --control-frequency 50 \
    --enable-safety-shield \
    --log-dir ./logs/deployment
```

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Training pipeline
- [Training Datasets](training_datasets.md) - Dataset information
- [Whole-Body Control](training_whole_body.md) - Control architecture
