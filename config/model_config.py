"""
Model Configuration Classes

Defines configuration dataclasses for all VLA model variants.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def _validate_positive(value: int, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ConfigValidationError(f"{name} must be positive, got {value}")


def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ConfigValidationError(f"{name} must be non-negative, got {value}")


def _validate_probability(value: float, name: str) -> None:
    """Validate that a value is a valid probability [0, 1]."""
    if not 0 <= value <= 1:
        raise ConfigValidationError(f"{name} must be in [0, 1], got {value}")


@dataclass
class VLAConfig:
    """
    Configuration for the base VLA model.

    Attributes:
        vision_model_name: HuggingFace vision encoder name
        llm_model_name: HuggingFace LLM name
        action_dim: Dimension of action space
        hidden_dim: Hidden dimension for action head
        num_vision_tokens: Number of vision tokens after projection
        action_chunk_size: Number of future actions to predict
        dropout: Dropout rate
        freeze_vision: Whether to freeze vision encoder
        freeze_llm: Whether to freeze LLM
        use_flash_attention: Use Flash Attention 2
    """
    # Vision encoder
    vision_model_name: str = "google/siglip-base-patch16-224"

    # Language model
    llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct"

    # Action space
    action_dim: int = 7  # [x, y, z, roll, pitch, yaw, gripper]
    action_chunk_size: int = 1

    # Architecture
    hidden_dim: int = 512
    num_vision_tokens: int = 64
    dropout: float = 0.1

    # Freezing
    freeze_vision: bool = False
    freeze_llm: bool = False

    # Optimizations
    use_flash_attention: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        _validate_positive(self.action_dim, "action_dim")
        _validate_positive(self.action_chunk_size, "action_chunk_size")
        _validate_positive(self.hidden_dim, "hidden_dim")
        _validate_positive(self.num_vision_tokens, "num_vision_tokens")
        _validate_probability(self.dropout, "dropout")

        if not self.vision_model_name:
            raise ConfigValidationError("vision_model_name cannot be empty")
        if not self.llm_model_name:
            raise ConfigValidationError("llm_model_name cannot be empty")

    # Preset configurations
    @classmethod
    def small(cls) -> "VLAConfig":
        """Small config for quick experiments."""
        return cls(
            vision_model_name="google/siglip-base-patch16-224",
            llm_model_name="Qwen/Qwen2-1.5B-Instruct",
            hidden_dim=256,
            num_vision_tokens=32,
            freeze_vision=True,
            freeze_llm=True,
        )

    @classmethod
    def medium(cls) -> "VLAConfig":
        """Medium config for standard training."""
        return cls(
            vision_model_name="google/siglip-base-patch16-224",
            llm_model_name="Qwen/Qwen2-1.5B-Instruct",
            hidden_dim=512,
            num_vision_tokens=64,
            freeze_vision=True,
            freeze_llm=False,
        )

    @classmethod
    def large(cls) -> "VLAConfig":
        """Large config for best performance."""
        return cls(
            vision_model_name="google/siglip-large-patch16-384",
            llm_model_name="Qwen/Qwen2-7B-Instruct",
            hidden_dim=1024,
            num_vision_tokens=128,
            freeze_vision=False,
            freeze_llm=False,
            use_flash_attention=True,
        )


@dataclass
class MultiSensorVLAConfig:
    """
    Configuration for Multi-Sensor VLA (autonomous vehicles/robots).

    Supports:
        - Camera (RGB)
        - Depth Camera (depth images)
        - LiDAR (point clouds)
        - Radar (range-doppler)
        - IMU (accelerometer/gyroscope)
    """
    # Base model
    vision_model_name: str = "google/siglip-base-patch16-224"
    llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct"

    # Action space
    action_dim: int = 7
    hidden_dim: int = 512
    action_chunk_size: int = 1

    # Sensor configuration
    use_depth: bool = False
    use_lidar: bool = True
    use_radar: bool = True
    use_imu: bool = True

    # Depth camera config
    depth_input_channels: int = 1
    depth_image_size: int = 224
    depth_output_dim: int = 256
    depth_num_tokens: int = 16

    # LiDAR config
    lidar_input_dim: int = 4  # x, y, z, intensity
    lidar_num_points: int = 4096
    lidar_output_dim: int = 512

    # Radar config
    radar_input_channels: int = 2  # magnitude, velocity
    radar_output_dim: int = 256

    # IMU config
    imu_input_dim: int = 6  # 3 accel + 3 gyro
    imu_seq_len: int = 100
    imu_output_dim: int = 256

    # Freezing
    freeze_vision: bool = False
    freeze_llm: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        _validate_positive(self.action_dim, "action_dim")
        _validate_positive(self.hidden_dim, "hidden_dim")
        _validate_positive(self.action_chunk_size, "action_chunk_size")

        if self.use_depth:
            _validate_positive(self.depth_input_channels, "depth_input_channels")
            _validate_positive(self.depth_image_size, "depth_image_size")
            _validate_positive(self.depth_output_dim, "depth_output_dim")
            _validate_positive(self.depth_num_tokens, "depth_num_tokens")

        if self.use_lidar:
            _validate_positive(self.lidar_input_dim, "lidar_input_dim")
            _validate_positive(self.lidar_num_points, "lidar_num_points")
            _validate_positive(self.lidar_output_dim, "lidar_output_dim")

        if self.use_radar:
            _validate_positive(self.radar_input_channels, "radar_input_channels")
            _validate_positive(self.radar_output_dim, "radar_output_dim")

        if self.use_imu:
            _validate_positive(self.imu_input_dim, "imu_input_dim")
            _validate_positive(self.imu_seq_len, "imu_seq_len")
            _validate_positive(self.imu_output_dim, "imu_output_dim")

    @classmethod
    def autonomous_driving(cls) -> "MultiSensorVLAConfig":
        """Config for autonomous driving."""
        return cls(
            action_dim=3,  # steering, throttle, brake
            use_lidar=True,
            use_radar=True,
            use_imu=True,
            lidar_num_points=16384,
            action_chunk_size=10,  # Predict 10 future waypoints
        )

    @classmethod
    def mobile_robot(cls) -> "MultiSensorVLAConfig":
        """Config for mobile robot navigation."""
        return cls(
            action_dim=2,  # linear velocity, angular velocity
            use_lidar=True,
            use_radar=False,
            use_imu=True,
            lidar_num_points=2048,
        )

    @classmethod
    def drone(cls) -> "MultiSensorVLAConfig":
        """Config for drone control."""
        return cls(
            action_dim=4,  # roll, pitch, yaw, throttle
            use_depth=False,
            use_lidar=False,
            use_radar=False,
            use_imu=True,
            imu_seq_len=50,
        )

    @classmethod
    def rgbd_manipulation(cls) -> "MultiSensorVLAConfig":
        """Config for RGB-D robot manipulation."""
        return cls(
            action_dim=7,  # [x, y, z, roll, pitch, yaw, gripper]
            use_depth=True,
            use_lidar=False,
            use_radar=False,
            use_imu=False,
            depth_num_tokens=16,
        )

    @classmethod
    def full_sensor(cls) -> "MultiSensorVLAConfig":
        """Config with all sensors enabled."""
        return cls(
            action_dim=7,
            use_depth=True,
            use_lidar=True,
            use_radar=True,
            use_imu=True,
        )


@dataclass
class OpenVLAConfig:
    """
    Configuration for OpenVLA-7B fine-tuning.

    Uses LoRA and quantization for memory-efficient training.
    """
    model_name: str = "openvla/openvla-7b"
    action_dim: int = 7

    # LoRA config
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Device mapping
    device_map: str = "auto"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        _validate_positive(self.action_dim, "action_dim")

        if self.use_lora:
            _validate_positive(self.lora_r, "lora_r")
            _validate_positive(self.lora_alpha, "lora_alpha")
            _validate_probability(self.lora_dropout, "lora_dropout")
            if not self.lora_target_modules:
                raise ConfigValidationError("lora_target_modules cannot be empty when use_lora=True")

        if self.load_in_8bit and self.load_in_4bit:
            raise ConfigValidationError("Cannot use both 8-bit and 4-bit quantization")

        valid_quant_types = ["nf4", "fp4"]
        if self.bnb_4bit_quant_type not in valid_quant_types:
            raise ConfigValidationError(
                f"bnb_4bit_quant_type must be one of {valid_quant_types}, got {self.bnb_4bit_quant_type}"
            )

    @classmethod
    def memory_efficient(cls) -> "OpenVLAConfig":
        """Most memory-efficient config (4-bit + LoRA)."""
        return cls(
            use_lora=True,
            load_in_4bit=True,
            lora_r=16,
        )

    @classmethod
    def balanced(cls) -> "OpenVLAConfig":
        """Balanced quality/memory config."""
        return cls(
            use_lora=True,
            load_in_4bit=True,
            lora_r=32,
        )

    @classmethod
    def high_quality(cls) -> "OpenVLAConfig":
        """Higher quality with 8-bit."""
        return cls(
            use_lora=True,
            load_in_8bit=True,
            load_in_4bit=False,
            lora_r=64,
        )


@dataclass
class SmolVLAConfig:
    """
    Configuration for SmolVLA-450M.

    Lightweight model for consumer hardware.
    """
    model_name: str = "HuggingFaceTB/SmolVLA-450M"
    action_dim: int = 7

    # Input configuration
    image_size: List[int] = field(default_factory=lambda: [3, 224, 224])
    state_dim: int = 0  # Additional state input dimension

    # Training
    use_lerobot: bool = True  # Use LeRobot dataloader

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        _validate_positive(self.action_dim, "action_dim")
        _validate_non_negative(self.state_dim, "state_dim")

        if len(self.image_size) != 3:
            raise ConfigValidationError(
                f"image_size must have 3 elements [C, H, W], got {len(self.image_size)}"
            )
        for i, dim in enumerate(self.image_size):
            if dim <= 0:
                raise ConfigValidationError(f"image_size[{i}] must be positive, got {dim}")

    @classmethod
    def default(cls) -> "SmolVLAConfig":
        """Default SmolVLA config."""
        return cls()


# Mapping of model types to configs
MODEL_CONFIGS = {
    "vla-small": VLAConfig.small,
    "vla-medium": VLAConfig.medium,
    "vla-large": VLAConfig.large,
    "multi-sensor-driving": MultiSensorVLAConfig.autonomous_driving,
    "multi-sensor-robot": MultiSensorVLAConfig.mobile_robot,
    "multi-sensor-drone": MultiSensorVLAConfig.drone,
    "multi-sensor-rgbd": MultiSensorVLAConfig.rgbd_manipulation,
    "multi-sensor-full": MultiSensorVLAConfig.full_sensor,
    "openvla-efficient": OpenVLAConfig.memory_efficient,
    "openvla-balanced": OpenVLAConfig.balanced,
    "openvla-quality": OpenVLAConfig.high_quality,
    "smolvla": SmolVLAConfig.default,
}


def get_model_config(name: str):
    """Get a model configuration by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]()
