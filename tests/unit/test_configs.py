"""
Unit Tests for Configuration Classes

Tests configuration validation and preset creation.
"""

import pytest
from dataclasses import asdict

from config.model_config import (
    VLAConfig,
    MultiSensorVLAConfig,
    OpenVLAConfig,
    SmolVLAConfig,
    ConfigValidationError,
    get_model_config,
    MODEL_CONFIGS,
)


class TestVLAConfig:
    """Tests for VLAConfig."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = VLAConfig()

        assert config.action_dim == 7
        assert config.hidden_dim == 512
        assert config.dropout == 0.1
        assert config.freeze_vision is False

    def test_preset_small(self):
        """Test small preset configuration."""
        config = VLAConfig.small()

        assert config.hidden_dim == 256
        assert config.num_vision_tokens == 32
        assert config.freeze_vision is True
        assert config.freeze_llm is True

    def test_preset_medium(self):
        """Test medium preset configuration."""
        config = VLAConfig.medium()

        assert config.hidden_dim == 512
        assert config.freeze_llm is False

    def test_preset_large(self):
        """Test large preset configuration."""
        config = VLAConfig.large()

        assert config.hidden_dim == 1024
        assert config.use_flash_attention is True

    def test_validation_action_dim(self):
        """Test that invalid action_dim raises error."""
        with pytest.raises(ConfigValidationError):
            VLAConfig(action_dim=0)

        with pytest.raises(ConfigValidationError):
            VLAConfig(action_dim=-1)

    def test_validation_hidden_dim(self):
        """Test that invalid hidden_dim raises error."""
        with pytest.raises(ConfigValidationError):
            VLAConfig(hidden_dim=0)

    def test_validation_dropout(self):
        """Test that invalid dropout raises error."""
        with pytest.raises(ConfigValidationError):
            VLAConfig(dropout=-0.1)

        with pytest.raises(ConfigValidationError):
            VLAConfig(dropout=1.5)

    def test_validation_empty_model_name(self):
        """Test that empty model names raise error."""
        with pytest.raises(ConfigValidationError):
            VLAConfig(vision_model_name="")

        with pytest.raises(ConfigValidationError):
            VLAConfig(llm_model_name="")

    def test_config_to_dict(self):
        """Test conversion to dictionary."""
        config = VLAConfig.small()
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert "action_dim" in config_dict
        assert "vision_model_name" in config_dict


class TestMultiSensorVLAConfig:
    """Tests for MultiSensorVLAConfig."""

    def test_autonomous_driving_preset(self):
        """Test autonomous driving preset."""
        config = MultiSensorVLAConfig.autonomous_driving()

        assert config.action_dim == 3
        assert config.use_lidar is True
        assert config.use_radar is True
        assert config.action_chunk_size == 10

    def test_mobile_robot_preset(self):
        """Test mobile robot preset."""
        config = MultiSensorVLAConfig.mobile_robot()

        assert config.action_dim == 2
        assert config.use_lidar is True
        assert config.use_radar is False

    def test_drone_preset(self):
        """Test drone preset."""
        config = MultiSensorVLAConfig.drone()

        assert config.action_dim == 4
        assert config.use_imu is True
        assert config.use_lidar is False

    def test_rgbd_preset(self):
        """Test RGB-D manipulation preset."""
        config = MultiSensorVLAConfig.rgbd_manipulation()

        assert config.use_depth is True
        assert config.use_lidar is False

    def test_validation_sensor_configs(self):
        """Test sensor configuration validation."""
        # Valid config
        config = MultiSensorVLAConfig(use_lidar=True, lidar_num_points=1024)
        assert config.lidar_num_points == 1024

        # Invalid lidar config
        with pytest.raises(ConfigValidationError):
            MultiSensorVLAConfig(use_lidar=True, lidar_num_points=0)


class TestOpenVLAConfig:
    """Tests for OpenVLAConfig."""

    def test_memory_efficient_preset(self):
        """Test memory efficient preset."""
        config = OpenVLAConfig.memory_efficient()

        assert config.use_lora is True
        assert config.load_in_4bit is True
        assert config.lora_r == 16

    def test_balanced_preset(self):
        """Test balanced preset."""
        config = OpenVLAConfig.balanced()

        assert config.lora_r == 32

    def test_high_quality_preset(self):
        """Test high quality preset."""
        config = OpenVLAConfig.high_quality()

        assert config.load_in_8bit is True
        assert config.load_in_4bit is False
        assert config.lora_r == 64

    def test_validation_conflicting_quantization(self):
        """Test that conflicting quantization raises error."""
        with pytest.raises(ConfigValidationError):
            OpenVLAConfig(load_in_8bit=True, load_in_4bit=True)

    def test_validation_lora_dropout(self):
        """Test LoRA dropout validation."""
        with pytest.raises(ConfigValidationError):
            OpenVLAConfig(use_lora=True, lora_dropout=1.5)

    def test_validation_quant_type(self):
        """Test quantization type validation."""
        # Valid types
        OpenVLAConfig(bnb_4bit_quant_type="nf4")
        OpenVLAConfig(bnb_4bit_quant_type="fp4")

        # Invalid type
        with pytest.raises(ConfigValidationError):
            OpenVLAConfig(bnb_4bit_quant_type="invalid")


class TestSmolVLAConfig:
    """Tests for SmolVLAConfig."""

    def test_default_preset(self):
        """Test default preset."""
        config = SmolVLAConfig.default()

        assert config.action_dim == 7
        assert config.image_size == [3, 224, 224]

    def test_validation_image_size(self):
        """Test image size validation."""
        # Valid
        SmolVLAConfig(image_size=[3, 256, 256])

        # Invalid length
        with pytest.raises(ConfigValidationError):
            SmolVLAConfig(image_size=[224, 224])

        # Invalid dimensions
        with pytest.raises(ConfigValidationError):
            SmolVLAConfig(image_size=[3, 0, 224])


class TestGetModelConfig:
    """Tests for get_model_config function."""

    def test_all_presets_available(self):
        """Test that all presets can be retrieved."""
        for name in MODEL_CONFIGS.keys():
            config = get_model_config(name)
            assert config is not None

    def test_unknown_config_raises_error(self):
        """Test that unknown config name raises error."""
        with pytest.raises(ValueError):
            get_model_config("unknown-config")

    def test_config_names(self):
        """Test expected config names exist."""
        expected_names = [
            "vla-small",
            "vla-medium",
            "vla-large",
            "multi-sensor-driving",
            "openvla-balanced",
            "smolvla",
        ]

        for name in expected_names:
            assert name in MODEL_CONFIGS
