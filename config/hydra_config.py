"""
Hydra/OmegaConf Configuration Integration

Provides CLI-based configuration management with:
- YAML config files
- Command-line overrides
- Experiment sweeps
- Config composition
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

try:
    from omegaconf import OmegaConf, DictConfig, MISSING
    import hydra
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    print("Hydra/OmegaConf not installed. Install with: pip install hydra-core omegaconf")


# ============================================================================
# Base Configuration Dataclasses
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration."""
    # Vision encoder
    vision_encoder: str = "siglip-base"
    vision_dim: int = 768
    freeze_vision: bool = True

    # Language model
    llm: str = "qwen2-1.5b"
    llm_dim: int = 1536
    freeze_llm: bool = True

    # Action head
    action_dim: int = 7
    action_head_type: str = "mlp"  # mlp, gaussian, diffusion, transformer
    action_chunk_size: int = 1
    hidden_dim: int = 256

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "lerobot/pusht"
    split: str = "train"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    use_augmentation: bool = True
    color_jitter: bool = True
    random_crop: bool = False

    # Action
    action_horizon: int = 1
    history_length: int = 1


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Schedule
    warmup_steps: int = 1000
    max_epochs: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Precision
    mixed_precision: str = "bf16"  # no, fp16, bf16

    # Checkpointing
    save_every_n_epochs: int = 10
    eval_every_n_epochs: int = 5
    log_every_n_steps: int = 10

    # Directories
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"


@dataclass
class RLConfig:
    """Reinforcement learning configuration."""
    algorithm: str = "ppo"  # ppo, sac, grpo

    # Common
    total_timesteps: int = 1000000
    buffer_size: int = 100000
    gamma: float = 0.99

    # PPO specific
    ppo_clip_range: float = 0.2
    ppo_epochs: int = 10
    n_steps: int = 2048
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # SAC specific
    sac_tau: float = 0.005
    sac_alpha: float = 0.2
    auto_entropy_tuning: bool = True


@dataclass
class OfflineRLConfig:
    """Offline RL configuration."""
    algorithm: str = "iql"  # cql, iql, td3bc, dt

    # Common
    num_epochs: int = 1000
    batch_size: int = 256
    learning_rate: float = 3e-4

    # IQL specific
    expectile: float = 0.7
    temperature: float = 3.0

    # CQL specific
    cql_alpha: float = 5.0
    cql_temp: float = 1.0

    # TD3+BC specific
    td3bc_alpha: float = 2.5


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Experiment info
    name: str = "vla_experiment"
    seed: int = 42
    device: str = "auto"

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Optional RL configs
    rl: Optional[RLConfig] = None
    offline_rl: Optional[OfflineRLConfig] = None

    # Logging
    use_wandb: bool = False
    wandb_project: str = "vla-training"
    wandb_entity: Optional[str] = None


# ============================================================================
# Config Registration (for Hydra)
# ============================================================================

def register_configs():
    """Register configs with Hydra ConfigStore."""
    if not HYDRA_AVAILABLE:
        return

    cs = ConfigStore.instance()

    # Register structured configs
    cs.store(name="base_model", node=ModelConfig)
    cs.store(name="base_data", node=DataConfig)
    cs.store(name="base_training", node=TrainingConfig)
    cs.store(name="base_rl", node=RLConfig)
    cs.store(name="base_offline_rl", node=OfflineRLConfig)
    cs.store(name="base_experiment", node=ExperimentConfig)


# ============================================================================
# Config Loading Utilities
# ============================================================================

def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    if not HYDRA_AVAILABLE:
        raise ImportError("OmegaConf required. Install with: pip install omegaconf")

    return OmegaConf.load(config_path)


def save_config(config: DictConfig, path: str):
    """Save configuration to YAML file."""
    if not HYDRA_AVAILABLE:
        raise ImportError("OmegaConf required. Install with: pip install omegaconf")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    OmegaConf.save(config, path)


def merge_configs(*configs) -> DictConfig:
    """Merge multiple configurations."""
    if not HYDRA_AVAILABLE:
        raise ImportError("OmegaConf required. Install with: pip install omegaconf")

    return OmegaConf.merge(*configs)


def to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf to regular dictionary."""
    if not HYDRA_AVAILABLE:
        return config if isinstance(config, dict) else {}

    return OmegaConf.to_container(config, resolve=True)


# ============================================================================
# Preset Configurations
# ============================================================================

PRESETS = {
    "pusht-bc": {
        "model": {
            "vision_encoder": "siglip-base",
            "llm": "qwen2-0.5b",
            "action_dim": 2,
            "action_head_type": "mlp",
        },
        "data": {
            "dataset_name": "lerobot/pusht",
            "batch_size": 64,
        },
        "training": {
            "learning_rate": 1e-4,
            "max_epochs": 100,
        },
    },
    "aloha-diffusion": {
        "model": {
            "vision_encoder": "siglip-large",
            "llm": "qwen2-1.5b",
            "action_dim": 14,
            "action_head_type": "diffusion",
            "action_chunk_size": 16,
        },
        "data": {
            "dataset_name": "lerobot/aloha_transfer_cube_human",
            "batch_size": 8,
        },
        "training": {
            "learning_rate": 1e-4,
            "max_epochs": 200,
            "gradient_accumulation_steps": 8,
        },
    },
    "driving-vla": {
        "model": {
            "vision_encoder": "siglip-base",
            "llm": "qwen2-1.5b",
            "action_dim": 3,
            "action_head_type": "mlp",
        },
        "data": {
            "dataset_name": "carla",
            "batch_size": 4,
            "image_size": 256,
        },
        "training": {
            "learning_rate": 5e-5,
            "max_epochs": 100,
            "gradient_accumulation_steps": 8,
        },
    },
    "d4rl-iql": {
        "model": {
            "action_dim": 3,
            "action_head_type": "gaussian",
            "hidden_dim": 256,
        },
        "offline_rl": {
            "algorithm": "iql",
            "expectile": 0.7,
            "temperature": 3.0,
            "num_epochs": 1000,
        },
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """Get a preset configuration."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESETS[name]


def list_presets() -> List[str]:
    """List available presets."""
    return list(PRESETS.keys())


# ============================================================================
# CLI Integration
# ============================================================================

def create_config_from_args(args) -> ExperimentConfig:
    """Create config from argparse arguments."""
    config = ExperimentConfig(
        name=getattr(args, "name", "experiment"),
        seed=getattr(args, "seed", 42),
        device=getattr(args, "device", "auto"),
    )

    # Model
    if hasattr(args, "vision_encoder"):
        config.model.vision_encoder = args.vision_encoder
    if hasattr(args, "llm"):
        config.model.llm = args.llm
    if hasattr(args, "action_dim"):
        config.model.action_dim = args.action_dim

    # Training
    if hasattr(args, "lr"):
        config.training.learning_rate = args.lr
    if hasattr(args, "batch_size"):
        config.data.batch_size = args.batch_size
    if hasattr(args, "epochs"):
        config.training.max_epochs = args.epochs

    return config


# ============================================================================
# Hydra App Decorator
# ============================================================================

def hydra_main(config_path: str = "conf", config_name: str = "config"):
    """Decorator for Hydra-enabled main functions."""
    if not HYDRA_AVAILABLE:
        def decorator(func):
            def wrapper():
                print("Hydra not available. Using default config.")
                config = ExperimentConfig()
                return func(config)
            return wrapper
        return decorator

    return hydra.main(version_base=None, config_path=config_path, config_name=config_name)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Hydra/OmegaConf Configuration")
    print("=" * 50)

    if HYDRA_AVAILABLE:
        # Create default config
        config = OmegaConf.structured(ExperimentConfig)
        print("\nDefault config:")
        print(OmegaConf.to_yaml(config))

        # Load preset
        print("\nAvailable presets:", list_presets())

        preset = get_preset("pusht-bc")
        print("\nPushT BC preset:")
        print(OmegaConf.to_yaml(OmegaConf.create(preset)))

    else:
        print("Install hydra-core and omegaconf for full functionality:")
        print("  pip install hydra-core omegaconf")
