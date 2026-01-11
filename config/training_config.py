"""
Training Configuration Classes

Defines configuration for different training schemes:
- Pretraining (VLM pretraining)
- Fine-tuning (supervised learning on robot data)
- Reinforcement Learning (PPO, SAC, GRPO)
- Imitation Learning (BC, DAgger, GAIL)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class OptimizerType(Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAFACTOR = "adafactor"


class SchedulerType(Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"
    COSINE_WARMUP = "cosine_with_warmup"
    POLYNOMIAL = "polynomial"


class RLAlgorithm(Enum):
    PPO = "ppo"
    SAC = "sac"
    GRPO = "grpo"  # Group Relative Policy Optimization
    REINFORCE = "reinforce"


class ILAlgorithm(Enum):
    BC = "bc"  # Behavioral Cloning
    DAGGER = "dagger"  # Dataset Aggregation
    GAIL = "gail"  # Generative Adversarial Imitation Learning
    IRL = "irl"  # Inverse Reinforcement Learning


@dataclass
class BaseTrainingConfig:
    """Base configuration shared across all training schemes."""
    # Output
    output_dir: str = "./output"
    experiment_name: str = "vla_experiment"

    # Hardware
    num_gpus: int = 1
    num_nodes: int = 1
    mixed_precision: str = "bf16"  # fp32, fp16, bf16

    # Optimization
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler: SchedulerType = SchedulerType.COSINE_WARMUP
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio

    # Training loop
    num_epochs: int = 10
    max_steps: int = -1  # If > 0, overrides num_epochs
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10

    # Checkpointing
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Logging
    use_wandb: bool = False
    wandb_project: str = "vla-training"
    wandb_entity: Optional[str] = None

    # Seed
    seed: int = 42


@dataclass
class PretrainingConfig(BaseTrainingConfig):
    """
    Configuration for VLM pretraining.

    Pretraining stages:
    1. Vision-Language Alignment: Train projector to align vision with LLM
    2. Visual Instruction Tuning: Fine-tune on multimodal instructions
    """
    # Stage configuration
    stage: str = "alignment"  # "alignment" or "instruction_tuning"

    # Data
    dataset_name: str = "liuhaotian/LLaVA-Pretrain"  # Stage 1
    instruction_dataset: str = "liuhaotian/LLaVA-Instruct-150K"  # Stage 2

    # Model freezing per stage
    freeze_vision_encoder: bool = True
    freeze_llm: bool = True  # Freeze in stage 1, unfreeze in stage 2

    # Stage 1 specific
    alignment_epochs: int = 1
    alignment_lr: float = 1e-3

    # Stage 2 specific
    instruction_epochs: int = 3
    instruction_lr: float = 2e-5

    # Sequence length
    max_seq_length: int = 2048

    @classmethod
    def stage1_alignment(cls) -> "PretrainingConfig":
        """Config for Stage 1: Vision-Language Alignment."""
        return cls(
            stage="alignment",
            freeze_vision_encoder=True,
            freeze_llm=True,
            num_epochs=1,
            learning_rate=1e-3,
            batch_size=32,
        )

    @classmethod
    def stage2_instruction(cls) -> "PretrainingConfig":
        """Config for Stage 2: Visual Instruction Tuning."""
        return cls(
            stage="instruction_tuning",
            freeze_vision_encoder=True,
            freeze_llm=False,
            num_epochs=3,
            learning_rate=2e-5,
            batch_size=16,
        )


@dataclass
class FineTuningConfig(BaseTrainingConfig):
    """
    Configuration for supervised fine-tuning on robot datasets.

    This is the standard approach after pretraining:
    - Load pretrained VLM
    - Add action head
    - Fine-tune on robot manipulation data
    """
    # Model
    pretrained_path: Optional[str] = None
    action_dim: int = 7
    freeze_vision: bool = True
    freeze_llm: bool = False

    # LoRA (optional)
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Data
    dataset_name: str = "lerobot/pusht"
    max_samples: Optional[int] = None
    num_workers: int = 4

    # Action prediction
    action_chunk_size: int = 1
    use_action_chunking: bool = False

    @classmethod
    def quick_test(cls) -> "FineTuningConfig":
        """Quick test configuration."""
        return cls(
            num_epochs=1,
            max_samples=100,
            batch_size=4,
            freeze_vision=True,
            freeze_llm=True,
        )

    @classmethod
    def standard(cls) -> "FineTuningConfig":
        """Standard fine-tuning configuration."""
        return cls(
            num_epochs=10,
            batch_size=8,
            learning_rate=1e-4,
            freeze_vision=True,
            freeze_llm=False,
        )

    @classmethod
    def full_finetune(cls) -> "FineTuningConfig":
        """Full fine-tuning (all parameters)."""
        return cls(
            num_epochs=20,
            batch_size=4,
            learning_rate=5e-5,
            freeze_vision=False,
            freeze_llm=False,
            use_lora=False,
        )

    @classmethod
    def lora_finetune(cls) -> "FineTuningConfig":
        """LoRA fine-tuning for memory efficiency."""
        return cls(
            num_epochs=10,
            batch_size=8,
            learning_rate=2e-4,
            freeze_vision=True,
            freeze_llm=True,  # LoRA adapters are trainable
            use_lora=True,
            lora_r=32,
        )

    @classmethod
    def memory_efficient(cls) -> "FineTuningConfig":
        """Memory-efficient configuration for limited GPU memory (8-16GB)."""
        return cls(
            num_epochs=10,
            batch_size=2,
            gradient_accumulation_steps=8,  # Effective batch size = 16
            learning_rate=1e-4,
            freeze_vision=True,
            freeze_llm=True,
            use_lora=True,
            lora_r=16,
            lora_alpha=16,
            mixed_precision="bf16",
        )


@dataclass
class RLConfig(BaseTrainingConfig):
    """
    Configuration for Reinforcement Learning training.

    Supported algorithms:
    - PPO: Proximal Policy Optimization
    - SAC: Soft Actor-Critic
    - GRPO: Group Relative Policy Optimization (for LLM-based)
    - REINFORCE: Basic policy gradient
    """
    # Algorithm
    algorithm: RLAlgorithm = RLAlgorithm.PPO

    # Environment
    env_name: str = "CartPole-v1"
    num_envs: int = 8  # Parallel environments
    max_episode_steps: int = 1000

    # PPO specific
    ppo_epochs: int = 4
    ppo_clip_range: float = 0.2
    ppo_clip_range_vf: Optional[float] = None
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_gae_lambda: float = 0.95

    # SAC specific
    sac_tau: float = 0.005  # Soft update coefficient
    sac_target_entropy: str = "auto"  # or float
    sac_init_temperature: float = 1.0
    sac_buffer_size: int = 1000000
    sac_learning_starts: int = 10000

    # GRPO specific (for VLA with LLM)
    grpo_group_size: int = 8  # Number of samples per prompt
    grpo_kl_coef: float = 0.1
    grpo_ref_model: Optional[str] = None  # Path to reference model

    # General RL
    discount_gamma: float = 0.99
    rollout_steps: int = 2048  # Steps before update
    total_timesteps: int = 1000000

    # Reward
    reward_normalization: bool = True
    reward_scaling: float = 1.0

    @classmethod
    def ppo_default(cls) -> "RLConfig":
        """Default PPO configuration."""
        return cls(
            algorithm=RLAlgorithm.PPO,
            ppo_epochs=4,
            ppo_clip_range=0.2,
            rollout_steps=2048,
            num_envs=8,
        )

    @classmethod
    def sac_default(cls) -> "RLConfig":
        """Default SAC configuration."""
        return cls(
            algorithm=RLAlgorithm.SAC,
            batch_size=256,
            sac_buffer_size=1000000,
            sac_learning_starts=10000,
        )

    @classmethod
    def grpo_vla(cls) -> "RLConfig":
        """GRPO for VLA fine-tuning."""
        return cls(
            algorithm=RLAlgorithm.GRPO,
            grpo_group_size=8,
            grpo_kl_coef=0.1,
            batch_size=4,
            learning_rate=5e-6,
        )


@dataclass
class ILConfig(BaseTrainingConfig):
    """
    Configuration for Imitation Learning.

    Supported algorithms:
    - BC: Behavioral Cloning (supervised learning)
    - DAgger: Dataset Aggregation
    - GAIL: Generative Adversarial Imitation Learning
    - IRL: Inverse Reinforcement Learning
    """
    # Algorithm
    algorithm: ILAlgorithm = ILAlgorithm.BC

    # Expert data
    expert_data_path: Optional[str] = None
    num_expert_episodes: int = 100
    expert_dataset: str = "lerobot/pusht"

    # BC specific
    bc_epochs: int = 100
    bc_validation_split: float = 0.2

    # DAgger specific
    dagger_iterations: int = 10
    dagger_episodes_per_iter: int = 20
    dagger_beta_schedule: str = "linear"  # linear, exponential, constant
    dagger_initial_beta: float = 1.0  # Expert weight

    # GAIL specific
    gail_disc_hidden_dim: int = 256
    gail_disc_lr: float = 1e-4
    gail_disc_updates: int = 5  # Discriminator updates per policy update
    gail_reward_scale: float = 1.0

    # IRL specific
    irl_reward_net_hidden: int = 256
    irl_reward_lr: float = 1e-4

    # Environment (for online algorithms)
    env_name: str = "CartPole-v1"
    max_episode_steps: int = 500

    @classmethod
    def behavioral_cloning(cls) -> "ILConfig":
        """Standard behavioral cloning."""
        return cls(
            algorithm=ILAlgorithm.BC,
            bc_epochs=100,
            batch_size=64,
            learning_rate=1e-3,
        )

    @classmethod
    def dagger(cls) -> "ILConfig":
        """DAgger configuration."""
        return cls(
            algorithm=ILAlgorithm.DAGGER,
            dagger_iterations=10,
            dagger_episodes_per_iter=20,
            bc_epochs=10,  # BC epochs per DAgger iteration
        )

    @classmethod
    def gail(cls) -> "ILConfig":
        """GAIL configuration."""
        return cls(
            algorithm=ILAlgorithm.GAIL,
            gail_disc_updates=5,
            batch_size=256,
            learning_rate=3e-4,
        )


# Training configuration presets
TRAINING_CONFIGS = {
    # Pretraining
    "pretrain-stage1": PretrainingConfig.stage1_alignment,
    "pretrain-stage2": PretrainingConfig.stage2_instruction,

    # Fine-tuning
    "finetune-quick": FineTuningConfig.quick_test,
    "finetune-standard": FineTuningConfig.standard,
    "finetune-full": FineTuningConfig.full_finetune,
    "finetune-lora": FineTuningConfig.lora_finetune,
    "finetune-memory-efficient": FineTuningConfig.memory_efficient,

    # RL
    "rl-ppo": RLConfig.ppo_default,
    "rl-sac": RLConfig.sac_default,
    "rl-grpo": RLConfig.grpo_vla,

    # IL
    "il-bc": ILConfig.behavioral_cloning,
    "il-dagger": ILConfig.dagger,
    "il-gail": ILConfig.gail,
}


def get_training_config(name: str):
    """Get a training configuration by name."""
    if name not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(TRAINING_CONFIGS.keys())}")
    return TRAINING_CONFIGS[name]()
