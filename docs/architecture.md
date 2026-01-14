# Architecture Guide

This document describes the architecture of the VLA Training Framework.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Vision Encoder](#vision-encoder)
4. [Language Model](#language-model)
5. [Vision Projector](#vision-projector)
6. [Action Heads](#action-heads)
7. [Sensor Encoders](#sensor-encoders)
8. [Fusion Modules](#fusion-modules)
9. [Temporal Modeling](#temporal-modeling)
10. [World Model](#world-model)

---

## Overview

The VLA (Vision-Language-Action) model architecture combines three modalities:

```
┌─────────────────────────────────────────────────────────────────┐
│                        VLA Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Image ──────┐                                                  │
│               │                                                  │
│               ▼                                                  │
│   ┌─────────────────┐    ┌─────────────────┐    ┌────────────┐  │
│   │ Vision Encoder  │───▶│ Vision Projector│───▶│            │  │
│   │ (SigLIP/CLIP)   │    │ (MLP/Perceiver) │    │            │  │
│   └─────────────────┘    └─────────────────┘    │            │  │
│                                                  │   LLM      │  │
│   Instruction ──────────────────────────────────▶│ (Qwen2)   │  │
│                                                  │            │  │
│   (Optional)                                     │            │  │
│   ┌─────────────────┐                           │            │  │
│   │ Sensor Encoders │───▶ Fusion ──────────────▶│            │  │
│   │ (LiDAR/Radar)   │                           │            │  │
│   └─────────────────┘                           └─────┬──────┘  │
│                                                       │         │
│                                                       ▼         │
│                                               ┌─────────────┐   │
│                                               │ Action Head │   │
│                                               │ (MLP/Diff)  │   │
│                                               └──────┬──────┘   │
│                                                      │          │
│                                                      ▼          │
│                                               Action Output     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### Component Hierarchy

```
model/
├── vlm/                    # Vision-Language Model
│   ├── vision_encoder.py   # Image feature extraction
│   └── vision_projector.py # Map to LLM space
├── vla/                    # Complete VLA models
│   ├── vla_model.py        # Core VLA
│   ├── multi_sensor_vla.py # Multi-modal VLA
│   ├── openvla_wrapper.py  # OpenVLA integration
│   └── smolvla_wrapper.py  # Lightweight VLA
├── action_head/            # Action prediction
│   ├── mlp_action_head.py
│   ├── diffusion_action_head.py
│   └── transformer_action_head.py
├── sensor/                 # Additional sensors
│   ├── lidar_encoder.py
│   ├── radar_encoder.py
│   └── imu_encoder.py
├── fusion/                 # Multi-modal fusion
│   └── sensor_fusion.py
├── temporal/               # Temporal modeling
│   ├── temporal_encoder.py
│   └── memory_buffer.py
└── world_model/            # World modeling
    ├── dynamics_model.py
    └── reward_predictor.py
```

---

## Vision Encoder

Extracts visual features from input images.

### Supported Encoders

| Encoder | Parameters | Output Dim | Best For |
|---------|------------|------------|----------|
| SigLIP-Base | 86M | 768 | General robotics |
| SigLIP-Large | 304M | 1024 | High accuracy |
| CLIP-ViT-B | 86M | 512 | Language grounding |
| CLIP-ViT-L | 304M | 768 | Language grounding |
| DINOv2-Base | 86M | 768 | Dense features |
| DINOv2-Large | 300M | 1024 | Dense features |

### Architecture

```python
class VisionEncoder(nn.Module):
    """
    Vision encoder using pretrained models.

    Input: (B, C, H, W) - Batch of images
    Output: (B, N, D) - Patch tokens
           N = (H/patch_size) * (W/patch_size)
           D = hidden dimension
    """

    def __init__(
        self,
        model_name: str = "siglip-base",
        output_dim: int = 768,
        freeze: bool = True,
    ):
        ...

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Returns patch tokens, not CLS token
        features = self.encoder(images)  # (B, N, D)
        return features
```

### Usage

```python
from model.vlm import VisionEncoder, VisionEncoderConfig

config = VisionEncoderConfig(
    model_name="siglip-base",
    image_size=224,
    freeze=True,
)

encoder = VisionEncoder(config)
features = encoder(images)  # (B, 196, 768) for 224x224 with patch=16
```

---

## Language Model

Backbone for understanding instructions and generating action-relevant features.

### Supported LLMs

| Model | Parameters | Context | Best For |
|-------|------------|---------|----------|
| Qwen2-0.5B | 500M | 32K | Edge devices |
| Qwen2-1.5B | 1.5B | 32K | Balanced |
| Qwen2-7B | 7B | 128K | High capability |
| LLaMA3-8B | 8B | 8K | General purpose |
| Phi-3-mini | 3.8B | 4K | Efficient |

### Integration Pattern

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMBackbone(nn.Module):
    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_features: torch.Tensor,  # Injected features
    ) -> torch.Tensor:
        # Merge vision features with text embeddings
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings = self.inject_vision(embeddings, vision_features)

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        return outputs.hidden_states[-1]  # Last layer features
```

---

## Vision Projector

Maps vision encoder features to LLM embedding space.

### Projector Types

#### 1. MLP Projector (Simple, Fast)

```python
class MLPProjector(nn.Module):
    """
    Simple 2-layer MLP for feature projection.

    Input: (B, N, vision_dim)
    Output: (B, N, llm_dim)
    """
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
```

#### 2. Attention Pooling Projector (Reduces Tokens)

```python
class AttentionPoolingProjector(nn.Module):
    """
    Reduces number of vision tokens using learned queries.

    Input: (B, N, vision_dim) - e.g., 196 tokens
    Output: (B, num_queries, llm_dim) - e.g., 64 tokens
    """
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_queries: int = 64,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim))
        self.cross_attn = nn.MultiheadAttention(llm_dim, num_heads=8)
        self.proj = nn.Linear(vision_dim, llm_dim)
```

#### 3. Perceiver Projector (Most Flexible)

```python
class PerceiverProjector(nn.Module):
    """
    Perceiver-style projector with iterative cross-attention.

    Benefits:
    - Handles variable input sizes
    - Learnable compression ratio
    - More expressive than simple pooling
    """
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_latents: int = 64,
        num_layers: int = 2,
    ):
        ...
```

### Comparison

| Projector | Speed | Token Reduction | Expressiveness |
|-----------|-------|-----------------|----------------|
| MLP | Fast | None | Low |
| Attention Pooling | Medium | Yes (e.g., 196→64) | Medium |
| Perceiver | Slow | Yes (configurable) | High |

---

## Action Heads

Convert LLM features to robot actions.

### 1. MLP Action Head

```python
class MLPActionHead(nn.Module):
    """
    Simple deterministic action prediction.

    Best for: Single-mode action distributions, fast inference

    Input: (B, D) - LLM features
    Output: (B, action_dim) or (B, chunk_size, action_dim)
    """
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * chunk_size),
        )
```

### 2. Gaussian MLP Action Head

```python
class GaussianMLPActionHead(nn.Module):
    """
    Stochastic action prediction with learned variance.

    Best for: RL training (policy gradient), uncertainty estimation

    Output: mean (B, action_dim), log_std (B, action_dim)
    """
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, features):
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(-20, 2)
        return mean, log_std

    def sample(self, features):
        mean, log_std = self.forward(features)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.rsample()
```

### 3. Diffusion Action Head

```python
class DiffusionActionHead(nn.Module):
    """
    Denoising diffusion for action generation.

    Best for: Multi-modal action distributions, precise manipulation

    Architecture:
    - Uses DDPM/DDIM for generation
    - Conditional on LLM features
    - Supports action horizon (sequence generation)
    """
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        horizon: int = 16,
        diffusion_steps: int = 100,
    ):
        ...

    @torch.no_grad()
    def predict(self, features, num_steps: int = 10):
        """DDIM sampling for fast inference."""
        ...
```

### 4. Transformer Action Head

```python
class TransformerActionHead(nn.Module):
    """
    Autoregressive action sequence generation.

    Best for: Long action sequences, temporal dependencies

    Uses causal masking for autoregressive generation.
    """
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        max_seq_len: int = 100,
        num_layers: int = 4,
    ):
        ...
```

### Comparison Table

| Head Type | Multi-modal | Speed | Best Use Case |
|-----------|-------------|-------|---------------|
| MLP | No | Very Fast | Simple tasks |
| Gaussian | No | Fast | RL training |
| Diffusion | Yes | Slow | Precise manipulation |
| Transformer | Partial | Medium | Action sequences |

---

## Sensor Encoders

For multi-sensor VLA (e.g., autonomous driving).

### LiDAR Encoder (PointNet++)

```python
class PointCloudEncoder(nn.Module):
    """
    Encodes 3D point clouds from LiDAR.

    Input: (B, N, 4) - N points with (x, y, z, intensity)
    Output: (B, output_dim) - Global feature
    """
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 512,
    ):
        # PointNet++ architecture
        self.sa1 = PointNetSetAbstraction(...)
        self.sa2 = PointNetSetAbstraction(...)
        self.fc = nn.Linear(1024, output_dim)
```

### Radar Encoder

```python
class RadarEncoder(nn.Module):
    """
    Encodes radar data (range-Doppler or point targets).

    Input: (B, C, H, W) - Range-Doppler map or
           (B, N, 5) - Point targets (x, y, vx, vy, rcs)
    """
```

### IMU Encoder

```python
class IMUEncoder(nn.Module):
    """
    Encodes IMU sequences (accelerometer + gyroscope).

    Input: (B, T, 6) - Time series of 6-DoF IMU
    Output: (B, output_dim) - Motion features
    """
    def __init__(self, output_dim: int = 128):
        self.lstm = nn.LSTM(6, output_dim, num_layers=2, batch_first=True)
```

---

## Fusion Modules

Combine features from multiple sensors.

### Cross-Modal Fusion

```python
class CrossModalFusion(nn.Module):
    """
    Fuses features using cross-attention.

    Each modality attends to all other modalities.
    """
    def __init__(self, feature_dim: int, num_heads: int = 8):
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads)

    def forward(
        self,
        vision: torch.Tensor,
        lidar: torch.Tensor,
        radar: torch.Tensor,
    ) -> torch.Tensor:
        # Vision attends to LiDAR and Radar
        fused = self.cross_attn(vision, torch.cat([lidar, radar], dim=1))
        return fused
```

### Hierarchical Fusion

```python
class HierarchicalFusion(nn.Module):
    """
    Fuses sensors in stages:
    1. LiDAR + Radar → 3D Scene
    2. 3D Scene + Vision → Complete Understanding
    """
```

---

## Temporal Modeling

Handle action sequences and history.

### History Encoder

```python
class HistoryEncoder(nn.Module):
    """
    Encodes past observations and actions.

    Input:
    - observations: (B, T, obs_dim) - Past T observations
    - actions: (B, T-1, action_dim) - Past T-1 actions

    Output: (B, context_dim) - Temporal context
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        context_dim: int = 256,
        num_layers: int = 2,
    ):
        self.transformer = nn.TransformerEncoder(...)
```

### Action Chunking

```python
class ActionChunker(nn.Module):
    """
    Predicts multiple future actions at once.

    Benefits:
    - More stable control (temporal consistency)
    - Better for long-horizon tasks
    - Common in diffusion policies
    """
    def __init__(self, chunk_size: int = 16):
        self.chunk_size = chunk_size

    def forward(self, features):
        # Returns (B, chunk_size, action_dim)
        return self.action_head(features)
```

---

## World Model

For model-based RL and planning.

### RSSM (Recurrent State-Space Model)

```python
class RSSM(nn.Module):
    """
    Dreamer-style world model.

    Components:
    - Representation: p(s_t | s_{t-1}, a_{t-1}, o_t)
    - Transition: p(s_t | s_{t-1}, a_{t-1})
    - Observation: p(o_t | s_t)
    - Reward: p(r_t | s_t)

    Used for imagination-based planning.
    """
    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 7,
        hidden_dim: int = 512,
    ):
        self.prior = nn.GRUCell(hidden_dim, state_dim)  # Transition
        self.posterior = nn.GRUCell(hidden_dim, state_dim)  # Representation
        self.decoder = ObservationDecoder(state_dim)
        self.reward_head = RewardPredictor(state_dim)
```

### Planning with World Model

```python
def plan(world_model, observation, horizon=15):
    """
    Model Predictive Control using world model.
    """
    # Encode current observation
    state = world_model.encode(observation)

    # Sample action sequences
    best_actions = None
    best_reward = -float('inf')

    for _ in range(num_samples):
        actions = sample_action_sequence(horizon)

        # Imagine future states
        states = world_model.imagine(state, actions)
        rewards = world_model.predict_rewards(states)

        if rewards.sum() > best_reward:
            best_reward = rewards.sum()
            best_actions = actions

    return best_actions[0]  # Return first action
```

---

## Model Variants

### VLAModel (Standard)

- Single camera input
- Language instruction
- Simple action head

### MultiSensorVLA (Autonomous Driving)

- Multi-camera (surround view)
- LiDAR + Radar + IMU
- BEV (Bird's Eye View) representation
- Trajectory prediction

### OpenVLAWrapper (Pretrained)

- Wraps OpenVLA-7B
- Fine-tuning interface
- LoRA support

### SmolVLAWrapper (Lightweight)

- Wraps SmolVLM (256M)
- Edge deployment ready
- Mobile-optimized

---

## Extension Points

### Adding New Vision Encoder

```python
# 1. Add to model/vlm/vision_encoder.py
ENCODER_REGISTRY["my_encoder"] = MyEncoderClass

# 2. Use in config
config = VisionEncoderConfig(model_name="my_encoder")
```

### Adding New Action Head

```python
# 1. Create class inheriting ActionHeadBase
class MyActionHead(ActionHeadBase):
    ...

# 2. Register
ACTION_HEAD_REGISTRY["my_head"] = MyActionHead
```

### Adding New Sensor

```python
# 1. Create encoder in model/sensor/
class MySensorEncoder(nn.Module):
    ...

# 2. Add to MultiSensorVLA
model = MultiSensorVLA(
    sensor_encoders={
        "my_sensor": MySensorEncoder(output_dim=256),
    }
)
```

---

## Performance Considerations

### Memory Optimization

```python
# Gradient checkpointing
model.enable_gradient_checkpointing()

# Mixed precision
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(input)

# LoRA (freeze most parameters)
model.enable_lora(r=16, alpha=32)
```

### Inference Optimization

```python
# TorchScript
scripted = torch.jit.script(model)

# ONNX export
torch.onnx.export(model, inputs, "model.onnx")

# TensorRT
# Use torch2trt or polygraphy
```

---

## Next Steps

- See [Training Recipes](training_recipes.md) for task-specific configurations
- See [Usage Guide](usage.md) for practical examples
