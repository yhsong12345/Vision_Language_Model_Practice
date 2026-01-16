# Temporal Module Training Documentation

This document provides comprehensive documentation for the temporal processing modules in the VLA framework. These modules enable temporal reasoning, history encoding, and memory-based decision making for sequential tasks.

## Overview

The temporal module provides components for processing sequential observations and maintaining context over time:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Temporal Processing Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Observation Sequence                                                        │
│  [o_t-n, ..., o_t-1, o_t]                                                   │
│          │                                                                   │
│          ▼                                                                   │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │ Temporal        │     │ History         │     │ Memory          │       │
│  │ Encoder         │     │ Encoder         │     │ Buffer          │       │
│  │ (Transformer/   │     │ (State+Action   │     │ (Episodic/      │       │
│  │  LSTM)          │     │  History)       │     │  Working)       │       │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘       │
│           │                       │                       │                 │
│           └───────────────────────┼───────────────────────┘                 │
│                                   │                                         │
│                                   ▼                                         │
│                          Aggregated Context                                 │
│                                   │                                         │
│                                   ▼                                         │
│                          Action Prediction                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Components

### Component Summary

| Component | File | Purpose |
|-----------|------|---------|
| TemporalEncoder | [temporal_encoder.py](../../model/temporal/temporal_encoder.py) | Encode observation sequences |
| TemporalTransformer | [temporal_encoder.py](../../model/temporal/temporal_encoder.py) | Transformer-based temporal encoding |
| TemporalLSTM | [temporal_encoder.py](../../model/temporal/temporal_encoder.py) | LSTM-based temporal encoding |
| HistoryEncoder | [history_encoder.py](../../model/temporal/history_encoder.py) | Encode state-action history |
| ActionHistoryEncoder | [history_encoder.py](../../model/temporal/history_encoder.py) | Encode past actions |
| StateHistoryEncoder | [history_encoder.py](../../model/temporal/history_encoder.py) | Encode past observations |
| MemoryBuffer | [memory_buffer.py](../../model/temporal/memory_buffer.py) | Key-value memory storage |
| EpisodicMemory | [memory_buffer.py](../../model/temporal/memory_buffer.py) | Store complete episodes |
| WorkingMemory | [memory_buffer.py](../../model/temporal/memory_buffer.py) | Short-term task context |
| HierarchicalMemory | [memory_buffer.py](../../model/temporal/memory_buffer.py) | Multi-scale memory |

---

## 1. Temporal Encoder

The `TemporalEncoder` provides a unified interface for encoding sequences of observations over time.

### Architecture Options

| Encoder Type | Best For | Characteristics |
|--------------|----------|-----------------|
| Transformer | Parallel processing, attention patterns | O(n²) complexity, captures long-range dependencies |
| LSTM | Sequential processing, streaming | O(n) complexity, natural for online inference |

### Configuration

```python
from model.temporal import TemporalEncoder
from model.temporal.temporal_encoder import TemporalEncoderConfig

config = TemporalEncoderConfig(
    input_dim=768,              # Input feature dimension
    hidden_dim=512,             # Hidden layer dimension
    output_dim=768,             # Output dimension
    num_layers=4,               # Number of encoder layers
    num_heads=8,                # Attention heads (Transformer only)
    max_seq_len=64,             # Maximum sequence length
    dropout=0.1,                # Dropout probability
    use_causal_mask=True,       # Causal attention for autoregressive
    encoder_type="transformer", # "transformer" or "lstm"
    use_positional_encoding=True,
    use_learned_pos_embed=True, # Learned vs sinusoidal positions
)

encoder = TemporalEncoder(config)
```

### Usage

```python
import torch

# Input: batch of observation sequences
# Shape: (batch_size, seq_len, input_dim)
observations = torch.randn(4, 32, 768)
attention_mask = torch.ones(4, 32)  # Optional padding mask

# Forward pass
outputs = encoder(observations, attention_mask, return_sequence=True)

# Available outputs
hidden_states = outputs["hidden_states"]      # (batch, seq_len, output_dim)
last_hidden = outputs["last_hidden_state"]    # (batch, output_dim)
aggregated = outputs["aggregated"]            # (batch, output_dim)
```

### TemporalTransformer Details

```python
from model.temporal import TemporalTransformer
from model.temporal.temporal_encoder import TemporalEncoderConfig

config = TemporalEncoderConfig(
    input_dim=768,
    hidden_dim=512,
    output_dim=768,
    num_layers=4,
    num_heads=8,
    use_causal_mask=True,  # For autoregressive generation
)

transformer = TemporalTransformer(config)

# Get the last hidden state (useful for decision making)
last_state = transformer.get_last_hidden_state(observations, attention_mask)
```

### TemporalLSTM Details

```python
from model.temporal import TemporalLSTM
from model.temporal.temporal_encoder import TemporalEncoderConfig

config = TemporalEncoderConfig(
    input_dim=768,
    hidden_dim=512,
    output_dim=768,
    num_layers=4,
    encoder_type="lstm",
)

lstm = TemporalLSTM(config)

# Initialize hidden state for streaming inference
batch_size = 4
hidden_state = lstm.init_hidden(batch_size, device=torch.device("cuda"))

# Process sequence with hidden state
output, new_hidden = lstm(observations, hidden_state)
```

---

## 2. History Encoder

The `HistoryEncoder` encodes both state and action history for informed decision making.

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      History Encoder                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  States: [s_0, s_1, ..., s_t]    Actions: [a_0, a_1, ..., a_t]  │
│           │                               │                      │
│           ▼                               ▼                      │
│  ┌─────────────────┐             ┌─────────────────┐            │
│  │ StateHistory    │             │ ActionHistory   │            │
│  │ Encoder         │             │ Encoder         │            │
│  └────────┬────────┘             └────────┬────────┘            │
│           │                               │                      │
│           └───────────┬───────────────────┘                      │
│                       │                                          │
│                       ▼                                          │
│              ┌─────────────────┐                                 │
│              │ Fusion Module   │                                 │
│              │ (Concat + MLP + │                                 │
│              │  Cross-Attn)    │                                 │
│              └────────┬────────┘                                 │
│                       │                                          │
│                       ▼                                          │
│              Fused History Context                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```python
from model.temporal import HistoryEncoder
from model.temporal.history_encoder import HistoryEncoderConfig

config = HistoryEncoderConfig(
    obs_dim=768,                # Observation dimension
    action_dim=7,               # Action dimension
    hidden_dim=512,             # Hidden layer dimension
    output_dim=768,             # Output dimension
    history_length=16,          # Number of timesteps to track
    num_layers=2,               # Transformer layers
    num_heads=8,                # Attention heads
    dropout=0.1,                # Dropout probability
    use_action_embedding=True,  # Embed actions
    use_timestep_embedding=True,# Add timestep information
)

history_encoder = HistoryEncoder(config)
```

### Usage

```python
# Batch of state-action sequences
states = torch.randn(4, 16, 768)   # (batch, history_len, obs_dim)
actions = torch.randn(4, 16, 7)    # (batch, history_len, action_dim)

# Encode history
outputs = history_encoder(states, actions)

# Available outputs
state_encoding = outputs["state_encoding"]     # (batch, output_dim)
action_encoding = outputs["action_encoding"]   # (batch, output_dim)
fused = outputs["fused"]                       # (batch, output_dim)
cross_attended = outputs["cross_attended"]     # (batch, seq_len, output_dim)
state_sequence = outputs["state_sequence"]     # (batch, seq_len, output_dim)
```

### Online History Tracking

```python
# For online/streaming inference with internal buffers
history_encoder = HistoryEncoder(config)

# During rollout, add transitions
for step in range(episode_length):
    state = env.get_observation()
    action = policy(state)

    # Add to internal buffer
    history_encoder.add_transition(state, action)

    # Get history tensors
    hist_states, hist_actions = history_encoder.get_history(device)

    # Use in policy
    if hist_states is not None:
        context = history_encoder(hist_states, hist_actions)

# Reset at episode end
history_encoder.reset()
```

### History-Aware Policy Wrapper

```python
from model.temporal.history_encoder import HistoryAwarePolicy

# Wrap any policy with history awareness
base_policy = MLPActionHead(input_dim=768, action_dim=7)

history_policy = HistoryAwarePolicy(
    base_policy=base_policy,
    obs_dim=768,
    action_dim=7,
    history_dim=512,
    history_length=16,
)

# Forward with automatic history tracking
action = history_policy(current_observation)

# Update history after action execution
history_policy.update_history(state, action)

# Reset at episode boundary
history_policy.reset_history()
```

---

## 3. Memory Buffer

The memory module provides various memory structures for long-horizon reasoning.

### MemoryBuffer (Key-Value Memory)

Differentiable memory with content-based addressing (similar to Neural Turing Machine).

```python
from model.temporal import MemoryBuffer
from model.temporal.memory_buffer import MemoryConfig

config = MemoryConfig(
    hidden_dim=512,         # Input/query dimension
    memory_size=1000,       # Number of memory slots
    key_dim=64,             # Key dimension for addressing
    value_dim=512,          # Value dimension
    num_heads=8,            # Attention heads
    num_read_heads=4,       # Number of read heads
)

memory = MemoryBuffer(config)

# Write to memory
content = torch.randn(512)
memory.write(content)

# Read from memory with query
query = torch.randn(4, 512)  # Batch of queries
retrieved = memory.read(query)  # (batch, hidden_dim)

# Reset memory
memory.reset()
```

### EpisodicMemory

Stores complete episodes for experience replay and retrieval-based learning.

```python
from model.temporal import EpisodicMemory

episodic_memory = EpisodicMemory(
    obs_dim=768,
    action_dim=7,
    max_episodes=100,        # Maximum episodes to store
    max_episode_length=1000, # Max steps per episode
    hidden_dim=512,
)

# During rollout, add transitions
episodic_memory.add_transition(
    obs=current_obs,
    action=action,
    reward=reward,
    done=done,
)

# Retrieve similar episodes for context
query_obs = current_observation
similar_episodes = episodic_memory.retrieve_similar(query_obs, k=5)

# Get all transitions for training
all_data = episodic_memory.get_all_transitions()
```

### WorkingMemory

Short-term memory for current task context with attention-based retrieval.

```python
from model.temporal import WorkingMemory

working_memory = WorkingMemory(
    hidden_dim=512,
    memory_slots=32,    # Fixed-size buffer
    num_heads=8,
)

# Update with new states
working_memory.update(new_state)

# Read with attention
context = working_memory.read(query)

# Gated update (combines current state with memory)
updated_state = working_memory.get_gated_update(current_state)

# Reset
working_memory.reset()
```

### HierarchicalMemory

Multi-scale memory combining short-term and long-term storage.

```python
from model.temporal.memory_buffer import HierarchicalMemory

hier_memory = HierarchicalMemory(
    hidden_dim=512,
    short_term_slots=32,    # Recent observations
    long_term_slots=256,    # Important/consolidated memories
    num_heads=8,
)

# Update (automatically consolidates important states)
hier_memory.update(state)

# Read from both levels
outputs = hier_memory.read(query)
short_term_ctx = outputs["short_term"]
long_term_ctx = outputs["long_term"]
fused_ctx = outputs["fused"]

# Reset (optionally keep long-term)
hier_memory.reset(reset_long_term=False)
```

---

## 4. Integration with VLA Models

### Adding Temporal Processing to VLA

```python
from model import VLAModel
from model.temporal import TemporalEncoder, HistoryEncoder
from model.temporal.temporal_encoder import TemporalEncoderConfig
from model.temporal.history_encoder import HistoryEncoderConfig

# Create VLA model
vla = VLAModel(
    vision_model_name="google/siglip-base-patch16-224",
    llm_model_name="Qwen/Qwen2-1.5B-Instruct",
    action_dim=7,
)

# Add temporal encoder
temporal_config = TemporalEncoderConfig(
    input_dim=vla.hidden_dim,
    hidden_dim=512,
    output_dim=vla.hidden_dim,
)
temporal_encoder = TemporalEncoder(temporal_config)

# Add history encoder
history_config = HistoryEncoderConfig(
    obs_dim=vla.hidden_dim,
    action_dim=7,
)
history_encoder = HistoryEncoder(history_config)

# Combined forward pass
class TemporalVLA(nn.Module):
    def __init__(self, vla, temporal_encoder, history_encoder):
        super().__init__()
        self.vla = vla
        self.temporal_encoder = temporal_encoder
        self.history_encoder = history_encoder

    def forward(self, images, text, past_states=None, past_actions=None):
        # Get VLA features
        features = self.vla.get_features(images, text)

        # Add temporal context if available
        if past_states is not None:
            temporal_ctx = self.temporal_encoder(past_states)
            features = features + temporal_ctx["aggregated"]

        # Add history context if available
        if past_states is not None and past_actions is not None:
            history_ctx = self.history_encoder(past_states, past_actions)
            features = features + history_ctx["fused"]

        # Predict action
        return self.vla.action_head(features)
```

---

## 5. Training Temporal Modules

### Training Configuration

```python
from config.training_config import TemporalTrainingConfig

config = TemporalTrainingConfig(
    # Sequence settings
    sequence_length=32,
    history_length=16,

    # Training
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=100,

    # Temporal encoder
    temporal_encoder_type="transformer",
    num_temporal_layers=4,

    # History encoder
    use_history_encoder=True,
    history_hidden_dim=512,

    # Memory
    use_memory=True,
    memory_size=1000,
)
```

### Training Script Example

```bash
# Train VLA with temporal processing
python train/finetune/vla_finetuner.py \
    --pretrained-vlm ./output/stage1b/best \
    --dataset nuscenes \
    --use-temporal-encoder \
    --temporal-encoder-type transformer \
    --sequence-length 32 \
    --use-history-encoder \
    --history-length 16 \
    --output-dir ./output/temporal_vla
```

### Loss Functions for Temporal Training

```python
def temporal_loss(
    predictions,
    targets,
    temporal_encoder,
    lambda_temporal=0.1,
):
    """
    Combined loss with temporal consistency regularization.
    """
    # Action prediction loss
    action_loss = F.mse_loss(predictions, targets)

    # Temporal consistency loss (smooth action sequences)
    if predictions.dim() == 3:  # (batch, seq_len, action_dim)
        temporal_diff = predictions[:, 1:] - predictions[:, :-1]
        temporal_loss = temporal_diff.pow(2).mean()
    else:
        temporal_loss = 0.0

    return action_loss + lambda_temporal * temporal_loss
```

---

## 6. Use Cases

### Autonomous Driving with Temporal Context

```python
# Process multiple frames for driving decisions
class DrivingVLAWithTemporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.vla = VLAModel(...)
        self.temporal = TemporalEncoder(...)
        self.frame_buffer = []
        self.max_frames = 8

    def forward(self, current_image, instruction):
        # Get current features
        current_features = self.vla.get_features(current_image, instruction)

        # Add to buffer
        self.frame_buffer.append(current_features)
        if len(self.frame_buffer) > self.max_frames:
            self.frame_buffer.pop(0)

        # Encode temporal context
        if len(self.frame_buffer) > 1:
            temporal_input = torch.stack(self.frame_buffer, dim=1)
            temporal_ctx = self.temporal(temporal_input)
            features = current_features + temporal_ctx["aggregated"]
        else:
            features = current_features

        return self.vla.action_head(features)
```

### Manipulation with Action History

```python
# Use action history for smooth manipulation
class ManipulationVLAWithHistory(nn.Module):
    def __init__(self):
        super().__init__()
        self.vla = VLAModel(...)
        self.history_encoder = HistoryEncoder(...)

    def forward(self, image, instruction, past_states, past_actions):
        # Get current features
        features = self.vla.get_features(image, instruction)

        # Encode history for action consistency
        history_ctx = self.history_encoder(past_states, past_actions)

        # Combine
        combined = features + history_ctx["fused"]

        return self.vla.action_head(combined)
```

---

## 7. Best Practices

### Choosing Temporal Architecture

| Scenario | Recommended Architecture |
|----------|-------------------------|
| Real-time inference | LSTM (streaming) |
| Offline training | Transformer (parallel) |
| Long sequences (>64) | LSTM or chunked Transformer |
| Short sequences (<32) | Transformer |
| Variable length | LSTM with padding |

### Memory Considerations

- **Working Memory**: Use for recent context (last 16-32 steps)
- **Episodic Memory**: Use for experience replay and retrieval
- **Hierarchical Memory**: Use when both short and long-term context matter

### Hyperparameter Guidelines

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| sequence_length | 8-64 | Longer for driving, shorter for manipulation |
| history_length | 8-32 | Based on task temporal dependency |
| num_layers | 2-6 | More for complex temporal patterns |
| hidden_dim | 256-1024 | Match VLA hidden dimension |
| num_heads | 4-16 | Standard transformer guidance |

---

## Next Steps

- [Training VLA Recipe](training_vla_recipe.md) - Complete training pipeline
- [Training World Model](training_world_model.md) - World model with temporal dynamics
- [Training Imitation Learning](training_imitation_learning.md) - IL with temporal context
- [Deployment](deployment.md) - Deploy temporal models
