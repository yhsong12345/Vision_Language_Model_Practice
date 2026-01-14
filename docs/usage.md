# Usage Guide

This guide provides detailed instructions for using the VLA Training Framework.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Training Workflows](#training-workflows)
5. [Inference](#inference)
6. [Deployment](#deployment)

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM (32GB recommended)
- 8GB+ VRAM (24GB+ recommended for full models)

### Step 1: Clone Repository

```bash
git clone https://github.com/yhsong12345/Vision_Language_Action_Model_Practice.git
cd Vision_Language_Action_Model_Practice
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n vla python=3.10
conda activate vla

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

```bash
# Core installation
pip install -r requirements.txt

# For GPU support (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: Flash Attention (faster attention on NVIDIA GPUs)
pip install flash-attn --no-build-isolation
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from model import VLAModel; print('VLA Framework loaded successfully!')"
```

---

## Quick Start

### Example 1: Train a Simple VLA Model

```python
from model import create_vla_model
from train.il import BehavioralCloning
from config import ILConfig

# 1. Create model
model = create_vla_model(
    vision_encoder="siglip-base",
    llm="qwen2-1.5b",
    action_dim=7,  # 6 DoF + gripper
    action_chunk_size=1,
)

# 2. Configure training
config = ILConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=100,
)

# 3. Train (with your dataset)
trainer = BehavioralCloning(model, config)
# trainer.train(dataloader)
```

### Example 2: Run Inference

```python
from model import create_vla_model
import torch
from PIL import Image

# Load trained model
model = create_vla_model(vision_encoder="siglip-base", llm="qwen2-1.5b", action_dim=7)
model.load_state_dict(torch.load("./checkpoints/vla_model.pt"))
model.eval()

# Prepare inputs
image = Image.open("robot_view.jpg")
instruction = "Pick up the red cube"

# Get action
with torch.no_grad():
    action = model.get_action(image, instruction)
print(f"Predicted action: {action}")
```

---

## Configuration

### Model Configuration

```python
from config import VLAConfig

config = VLAConfig(
    # Vision Encoder
    vision_encoder="siglip-base",  # Options: siglip-base, clip-vit-large, dinov2-base
    vision_dim=768,

    # Language Model
    llm="qwen2-1.5b",  # Options: qwen2-1.5b, qwen2-7b, llama3-8b
    llm_dim=1536,

    # Action Head
    action_dim=7,
    action_head_type="mlp",  # Options: mlp, gaussian, diffusion, transformer
    action_chunk_size=1,

    # Training
    freeze_vision=True,
    freeze_llm=True,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
)
```

### Training Configuration

```python
from config import TrainingConfig

config = TrainingConfig(
    # Optimizer
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,

    # Training
    batch_size=32,
    gradient_accumulation_steps=4,
    max_epochs=100,

    # Precision
    mixed_precision="bf16",  # Options: no, fp16, bf16

    # Logging
    log_every_n_steps=10,
    eval_every_n_epochs=5,
    save_every_n_epochs=10,

    # Directories
    output_dir="./outputs",
    checkpoint_dir="./checkpoints",
)
```

### Dataset Configuration

```python
from config import DatasetConfig

config = DatasetConfig(
    # Dataset
    dataset_name="lerobot/pusht",
    split="train",

    # Preprocessing
    image_size=(224, 224),
    normalize=True,

    # Action
    action_dim=2,
    action_horizon=1,  # Number of future actions to predict

    # History
    history_length=1,  # Number of past observations

    # Depth Camera (Optional)
    use_depth=False,          # Enable depth processing
    depth_size=224,           # Depth image resolution
    depth_normalization="minmax",  # Normalization method
)
```

### Multi-Sensor VLA Configuration

```python
from config import MultiSensorVLAConfig

# For RGB-D manipulation
config = MultiSensorVLAConfig.rgbd_manipulation()
# Enables: RGB camera + Depth camera
# Action dim: 7 (6DoF + gripper)

# For autonomous driving
config = MultiSensorVLAConfig.autonomous_driving()
# Enables: Camera + LiDAR + Radar + IMU

# For full sensor suite
config = MultiSensorVLAConfig.full_sensor()
# Enables: Camera + Depth + LiDAR + Radar + IMU
```

---

## Training Workflows

### Workflow 1: Behavioral Cloning (Recommended Start)

```python
from model import create_vla_model
from train.il import BehavioralCloning
from train.datasets import LeRobotDataset
from config import ILConfig

# 1. Prepare dataset
dataset = LeRobotDataset(
    repo_id="lerobot/pusht",
    split="train",
)

# 2. Create model
model = create_vla_model(
    vision_encoder="siglip-base",
    llm="qwen2-1.5b",
    action_dim=2,
)

# 3. Train
config = ILConfig(learning_rate=1e-4, num_epochs=100)
trainer = BehavioralCloning(model, config)
trainer.train(dataset)

# 4. Save
trainer.save("./checkpoints/bc_model.pt")
```

### Workflow 2: Offline RL (IQL)

```python
from train.offline_rl import IQLTrainer
from train.datasets import D4RLDataset
from config import OfflineRLConfig

# 1. Load offline dataset
dataset = D4RLDataset(env_name="hopper-medium-expert-v2")

# 2. Configure
config = OfflineRLConfig(
    expectile=0.7,
    temperature=3.0,
    learning_rate=3e-4,
)

# 3. Train
trainer = IQLTrainer(model, dataset, config)
trainer.train(num_epochs=1000)
```

### Workflow 3: Online RL (PPO)

```python
from train.online_rl import PPOTrainer
from config import RLConfig
import gymnasium as gym

# 1. Setup environment
env = gym.make("FetchPickAndPlace-v2")

# 2. Configure
config = RLConfig(
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    ppo_clip_range=0.2,
)

# 3. Train
trainer = PPOTrainer(model, env, config)
trainer.train()
```

---

## Inference

### Single Image Inference

```python
from infer import VLAInference
from PIL import Image

# Initialize
infer = VLAInference(
    model_path="./checkpoints/vla_model.pt",
    device="cuda",
)

# Run inference
image = Image.open("robot_camera.jpg")
instruction = "Pick up the blue block"

action = infer.predict(image, instruction)
print(f"Action: {action}")  # [x, y, z, rx, ry, rz, gripper]
```

### Batch Inference

```python
images = [Image.open(f"frame_{i}.jpg") for i in range(10)]
instruction = "Stack the blocks"

actions = infer.predict_batch(images, instruction)
print(f"Actions shape: {actions.shape}")  # (10, 7)
```

### Video/Sequence Inference

```python
# With temporal context
actions = infer.predict_sequence(
    video_path="robot_task.mp4",
    instruction="Complete the assembly task",
    fps=30,
)
```

---

## Deployment

### ROS2 Integration

```python
from integration import ROSBridge

# Initialize ROS node
ros = ROSBridge(
    node_name="vla_controller",
    image_topic="/camera/image_raw",
    action_topic="/robot/command",
)

# Load model
model = VLAInference("./checkpoints/vla_model.pt")

# Run control loop
ros.run(model, control_rate=30)  # 30 Hz
```

### CARLA Simulator

```python
from integration import CARLABridge

# Connect to CARLA
carla = CARLABridge(
    host="localhost",
    port=2000,
)

# Run autonomous driving
carla.run_vla(
    model_path="./checkpoints/driving_vla.pt",
    route="Town01_route1",
)
```

### MuJoCo Simulation

```python
from integration import MuJoCoBridge

# Setup environment
mujoco = MuJoCoBridge(
    env_name="FetchPickAndPlace-v2",
    render=True,
)

# Run policy
mujoco.run_policy(
    model_path="./checkpoints/manipulation_vla.pt",
    num_episodes=10,
)
```

---

## Common Issues

### Out of Memory

```python
# Use gradient checkpointing
config.gradient_checkpointing = True

# Reduce batch size
config.batch_size = 4
config.gradient_accumulation_steps = 16

# Use LoRA instead of full fine-tuning
config.use_lora = True
config.freeze_llm = True
```

### Slow Training

```python
# Enable mixed precision
config.mixed_precision = "bf16"

# Use Flash Attention
config.use_flash_attention = True

# Optimize data loading
config.num_workers = 8
config.pin_memory = True
```

### Poor Performance

```python
# Increase model capacity
config.action_head_hidden_dim = 512

# Use action chunking
config.action_chunk_size = 16

# Add temporal context
config.history_length = 4
```

---

## Next Steps

- See [Architecture Guide](architecture.md) for model details
- See [Training Recipes](training_recipes.md) for task-specific configurations
- Check [examples/](../examples/) for complete working examples
