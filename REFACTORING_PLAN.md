# Refactoring Plan for VLA Training Framework

## Overview

This plan reorganizes the codebase to improve clarity, reduce duplication, and simplify the training script structure.

---

## Phase 1: Rename `vla/` to `core/` (Framework Utilities)

### Problem
- `vla/` at root level conflicts with `model/vla/` (actual VLA models)
- Causes import confusion: `from vla import ...` vs `from model.vla import ...`

### Changes

| Current Path | New Path |
|-------------|----------|
| `vla/__init__.py` | `core/__init__.py` |
| `vla/exceptions.py` | `core/exceptions.py` |
| `vla/registry.py` | `core/registry.py` |
| `vla/logging.py` | `core/logging.py` |

### Files to Update (imports)
- All files importing from `vla.exceptions`, `vla.registry`, `vla.logging`
- Update `pyproject.toml` if package name references `vla`

---

## Phase 2: Consolidate `device_utils.py`

### Problem
- Duplicate files:
  - `model/utils/device_utils.py`
  - `train/utils/device_utils.py`

### Changes
1. Keep single source in `core/device_utils.py`
2. Delete duplicates
3. Update all imports

| Current Path | Action |
|-------------|--------|
| `model/utils/device_utils.py` | DELETE |
| `train/utils/device_utils.py` | DELETE |
| `core/device_utils.py` | CREATE (merged) |

---

## Phase 3: Simplify Trainer Base Class Naming

### Problem
- Multiple `base_trainer.py` files with confusing names

### Changes

| Current Path | New Path |
|-------------|----------|
| `train/base_trainer.py` | `train/base_trainer.py` (keep - root base) |
| `train/il/base_trainer.py` | `train/il/trainer.py` |
| `train/online_rl/base_trainer.py` | `train/online_rl/trainer.py` |
| `train/offline_rl/base_trainer.py` | `train/offline_rl/trainer.py` |

### Update `__init__.py` files accordingly

---

## Phase 4: Unified Training Entry Points with Algorithm Selection

### Problem
- Too many separate scripts for each algorithm:
  - `scripts/run_offline_rl_cql.sh`
  - `scripts/run_offline_rl_iql.sh`
  - `scripts/run_offline_rl_td3bc.sh`
  - `scripts/run_offline_rl_dt.sh`
  - `scripts/run_online_rl_ppo.sh`
  - `scripts/run_online_rl_sac.sh`
  - `scripts/run_online_rl_grpo.sh`
  - `scripts/run_il_dagger.sh`
  - `scripts/run_il_gail.sh`

### Solution
Create unified `train.py` entry points in each training module:

```
train/
├── offline_rl/
│   └── train.py      # NEW - unified entry point
├── online_rl/
│   └── train.py      # NEW - unified entry point
└── il/
    └── train.py      # NEW - unified entry point
```

### Usage Examples

```bash
# Offline RL - select algorithm via --algo flag
python train/offline_rl/train.py --algo cql --config config.yaml
python train/offline_rl/train.py --algo iql --config config.yaml
python train/offline_rl/train.py --algo td3bc --config config.yaml
python train/offline_rl/train.py --algo dt --config config.yaml

# Online RL
python train/online_rl/train.py --algo ppo --config config.yaml
python train/online_rl/train.py --algo sac --config config.yaml
python train/online_rl/train.py --algo grpo --config config.yaml

# Imitation Learning
python train/il/train.py --algo bc --config config.yaml
python train/il/train.py --algo dagger --config config.yaml
python train/il/train.py --algo gail --config config.yaml
```

### New Consolidated Scripts

| Old Scripts | New Script |
|------------|------------|
| `run_offline_rl_cql.sh`, `run_offline_rl_iql.sh`, `run_offline_rl_td3bc.sh`, `run_offline_rl_dt.sh` | `scripts/run_offline_rl.sh` |
| `run_online_rl_ppo.sh`, `run_online_rl_sac.sh`, `run_online_rl_grpo.sh` | `scripts/run_online_rl.sh` |
| `run_il.sh`, `run_il_dagger.sh`, `run_il_gail.sh` | `scripts/run_il.sh` |

### Script Usage

```bash
# Run offline RL with specific algorithm
sbatch scripts/run_offline_rl.sh --algo cql
sbatch scripts/run_offline_rl.sh --algo grpo  # Note: GRPO is actually online RL

# Run online RL
sbatch scripts/run_online_rl.sh --algo ppo
sbatch scripts/run_online_rl.sh --algo sac

# Run IL
sbatch scripts/run_il.sh --algo bc
sbatch scripts/run_il.sh --algo dagger
```

---

## Phase 5: Create CLI Module

### Problem
- Entry points (`run.py`, `infer.py`) scattered at root level

### Changes

| Current Path | New Path |
|-------------|----------|
| `run.py` | `cli/run.py` |
| `infer.py` | `cli/infer.py` |

### Add `cli/__init__.py` and update `pyproject.toml` entry points

---

## Phase 6: Reorganize Tests to Mirror Source

### Current Structure
```
tests/
├── unit/
│   ├── test_action_heads.py
│   ├── test_configs.py
│   ├── test_utils.py
│   └── test_registry.py
├── integration/
│   └── test_training_loop.py
├── test_model.py
└── test_training.py
```

### New Structure
```
tests/
├── model/
│   ├── test_vla_model.py
│   ├── test_vlm_model.py
│   ├── action_head/
│   │   └── test_action_heads.py
│   └── sensor/
│       └── test_sensors.py
├── train/
│   ├── test_il.py
│   ├── test_online_rl.py
│   └── test_offline_rl.py
├── core/
│   ├── test_registry.py
│   └── test_utils.py
├── config/
│   └── test_configs.py
├── integration/
│   └── test_training_loop.py
└── conftest.py
```

---

## Phase 7: Scripts Cleanup

### Delete (replaced by unified scripts)
- `scripts/run_offline_rl_cql.sh`
- `scripts/run_offline_rl_iql.sh`
- `scripts/run_offline_rl_td3bc.sh`
- `scripts/run_offline_rl_dt.sh`
- `scripts/run_online_rl_ppo.sh`
- `scripts/run_online_rl_sac.sh`
- `scripts/run_online_rl_grpo.sh`
- `scripts/run_il_dagger.sh`
- `scripts/run_il_gail.sh`

### Keep (task-specific)
- `scripts/run_pretrain.sh`
- `scripts/run_finetune.sh`
- `scripts/run_eval.sh`
- `scripts/run_driving_vla.sh`
- `scripts/run_humanoid_vla.sh`
- `scripts/run_world_model.sh`
- `scripts/run_all_training.sh`
- `scripts/slurm_launcher.sh`

---

## Final Directory Structure

```
Vision_Language_Model_Practice/
├── core/                          # Framework utilities (renamed from vla/)
│   ├── __init__.py
│   ├── exceptions.py
│   ├── registry.py
│   ├── logging.py
│   └── device_utils.py            # Single source of truth
│
├── model/                         # Neural network architectures
│   ├── vla/                       # VLA models (unchanged)
│   ├── vlm/
│   ├── action_head/
│   ├── sensor/
│   ├── fusion/
│   ├── temporal/
│   ├── world_model/
│   ├── safety/
│   ├── embodiment/
│   └── utils/                     # (remove device_utils.py)
│
├── train/                         # Training pipelines
│   ├── base_trainer.py            # Root base class
│   ├── il/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Renamed from base_trainer.py
│   │   ├── behavioral_cloning.py
│   │   ├── dagger.py
│   │   ├── gail.py
│   │   └── train.py               # NEW unified entry point
│   ├── online_rl/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Renamed from base_trainer.py
│   │   ├── ppo_trainer.py
│   │   ├── sac_trainer.py
│   │   ├── grpo_trainer.py
│   │   └── train.py               # NEW unified entry point
│   ├── offline_rl/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Renamed from base_trainer.py
│   │   ├── cql_trainer.py
│   │   ├── iql_trainer.py
│   │   ├── td3_bc_trainer.py
│   │   ├── decision_transformer.py
│   │   └── train.py               # NEW unified entry point
│   ├── pretrain/
│   ├── finetune/
│   ├── world_model/
│   ├── embodiment/
│   ├── datasets/
│   └── utils/                     # (remove device_utils.py)
│
├── cli/                           # Entry points
│   ├── __init__.py
│   ├── run.py
│   └── infer.py
│
├── config/
├── eval/
├── integration/
├── examples/
│
├── tests/                         # Reorganized tests
│   ├── model/
│   ├── train/
│   ├── core/
│   ├── config/
│   ├── integration/
│   └── conftest.py
│
├── scripts/
│   ├── run_offline_rl.sh          # Unified (replaces 4 scripts)
│   ├── run_online_rl.sh           # Unified (replaces 3 scripts)
│   ├── run_il.sh                  # Unified (replaces 3 scripts)
│   ├── run_pretrain.sh
│   ├── run_finetune.sh
│   ├── run_eval.sh
│   ├── run_driving_vla.sh
│   ├── run_humanoid_vla.sh
│   ├── run_world_model.sh
│   ├── run_all_training.sh
│   └── slurm_launcher.sh
│
├── docs/
├── requirements/
├── pyproject.toml
└── README.md
```

---

## Implementation Order

1. **Phase 1**: Rename `vla/` → `core/` (low risk, clear benefit)
2. **Phase 2**: Consolidate `device_utils.py` (removes duplication)
3. **Phase 4**: Create unified `train.py` entry points (biggest improvement)
4. **Phase 7**: Clean up old scripts
5. **Phase 3**: Rename base trainers (clarity improvement)
6. **Phase 5**: Create CLI module (optional, lower priority)
7. **Phase 6**: Reorganize tests (optional, can be done incrementally)

---

## New Unified Training Entry Points

### `train/offline_rl/train.py`

```python
"""Unified Offline RL Training Entry Point"""
import argparse
from train.offline_rl import (
    CQLTrainer,
    IQLTrainer,
    TD3BCTrainer,
    DecisionTransformerTrainer,
)

TRAINERS = {
    "cql": CQLTrainer,
    "iql": IQLTrainer,
    "td3bc": TD3BCTrainer,
    "dt": DecisionTransformerTrainer,
}

def main():
    parser = argparse.ArgumentParser(description="Offline RL Training")
    parser.add_argument("--algo", type=str, required=True,
                        choices=list(TRAINERS.keys()),
                        help="Algorithm to use")
    parser.add_argument("--config", type=str, help="Config file path")
    # ... other args
    args, remaining = parser.parse_known_args()

    trainer_cls = TRAINERS[args.algo]
    trainer = trainer_cls.from_config(args.config)
    trainer.train()

if __name__ == "__main__":
    main()
```

### `train/online_rl/train.py`

```python
"""Unified Online RL Training Entry Point"""
import argparse
from train.online_rl import PPOTrainer, SACTrainer, GRPOTrainer

TRAINERS = {
    "ppo": PPOTrainer,
    "sac": SACTrainer,
    "grpo": GRPOTrainer,
}

def main():
    parser = argparse.ArgumentParser(description="Online RL Training")
    parser.add_argument("--algo", type=str, required=True,
                        choices=list(TRAINERS.keys()),
                        help="Algorithm to use")
    parser.add_argument("--config", type=str, help="Config file path")
    args, remaining = parser.parse_known_args()

    trainer_cls = TRAINERS[args.algo]
    trainer = trainer_cls.from_config(args.config)
    trainer.train()

if __name__ == "__main__":
    main()
```

### `train/il/train.py`

```python
"""Unified Imitation Learning Training Entry Point"""
import argparse
from train.il import BehavioralCloning, DAgger, GAIL

TRAINERS = {
    "bc": BehavioralCloning,
    "dagger": DAgger,
    "gail": GAIL,
}

def main():
    parser = argparse.ArgumentParser(description="Imitation Learning Training")
    parser.add_argument("--algo", type=str, required=True,
                        choices=list(TRAINERS.keys()),
                        help="Algorithm to use")
    parser.add_argument("--config", type=str, help="Config file path")
    args, remaining = parser.parse_known_args()

    trainer_cls = TRAINERS[args.algo]
    trainer = trainer_cls.from_config(args.config)
    trainer.train()

if __name__ == "__main__":
    main()
```

---

## New Unified Shell Scripts

### `scripts/run_offline_rl.sh`

```bash
#!/bin/bash
#SBATCH --job-name=vla_offline_rl
#SBATCH --comment="VLA Offline RL Training"
# ... SLURM config ...

# Parse algorithm from arguments
ALGO="cql"  # default
for arg in "$@"; do
    case $arg in
        --algo=*) ALGO="${arg#*=}"; shift ;;
        --algo) ALGO="$2"; shift 2 ;;
    esac
done

echo "Running Offline RL with algorithm: $ALGO"

# ... distributed training setup ...

accelerate launch \
    --config_file config/deepspeed_zero3.yaml \
    train/offline_rl/train.py --algo "$ALGO" "$@"
```

### `scripts/run_online_rl.sh`

```bash
#!/bin/bash
#SBATCH --job-name=vla_online_rl
#SBATCH --comment="VLA Online RL Training"
# ... SLURM config ...

ALGO="ppo"  # default
for arg in "$@"; do
    case $arg in
        --algo=*) ALGO="${arg#*=}"; shift ;;
        --algo) ALGO="$2"; shift 2 ;;
    esac
done

echo "Running Online RL with algorithm: $ALGO"

accelerate launch \
    --config_file config/deepspeed_zero3.yaml \
    train/online_rl/train.py --algo "$ALGO" "$@"
```

### `scripts/run_il.sh` (updated)

```bash
#!/bin/bash
#SBATCH --job-name=vla_il
#SBATCH --comment="VLA Imitation Learning Training"
# ... SLURM config ...

ALGO="bc"  # default
for arg in "$@"; do
    case $arg in
        --algo=*) ALGO="${arg#*=}"; shift ;;
        --algo) ALGO="$2"; shift 2 ;;
    esac
done

echo "Running IL with algorithm: $ALGO"

accelerate launch \
    --config_file config/deepspeed_zero3.yaml \
    train/il/train.py --algo "$ALGO" "$@"
```

---

## Summary of Benefits

1. **Clearer naming**: `core/` vs `model/vla/` removes ambiguity
2. **No duplication**: Single `device_utils.py` in `core/`
3. **Simpler scripts**: 3 unified scripts instead of 10+ separate ones
4. **Flexible CLI**: `--algo` flag selects algorithm at runtime
5. **Easier maintenance**: Less files to maintain
6. **Better discoverability**: Clear trainer hierarchy

---

## Implementation Status

### Completed

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Rename `vla/` to `core/` | DONE |
| Phase 2 | Consolidate `device_utils.py` | DONE |
| Phase 4 | Create unified `train.py` entry points | DONE |
| Phase 5 | Create unified shell scripts | DONE |

### Created Files

**New `core/` module:**
- `core/__init__.py` - Main exports
- `core/exceptions.py` - Custom exceptions
- `core/registry.py` - Component registry
- `core/logging.py` - Logging utilities
- `core/device_utils.py` - Device management (single source)

**Unified training entry points:**
- `train/offline_rl/train.py` - Supports: cql, iql, td3bc, dt
- `train/online_rl/train.py` - Supports: ppo, sac, grpo
- `train/il/train.py` - Supports: bc, dagger, gail

**Unified shell scripts:**
- `scripts/run_offline_rl.sh` - `--algo {cql,iql,td3bc,dt}`
- `scripts/run_online_rl.sh` - `--algo {ppo,sac,grpo}`
- `scripts/run_il.sh` - `--algo {bc,dagger,gail}`

**Backward compatibility shims:**
- `vla/__init__.py` - Deprecation warning, re-exports from `core`
- `model/utils/device_utils.py` - Re-exports from `core.device_utils`
- `train/utils/device_utils.py` - Re-exports from `core.device_utils`

### Remaining (Optional)

| Phase | Description | Notes |
|-------|-------------|-------|
| Phase 3 | Rename `base_trainer.py` → `trainer.py` | Optional clarity improvement |
| Phase 5 | Create `cli/` module | Optional, lower priority |
| Phase 6 | Reorganize tests | Can be done incrementally |
| Phase 7 | Delete old scripts | Safe to delete after testing |

### Scripts Safe to Delete (After Testing)

```bash
# Old per-algorithm scripts (replaced by unified scripts)
scripts/run_offline_rl_cql.sh
scripts/run_offline_rl_iql.sh
scripts/run_offline_rl_td3bc.sh
scripts/run_offline_rl_dt.sh
scripts/run_online_rl_ppo.sh
scripts/run_online_rl_sac.sh
scripts/run_online_rl_grpo.sh
scripts/run_il_dagger.sh
scripts/run_il_gail.sh
```

---

## Usage Examples

### Training with Unified Entry Points

```bash
# Offline RL
python train/offline_rl/train.py --algo cql --dataset hopper-medium-v2
python train/offline_rl/train.py --algo iql --num_epochs 200
python train/offline_rl/train.py --algo td3bc --batch_size 512
python train/offline_rl/train.py --algo dt --context_length 30

# Online RL
python train/online_rl/train.py --algo ppo --env CartPole-v1
python train/online_rl/train.py --algo sac --env HalfCheetah-v4
python train/online_rl/train.py --algo grpo --total_timesteps 500000

# Imitation Learning
python train/il/train.py --algo bc --env CartPole-v1
python train/il/train.py --algo dagger --dagger_iterations 20
python train/il/train.py --algo gail --gail_total_timesteps 1000000
```

### SLURM Submission

```bash
# Submit with algorithm selection
sbatch scripts/run_offline_rl.sh --algo cql
sbatch scripts/run_online_rl.sh --algo ppo --env HalfCheetah-v4
sbatch scripts/run_il.sh --algo dagger
```

---

## Migration Guide

### Updating Imports

```python
# Old imports (deprecated, will show warning)
from vla.exceptions import VLAError
from model.utils.device_utils import get_device
from train.utils.device_utils import move_to_device

# New imports (recommended)
from core.exceptions import VLAError
from core.device_utils import get_device, move_to_device
from core import VLAError, get_device  # or use top-level exports
```

### Running Training

```bash
# Old way (multiple scripts)
sbatch scripts/run_offline_rl_cql.sh
sbatch scripts/run_offline_rl_iql.sh

# New way (single script with --algo)
sbatch scripts/run_offline_rl.sh --algo cql
sbatch scripts/run_offline_rl.sh --algo iql
```
