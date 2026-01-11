#!/bin/bash
#===============================================================================
# SLURM Script for VLA Fine-tuning - Multi-Node Multi-GPU
#
# Usage:
#   sbatch slurm_finetune.sh [custom|openvla|smolvla]
#
# This script runs distributed VLA fine-tuning on robot manipulation datasets.
#===============================================================================

#SBATCH --job-name=vla_finetune
#SBATCH --partition=gpu           # GPU partition
#SBATCH --nodes=2                 # Number of nodes
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --gpus-per-node=8         # GPUs per node
#SBATCH --cpus-per-task=64        # CPUs per task
#SBATCH --mem=256G                # Memory per node
#SBATCH --time=24:00:00           # Max time
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

# Get model type from argument or default
MODEL_TYPE=${1:-"custom"}

# Configuration
case $MODEL_TYPE in
    "custom")
        VISION_MODEL="google/siglip-base-patch16-224"
        LLM_MODEL="Qwen/Qwen2-1.5B-Instruct"
        BATCH_SIZE_PER_GPU=8
        LEARNING_RATE=1e-4
        NUM_EPOCHS=10
        EXTRA_ARGS="--freeze_vision"
        ;;
    "openvla")
        MODEL_NAME="openvla/openvla-7b"
        BATCH_SIZE_PER_GPU=1
        LEARNING_RATE=2e-5
        NUM_EPOCHS=5
        EXTRA_ARGS="--use_lora --lora_r 32 --load_in_4bit"
        ;;
    "smolvla")
        MODEL_NAME="HuggingFaceTB/SmolVLA-450M"
        BATCH_SIZE_PER_GPU=4
        LEARNING_RATE=1e-4
        NUM_EPOCHS=10
        EXTRA_ARGS=""
        ;;
    *)
        echo "Unknown model type: ${MODEL_TYPE}"
        exit 1
        ;;
esac

# Directories
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/finetune_distributed/${MODEL_TYPE}"
CONFIG_DIR="${PROJECT_DIR}/config"

# Dataset
DATASET="lerobot/pusht"

# Training parameters
GRADIENT_ACCUMULATION=4

# Calculate distributed settings
WORLD_SIZE=$((SLURM_NNODES * 8))
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * WORLD_SIZE * GRADIENT_ACCUMULATION))

echo "============================================"
echo "VLA Distributed Fine-tuning"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model Type: ${MODEL_TYPE}"
echo "Nodes: ${SLURM_NNODES}"
echo "Total GPUs: ${WORLD_SIZE}"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Effective batch size: ${TOTAL_BATCH_SIZE}"
echo "Dataset: ${DATASET}"
echo "============================================"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Load modules
module purge
module load cuda/12.1
module load python/3.10
module load nccl

# Activate environment
source "${PROJECT_DIR}/venv/bin/activate"

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=${WORLD_SIZE}
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# DeepSpeed config
DEEPSPEED_CONFIG="${CONFIG_DIR}/deepspeed_zero3.yaml"

# Create Accelerate config
ACCELERATE_CONFIG="${OUTPUT_DIR}/accelerate_config.yaml"
cat > "${ACCELERATE_CONFIG}" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: ${DEEPSPEED_CONFIG}
  zero3_init_flag: true
machine_rank: \${SLURM_NODEID}
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}
main_training_function: main
mixed_precision: bf16
num_machines: ${SLURM_NNODES}
num_processes: ${WORLD_SIZE}
EOF

# Run training
cd "${PROJECT_DIR}"

echo ""
echo "Starting Fine-tuning..."
echo ""

if [ "$MODEL_TYPE" = "custom" ]; then
    srun --ntasks-per-node=1 \
        accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        --num_processes ${WORLD_SIZE} \
        --num_machines ${SLURM_NNODES} \
        --machine_rank ${SLURM_NODEID} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        -m train.finetune.vla_finetuner \
        --vision_model "${VISION_MODEL}" \
        --llm_model "${LLM_MODEL}" \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size ${BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
        --learning_rate ${LEARNING_RATE} \
        --num_epochs ${NUM_EPOCHS} \
        ${EXTRA_ARGS}
else
    srun --ntasks-per-node=1 \
        accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        --num_processes ${WORLD_SIZE} \
        --num_machines ${SLURM_NNODES} \
        --machine_rank ${SLURM_NODEID} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        -m train.finetune.vla_finetuner \
        --model_name "${MODEL_NAME}" \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size ${BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
        --learning_rate ${LEARNING_RATE} \
        --num_epochs ${NUM_EPOCHS} \
        ${EXTRA_ARGS}
fi

echo ""
echo "============================================"
echo "Fine-tuning Complete!"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
