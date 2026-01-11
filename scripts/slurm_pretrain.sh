#!/bin/bash
#===============================================================================
# SLURM Script for VLM Pretraining - Multi-Node Multi-GPU
#
# Usage:
#   sbatch slurm_pretrain.sh
#
# This script runs distributed VLM pretraining across multiple nodes using
# DeepSpeed ZeRO-3 for memory-efficient training.
#===============================================================================

#SBATCH --job-name=vla_pretrain
#SBATCH --partition=gpu           # GPU partition
#SBATCH --nodes=4                 # Number of nodes
#SBATCH --ntasks-per-node=1       # 1 task per node (Accelerate handles GPUs)
#SBATCH --gpus-per-node=8         # GPUs per node
#SBATCH --cpus-per-task=64        # CPUs per task
#SBATCH --mem=512G                # Memory per node
#SBATCH --time=48:00:00           # Max time
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

# Configuration
VISION_MODEL="google/siglip-base-patch16-224"
LLM_MODEL="Qwen/Qwen2-1.5B-Instruct"
ALIGNMENT_DATASET="liuhaotian/LLaVA-Pretrain"
INSTRUCTION_DATASET="liuhaotian/LLaVA-Instruct-150K"

# Directories
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/pretrain_distributed"
CONFIG_DIR="${PROJECT_DIR}/config"

# Training parameters
BATCH_SIZE_PER_GPU=8
GRADIENT_ACCUMULATION=4
LEARNING_RATE_STAGE1=1e-3
LEARNING_RATE_STAGE2=2e-5

# Calculate total batch size
WORLD_SIZE=$((SLURM_NNODES * 8))  # Nodes * GPUs per node
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * WORLD_SIZE * GRADIENT_ACCUMULATION))

echo "============================================"
echo "VLM Distributed Pretraining"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: 8"
echo "Total GPUs: ${WORLD_SIZE}"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "Effective batch size: ${TOTAL_BATCH_SIZE}"
echo "============================================"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Load modules (adjust for your cluster)
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
echo "Starting Stage 1: Vision-Language Alignment..."
echo ""

srun --ntasks-per-node=1 \
    accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes ${WORLD_SIZE} \
    --num_machines ${SLURM_NNODES} \
    --machine_rank ${SLURM_NODEID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    -m train.pretrain.vlm_pretrainer \
    --stage alignment \
    --vision_model "${VISION_MODEL}" \
    --llm_model "${LLM_MODEL}" \
    --dataset "${ALIGNMENT_DATASET}" \
    --output_dir "${OUTPUT_DIR}/stage1" \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE_STAGE1} \
    --num_epochs 1 \
    --freeze_vision \
    --freeze_llm

echo ""
echo "Starting Stage 2: Visual Instruction Tuning..."
echo ""

srun --ntasks-per-node=1 \
    accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes ${WORLD_SIZE} \
    --num_machines ${SLURM_NNODES} \
    --machine_rank ${SLURM_NODEID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    -m train.pretrain.vlm_pretrainer \
    --stage instruction_tuning \
    --vision_model "${VISION_MODEL}" \
    --llm_model "${LLM_MODEL}" \
    --dataset "${INSTRUCTION_DATASET}" \
    --output_dir "${OUTPUT_DIR}/stage2" \
    --pretrained_path "${OUTPUT_DIR}/stage1/model.pt" \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE_STAGE2} \
    --num_epochs 3 \
    --freeze_vision

echo ""
echo "============================================"
echo "Pretraining Complete!"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
