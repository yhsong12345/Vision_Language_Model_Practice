#!/bin/bash
#===============================================================================
# SLURM Script for VLA Reinforcement Learning
#
# Usage:
#   sbatch slurm_rl.sh [ppo|sac|grpo]
#
# This script runs RL training for VLA models.
# - PPO: Proximal Policy Optimization
# - SAC: Soft Actor-Critic
# - GRPO: Group Relative Policy Optimization
#===============================================================================

#SBATCH --job-name=vla_rl
#SBATCH --partition=gpu           # GPU partition
#SBATCH --nodes=1                 # Number of nodes (RL typically single-node)
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --gpus-per-node=4         # GPUs per node
#SBATCH --cpus-per-task=32        # CPUs for environment workers
#SBATCH --mem=128G                # Memory per node
#SBATCH --time=48:00:00           # Max time
#SBATCH --output=logs/rl_%j.out
#SBATCH --error=logs/rl_%j.err

# Get algorithm from argument
ALGORITHM=${1:-"ppo"}

# Directories
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/rl/${ALGORITHM}"

# Configuration
case $ALGORITHM in
    "ppo")
        TOTAL_TIMESTEPS=1000000
        BATCH_SIZE=2048
        LEARNING_RATE=3e-4
        NUM_ENVS=16
        EXTRA_ARGS="--ppo_epochs 4 --ppo_clip_range 0.2 --gae_lambda 0.95"
        ;;
    "sac")
        TOTAL_TIMESTEPS=1000000
        BATCH_SIZE=256
        LEARNING_RATE=3e-4
        NUM_ENVS=1
        EXTRA_ARGS="--sac_tau 0.005 --buffer_size 1000000"
        ;;
    "grpo")
        TOTAL_TIMESTEPS=100000
        BATCH_SIZE=4
        LEARNING_RATE=5e-6
        NUM_ENVS=1
        EXTRA_ARGS="--grpo_group_size 8 --grpo_kl_coef 0.1"
        ;;
    *)
        echo "Unknown algorithm: ${ALGORITHM}"
        echo "Supported: ppo, sac, grpo"
        exit 1
        ;;
esac

echo "============================================"
echo "VLA Reinforcement Learning"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Algorithm: ${ALGORITHM}"
echo "Total Timesteps: ${TOTAL_TIMESTEPS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Num Envs: ${NUM_ENVS}"
echo "============================================"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Load modules
module purge
module load cuda/12.1
module load python/3.10

# Activate environment
source "${PROJECT_DIR}/venv/bin/activate"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

cd "${PROJECT_DIR}"

echo ""
echo "Starting RL Training..."
echo ""

python -m train.rl.${ALGORITHM}_trainer \
    --output_dir "${OUTPUT_DIR}" \
    --total_timesteps ${TOTAL_TIMESTEPS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_envs ${NUM_ENVS} \
    --seed ${SLURM_JOB_ID} \
    ${EXTRA_ARGS}

echo ""
echo "============================================"
echo "RL Training Complete!"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
