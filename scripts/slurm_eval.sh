#!/bin/bash
#===============================================================================
# SLURM Script for VLA Evaluation
#
# Usage:
#   sbatch slurm_eval.sh <model_path> [benchmark]
#
# Evaluates a trained VLA model on benchmarks.
#===============================================================================

#SBATCH --job-name=vla_eval
#SBATCH --partition=gpu           # GPU partition
#SBATCH --nodes=1                 # Single node for evaluation
#SBATCH --ntasks-per-node=1       # 1 task
#SBATCH --gpus-per-node=1         # 1 GPU for evaluation
#SBATCH --cpus-per-task=16        # CPUs
#SBATCH --mem=64G                 # Memory
#SBATCH --time=12:00:00           # Max time
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Arguments
MODEL_PATH=${1:-"outputs/finetune/custom/best_policy.pt"}
BENCHMARK=${2:-"lerobot"}

# Directories
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/eval/${SLURM_JOB_ID}"

# Evaluation parameters
NUM_EPISODES=100
SEED=42

echo "============================================"
echo "VLA Evaluation"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model: ${MODEL_PATH}"
echo "Benchmark: ${BENCHMARK}"
echo "Episodes: ${NUM_EPISODES}"
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

cd "${PROJECT_DIR}"

echo ""
echo "Starting Evaluation..."
echo ""

python -c "
import torch
import json
import sys
sys.path.insert(0, '${PROJECT_DIR}')

from eval.evaluator import VLAEvaluator, EvalConfig
from eval.benchmark import VLABenchmark, BenchmarkConfig
from model.vla_base import VLAModel

# Load model
print('Loading model...')
model = VLAModel()
model.load_state_dict(torch.load('${MODEL_PATH}', map_location='cpu'))
model.eval()

# Run benchmark
config = BenchmarkConfig(
    benchmark_name='${BENCHMARK}',
    num_episodes=${NUM_EPISODES},
    seed=${SEED},
    output_dir='${OUTPUT_DIR}',
)

benchmark = VLABenchmark(model, config)
results = benchmark.run_benchmark()

# Save results
with open('${OUTPUT_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print('Results saved to ${OUTPUT_DIR}/results.json')
"

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "Results: ${OUTPUT_DIR}"
echo "============================================"
