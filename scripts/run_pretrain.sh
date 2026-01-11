#!/bin/bash
#===============================================================================
# VLM Pretraining Script
#
# Usage:
#   bash run_pretrain.sh [stage1|stage2|full]
#
# Examples:
#   bash run_pretrain.sh stage1    # Run Stage 1: Vision-Language Alignment
#   bash run_pretrain.sh stage2    # Run Stage 2: Visual Instruction Tuning
#   bash run_pretrain.sh full      # Run both stages
#===============================================================================

set -e

# Configuration
STAGE=${1:-"full"}
VISION_MODEL="google/siglip-base-patch16-224"
LLM_MODEL="Qwen/Qwen2-1.5B-Instruct"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/outputs/pretrain"

# Training parameters
BATCH_SIZE=32
LEARNING_RATE_STAGE1=1e-3
LEARNING_RATE_STAGE2=2e-5
NUM_EPOCHS_STAGE1=1
NUM_EPOCHS_STAGE2=3

# Datasets
ALIGNMENT_DATASET="liuhaotian/LLaVA-Pretrain"
INSTRUCTION_DATASET="liuhaotian/LLaVA-Instruct-150K"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "VLM Pretraining"
echo "============================================"
echo "Vision Model: ${VISION_MODEL}"
echo "LLM Model: ${LLM_MODEL}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Stage: ${STAGE}"
echo "============================================"

# Activate virtual environment if exists
if [ -d "${PROJECT_DIR}/venv" ]; then
    source "${PROJECT_DIR}/venv/bin/activate"
fi

# Install requirements if needed
pip install -q transformers accelerate datasets

# Run training
cd "${PROJECT_DIR}"

if [ "$STAGE" = "stage1" ] || [ "$STAGE" = "full" ]; then
    echo ""
    echo "Running Stage 1: Vision-Language Alignment"
    echo "============================================"

    python -m train.pretrain.vlm_pretrainer \
        --stage alignment \
        --vision_model "${VISION_MODEL}" \
        --llm_model "${LLM_MODEL}" \
        --dataset "${ALIGNMENT_DATASET}" \
        --output_dir "${OUTPUT_DIR}/stage1" \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE_STAGE1} \
        --num_epochs ${NUM_EPOCHS_STAGE1} \
        --freeze_vision \
        --freeze_llm
fi

if [ "$STAGE" = "stage2" ] || [ "$STAGE" = "full" ]; then
    echo ""
    echo "Running Stage 2: Visual Instruction Tuning"
    echo "============================================"

    # Load Stage 1 checkpoint if running full
    PRETRAINED_PATH=""
    if [ "$STAGE" = "full" ]; then
        PRETRAINED_PATH="${OUTPUT_DIR}/stage1/model.pt"
    fi

    python -m train.pretrain.vlm_pretrainer \
        --stage instruction_tuning \
        --vision_model "${VISION_MODEL}" \
        --llm_model "${LLM_MODEL}" \
        --dataset "${INSTRUCTION_DATASET}" \
        --output_dir "${OUTPUT_DIR}/stage2" \
        --batch_size $((BATCH_SIZE / 2)) \
        --learning_rate ${LEARNING_RATE_STAGE2} \
        --num_epochs ${NUM_EPOCHS_STAGE2} \
        --freeze_vision \
        ${PRETRAINED_PATH:+--pretrained_path "$PRETRAINED_PATH"}
fi

echo ""
echo "============================================"
echo "Pretraining Complete!"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
