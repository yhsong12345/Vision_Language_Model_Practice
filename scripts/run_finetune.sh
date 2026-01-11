#!/bin/bash
#===============================================================================
# VLA Fine-tuning Script
#
# Usage:
#   bash run_finetune.sh [custom|openvla|smolvla]
#
# Examples:
#   bash run_finetune.sh custom     # Fine-tune custom VLA model
#   bash run_finetune.sh openvla    # Fine-tune OpenVLA-7B with LoRA
#   bash run_finetune.sh smolvla    # Fine-tune SmolVLA-450M
#===============================================================================

set -e

# Configuration
MODEL_TYPE=${1:-"custom"}

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/outputs/finetune/${MODEL_TYPE}"

# Dataset
DATASET="lerobot/pusht"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "VLA Fine-tuning"
echo "============================================"
echo "Model Type: ${MODEL_TYPE}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Activate virtual environment if exists
if [ -d "${PROJECT_DIR}/venv" ]; then
    source "${PROJECT_DIR}/venv/bin/activate"
fi

cd "${PROJECT_DIR}"

case $MODEL_TYPE in
    "custom")
        # Custom VLA (SigLIP + Qwen2-1.5B)
        VISION_MODEL="google/siglip-base-patch16-224"
        LLM_MODEL="Qwen/Qwen2-1.5B-Instruct"
        BATCH_SIZE=8
        LEARNING_RATE=1e-4
        NUM_EPOCHS=10

        python -m train.finetune.vla_finetuner \
            --vision_model "${VISION_MODEL}" \
            --llm_model "${LLM_MODEL}" \
            --dataset "${DATASET}" \
            --output_dir "${OUTPUT_DIR}" \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --num_epochs ${NUM_EPOCHS} \
            --freeze_vision \
            --action_dim 7
        ;;

    "openvla")
        # OpenVLA-7B with LoRA and 4-bit quantization
        MODEL_NAME="openvla/openvla-7b"
        BATCH_SIZE=2
        LEARNING_RATE=2e-5
        NUM_EPOCHS=5

        python -m train.finetune.vla_finetuner \
            --model_name "${MODEL_NAME}" \
            --dataset "${DATASET}" \
            --output_dir "${OUTPUT_DIR}" \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --num_epochs ${NUM_EPOCHS} \
            --use_lora \
            --lora_r 32 \
            --lora_alpha 32 \
            --load_in_4bit \
            --gradient_accumulation_steps 8
        ;;

    "smolvla")
        # SmolVLA-450M
        MODEL_NAME="HuggingFaceTB/SmolVLA-450M"
        BATCH_SIZE=8
        LEARNING_RATE=1e-4
        NUM_EPOCHS=10

        python -m train.finetune.vla_finetuner \
            --model_name "${MODEL_NAME}" \
            --dataset "${DATASET}" \
            --output_dir "${OUTPUT_DIR}" \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --num_epochs ${NUM_EPOCHS} \
            --gradient_accumulation_steps 4
        ;;

    *)
        echo "Unknown model type: ${MODEL_TYPE}"
        echo "Supported: custom, openvla, smolvla"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Fine-tuning Complete!"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
