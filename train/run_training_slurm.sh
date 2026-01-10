#!/bin/bash
#SBATCH --job-name=vla_train
#SBATCH --comment="VLA Model Training"
#SBATCH --partition=hopper
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=8G

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

# ##################
# echo "Make symlink for libcuda.so.1"
# mkdir -p ~/lib
# ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 ~/lib/libcuda.so 2>/dev/null || true

# # ld wrapper script creation
# echo "Creating ld wrapper script..."
# if [ ! -d ~/bin ]; then
#     echo "Creating ~/bin directory..."
#     mkdir -p ~/bin
# fi

# echo "Writing ld wrapper to ~/bin/ld..."
# cat > ~/bin/ld << 'EOF'
# #!/bin/bash
# /usr/bin/ld -L"$HOME"/lib "$@"
# EOF
# chmod +x ~/bin/ld
# export PATH=~/bin:$PATH

# # Attempt to locate libcuda.so or libcuda.so.1
# libcuda_path=$(find /usr -name 'libcuda.so*' 2>/dev/null | head -n 1)
# if [ -z "$libcuda_path" ]; then
#   echo "libcuda.so not found. CUDA may not be installed or GPU drivers missing."
#   exit 1
# fi

# lib_dir=$(dirname "$libcuda_path")

# # Export the path for this session
# export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
# echo "LD_LIBRARY_PATH updated: $LD_LIBRARY_PATH"

# ############################

# Huggingface cache directory
export HF_HOME=/purestorage/AILAB/AI_2/youhans/huggingface

# Configuration
MODEL_TYPE=${1:-"custom"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"

echo "================================"
echo "VLA Training Script (SLURM)"
echo "Model: ${MODEL_TYPE}"
echo "Output: ${OUTPUT_DIR}"
echo "================================"

# Calculate world size
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
echo "World size: $NUM_NODES x $GPUS_PER_NODE = $WORLD_SIZE"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "$nodes"

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_port=29500

# NCCL network configuration (for Infiniband)
export NCCL_SOCKET_IFNAME=eno1

echo "****Starting HEAD at $head_node, $head_node_ip:$head_port"

# Select training script and arguments based on model type
if [ "$MODEL_TYPE" == "custom" ]; then
    TRAIN_SCRIPT="train_vla.py"
    TRAIN_ARGS="--vision_model google/siglip-base-patch16-224 \
        --llm_model Qwen/Qwen2-1.5B-Instruct \
        --dataset_name lerobot/pusht \
        --action_dim 7 \
        --freeze_vision \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --num_epochs 10 \
        --gradient_accumulation_steps 4 \
        --logging_steps 10 \
        --save_steps 500"

elif [ "$MODEL_TYPE" == "openvla" ]; then
    TRAIN_SCRIPT="train.py"
    TRAIN_ARGS="--model_name_or_path openvla/openvla-7b \
        --dataset_name berkeley-autolab/bridge_data_v2 \
        --use_lora True \
        --lora_r 32 \
        --lora_alpha 32 \
        --load_in_4bit True \
        --output_dir ${OUTPUT_DIR} \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --warmup_ratio 0.03 \
        --logging_steps 10 \
        --save_steps 500 \
        --bf16 True \
        --tf32 True \
        --dataloader_num_workers 4 \
        --report_to wandb"

elif [ "$MODEL_TYPE" == "smolvla" ]; then
    TRAIN_SCRIPT="train_smolvla.py"
    TRAIN_ARGS="--model_name HuggingFaceTB/SmolVLA-450M \
        --dataset_name lerobot/pusht \
        --output_dir ${OUTPUT_DIR} \
        --learning_rate 1e-4 \
        --batch_size 8 \
        --num_epochs 10 \
        --gradient_accumulation_steps 4"

else
    echo "Unknown model type: ${MODEL_TYPE}"
    echo "Usage: sbatch run_training_slurm.sh [openvla|smolvla|custom]"
    exit 1
fi

# Start head node (rank 0)
srun --nodes=1 --ntasks=1 -w "$head_node" \
    accelerate launch \
        --config_file ${SCRIPT_DIR}/deepspeed_zero3.yaml \
        --num_machines "$NUM_NODES" \
        --num_processes "$WORLD_SIZE" \
        --main_process_ip "$head_node_ip" \
        --main_process_port "$head_port" \
        --machine_rank 0 \
    ${SCRIPT_DIR}/${TRAIN_SCRIPT} ${TRAIN_ARGS} &
sleep 15

# Start worker nodes (rank 1 to N-1)
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "****Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
      accelerate launch \
        --config_file ${SCRIPT_DIR}/deepspeed_zero3.yaml \
        --num_machines "$NUM_NODES" \
        --num_processes "$WORLD_SIZE" \
        --main_process_ip "$head_node_ip" \
        --main_process_port "$head_port" \
        --machine_rank "$i" \
      ${SCRIPT_DIR}/${TRAIN_SCRIPT} ${TRAIN_ARGS} &
    sleep 5
done

# Wait for all background processes to complete
wait

echo "================================"
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "================================"
