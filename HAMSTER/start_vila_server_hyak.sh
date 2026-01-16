#!/bin/bash
#
# VILA Server Startup Script for Hyak HPC
#
# Usage:
#   1. Get a GPU node first:
#      srun -p gpu-a40 -A escience --nodes=1 --cpus-per-task=32 --mem=400G --time=24:00:00 --gpus=1 --pty /bin/bash
#
#   2. Run this script:
#      ./start_vila_server_hyak.sh
#
# The server will be available at http://localhost:8000
#

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Model path
MODEL_PATH="$SCRIPT_DIR/Hamster_dev/VILA1.5-13b-robopoint_1432k+rlbench_all_tasks_256_1000_eps_sketch_v5_alpha+droid_train99_sketch_v5_alpha_fix+bridge_data_v2_train90_10k_sketch_v5_alpha-e1-LR1e-5"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please run: git clone https://huggingface.co/yili18/Hamster_dev"
    exit 1
fi

echo "=========================================="
echo "VILA Server (HAMSTER finetuned)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Port: 8000"
echo "Conv mode: vicuna_v1"
echo ""

# Check if Singularity module is loaded
if ! command -v singularity &> /dev/null; then
    echo "Loading Singularity module..."
    module load singularity
fi

# Singularity image path (using hamster-maniflow for PyTorch 2.5 + CUDA 13.0 compatibility)
SIF_PATH="/gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif"

if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: Singularity image not found at $SIF_PATH"
    exit 1
fi

echo "Singularity image: $SIF_PATH"
echo ""
echo "Starting VILA server..."
echo "=========================================="

# Set PYTHONPATH to include VILA repository (for llava package)
export PYTHONPATH="$SCRIPT_DIR/VILA:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Set HuggingFace cache directory
export HF_HOME="/gscratch/scrubbed/naoto03/.cache/huggingface"
export TRANSFORMERS_CACHE="/gscratch/scrubbed/naoto03/.cache/huggingface"

# Run server inside Singularity container
# PYTHONNOUSERSITE=1 prevents loading user-local packages (~/.local) to avoid version conflicts
singularity exec --nv \
    --bind /gscratch/:/gscratch/:rw \
    --env PYTHONPATH="/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila:$SCRIPT_DIR/VILA" \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    "$SIF_PATH" \
    python -W ignore "$SCRIPT_DIR/server.py" \
        --port 8000 \
        --model-path "$MODEL_PATH" \
        --conv-mode vicuna_v1
