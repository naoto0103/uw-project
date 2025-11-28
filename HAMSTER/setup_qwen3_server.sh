#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Store IP address
ifconfig eth0 | grep 'inet ' | awk '{print $2}' > ip_eth0.txt
echo "IP address saved to ip_eth0.txt"

# Set default model path
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"

echo "Using model: $MODEL_PATH"
echo "Model will be auto-downloaded from Hugging Face if not cached"

# Activate qwen3 conda environment
echo "Activating qwen3 conda environment..."
source /home/naoto/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

# Run Qwen3 server
echo "Starting Qwen3-VL server on port 8001..."
python -W ignore server_qwen3.py \
    --model-path "$MODEL_PATH" \
    --port 8001 \
    --host 0.0.0.0
