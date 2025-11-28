#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Store IP address
ifconfig eth0 | grep 'inet ' | awk '{print $2}' > ip_eth0.txt
echo "IP address saved to ip_eth0.txt"

# Clone the model from Hugging Face (if not already cloned)
if [ ! -d "Hamster_dev" ]; then
    git lfs install
    git clone https://huggingface.co/yili18/Hamster_dev
fi

# Find the actual model directory (it's a subdirectory of Hamster_dev)
MODEL_PATH=$(ls -d Hamster_dev/VILA* 2>/dev/null | head -1)
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model directory not found in Hamster_dev/"
    exit 1
fi

echo "Using model: $MODEL_PATH"

# Run our custom server
python -W ignore server.py \
    --port 8000 \
    --model-path "$MODEL_PATH" \
    --conv-mode vicuna_v1
