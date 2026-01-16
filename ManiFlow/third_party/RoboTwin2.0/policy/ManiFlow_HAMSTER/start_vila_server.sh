#!/bin/bash
#
# Start a single VILA server for ManiFlow+HAMSTER evaluation.
#
# Usage:
#   ./start_vila_server.sh [GPU_ID] [PORT]
#
# Arguments:
#   GPU_ID  - GPU to use (default: 0)
#   PORT    - Port to run server on (default: 8000)
#
# Prerequisites:
#   - GPU available
#   - Singularity module loaded
#

set -e

# Arguments
GPU_ID=${1:-0}
PORT=${2:-8000}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Paths (same as parallel_vila)
PROJECT_DIR="/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration"
HAMSTER_DIR="$PROJECT_DIR/HAMSTER"
MODEL_PATH="$HAMSTER_DIR/Hamster_dev/VILA1.5-13b-robopoint_1432k+rlbench_all_tasks_256_1000_eps_sketch_v5_alpha+droid_train99_sketch_v5_alpha_fix+bridge_data_v2_train90_10k_sketch_v5_alpha-e1-LR1e-5"
SIF_PATH="/gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif"
SITE_PACKAGES_VILA="/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila"
HF_HOME="/gscratch/scrubbed/naoto03/.cache/huggingface"

# Log file
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vila_server_gpu${GPU_ID}_port${PORT}.log"

echo "============================================================"
echo "Starting VILA Server (Single GPU Mode)"
echo "============================================================"
echo "GPU: $GPU_ID"
echo "Port: $PORT"
echo "Model: $MODEL_PATH"
echo "Log: $LOG_FILE"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if Singularity image exists
if [ ! -f "$SIF_PATH" ]; then
    echo "ERROR: Singularity image not found at $SIF_PATH"
    exit 1
fi

# Check if Singularity is available
if ! command -v singularity &> /dev/null; then
    echo "Loading Singularity module..."
    module load singularity 2>/dev/null || true
fi

# Check if port is already in use
if curl -s --max-time 2 "http://localhost:$PORT/" > /dev/null 2>&1; then
    echo "WARNING: Port $PORT is already in use!"
    echo "Either a server is already running or stop it first."
    exit 1
fi

echo "Starting server..."
echo ""

# Start server in background
singularity exec --nv \
    --bind /gscratch/:/gscratch/:rw \
    --env PYTHONPATH="$SITE_PACKAGES_VILA:$HAMSTER_DIR/VILA" \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$HF_HOME" \
    --env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$SIF_PATH" \
    python -W ignore "$HAMSTER_DIR/server.py" \
        --port "$PORT" \
        --model-path "$MODEL_PATH" \
        --conv-mode vicuna_v1 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$SCRIPT_DIR/server_pid.txt"

echo "Server PID: $PID"
echo ""
echo "Waiting for server to initialize (this takes ~3-4 minutes)..."
echo ""

# Wait for server to be ready
MAX_WAIT=300  # 5 minutes max
WAIT_INTERVAL=5
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if server responds
    if curl -s --max-time 5 "http://localhost:$PORT/" > /dev/null 2>&1; then
        echo ""
        echo "============================================================"
        echo "Server READY!"
        echo "============================================================"
        echo "URL: http://localhost:$PORT/v1"
        echo "PID: $PID"
        echo ""
        echo "To stop the server:"
        echo "  ./stop_vila_server.sh"
        echo "  # or: kill $PID"
        echo ""
        echo "To check logs:"
        echo "  tail -f $LOG_FILE"
        echo "============================================================"
        exit 0
    fi

    # Check if process is still running
    if ! kill -0 $PID 2>/dev/null; then
        echo ""
        echo "ERROR: Server process died!"
        echo "Check log: $LOG_FILE"
        echo ""
        tail -50 "$LOG_FILE"
        exit 1
    fi

    sleep $WAIT_INTERVAL
    WAITED=$((WAITED + WAIT_INTERVAL))
    echo -n "."
done

echo ""
echo "ERROR: Timeout waiting for server!"
echo "Check log: $LOG_FILE"
exit 1
