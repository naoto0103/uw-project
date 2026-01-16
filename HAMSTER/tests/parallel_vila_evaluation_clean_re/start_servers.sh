#!/bin/bash
#
# Start 4 VILA servers on different GPUs (ports 8008-8011)
#
# Usage:
#   ./start_servers.sh
#
# Prerequisites:
#   - 4 GPUs available (CUDA_VISIBLE_DEVICES will be set per server)
#   - Singularity module loaded or available
#

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HAMSTER_DIR="$SCRIPT_DIR/../.."

# Configuration
NUM_GPUS=4
BASE_PORT=8008
MODEL_PATH="$HAMSTER_DIR/Hamster_dev/VILA1.5-13b-robopoint_1432k+rlbench_all_tasks_256_1000_eps_sketch_v5_alpha+droid_train99_sketch_v5_alpha_fix+bridge_data_v2_train90_10k_sketch_v5_alpha-e1-LR1e-5"
SIF_PATH="/gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif"
SITE_PACKAGES_VILA="/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila"
HF_HOME="/gscratch/scrubbed/naoto03/.cache/huggingface"

# Log directory
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# PID file to track server processes
PID_FILE="$SCRIPT_DIR/server_pids.txt"

echo "============================================================"
echo "Starting $NUM_GPUS VILA Servers (Parallel Mode)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Ports: $BASE_PORT - $((BASE_PORT + NUM_GPUS - 1))"
echo "Log directory: $LOG_DIR"
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

# Clear old PID file
> "$PID_FILE"

# Start servers
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_ID))
    LOG_FILE="$LOG_DIR/server_gpu${GPU_ID}_port${PORT}.log"

    echo "Starting server on GPU $GPU_ID, port $PORT..."
    echo "  Log: $LOG_FILE"

    # Start server in background
    # Use --env to pass CUDA_VISIBLE_DEVICES to the container
    # PORT is expanded here before singularity exec runs
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

    # Save PID
    PID=$!
    echo "$PID $GPU_ID $PORT" >> "$PID_FILE"
    echo "  PID: $PID"
done

echo ""
echo "============================================================"
echo "Waiting for servers to initialize..."
echo "============================================================"

# Wait for all servers to be ready
MAX_WAIT=300  # 5 minutes max wait (model loading takes ~3-4 min per GPU)
WAIT_INTERVAL=5

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_ID))
    echo -n "Checking GPU $GPU_ID (port $PORT)..."

    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        # Try to connect to the server (check if it responds to any request)
        if curl -s --max-time 5 "http://localhost:$PORT/" > /dev/null 2>&1; then
            echo " Ready!"
            break
        fi

        # Check if process is still running
        PID=$(grep " $GPU_ID $PORT" "$PID_FILE" | cut -d' ' -f1)
        if [ -n "$PID" ] && ! kill -0 $PID 2>/dev/null; then
            echo " FAILED (process died)"
            echo "Check log: $LOG_DIR/server_gpu${GPU_ID}_port${PORT}.log"
            break
        fi

        sleep $WAIT_INTERVAL
        WAITED=$((WAITED + WAIT_INTERVAL))
        echo -n "."
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo " TIMEOUT"
        echo "Check log: $LOG_DIR/server_gpu${GPU_ID}_port${PORT}.log"
    fi
done

echo ""
echo "============================================================"
echo "Server Status Summary"
echo "============================================================"

READY_COUNT=0
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_ID))
    PID=$(grep " $GPU_ID $PORT" "$PID_FILE" | cut -d' ' -f1)

    if curl -s --max-time 5 "http://localhost:$PORT/" > /dev/null 2>&1; then
        echo "  GPU $GPU_ID (port $PORT): READY (PID: $PID)"
        READY_COUNT=$((READY_COUNT + 1))
    else
        echo "  GPU $GPU_ID (port $PORT): NOT READY (PID: $PID)"
    fi
done

echo ""
echo "$READY_COUNT / $NUM_GPUS servers ready"

if [ $READY_COUNT -eq $NUM_GPUS ]; then
    echo ""
    echo "All servers ready! You can now run:"
    echo "  python generate_paths.py --episodes 50"
else
    echo ""
    echo "WARNING: Not all servers are ready. Check logs in $LOG_DIR"
fi

echo ""
echo "To stop all servers:"
echo "  ./stop_servers.sh"
echo "============================================================"
