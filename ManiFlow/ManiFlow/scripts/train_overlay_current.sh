#!/bin/bash
# Train ManiFlow with RoboTwin 2.0 Overlay Images (Current Only)
#
# This is for conditions 2 and 5 (path guidance without Memory Function).
#
# Usage:
#   ./train_overlay_current.sh [env] [task] [gpu_id] [seed]
#
# Example:
#   ./train_overlay_current.sh clean beat_block_hammer 0 42
#   ./train_overlay_current.sh cluttered beat_block_hammer 1 123

set -e

ENV=${1:-clean}
TASK=${2:-beat_block_hammer}
GPU_ID=${3:-0}
SEED=${4:-42}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFLOW_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$MANIFLOW_DIR/maniflow/workspace"

# Check if Zarr file exists
ZARR_PATH="$MANIFLOW_DIR/data/zarr/${ENV}_overlay_${TASK}.zarr"
if [ ! -d "$ZARR_PATH" ]; then
    echo "ERROR: Zarr file not found at $ZARR_PATH"
    echo "Run convert_overlay_to_zarr.py first!"
    exit 1
fi

# Create log directory
LOG_DIR="$MANIFLOW_DIR/data/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_overlay_current_${ENV}_${TASK}_seed${SEED}_${TIMESTAMP}.log"

echo "=============================================="
echo "Training ManiFlow with Overlay Images (Current Only)"
echo "=============================================="
echo "Environment: $ENV"
echo "Task: $TASK"
echo "GPU: $GPU_ID"
echo "Seed: $SEED"
echo "Zarr: $ZARR_PATH"
echo "Config: maniflow_overlay_current_robotwin2"
echo "Log: $LOG_FILE"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

cd "$WORKSPACE_DIR"

# Run training with unbuffered output, tee to both console and log file
# Output dir includes env, task, and seed for easy identification
OUTPUT_DIR="data/outputs/${ENV}_${TASK}/overlay_current_seed${SEED}"

python -u train_maniflow_robotwin_workspace.py \
    --config-name=maniflow_overlay_current_robotwin2 \
    training.seed=$SEED \
    training.device="cuda:0" \
    robotwin_task.dataset.zarr_path="$ZARR_PATH" \
    hydra.run.dir="$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "=============================================="
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
echo "=============================================="
