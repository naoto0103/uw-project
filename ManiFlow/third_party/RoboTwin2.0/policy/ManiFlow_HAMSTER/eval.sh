#!/bin/bash
#
# Unified evaluation script for ManiFlow + HAMSTER on RoboTwin 2.0
#
# Usage:
#   ./eval.sh --task <task_name> --mode <mode> --env <env> [options]
#
# Examples:
#   # Condition 1: cluttered + original (baseline, no VILA)
#   ./eval.sh --task click_bell --mode original --env cluttered --seed 42
#
#   # Condition 2: cluttered + overlay current
#   ./eval.sh --task click_bell --mode current --env cluttered --seed 42
#
#   # Condition 3: cluttered + overlay initial+current (Memory Function)
#   ./eval.sh --task click_bell --mode initial_current --env cluttered --seed 42
#
#   # Condition 4-6: clean training data evaluated on cluttered
#   ./eval.sh --task click_bell --mode original --env clean --seed 42
#
# Prerequisites:
#   - For modes 'current' and 'initial_current': VILA server must be running
#     ./start_vila_server.sh 0 8000
#   - GPU node with enough VRAM (26GB for VILA + ManiFlow inference)
#

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROBOTWIN_DIR="$SCRIPT_DIR/../.."

# Default values
TASK=""
MODE=""
ENV=""
EVAL_ENV="cluttered"  # Evaluation environment: cluttered (default) or clean
SEED=42
EPISODES=100
CHECKPOINT=""
VILA_PORT=8000
OUTPUT_DIR="$SCRIPT_DIR/eval_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --env)
            ENV="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --vila_port)
            VILA_PORT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eval_env)
            EVAL_ENV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --task <task> --mode <mode> --env <env> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --task TASK       Task name (click_bell, turn_switch, move_can_pot,"
            echo "                    open_microwave, adjust_bottle, beat_block_hammer)"
            echo "  --mode MODE       Evaluation mode (original, current, initial_current)"
            echo "  --env ENV         Training data environment (clean, cluttered)"
            echo ""
            echo "Optional arguments:"
            echo "  --seed SEED       Evaluation seed (default: 42)"
            echo "  --episodes N      Number of episodes (default: 100)"
            echo "  --checkpoint PATH Path to checkpoint (default: auto-find epoch 500)"
            echo "  --vila_port PORT  VILA server port (default: 8000)"
            echo "  --output_dir DIR  Output directory (default: ./eval_results)"
            echo "  --eval_env ENV    Evaluation environment: cluttered or clean (default: cluttered)"
            echo ""
            echo "Modes:"
            echo "  original         Raw RGB images, no VILA (conditions 1,4)"
            echo "  current          Current overlay only (conditions 2,5)"
            echo "  initial_current  Initial + current overlay (conditions 3,6)"
            echo ""
            echo "Examples:"
            echo "  # Condition 1: cluttered training, original mode"
            echo "  $0 --task click_bell --mode original --env cluttered"
            echo ""
            echo "  # Condition 6: clean training, initial+current mode"
            echo "  $0 --task click_bell --mode initial_current --env clean"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$TASK" ]; then
    echo "ERROR: --task is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "ERROR: --mode is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$ENV" ]; then
    echo "ERROR: --env is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate mode
if [[ "$MODE" != "original" && "$MODE" != "current" && "$MODE" != "initial_current" ]]; then
    echo "ERROR: Invalid mode '$MODE'. Must be one of: original, current, initial_current"
    exit 1
fi

# Validate env (training data)
if [[ "$ENV" != "clean" && "$ENV" != "cluttered" ]]; then
    echo "ERROR: Invalid env '$ENV'. Must be one of: clean, cluttered"
    exit 1
fi

# Validate eval_env (evaluation environment)
if [[ "$EVAL_ENV" != "clean" && "$EVAL_ENV" != "cluttered" ]]; then
    echo "ERROR: Invalid eval_env '$EVAL_ENV'. Must be one of: clean, cluttered"
    exit 1
fi

# Select task_config based on eval_env
if [[ "$EVAL_ENV" == "cluttered" ]]; then
    TASK_CONFIG="single_arm_eval_cluttered"
else
    TASK_CONFIG="single_arm_eval_clean"
fi

# Validate task
VALID_TASKS="click_bell turn_switch move_can_pot open_microwave adjust_bottle beat_block_hammer"
if [[ ! " $VALID_TASKS " =~ " $TASK " ]]; then
    echo "ERROR: Invalid task '$TASK'"
    echo "Valid tasks: $VALID_TASKS"
    exit 1
fi

# Check VILA server for overlay modes
if [[ "$MODE" != "original" ]]; then
    if ! curl -s --max-time 2 "http://localhost:$VILA_PORT/" > /dev/null 2>&1; then
        echo "ERROR: VILA server not responding on port $VILA_PORT"
        echo ""
        echo "For modes 'current' and 'initial_current', VILA server is required."
        echo "Start the server first:"
        echo "  ./start_vila_server.sh 0 $VILA_PORT"
        exit 1
    fi
    echo "VILA server: http://localhost:$VILA_PORT (connected)"
fi

# Determine condition number
if [[ "$ENV" == "cluttered" ]]; then
    case $MODE in
        original) CONDITION_NUM=1 ;;
        current) CONDITION_NUM=2 ;;
        initial_current) CONDITION_NUM=3 ;;
    esac
else
    case $MODE in
        original) CONDITION_NUM=4 ;;
        current) CONDITION_NUM=5 ;;
        initial_current) CONDITION_NUM=6 ;;
    esac
fi

# Create log directory
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/eval_${TASK}_condition${CONDITION_NUM}_seed${SEED}_${TIMESTAMP}.log"

echo "============================================================"
echo "ManiFlow + HAMSTER Evaluation"
echo "============================================================"
echo "Task:        $TASK"
echo "Mode:        $MODE"
echo "Train Env:   $ENV (training data)"
echo "Eval Env:    $EVAL_ENV (evaluation environment)"
echo "Condition:   $CONDITION_NUM"
echo "Seed:        $SEED"
echo "Episodes:    $EPISODES"
echo "Output:      $OUTPUT_DIR"
echo "Log:         $LOG_FILE"
echo ""

# Build checkpoint path argument
CHECKPOINT_ARG=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARG="--checkpoint_path $CHECKPOINT"
fi

# Change to RoboTwin directory
cd "$ROBOTWIN_DIR"

# Run evaluation with tee for real-time logging
python -u script/eval_policy.py \
    --config policy/ManiFlow_HAMSTER/deploy_policy.yml \
    --overrides \
    --task_name "$TASK" \
    --task_config "$TASK_CONFIG" \
    --mode "$MODE" \
    --env "$ENV" \
    --eval_env "$EVAL_ENV" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --vila_server_url "http://localhost:$VILA_PORT/v1" \
    $CHECKPOINT_ARG \
    2>&1 | tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOG_FILE")

# Print completion message
echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR/$TASK/condition${CONDITION_NUM}_${ENV}_*"
echo "Log file: $LOG_FILE"
