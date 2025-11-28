
#!/bin/bash
# Example usage:
# bash scripts/gen_demonstrations_robotwin1.0.sh pick_apple_messy 0
# bash scripts/gen_demonstrations_robotwin1.0.sh diverse_bottles_pick 0
# bash scripts/gen_demonstrations_robotwin1.0.sh dual_bottles_pick_hard 0

cd third_party/RoboTwin1.0

task_name=${1}
gpu_id=${2}
num_demos=50  # Number of demonstrations to generate
camera_type="D435"  # Camera type, e.g., D435

export CUDA_VISIBLE_DEVICES=${gpu_id}

# Generate demonstrations
echo "Generating demonstrations for task: ${task_name} on GPU: ${gpu_id}"
echo ${task_name} | python script/run_task.py ${task_name} ${camera_type} ${num_demos}

# Process the generated data into zarr format
echo "Processing generated data into zarr format for task: ${task_name}"
python script/pkl2zarr_maniflow.py ${task_name} ${camera_type} ${num_demos}