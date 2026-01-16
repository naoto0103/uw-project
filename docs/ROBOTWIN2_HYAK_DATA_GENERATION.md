# RoboTwin 2.0 Data Generation on Hyak

This guide explains how to generate simulation data using RoboTwin 2.0 on the Hyak cluster.

## Prerequisites

- Hyak cluster access with GPU node (tested on RTX A6000, L40S)
- CUDA 12.4.1
- Conda environment `robotwin` already set up

## Quick Start

### 1. Activate Environment

```bash
# Load required modules and activate conda environment
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
module load cuda/12.4.1 gcc/13.2.0
```

### 2. Navigate to RoboTwin Directory

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0
```

### 3. Run Data Collection

```bash
python script/collect_data.py <task_name> <config_name>
```

**Example:**
```bash
# Generate 2 episodes for click_bell task
python script/collect_data.py click_bell test_2eps
```

## Available Tasks

The following tasks are available in `task_config/`:

| Task Name | Description |
|-----------|-------------|
| `click_bell` | Click a bell on the table |
| `blocks_stack` | Stack blocks |
| `container_sort` | Sort objects into containers |
| `diverse_bottles` | Manipulate various bottles |
| `dual_bottles` | Handle two bottles |
| `empty_cup` | Empty a cup |
| `mug_hanging` | Hang a mug |
| `pick_apple` | Pick up an apple |
| `shoe_place` | Place shoes |
| ... | (see `task_config/` for full list) |

## Configuration Files

### Task Configuration

Task configs are located in `task_config/<task_name>.yml`. Example structure:

```yaml
task_name: click_bell
episode_num: 100
head_camera_type: D435
wrist_camera_type: D435
embodiment_type: aloha-agilex
...
```

### Creating Custom Configs

To create a custom config (e.g., for testing with fewer episodes):

```bash
cp task_config/click_bell.yml task_config/click_bell_test.yml
# Edit episode_num and other parameters as needed
```

## Output Structure

Generated data is saved to `./dataset/test/<task_name>/<config_name>/`:

```
dataset/test/click_bell/test_2eps/
├── data/
│   ├── episode0.hdf5    # Observation and action data
│   └── episode1.hdf5
├── video/
│   ├── episode0.mp4     # Visualization video
│   └── episode1.mp4
├── _traj_data/          # Raw trajectory data
├── scene_info.json      # Scene configuration
└── seed.txt             # Random seeds used
```

### HDF5 Data Format

Each `episodeX.hdf5` contains:

- `observation/`
  - `head_camera/rgb` - Head camera RGB images
  - `head_camera/depth` - Head camera depth images
  - `wrist_camera/rgb` - Wrist camera RGB images
  - `wrist_camera/depth` - Wrist camera depth images
  - `joint_positions` - Robot joint positions
  - `ee_pose` - End-effector pose
- `action/` - Action commands

## Batch Data Generation

For generating large datasets, use a SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=robotwin_data
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G

source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
module load cuda/12.4.1 gcc/13.2.0

cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0

# Generate data for multiple tasks
for task in click_bell blocks_stack container_sort; do
    python script/collect_data.py $task default
done
```

## Troubleshooting

### Common Issues

1. **Vulkan ICD Warning**
   ```
   UserWarning: Failed to find Vulkan ICD file
   ```
   This is expected on headless nodes and doesn't affect functionality.

2. **Missing pytorch3d**
   ```
   missing pytorch3d
   ```
   This warning can be ignored for data collection.

3. **CUDA Version Mismatch Warning**
   ```
   UserWarning: The detected CUDA version...
   ```
   This is expected due to our cpp_extension patch and is safe to ignore.

### Environment Issues

If you encounter environment issues, verify the setup:

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check SAPIEN
python -c "import sapien; print(f'SAPIEN: {sapien.__version__}')"

# Check ffmpeg encoder
ffmpeg -encoders 2>/dev/null | grep x264
```

## Environment Setup Details

The `robotwin` conda environment includes:

- Python 3.10
- PyTorch 2.6.0+cu124
- SAPIEN 3.0.0b1
- mplib 0.2.1
- CuRobo (source build)
- ffmpeg 7.1.0 (with libx264)

### Key Patches Applied

1. **cpp_extension.py**: CUDA version mismatch downgraded to warning
2. **sapien/urdf_loader.py**: Fixed `.srdf` extension bug
3. **mplib/planner.py**: Relaxed collision check for data generation
4. **curobo configs**: `${ASSETS_PATH}` replaced with absolute paths

## File Locations

| Component | Path |
|-----------|------|
| RoboTwin Code | `/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0` |
| Task Configs | `./task_config/` |
| Assets | `./assets/` |
| Output Data | `./dataset/` |
| Conda Env | `/gscratch/scrubbed/naoto03/miniconda3/envs/robotwin` |

## References

- [RoboTwin GitHub](https://github.com/TianxingChen/RoboTwin)
- [SAPIEN Documentation](https://sapien.ucsd.edu/)
- [CuRobo Documentation](https://curobo.org/)
