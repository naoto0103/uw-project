# RoboTwin Overview: 1.0 vs 2.0

**Last Updated**: 2025-11-20

---

## Summary

RoboTwin is a simulation framework for dual-arm robotic manipulation benchmarking. The project has evolved from version 1.0 (ECCV Workshop 2024 Best Paper) to version 2.0 (CVPR 2025 Highlight), with significant improvements in scalability and automation.

---

## RoboTwin 1.0 (Current in Project)

### Overview
- **Location**: `/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin1.0/`
- **Status**: Early version, ECCV Workshop 2024 Best Paper
- **Release**: September 2024
- **Repository**: https://github.com/TianxingChen/RoboTwin (migrated)

### Simulation Engine
**SAPIEN 3.0.0b1** - Physics simulation platform

### Key Features
- **16 Dual-Arm Tasks**: Manual expert data collection
- **Baseline Policies**: Diffusion Policy (DP), 3D Diffusion Policy (DP3)
- **Camera Support**: Intel RealSense L515 (LiDAR), D455, ZED 2i
- **Data Format**: Zarr-based efficient storage

### Tasks (16 Total)

| Task Name | `${task_name}` | Complexity |
|-----------|----------------|------------|
| Apple Cabinet Storage | `apple_cabinet_storage` | Medium |
| Block Hammer Beat | `block_hammer_beat` | Easy |
| Block Handover | `block_handover` | Medium |
| Blocks Stack (Easy) | `blocks_stack_easy` | Easy |
| Blocks Stack (Hard) | `blocks_stack_hard` | Hard |
| Bottle Adjust | `bottle_adjust` | Medium |
| Container Place | `container_place` | Medium |
| Diverse Bottles Pick | `diverse_bottles_pick` | Medium |
| Dual Bottles Pick (Easy) | `dual_bottles_pick_easy` | Easy |
| Dual Bottles Pick (Hard) | `dual_bottles_pick_hard` | Hard |
| Dual Shoes Place | `dual_shoes_place` | Medium |
| Empty Cup Place | `empty_cup_place` | Easy |
| Mug Hanging (Easy) | `mug_hanging_easy` | Medium |
| Mug Hanging (Hard) | `mug_hanging_hard` | Hard |
| Pick Apple Messy | `pick_apple_messy` | Easy |
| Shoe Place | `shoe_place` | Easy |

### Installation (RoboTwin 1.0)

```bash
# Create environment
conda create -n RoboTwin python=3.8
conda activate RoboTwin

# Install dependencies
pip install torch==2.4.1 torchvision sapien==3.0.0b1 scipy==1.10.1 \
    mplib==0.1.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 \
    imageio==2.34.2 pydantic zarr openai huggingface_hub==0.25.0

# Install pytorch3d
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..

# Download assets
python ./script/download_asset.py
unzip aloha_urdf.zip && unzip main_models.zip

# Modify mplib (required)
# See INSTALLATION.md for details
```

### Usage (RoboTwin 1.0)

#### Data Collection
```bash
# Collect 100 trajectories for a task
bash run_task.sh ${task_name} ${gpu_id}

# Example
bash run_task.sh pick_apple_messy 0
```

#### Policy Training
```bash
# Diffusion Policy (DP)
cd policy/Diffusion-Policy
bash train.sh ${task_name} ${camera_type} ${data_num} ${seed} ${gpu_id}

# 3D Diffusion Policy (DP3)
cd policy/3D-Diffusion-Policy
bash train.sh ${task_name} ${camera_type} ${data_num} ${seed} ${gpu_id}
```

#### Policy Evaluation
```bash
# Modify script/eval_policy.py to load your model
bash script/run_eval_policy.sh ${task_name} ${gpu_id}
```

### Data Format

```python
# Zarr structure
dataset/
├── data/
│   ├── head_camera/     # [N_episodes, T, 3, H, W] RGB images
│   ├── point_cloud/     # [N_episodes, T, N_points, 3/6] XYZ or XYZRGB
│   ├── state/           # [N_episodes, T, 14] robot state
│   └── action/          # [N_episodes, T, 14] robot actions
└── meta/
    └── episode_ends     # [N_episodes] episode lengths
```

---

## RoboTwin 2.0 (Next Generation)

### Overview
- **Status**: CVPR 2025 Highlight paper
- **Release**: June 2025 (arXiv 2506.18088)
- **Repository**: https://github.com/RoboTwin-Platform/RoboTwin
- **Website**: https://robotwin-platform.github.io/

### Major Improvements

#### 1. Scale
- **Tasks**: 16 → **50** dual-arm manipulation tasks
- **Object Library**: RoboTwin-OD with **731 instances** across **147 categories**
- **Pre-collected Data**: Over **100,000 trajectories** (HuggingFace)

#### 2. Automation
- **LLM-based Code Generation**: Automatic expert data collector synthesis
- **Simulation-in-the-loop Refinement**: Iterative improvement
- **10.9% improvement** in code generation success rate

#### 3. Domain Randomization
Structured randomization across **5 axes**:
- **Clutter**: Object arrangement variations
- **Lighting**: Illumination changes
- **Background**: Scene variations
- **Tabletop Height**: Workspace adjustments
- **Language**: Task description variations

#### 4. Multi-Embodiment Support
**5 Robot Embodiments**:
- Dual-arm configurations
- AgileX Robotics hardware support
- Cross-embodiment evaluation

#### 5. Expanded Baselines
**10+ Policy Architectures**:
- DP (Diffusion Policy)
- ACT (Action Chunking Transformer)
- DP3 (3D Diffusion Policy)
- RDT (Robotic Diffusion Transformer)
- PI0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA

### Performance Gains

| Training Data | Improvement |
|--------------|-------------|
| Synthetic + 10 real demos | **+367%** vs 10-demo baseline |
| Synthetic only (zero-shot) | **+228%** vs baseline |

### Installation (RoboTwin 2.0)

```bash
# Create environment
conda create -n RoboTwin python=3.8
conda activate RoboTwin

# Install core dependencies
pip install torch==2.3.1 sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 \
    gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 \
    pydantic openai gdown

# Follow official documentation for:
# - mplib library modifications
# - Asset downloads
# - Policy baseline installations
```

### Usage (RoboTwin 2.0)

#### Data Collection
```bash
# Collect data for a specific task
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
```

#### Control Robot
- **Endpose Control Mode**: Updated July 2025
- Documentation: https://robotwin-platform.github.io/doc/usage/control-robot.html

#### Pre-collected Data
- Available on Hugging Face
- 100,000+ trajectories
- Custom data collection recommended for task-specific configurations

---

## Comparison: RoboTwin 1.0 vs 2.0

| Feature | RoboTwin 1.0 | RoboTwin 2.0 |
|---------|-------------|-------------|
| **Tasks** | 16 | 50 |
| **Object Library** | Limited | 731 instances, 147 categories |
| **Data Collection** | Manual | Automated (LLM-based) |
| **Domain Randomization** | Basic | 5-axis structured |
| **Embodiments** | 1-2 | 5 |
| **Baselines** | DP, DP3 | 10+ (DP, ACT, RDT, VLAs) |
| **Pre-collected Data** | N/A | 100,000+ trajectories |
| **Publication** | ECCV Workshop 2024 | CVPR 2025 Highlight |
| **Status in Project** | ✅ Installed | ⏳ Not yet installed |

---

## Integration with ManiFlow

### Current (RoboTwin 1.0)
- **Location**: `ManiFlow/third_party/RoboTwin1.0/`
- **Used for**: Phase 2 data collection (6 tasks × 50 episodes)
- **Tasks Used**:
  - pick_apple_messy
  - diverse_bottles_pick
  - dual_bottles_pick_hard
  - empty_cup_place
  - block_hammer_beat
  - shoe_place

### Potential (RoboTwin 2.0)
- **Video Frame Testing**: Ideal for evaluating Qwen3 path generation robustness
- **Expanded Tasks**: 50 tasks for comprehensive evaluation
- **Domain Randomization**: Better simulation-to-real transfer
- **Automated Data Generation**: Faster experimentation

---

## Next Steps: Video-based Path Generation Testing

### Objective
Test Qwen3-VL path generation on video frames from RoboTwin to evaluate:
1. **Temporal Consistency**: Path stability across frames
2. **Position Robustness**: Adaptation to object/robot movement
3. **Dynamic Scene Understanding**: Response to changing environments

### Approach Options

#### Option 1: Use RoboTwin 1.0 (Current)
**Pros**:
- Already installed
- Known tasks
- Existing data (6 tasks × 50 episodes)

**Cons**:
- Limited domain randomization
- 16 tasks only

#### Option 2: Upgrade to RoboTwin 2.0
**Pros**:
- 50 tasks
- Strong domain randomization (ideal for robustness testing)
- Modern baselines

**Cons**:
- Installation required (~20 min)
- New codebase
- Migration effort

### Recommendation

**Start with RoboTwin 1.0** for initial video testing:
1. Use existing zarr datasets (`head_camera` frames)
2. Extract video sequences
3. Apply Qwen3 VERSION 18 prompt per frame
4. Analyze temporal consistency

**Upgrade to RoboTwin 2.0** if:
- Need more diverse tasks
- Require stronger domain randomization
- Want automated data generation

---

## Technical Details

### Simulation Engine
**SAPIEN 3.0.0b1**:
- Physics-based simulation
- Ray tracing support (NVIDIA RTX)
- Vulkan rendering
- GPU acceleration

### Camera Types
- **L515**: Intel RealSense LiDAR L515 (default)
- **D455**: Intel RealSense D455 RGB-D
- **ZED 2i**: Stereo camera

### Observation Modalities
- **RGB Images**: [3, H, W]
- **Point Clouds**: [N, 3] or [N, 6] (XYZ or XYZRGB)
- **Depth Maps**: Available via RGB-D cameras
- **Robot State**: 14-DoF (joint positions)

### Action Space
- **Dimensions**: 14-DoF
- **Format**: Joint positions or end-effector poses
- **Control**: Position control or velocity control

---

## Links

### RoboTwin 1.0
- **Repository**: https://github.com/TianxingChen/RoboTwin (early_version branch)
- **Paper**: https://arxiv.org/abs/2409.02920
- **Website**: https://robotwin-benchmark.github.io/early-version

### RoboTwin 2.0
- **Repository**: https://github.com/RoboTwin-Platform/RoboTwin
- **Paper**: https://arxiv.org/abs/2506.18088
- **Website**: https://robotwin-platform.github.io/
- **Challenge**: https://robotwin-benchmark.github.io/cvpr-2025-challenge/

---

## Related Projects
- **G3Flow**: Uses 5 RoboTwin tasks for benchmarking (arXiv 2411.18369)
- **ARIO Dataset**: Real robot data via teleoperation (https://ario-dataset.github.io/)
- **Deemos Rodin**: Digital twin generation (https://hyperhuman.deemos.com/rodin)
