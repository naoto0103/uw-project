# ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training

<a href="https://maniflow-policy.github.io/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/pdf/2509.01819"><strong>Paper</strong></a>
  |
  <a href="https://x.com/GeYan_21/status/1963638893649301678"><strong>Twitter</strong></a> | <a href="Place Holder"><strong>YouTube</strong></a>

  <a href="https://geyan21.github.io/">Ge Yan</a>, 
  <a href="https://jiyuezh.github.io/">Jiyue Zhu*</a>, 
  <a href="https://www.linkedin.com/in/yuquand/">Yuquan Deng*</a>, 
  <a href="https://aaronyang1223.github.io/">Shiqi Yang</a>, 
  <a href="https://rogerqi.github.io/">Ri-Zhao Qiu</a>, 
  <a href="https://chengxuxin.github.io/">Xuxin Cheng</a>, 
  <a href="https://memmelma.github.io/">Marius Memmel</a>, 
  <a href="https://ranjaykrishna.com/index.html/">Ranjay Krishna†</a>, 
  <a href="https://imankgoyal.github.io/">Ankit Goyal†</a>, 
  <a href="https://xiaolonw.github.io/">Xiaolong Wang†</a>, 
  <a href="https://homes.cs.washington.edu/~fox/">Dieter Fox†</a>


**Conference on Robot Learning (CoRL) 2025**

<div align="center">
  <img src="assets/maniflow_figure.png" alt="maniflow" width="100%">
</div>

---

## 🔥 Overview

ManiFlow is a visuomotor imitation learning policy for general robot manipulation that generates precise, high-dimensional, and dexterous actions from visual, language, and proprioceptive inputs. It features a general architecture supporting both 2D and 3D inputs, combines flow matching with consistency training for 1-2 step inference, and demonstrates strong scaling behavior and generalization across diverse robots and environments.

### Key Features
- **Superior Performance** across 60+ simulation tasks on 4 benchmarks, **98.3% improvement** on 8 real-world tasks with **strong generalization** to novel objects, backgrounds, and cluttered scenes.

- **Diverse Robot Embodiments**: Single policy works across diverse platforms with increasing dexterity - **humanoid robots** (Unitree H1), **bimanual systems** (dual xArm), and **single-arm setups** (Franka Panda)

- **Universal Policy Architecture**: Unified framework handles both **2D RGB images** and **3D point clouds**.

- **Strong Scaling Capability**: Shows superior data efficiency and scaling performance, achieving **99.7%** success on a single task with 500 demos while outperforming large-scale pre-trained π0 model.

- **Few-Step Inference**: Generates precise **dexterous** actions in just **1-2 inference steps** vs. 10+ steps for diffusion baselines through consistency flow training in a **single run without pre-trained teacher models**.

- **General Applicability**: ManiFlow's consistency flow training objective can be seamlessly integrated into existing VLA model by replacing the diffusion loss, enabling faster inference and improved performance.

---

## ✨ News

* ❗**2025-09-03, Released ManiFlow**, a general robot manipulation policy via consistency flow training. Check out the [Project Page](https://maniflow-policy.github.io/) and [Paper](https://arxiv.org/pdf/2509.01819v1).

---

## 📝 Upcoming Features

* 🐳 **Docker containerization** for simplified environment setup
* 🔄 **RoboTwin 2.0 support** with enhanced task diversity and domain randomization
* 🤖 **Real-Time Execution** with [RTC](https://www.pi.website/research/real_time_chunking) and varied temporal ensembling

---


## 📋 Table of Contents
- [Installation](#️-installation)
- [Data Preparation](#-data-preparation)
- [Training & Evaluation](#-training--evaluation)
- [Practice Tips](#-practice-tips)
- [Real-World Deployment](#-real-world-deployment)
- [Q&A](#-qa)
- [Adding Your Own Method](#-adding-your-own-method)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🛠️ Installation
Please follow the detailed instructions in [INSTALL.md](INSTALL.md) to set up the environment and install dependencies.

---
## 📊 Data Preparation

The scripts for generating data are provided in the `scripts/` directory. Generated data will be stored in the `ManiFlow_Policy/ManiFlow/data/` directory by default as `.zarr` format files for efficient data loading.

### 🤖 RoboTwin Environment

RoboTwin provides a comprehensive suite of bimanual dexterous manipulation tasks featuring visually realistic environments and diverse scene configurations. The benchmark includes complex scenarios with varying object properties, lighting conditions, and environmental clutter to test policy robustness and generalization.

🔹 **Current Support - RoboTwin 1.0:**
Generate demonstrations for the example `diverse_bottles_pick` task using the provided scripts:

```bash
bash scripts/gen_demonstrations_RoboTwin1.0.sh diverse_bottles_pick 0
```
⬇️ **Download pre-generated data:**
For convenience, pre-generated RoboTwin 1.0 datasets are available on [Google Drive](https://drive.google.com/file/d/1YDOSyL3YT5DYyGZF-0xV-nAs1GklUr0d/view?usp=drive_link). Download and extract the `.zarr` files directly to your data directory.

**🔥 Coming Soon - RoboTwin 2.0:**
We will be releasing ManiFlow support for RoboTwin 2.0, featuring enhanced task diversity, comprehensive domain randomization, and more challenging manipulation scenarios. Stay tuned for updates!


### 🎮 Other Simulation Environments (Adroit, DexArt, MetaWorld)

These three simulation environments provide diverse manipulation tasks for training and evaluation. Each environment offers different types of challenges:

- Adroit: Hand manipulation tasks with dexterous control
- DexArt: Complex dexterous manipulation with articulated objects  
- MetaWorld: Multi-task manipulation benchmark with 50+ tasks

Generate demonstrations for these environments using the corresponding scripts in the `scripts/` directory.

Please refer to the individual scripts for specific task names and available parameters. Each script allows customization of demonstration count, environment settings, and data collection parameters.


---

## 🚀 Training & Evaluation

ManiFlow supports both 2D RGB and 3D point cloud policy training across different benchmarks. Use the provided training scripts in the `scripts/` directory to train policies:

**Train on RoboTwin Environment:**
1. Train a 3D ManiFlow Policy for a single task with seed 0 on GPU 1 in the robotwin environment:
    ```bash
    bash scripts/train_eval_robotwin.sh maniflow_pointcloud_policy_robotwin pick_apple_messy_pointcloud 0901 0 1
    ```

2. Train a 2D ManiFlow Policy using the same script:
    ```bash
    bash scripts/train_eval_robotwin.sh maniflow_image_timm_policy_robotwin pick_apple_messy_image 0901 0 1
    ```

**Train on Adroit and DexArt Environments:**

For Adroit and DexArt environments, use the same training scripts with different environment and task names. For example:
1. Adroit:
    ```bash
    bash scripts/train_eval_dex.sh maniflow_pointcloud_policy_dex adroit_door_pointcloud 0901 0 1
    ```
2. DexArt:
    ```bash
    bash scripts/train_eval_dex.sh maniflow_pointcloud_policy_dex dexart_laptop_pointcloud 0901 0 1
    ```

**Train on MetaWorld Environment:**

Train a language-conditioned multi-task ManiFlow policy on MetaWorld with multiple GPUs:

```bash
bash scripts/train_eval_metaworld.sh maniflow_pointcloud_policy_metaworld metaworld_multitask_mp debug 0 1_2_3
```

**Multi-GPU Configuration:**
- Set GPU IDs as a list format (e.g., `0_1_2`) in the training scripts to utilize multiple GPUs.
- Evaluation runs automatically on multiple GPUs to accelerate the evaluation process.
- Use `eval_env_processes` parameter in scripts to control the number of parallel evaluation environments across all GPUs.

### Evaluation

**Automatic Evaluation:**
Set `eval=True` in RoboTwin and MetaWorld training scripts to automatically evaluate the saved trained model on the specified task after training completion.

**Online Evaluation:**
Adroit and DexArt environments use online evaluation during training with results logged to `wandb`.

**Customization:**
Check individual scripts for detailed parameter descriptions and customization options.


---

## 💡 Practice Tips

- 🔍 **Dense representation matters.** Avoid pooling operation in the point cloud encoder to preserve fine-grained geometric details. Use 128-256 points for well-calibrated cropped scenes, and 2048-4096 points for cluttered environments.
- 🌈 **Apply color jitter** (brightness, contrast, saturation) with 0.2 probability during training to prevent overfitting to specific lighting conditions and improve real-world robustness.
- 👁️ **For active sensing cameras** with dynamic viewpoints, collect demonstrations with head movements to enable coordinated head-arm control for humanoid platforms and better generalization to novel viewpoints.
- 🔧 **Use temporal ensembling** for real-world deployment to ensure safety, smoother transitions and account for execution delays.
- ⏱️ **Use longer action horizons** (16+ steps) for complex long-horizon tasks to capture full temporal dynamics.
- 📈 **Data diversity is crucial** for generalization. Collect demonstrations with varying object properties, backgrounds, and clutter to improve robustness.


---


## 🦾 Real-World Deployment

ManiFlow has been successfully deployed across three distinct real-world robot platforms, demonstrating superior performance in challenging manipulation tasks with increasing dexterity requirements.

### 🤖 Supported Robot Platforms

#### **Humanoid Setup - Unitree H1**
- **Hardware**: Full-sized humanoid with 28-DoF (7-DoF arms + 6-DoF anthropomorphic Inspire hands + 2-DoF active head). Use upper-body only for manipulation tasks.
- **Perception**: Gimbal-mounted ZED stereo camera for active egocentric sensing
- **Teleoperation**: Apple Vision Pro - refer to [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision) for setup
- **Key Features**:
   - Coordinated head-arm movements for dynamic viewpoint handling
   - Multi-finger coordination with 6-DoF per anthropomorphic hand
   - Active sensing capabilities with moving camera perspectives

#### **Bimanual Setup - Dual xArm7 + Ability Hands**
- **Hardware**: Two UFACTORY xArm 7 robotic arms with PSYONIC Ability Hands (26-DoF total)
- **Perception**: Intel RealSense LiDAR L515 camera with front-facing static view
- **Teleoperation**: Apple Vision Pro - refer to [Bunny-VisionPro](https://github.com/Dingry/BunnyVisionPro) for setup
- **Key Features**:
   - Precise bimanual coordination and synchronization
   - Dexterous manipulation with 6-DoF hands
   - Consistent third-person visual perspective

#### **Single-Arm Setup - Franka Panda**
- **Hardware**: 7-DoF Franka Emika Panda with Robotiq parallel gripper
- **Perception**: Intel RealSense D455 RGB-D camera (statically mounted)
- **Teleoperation**: Oculus VR for data collection
- **Key Features**:
   - Industrial-grade precision and reliability
   - External visual observations for consistent viewpoint


---

## ❓ Q&A

Q: What is the limitation of 3D perception in ManiFlow?

A: ManiFlow currently leverages dense point cloud representation, which can be computationally expensive for larger scale scenes requiring more points. Future work could explore more efficient 3D encoder to compress the point cloud while preserving geometric details.

Q: How much demonstration data does ManiFlow need?

A: For simple tasks, ManiFlow can achieve reasonable performance with as few as 20-50 demos. For complex long-horizon tasks, 100-200 demos are recommended for optimal performance. ManiFlow shows strong scaling behavior, so more data generally leads to better results.

---

## 🔧 Adding Your Own Method

To integrate your custom manipulation policy:

1. **Implement your policy** inheriting from `BasePolicy` in `maniflow/policy/`, e.g., `maniflow/policy/maniflow_pointcloud_policy.py`.
2. **Register** policy config in `config/policy/`, e.g., `maniflow/config/maniflow_pointcloud_policy_robotwin.yaml`.
3. **Train** using existing scripts: `bash scripts/train_eval_*.sh your_method task_name`

---


## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements
We thank the authors of the following open-source projects, whose code were invaluable in developing ManiFlow: [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), 
[UMI](https://github.com/real-stanford/universal_manipulation_interface),
[3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy/), 
[Improved 3D Diffusion Policy](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy), 
[RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), 
[RoboTwin](https://RoboTwin.github.io),
[consistency models](https://github.com/openai/consistency_models),
[shortcut models](https://github.com/kvfrans/shortcut-models)

---

## 📖 Citation
If you find this repository useful for your research, please consider citing the following paper: 

```
@inproceedings{yan2025maniflow,
  title={{ManiFlow}: A General Robot Manipulation Policy via Consistency Flow Training},
  author={Yan, Ge and Zhu, Jiyue and Deng, Yuquan and Yang, Shiqi and Qiu, Ri-Zhao and Cheng, Xuxin and Memmel, Marius and Krishna, Ranjay and Goyal, Ankit and Wang, Xiaolong and Fox, Dieter},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2025}
}
```

---

## 📧 Contact

For questions, issues, or collaboration inquiries:
- 🐛 **Open an issue** in this repository
- 💬 **Reach out directly** to [Ge Yan](https://geyan21.github.io/)

---

<div align="center">
  <i>🤖 Advancing the frontier of general robotic manipulation</i>
</div>


