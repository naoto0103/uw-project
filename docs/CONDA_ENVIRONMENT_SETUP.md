# Conda Environment Setup Guide

This guide provides step-by-step instructions for setting up the `robotwin` conda environment from scratch on Hyak, **reproducing the exact same environment**.

## Overview

The `robotwin` environment is used for:
- RoboTwin 2.0 simulation and data generation
- ManiFlow training and evaluation
- HAMSTER-ManiFlow integration experiments

---

## Step 1: Install Miniconda

```bash
cd /gscratch/scrubbed/naoto03

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh -b -p /gscratch/scrubbed/naoto03/miniconda3

# Clean up
rm Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc
```

---

## Step 2: Create Environment with Python 3.10

```bash
conda create -n robotwin python=3.10.19 -y
conda activate robotwin
```

---

## Step 3: Install ffmpeg via conda

```bash
conda install -c conda-forge ffmpeg=7.1 x264 -y
```

---

## Step 4: Create requirements.txt

Create the following `requirements.txt` file with **exact versions**:

```bash
cat > /tmp/requirements.txt << 'EOF'
# PyTorch (install separately with --index-url)
# torch==2.6.0+cu124
# torchvision==0.21.0+cu124
# torchaudio==2.6.0+cu124

# Core Scientific
numpy==1.26.4
scipy==1.15.3
pandas==2.3.3
scikit-learn==1.7.2
scikit-image==0.25.2
numba==0.63.1
llvmlite==0.46.0
sympy==1.13.1
mpmath==1.3.0

# Robotics & Simulation
sapien==3.0.0b1
mplib==0.2.1
open3d==0.18.0
gymnasium==1.2.3
Farama-Notifications==0.0.4

# Deep Learning
diffusers==0.36.0
timm==1.0.22
einops==0.8.1
hydra-core==1.3.2
omegaconf==2.3.0
huggingface-hub==1.2.3
safetensors==0.7.0
transformers
triton==3.2.0

# Geometry & Mesh
trimesh==4.10.1
shapely==2.1.2
rtree==1.4.1
manifold3d==3.3.2
embreex==2.17.7.post7
vhacdx==0.0.10
pycollada==0.9.2
yourdfpy==0.0.58
transforms3d==0.4.2
pyquaternion==0.9.9
numpy-quaternion==2024.0.13
numpy-stl==3.2.0
mapbox-earcut==2.0.0
svg-path==7.0

# Motion Planning
toppra==0.6.3

# NVIDIA Warp
warp-lang==1.10.1

# Data Storage
h5py==3.15.1
zarr==2.18.3
numcodecs==0.13.1
asciitree==0.3.3
fasteners==0.20

# Visualization
matplotlib==3.10.8
plotly==6.5.0
contourpy==1.3.2
cycler==0.12.1
kiwisolver==1.4.9
fonttools==4.61.1
pillow==12.0.0
imageio==2.37.2
imageio-ffmpeg==0.6.0
opencv-python==4.11.0.86
tifffile==2025.5.10
lazy-loader==0.4

# Web & API
requests==2.32.5
httpx==0.28.1
httpcore==1.0.9
h11==0.16.0
openai==2.14.0
urllib3==2.6.2
certifi==2025.11.12
charset-normalizer==3.4.4
idna==3.11
anyio==4.12.0
sniffio==1.3.1
hf-xet==1.2.0

# Pydantic & Validation
pydantic==2.12.5
pydantic-core==2.41.5
annotated-types==0.7.0
typing-extensions==4.15.0
typing-inspection==0.4.2

# Config & CLI
pyyaml==6.0.3
click==8.3.1
typer-slim==0.20.1
shellingham==1.5.4
configargparse==1.7.1

# Logging & Monitoring
wandb==0.23.1
sentry-sdk==2.48.0
tqdm==4.67.1
termcolor==3.2.0
colorlog==6.10.1
psutil==7.2.0

# Jupyter & IPython
ipython==8.37.0
ipykernel==7.1.0
ipywidgets==8.1.8
jupyter-client==8.7.0
jupyter-core==5.9.1
jupyterlab-widgets==3.0.16
widgetsnbextension==4.0.15
nbformat==5.10.4
traitlets==5.14.3
comm==0.2.3
debugpy==1.8.19
nest-asyncio==1.6.0
tornado==6.5.4
pyzmq==27.1.0

# IPython dependencies
jedi==0.19.2
parso==0.8.5
prompt-toolkit==3.0.52
pygments==2.19.2
decorator==5.2.1
asttokens==3.0.1
executing==2.2.1
pure-eval==0.2.3
stack-data==0.6.3
matplotlib-inline==0.2.1
pexpect==4.9.0
ptyprocess==0.7.0
wcwidth==0.2.14

# JSON & Schema
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
referencing==0.37.0
rpds-py==0.30.0
fastjsonschema==2.21.2
jiter==0.12.0
attrs==25.4.0

# Template & Web Framework
jinja2==3.1.6
MarkupSafe==2.1.5
flask==3.1.2
werkzeug==3.1.4
itsdangerous==2.2.0
blinker==1.9.0
dash==3.3.0

# File & Network
filelock==3.20.0
fsspec==2025.12.0
portalocker==3.2.0
gdown==5.2.0
PySocks==1.7.1

# Git
gitpython==3.1.45
gitdb==4.0.12
smmap==5.0.2

# XML & Parsing
lxml==6.0.2
beautifulsoup4==4.14.3
soupsieve==2.8.1
pyparsing==3.2.5
antlr4-python3-runtime==4.9.3

# Misc Utilities
addict==2.4.0
cloudpickle==3.1.2
dill==0.4.0
joblib==1.5.3
six==1.17.0
python-dateutil==2.9.0.post0
pytz==2025.2
tzdata==2025.3
packaging==25.0
platformdirs==4.5.1
importlib-metadata==8.7.0
importlib-resources==6.5.2
zipp==3.23.0
tomli==2.3.0
exceptiongroup==1.3.1
distro==1.9.0
regex==2025.11.3
protobuf==6.33.2
networkx==3.4.2
iopath==0.1.10
python-utils==3.9.1
retrying==1.4.2
pyperclip==1.11.0
narwhals==2.14.0
xxhash==3.6.0
pybind11==3.0.1
setuptools-scm==9.2.2
rdp==0.8
EOF
```

---

## Step 5: Install PyTorch

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

---

## Step 6: Install All Packages from requirements.txt

```bash
pip install -r /tmp/requirements.txt
```

---

## Step 7: Install Source Packages (Editable)

These packages are installed from local source in editable mode.

```bash
# Load CUDA module
module load cuda/12.4.1 gcc/13.2.0

PROJECT_DIR="/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration"

# PyTorch3D (version 0.7.8)
cd ${PROJECT_DIR}/ManiFlow/third_party/pytorch3d
pip install -e .

# CuRobo
cd ${PROJECT_DIR}/ManiFlow/third_party/curobo
pip install -e src/

# R3M
cd ${PROJECT_DIR}/ManiFlow/third_party/r3m
pip install -e .

# ManiFlow
cd ${PROJECT_DIR}/ManiFlow/ManiFlow
pip install -e .
```

---

## Step 8: Apply Patches

### Patch 1: PyTorch cpp_extension.py

CUDA version mismatch を warning に変更:

```bash
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
CPP_EXT="${TORCH_PATH}/utils/cpp_extension.py"

# Backup
cp "${CPP_EXT}" "${CPP_EXT}.backup"

# Find and replace RuntimeError with warnings.warn in _check_cuda_version()
# The exact line varies by version, but look for:
#   raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(...))
# Change to:
#   warnings.warn(CUDA_MISMATCH_MESSAGE.format(...))
```

### Patch 2: CuRobo Config Paths

`${ASSETS_PATH}` を絶対パスに置換:

```bash
CUROBO_PATH="${PROJECT_DIR}/ManiFlow/third_party/curobo"
find "${CUROBO_PATH}/src/curobo/content" -name "*.yml" -exec sed -i \
    "s|\${ASSETS_PATH}|${CUROBO_PATH}/src/curobo/content/assets|g" {} \;
```

---

## Step 9: Verify Installation

```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
module load cuda/12.4.1 gcc/13.2.0

# Core checks
python -c "import torch; print(f'torch=={torch.__version__}')"
python -c "import numpy; print(f'numpy=={numpy.__version__}')"
python -c "import sapien; print(f'sapien=={sapien.__version__}')"
python -c "import mplib; print('mplib OK')"
python -c "import open3d; print(f'open3d=={open3d.__version__}')"
python -c "import diffusers; print(f'diffusers=={diffusers.__version__}')"
python -c "import pytorch3d; print(f'pytorch3d=={pytorch3d.__version__}')"
python -c "import maniflow; print('maniflow OK')"
python -c "import r3m; print('r3m OK')"

# ffmpeg check
ffmpeg -encoders 2>/dev/null | grep x264
```

---

## Complete Package Version List

以下は現在の robotwin 環境にインストールされている**全パッケージとバージョン**の完全なリスト:

| Package | Version |
|---------|---------|
| addict | 2.4.0 |
| annotated-types | 0.7.0 |
| antlr4-python3-runtime | 4.9.3 |
| anyio | 4.12.0 |
| asciitree | 0.3.3 |
| asttokens | 3.0.1 |
| attrs | 25.4.0 |
| beautifulsoup4 | 4.14.3 |
| blinker | 1.9.0 |
| certifi | 2025.11.12 |
| charset-normalizer | 3.4.4 |
| click | 8.3.1 |
| cloudpickle | 3.1.2 |
| colorlog | 6.10.1 |
| comm | 0.2.3 |
| configargparse | 1.7.1 |
| contourpy | 1.3.2 |
| cycler | 0.12.1 |
| dash | 3.3.0 |
| debugpy | 1.8.19 |
| decorator | 5.2.1 |
| diffusers | 0.36.0 |
| dill | 0.4.0 |
| distro | 1.9.0 |
| einops | 0.8.1 |
| embreex | 2.17.7.post7 |
| exceptiongroup | 1.3.1 |
| executing | 2.2.1 |
| Farama-Notifications | 0.0.4 |
| fasteners | 0.20 |
| fastjsonschema | 2.21.2 |
| filelock | 3.20.0 |
| flask | 3.1.2 |
| fonttools | 4.61.1 |
| fsspec | 2025.12.0 |
| gdown | 5.2.0 |
| gitdb | 4.0.12 |
| gitpython | 3.1.45 |
| gymnasium | 1.2.3 |
| h11 | 0.16.0 |
| h5py | 3.15.1 |
| hf-xet | 1.2.0 |
| httpcore | 1.0.9 |
| httpx | 0.28.1 |
| huggingface-hub | 1.2.3 |
| hydra-core | 1.3.2 |
| idna | 3.11 |
| imageio | 2.37.2 |
| imageio-ffmpeg | 0.6.0 |
| importlib-metadata | 8.7.0 |
| importlib-resources | 6.5.2 |
| iopath | 0.1.10 |
| ipykernel | 7.1.0 |
| ipython | 8.37.0 |
| ipywidgets | 8.1.8 |
| itsdangerous | 2.2.0 |
| jedi | 0.19.2 |
| jinja2 | 3.1.6 |
| jiter | 0.12.0 |
| joblib | 1.5.3 |
| jsonschema | 4.25.1 |
| jsonschema-specifications | 2025.9.1 |
| jupyter-client | 8.7.0 |
| jupyter-core | 5.9.1 |
| jupyterlab-widgets | 3.0.16 |
| kiwisolver | 1.4.9 |
| lazy-loader | 0.4 |
| llvmlite | 0.46.0 |
| lxml | 6.0.2 |
| maniflow | 0.0.0 (editable) |
| manifold3d | 3.3.2 |
| mapbox-earcut | 2.0.0 |
| MarkupSafe | 2.1.5 |
| matplotlib | 3.10.8 |
| matplotlib-inline | 0.2.1 |
| mplib | 0.2.1 |
| mpmath | 1.3.0 |
| narwhals | 2.14.0 |
| nbformat | 5.10.4 |
| nest-asyncio | 1.6.0 |
| networkx | 3.4.2 |
| numba | 0.63.1 |
| numcodecs | 0.13.1 |
| numpy | 1.26.4 |
| numpy-quaternion | 2024.0.13 |
| numpy-stl | 3.2.0 |
| nvidia-cublas-cu12 | 12.4.5.8 |
| nvidia-cuda-cupti-cu12 | 12.4.127 |
| nvidia-cuda-nvrtc-cu12 | 12.4.127 |
| nvidia-cuda-runtime-cu12 | 12.4.127 |
| nvidia-cudnn-cu12 | 9.1.0.70 |
| nvidia-cufft-cu12 | 11.2.1.3 |
| nvidia-curand-cu12 | 10.3.5.147 |
| nvidia-curobo | 0.0.0 (editable) |
| nvidia-cusolver-cu12 | 11.6.1.9 |
| nvidia-cusparse-cu12 | 12.3.1.170 |
| nvidia-cusparselt-cu12 | 0.6.2 |
| nvidia-nccl-cu12 | 2.21.5 |
| nvidia-nvjitlink-cu12 | 12.4.127 |
| nvidia-nvtx-cu12 | 12.4.127 |
| omegaconf | 2.3.0 |
| open3d | 0.18.0 |
| openai | 2.14.0 |
| opencv-python | 4.11.0.86 |
| packaging | 25.0 |
| pandas | 2.3.3 |
| parso | 0.8.5 |
| pexpect | 4.9.0 |
| pillow | 12.0.0 |
| pip | 25.3 |
| platformdirs | 4.5.1 |
| plotly | 6.5.0 |
| portalocker | 3.2.0 |
| prompt-toolkit | 3.0.52 |
| protobuf | 6.33.2 |
| psutil | 7.2.0 |
| ptyprocess | 0.7.0 |
| pure-eval | 0.2.3 |
| pybind11 | 3.0.1 |
| pycollada | 0.9.2 |
| pydantic | 2.12.5 |
| pydantic-core | 2.41.5 |
| pygments | 2.19.2 |
| pyparsing | 3.2.5 |
| pyperclip | 1.11.0 |
| pyquaternion | 0.9.9 |
| PySocks | 1.7.1 |
| python-dateutil | 2.9.0.post0 |
| python-utils | 3.9.1 |
| pytorch3d | 0.7.8 (editable) |
| pytz | 2025.2 |
| pyyaml | 6.0.3 |
| pyzmq | 27.1.0 |
| r3m | 0.0.0 (editable) |
| rdp | 0.8 |
| referencing | 0.37.0 |
| regex | 2025.11.3 |
| requests | 2.32.5 |
| retrying | 1.4.2 |
| rpds-py | 0.30.0 |
| rtree | 1.4.1 |
| safetensors | 0.7.0 |
| sapien | 3.0.0b1 |
| scikit-image | 0.25.2 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| sentry-sdk | 2.48.0 |
| setuptools-scm | 9.2.2 |
| shapely | 2.1.2 |
| shellingham | 1.5.4 |
| six | 1.17.0 |
| smmap | 5.0.2 |
| sniffio | 1.3.1 |
| soupsieve | 2.8.1 |
| stack-data | 0.6.3 |
| svg-path | 7.0 |
| sympy | 1.13.1 |
| termcolor | 3.2.0 |
| threadpoolctl | 3.6.0 |
| tifffile | 2025.5.10 |
| timm | 1.0.22 |
| tomli | 2.3.0 |
| toppra | 0.6.3 |
| torch | 2.6.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| tornado | 6.5.4 |
| tqdm | 4.67.1 |
| traitlets | 5.14.3 |
| transforms3d | 0.4.2 |
| trimesh | 4.10.1 |
| triton | 3.2.0 |
| typer-slim | 0.20.1 |
| typing-extensions | 4.15.0 |
| typing-inspection | 0.4.2 |
| tzdata | 2025.3 |
| urllib3 | 2.6.2 |
| vhacdx | 0.0.10 |
| wandb | 0.23.1 |
| warp-lang | 1.10.1 |
| wcwidth | 0.2.14 |
| werkzeug | 3.1.4 |
| wheel | 0.45.1 |
| widgetsnbextension | 4.0.15 |
| xxhash | 3.6.0 |
| yourdfpy | 0.0.58 |
| zarr | 2.18.3 |
| zipp | 3.23.0 |

### Editable Packages (from source)

| Package | Version | Source Path |
|---------|---------|-------------|
| maniflow | 0.0.0 | `ManiFlow/ManiFlow/` |
| nvidia-curobo | 0.0.0 | `ManiFlow/third_party/curobo/src/` |
| r3m | 0.0.0 | `ManiFlow/third_party/r3m/` |
| pytorch3d | 0.7.8 | `ManiFlow/third_party/pytorch3d/` |

---

## Quick Setup Script

```bash
#!/bin/bash
# robotwin_setup.sh - Complete environment setup

set -e

SCRATCH="/gscratch/scrubbed/naoto03"
PROJECT="${SCRATCH}/projects/HAMSTER-ManiFlow-Integration"

echo "=== Step 1: Install Miniconda ==="
cd ${SCRATCH}
if [ ! -d "${SCRATCH}/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p ${SCRATCH}/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
fi

source ${SCRATCH}/miniconda3/etc/profile.d/conda.sh

echo "=== Step 2: Create environment ==="
conda create -n robotwin python=3.10.19 -y
conda activate robotwin

echo "=== Step 3: Install ffmpeg ==="
conda install -c conda-forge ffmpeg=7.1 x264 -y

echo "=== Step 4: Install PyTorch ==="
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== Step 5: Install packages ==="
pip install -r ${PROJECT}/docs/requirements_robotwin.txt

echo "=== Step 6: Install source packages ==="
module load cuda/12.4.1 gcc/13.2.0

cd ${PROJECT}/ManiFlow/third_party/pytorch3d && pip install -e .
cd ${PROJECT}/ManiFlow/third_party/curobo && pip install -e src/
cd ${PROJECT}/ManiFlow/third_party/r3m && pip install -e .
cd ${PROJECT}/ManiFlow/ManiFlow && pip install -e .

echo "=== Setup Complete ==="
```

---

## Troubleshooting

### Python encodings error
```
Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encoding
```
→ conda環境が壊れている。環境を削除して再作成が必要。

### CUDA version mismatch
→ Patch 1 (cpp_extension.py) を適用。

### PyTorch3D build fails
```bash
module load cuda/12.4.1 gcc/13.2.0
cd ${PROJECT}/ManiFlow/third_party/pytorch3d
rm -rf build/ pytorch3d.egg-info/
pip install -e .
```

---

## References

- [Miniconda](https://docs.anaconda.com/miniconda/)
- [PyTorch](https://pytorch.org/)
- [SAPIEN](https://sapien.ucsd.edu/)
- [CuRobo](https://curobo.org/)
