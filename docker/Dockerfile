# HAMSTER-ManiFlow Integration Docker Image
# For Hyak HPC (A40/L40s GPUs) via Singularity

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# System packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    # Vulkan for SAPIEN
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    # OpenGL/Graphics
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglew-dev \
    libosmesa6-dev \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    patchelf \
    # MuJoCo dependencies
    libglfw3 \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements first
COPY requirements_docker.txt /tmp/requirements_docker.txt

# Install PyTorch first
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
RUN pip3 install -r /tmp/requirements_docker.txt

# Install PyTorch3D (disable build isolation to use system PyTorch)
RUN pip3 install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install MuJoCo 2.1.0
RUN mkdir -p /root/.mujoco && \
    wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz && \
    tar -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco210.tar.gz

# Environment variables for MuJoCo
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# HuggingFace cache directory (will be mounted)
ENV HF_HOME=/workspace/cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/cache/huggingface

# Working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
