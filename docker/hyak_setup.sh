#!/bin/bash
# Hyak Setup Script for HAMSTER-ManiFlow Integration
# Run this after pulling the Singularity image

set -e

# ===== Configuration =====
# Change these to your account
USER_NAME="${USER:-your_username}"
SCRATCH_DIR="/gscratch/scrubbed/${USER_NAME}"
IMAGE_NAME="hamster-maniflow_latest.sif"
INSTANCE_NAME="hamster_maniflow"

# ===== Create Directory Structure =====
echo "Creating directory structure..."
mkdir -p ${SCRATCH_DIR}/singularity/{cache,tmp,images}
mkdir -p ${SCRATCH_DIR}/models
mkdir -p ${SCRATCH_DIR}/data
mkdir -p ${SCRATCH_DIR}/code
mkdir -p ${SCRATCH_DIR}/cache/huggingface

# ===== Set Environment Variables =====
echo "Setting environment variables..."
export SINGULARITY_CACHEDIR=${SCRATCH_DIR}/singularity/cache
export SINGULARITY_TMPDIR=${SCRATCH_DIR}/singularity/tmp
export APPTAINER_CACHEDIR=${SCRATCH_DIR}/singularity/cache

# ===== Load Singularity Module =====
echo "Loading Singularity module..."
module load singularity

# ===== Pull Docker Image (if not exists) =====
cd ${SCRATCH_DIR}/singularity/images
if [ ! -f "${IMAGE_NAME}" ]; then
    echo "Pulling Docker image..."
    # Replace with your DockerHub username
    singularity pull docker://naototo0103/hamster-maniflow:latest
else
    echo "Image already exists: ${IMAGE_NAME}"
fi

# ===== Start Singularity Instance =====
echo "Starting Singularity instance..."
singularity instance start --nv \
    --bind /gscratch/:/gscratch/:rw \
    --bind ${SCRATCH_DIR}/cache/huggingface:/workspace/cache/huggingface:rw \
    ${IMAGE_NAME} ${INSTANCE_NAME}

echo ""
echo "===== Setup Complete ====="
echo "To enter the container:"
echo "  singularity shell instance://${INSTANCE_NAME}"
echo ""
echo "To stop the instance:"
echo "  singularity instance stop ${INSTANCE_NAME}"
echo ""
echo "To check running instances:"
echo "  singularity instance list"
