#!/bin/bash
# Initialize Container Script
# Run this inside the Singularity container after first start

set -e

# ===== Configuration =====
USER_NAME="${USER:-your_username}"
SCRATCH_DIR="/gscratch/scrubbed/${USER_NAME}"
CODE_DIR="${SCRATCH_DIR}/code/HAMSTER-ManiFlow-Integration"

echo "===== Initializing HAMSTER-ManiFlow Container ====="

# ===== Clone Repository (if not exists) =====
if [ ! -d "${CODE_DIR}" ]; then
    echo "Cloning repository..."
    cd ${SCRATCH_DIR}/code
    git clone https://github.com/YOUR_GITHUB_USER/HAMSTER-ManiFlow-Integration.git
else
    echo "Repository already exists at ${CODE_DIR}"
fi

cd ${CODE_DIR}

# ===== Install ManiFlow as editable package =====
echo "Installing ManiFlow..."
cd ManiFlow/ManiFlow
pip install -e .
cd ../..

# ===== Install mujoco-py =====
echo "Installing mujoco-py..."
cd ManiFlow/third_party/mujoco-py-2.1.2.14
pip install -e .
cd ../../..

# ===== Install simulation environments =====
echo "Installing simulation environments..."
cd ManiFlow/third_party
cd gym-0.21.0 && pip install -e . && cd ..
cd Metaworld && pip install -e . && cd ..
cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ..
cd r3m && pip install -e . && cd ../..

# ===== Modify mplib (for RoboTwin) =====
echo "Modifying mplib for RoboTwin compatibility..."
MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib
PLANNER=${MPLIB_LOCATION}/planner.py

if [ -f "$PLANNER" ]; then
    # Comment out convex=True parameter
    sed -i -E 's/^(\s*)(.*convex=True.*)/\1# \2/' $PLANNER
    # Remove 'or collide' from the condition
    sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER
    echo "mplib modified successfully"
else
    echo "Warning: mplib planner.py not found at $PLANNER"
fi

# ===== Test Qwen3-VL Import =====
echo "Testing Qwen3-VL import..."
python -c "from transformers import AutoModelForImageTextToText, AutoProcessor; print('Qwen3-VL imports OK')"

# ===== Test PyTorch3D Import =====
echo "Testing PyTorch3D import..."
python -c "import pytorch3d; print('PyTorch3D version:', pytorch3d.__version__)"

# ===== Test ManiFlow Import =====
echo "Testing ManiFlow import..."
python -c "import maniflow; print('ManiFlow imports OK')"

echo ""
echo "===== Initialization Complete ====="
echo ""
echo "Next steps:"
echo "1. Download Qwen3-VL model (auto-downloads on first use)"
echo "2. Transfer data: rsync -avzP source:/path/to/data ${SCRATCH_DIR}/data/"
echo "3. Start Qwen3 server: python HAMSTER/server_qwen3.py"
echo "4. Run ManiFlow training: bash scripts/train_eval_hamster_maniflow.sh"
