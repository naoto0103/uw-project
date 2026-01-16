"""
Configuration for parallel VILA path generation.
"""

from pathlib import Path

# ============================================================
# Server Configuration
# ============================================================
NUM_GPUS = 4
BASE_PORT = 8000  # Ports will be 8000, 8001, 8002, 8003

# VILA model settings
VILA_MODEL = "HAMSTER_dev"

# ============================================================
# Paths
# ============================================================
# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
HAMSTER_DIR = SCRIPT_DIR.parent.parent  # HAMSTER/
PROJECT_DIR = HAMSTER_DIR.parent  # HAMSTER-ManiFlow-Integration/

# Model path
MODEL_PATH = HAMSTER_DIR / "Hamster_dev" / "VILA1.5-13b-robopoint_1432k+rlbench_all_tasks_256_1000_eps_sketch_v5_alpha+droid_train99_sketch_v5_alpha_fix+bridge_data_v2_train90_10k_sketch_v5_alpha-e1-LR1e-5"

# Singularity image
SIF_PATH = Path("/gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif")

# Site packages for VILA
SITE_PACKAGES_VILA = Path("/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila")

# Results directory
RESULTS_DIR = HAMSTER_DIR / "results" / "robotwin2_single_6tasks_vila"

# ============================================================
# Tasks
# ============================================================
SINGLE_ARM_TASKS = [
    "beat_block_hammer",
    "click_bell",
    "move_can_pot",
    "place_object_stand",
    "open_microwave",
    "turn_switch",
]

SINGLE_ARM_INSTRUCTIONS = {
    "beat_block_hammer": "Pick up the hammer and beat the block",
    "click_bell": "click the bell's top center on the table",
    "move_can_pot": "pick up the can and move it to beside the pot",
    "place_object_stand": "place the object on the stand",
    "open_microwave": "open the microwave",
    "turn_switch": "click the switch",
}

# ============================================================
# Generation Parameters
# ============================================================
VILA_TEMPERATURE = 0.0
VILA_TOP_P = 0.95
VILA_MAX_TOKENS = 256

# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1
