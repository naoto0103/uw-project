"""
Configuration for parallel VILA path generation - Evaluation Tasks (Cluttered).
"""

from pathlib import Path

# ============================================================
# Server Configuration
# ============================================================
NUM_GPUS = 4
BASE_PORT = 8012  # Ports will be 8012, 8013, 8014, 8015

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

# Results directory - Cluttered evaluation data
RESULTS_DIR = HAMSTER_DIR / "results" / "evaluation_tasks_cluttered"

# ============================================================
# Tasks - 6 evaluation tasks from RESEARCH_OVERVIEW.md
# ============================================================
SINGLE_ARM_TASKS = [
    "beat_block_hammer",
    "click_bell",
    "move_can_pot",
    "open_microwave",
    "turn_switch",
    "adjust_bottle",
    "open_laptop",
]

SINGLE_ARM_INSTRUCTIONS = {
    "beat_block_hammer": "there is a hammer and a block on the table, use the arm to grab the hammer and beat the block",
    "click_bell": "click the bell's top center on the table",
    "move_can_pot": "there is a can and a pot on the table, use one arm to pick up the can and move it to the side of the pot closest to the can",
    "open_microwave": "Use one arm to open the microwave.",
    "turn_switch": "use the robotic arm to click the switch",
    "adjust_bottle": "Pick up the bottle on the table headup with the correct arm",
    "open_laptop": "use one arm to open the laptop",
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
