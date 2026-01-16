"""
Configuration for parallel PEEK (VILA-3B) path generation.
"""

from pathlib import Path

# ============================================================
# Server Configuration
# ============================================================
NUM_GPUS = 4
BASE_PORT = 8010  # Ports will be 8010, 8011, 8012, 8013 (avoid conflict with VILA 13B)

# PEEK model settings
PEEK_MODEL = "peek_3b"

# ============================================================
# Paths
# ============================================================
# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
HAMSTER_DIR = SCRIPT_DIR.parent.parent  # HAMSTER/
PROJECT_DIR = HAMSTER_DIR.parent  # HAMSTER-ManiFlow-Integration/

# PEEK model path (HuggingFace cache)
MODEL_PATH = Path("/gscratch/scrubbed/naoto03/.cache/huggingface/models--memmelma--peek_3b/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d")

# Singularity image
SIF_PATH = Path("/gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif")

# Site packages for VILA
SITE_PACKAGES_VILA = Path("/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila")

# PEEK repository path
PEEK_DIR = PROJECT_DIR / "PEEK"
PEEK_VLM_DIR = PEEK_DIR / "peek_vlm"

# Results directory
RESULTS_DIR = HAMSTER_DIR / "results" / "robotwin2_single_6tasks_peek"

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
PEEK_TEMPERATURE = 0.0
PEEK_TOP_P = 0.95
PEEK_MAX_TOKENS = 1024  # PEEK outputs both trajectory and mask, needs more tokens

# ============================================================
# PEEK Prompt (from peek_vlm/peek_vlm/utils/encode.py)
# ============================================================
def get_peek_prompt(instruction: str) -> str:
    """
    Get PEEK path_mask prompt format.
    This is the exact prompt from PEEK's encode.py
    """
    return (
        f"<image>\n"
        f"In the image, please execute the command described in <quest>{instruction}</quest>.\n"
        f"Provide a sequence of points denoting the trajectory of a robot gripper and a set of points denoting the areas the robot must see to achieve the goal.\n"
        f"Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
        f"TRAJECTORY: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans> MASK: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans>\n"
        f"The tuple denotes point x and y location of the end effector of the gripper in the image.\n"
        f"The coordinates should be integers ranging between 0.0 and 1.0, indicating the relative locations of the points in the image.\n"
    )
