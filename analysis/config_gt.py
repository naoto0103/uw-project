"""
Configuration for GT Path trained models analysis.
(Models trained with Ground Truth paths instead of VILA-generated paths)
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

# Base directory
ANALYSIS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = ANALYSIS_DIR.parent

# Data directories
RAW_DATA_DIR = ANALYSIS_DIR / "raw_data"

# Output directories
OUTPUTS_DIR = ANALYSIS_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# =============================================================================
# Tasks
# =============================================================================

# Tasks to analyze (論文で報告する3タスク)
TASKS = [
    "click_bell",
    "move_can_pot",
    "beat_block_hammer",
]

# Task descriptions (for reports/figures)
TASK_DESCRIPTIONS = {
    "click_bell": "Click Bell",
    "move_can_pot": "Move Can Pot",
    "beat_block_hammer": "Beat Block Hammer",
}

# =============================================================================
# GT Path Experimental Conditions
# =============================================================================

# 4 GT path conditions (GT-C2, GT-C3, GT-C5, GT-C6)
# Note: No GT-C1 and GT-C4 because those are "original" mode without path overlay
CONDITIONS = {
    2: {
        "name": "gt_condition2",
        "train_env": "cluttered",
        "mode": "overlay_current",
        "description": "GT+MF current (cluttered)",
        "short_name": "GT+MF-cur-clut",
        "uses_vila": True,  # Uses VILA at evaluation time
    },
    3: {
        "name": "gt_condition3",
        "train_env": "cluttered",
        "mode": "overlay_initial_current",
        "description": "GT+MF init+curr (cluttered)",
        "short_name": "GT+MF-ic-clut",
        "uses_vila": True,
    },
    5: {
        "name": "gt_condition5",
        "train_env": "clean",
        "mode": "overlay_current",
        "description": "GT+MF current (clean→cluttered)",
        "short_name": "GT+MF-cur-clean",
        "uses_vila": True,
    },
    6: {
        "name": "gt_condition6",
        "train_env": "clean",
        "mode": "overlay_initial_current",
        "description": "GT+MF init+curr (clean→cluttered)",
        "short_name": "GT+MF-ic-clean",
        "uses_vila": True,
    },
}

# Condition directory naming pattern for GT path
def get_condition_dir_name(condition_id: int, eval_env: str = "cluttered") -> str:
    """Get the directory name for a GT condition."""
    cond = CONDITIONS[condition_id]
    return f"gt_condition{condition_id}_{cond['train_env']}_{cond['mode']}_eval{eval_env}"

# =============================================================================
# Evaluation Settings
# =============================================================================

# Evaluation seed (fixed due to time constraints)
EVAL_SEED = 42

# Number of episodes per evaluation
NUM_EPISODES = 100

# Evaluation environment (all conditions evaluated on cluttered)
EVAL_ENV = "cluttered"

# =============================================================================
# Hypotheses (GT Path specific)
# =============================================================================

HYPOTHESES = {
    "H3": {
        "name": "Initial path benefit (GT trained)",
        "description": "Initial path addition improves over current-only (GT path trained models)",
        "comparisons": [
            {"baseline": 2, "treatment": 3, "label": "GT-C2 vs GT-C3 (in-domain)"},
            {"baseline": 5, "treatment": 6, "label": "GT-C5 vs GT-C6 (cross-domain)"},
        ],
    },
}

# =============================================================================
# Visualization Settings
# =============================================================================

# Colors for GT conditions (colorblind-friendly)
CONDITION_COLORS = {
    2: "#ff7f0e",  # orange - GT-C2
    3: "#2ca02c",  # green - GT-C3
    5: "#9467bd",  # purple - GT-C5
    6: "#8c564b",  # brown - GT-C6
}

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
