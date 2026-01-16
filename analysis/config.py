"""
Configuration for HAMSTER + ManiFlow analysis.
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
    "move_can_pot": "Move Can to Pot",
    "beat_block_hammer": "Beat Block with Hammer",
}

# =============================================================================
# Experimental Conditions
# =============================================================================

# 6 experimental conditions
CONDITIONS = {
    1: {
        "name": "condition1",
        "train_env": "cluttered",
        "mode": "original",
        "description": "ManiFlow (cluttered)",
        "short_name": "MF-clut",
        "uses_vila": False,
    },
    2: {
        "name": "condition2",
        "train_env": "cluttered",
        "mode": "overlay_current",
        "description": "VILA+MF current (cluttered)",
        "short_name": "V+MF-cur-clut",
        "uses_vila": True,
    },
    3: {
        "name": "condition3",
        "train_env": "cluttered",
        "mode": "overlay_initial_current",
        "description": "VILA+MF init+curr (cluttered)",
        "short_name": "V+MF-ic-clut",
        "uses_vila": True,
    },
    4: {
        "name": "condition4",
        "train_env": "clean",
        "mode": "original",
        "description": "ManiFlow (clean→cluttered)",
        "short_name": "MF-clean",
        "uses_vila": False,
    },
    5: {
        "name": "condition5",
        "train_env": "clean",
        "mode": "overlay_current",
        "description": "VILA+MF current (clean→cluttered)",
        "short_name": "V+MF-cur-clean",
        "uses_vila": True,
    },
    6: {
        "name": "condition6",
        "train_env": "clean",
        "mode": "overlay_initial_current",
        "description": "VILA+MF init+curr (clean→cluttered)",
        "short_name": "V+MF-ic-clean",
        "uses_vila": True,
    },
}

# Condition directory naming pattern
def get_condition_dir_name(condition_id: int, eval_env: str = "cluttered") -> str:
    """Get the directory name for a condition."""
    cond = CONDITIONS[condition_id]
    return f"condition{condition_id}_{cond['train_env']}_{cond['mode']}_eval{eval_env}"

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
# Hypotheses
# =============================================================================

HYPOTHESES = {
    "H1": {
        "name": "Single-task accuracy improvement",
        "description": "VILA path guidance improves single-task accuracy (in-domain)",
        "comparisons": [
            {"baseline": 1, "treatment": 2, "label": "cond1 vs cond2"},
            {"baseline": 1, "treatment": 3, "label": "cond1 vs cond3"},
        ],
    },
    "H2": {
        "name": "Generalization improvement",
        "description": "VILA path guidance improves generalization (cross-domain)",
        "comparisons": [
            {"baseline": 4, "treatment": 5, "label": "cond4 vs cond5"},
            {"baseline": 4, "treatment": 6, "label": "cond4 vs cond6"},
        ],
    },
    "H3": {
        "name": "Initial path benefit",
        "description": "Initial path addition improves over current-only",
        "comparisons": [
            {"baseline": 2, "treatment": 3, "label": "cond2 vs cond3 (in-domain)"},
            {"baseline": 5, "treatment": 6, "label": "cond5 vs cond6 (cross-domain)"},
        ],
    },
    "H4": {
        "name": "Larger effect in cross-domain",
        "description": "Path guidance effect is larger in cross-domain condition",
        "comparisons": [
            {
                "in_domain": {"baseline": 1, "treatment": 2},
                "cross_domain": {"baseline": 4, "treatment": 5},
                "label": "Δ(cond2-cond1) vs Δ(cond5-cond4)",
            },
            {
                "in_domain": {"baseline": 1, "treatment": 3},
                "cross_domain": {"baseline": 4, "treatment": 6},
                "label": "Δ(cond3-cond1) vs Δ(cond6-cond4)",
            },
        ],
    },
}

# =============================================================================
# Visualization Settings
# =============================================================================

# Colors for conditions (colorblind-friendly)
CONDITION_COLORS = {
    1: "#1f77b4",  # blue
    2: "#ff7f0e",  # orange
    3: "#2ca02c",  # green
    4: "#d62728",  # red
    5: "#9467bd",  # purple
    6: "#8c564b",  # brown
}

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
