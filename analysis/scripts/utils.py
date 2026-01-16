"""
Common utilities for HAMSTER + ManiFlow analysis.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    CONDITIONS,
    HYPOTHESES,
)


def ensure_output_dirs():
    """Ensure all output directories exist."""
    for dir_path in [OUTPUTS_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_date_stamp() -> str:
    """Get current date string for filenames."""
    return datetime.now().strftime("%Y%m%d")


# =============================================================================
# Hypothesis testing utilities
# =============================================================================

def compute_improvement(baseline: float, treatment: float) -> float:
    """Compute absolute improvement (treatment - baseline)."""
    return treatment - baseline


def compute_relative_improvement(baseline: float, treatment: float) -> float:
    """Compute relative improvement percentage."""
    if baseline == 0:
        return float('inf') if treatment > 0 else 0.0
    return ((treatment - baseline) / baseline) * 100


def evaluate_h1(
    task_results,
    conditions_to_check: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate H1: VILA path guidance improves single-task accuracy.
    Compare condition 1 vs 2, and condition 1 vs 3.
    """
    if conditions_to_check is None:
        conditions_to_check = HYPOTHESES["H1"]["comparisons"]

    results = {
        "hypothesis": "H1",
        "description": HYPOTHESES["H1"]["description"],
        "comparisons": [],
    }

    for comp in conditions_to_check:
        baseline_id = comp["baseline"]
        treatment_id = comp["treatment"]

        baseline_cond = task_results.get_condition(baseline_id)
        treatment_cond = task_results.get_condition(treatment_id)

        if baseline_cond is None or treatment_cond is None:
            continue

        baseline_rate = baseline_cond.success_rate_percent
        treatment_rate = treatment_cond.success_rate_percent
        improvement = compute_improvement(baseline_rate, treatment_rate)

        results["comparisons"].append({
            "label": comp["label"],
            "baseline": baseline_rate,
            "treatment": treatment_rate,
            "improvement": improvement,
            "improved": improvement > 0,
        })

    return results


def evaluate_h2(
    task_results,
    conditions_to_check: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate H2: VILA path guidance improves generalization.
    Compare condition 4 vs 5, and condition 4 vs 6.
    """
    if conditions_to_check is None:
        conditions_to_check = HYPOTHESES["H2"]["comparisons"]

    results = {
        "hypothesis": "H2",
        "description": HYPOTHESES["H2"]["description"],
        "comparisons": [],
    }

    for comp in conditions_to_check:
        baseline_id = comp["baseline"]
        treatment_id = comp["treatment"]

        baseline_cond = task_results.get_condition(baseline_id)
        treatment_cond = task_results.get_condition(treatment_id)

        if baseline_cond is None or treatment_cond is None:
            continue

        baseline_rate = baseline_cond.success_rate_percent
        treatment_rate = treatment_cond.success_rate_percent
        improvement = compute_improvement(baseline_rate, treatment_rate)

        results["comparisons"].append({
            "label": comp["label"],
            "baseline": baseline_rate,
            "treatment": treatment_rate,
            "improvement": improvement,
            "improved": improvement > 0,
        })

    return results


def evaluate_h3(
    task_results,
    conditions_to_check: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate H3: Initial path addition improves over current-only.
    Compare condition 2 vs 3, and condition 5 vs 6.
    """
    if conditions_to_check is None:
        conditions_to_check = HYPOTHESES["H3"]["comparisons"]

    results = {
        "hypothesis": "H3",
        "description": HYPOTHESES["H3"]["description"],
        "comparisons": [],
    }

    for comp in conditions_to_check:
        baseline_id = comp["baseline"]
        treatment_id = comp["treatment"]

        baseline_cond = task_results.get_condition(baseline_id)
        treatment_cond = task_results.get_condition(treatment_id)

        if baseline_cond is None or treatment_cond is None:
            continue

        baseline_rate = baseline_cond.success_rate_percent
        treatment_rate = treatment_cond.success_rate_percent
        improvement = compute_improvement(baseline_rate, treatment_rate)

        results["comparisons"].append({
            "label": comp["label"],
            "baseline": baseline_rate,
            "treatment": treatment_rate,
            "improvement": improvement,
            "improved": improvement > 0,
        })

    return results


def evaluate_h4(
    task_results,
    conditions_to_check: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate H4: Path guidance effect is larger in cross-domain condition.
    Compare improvement in in-domain vs cross-domain.
    """
    if conditions_to_check is None:
        conditions_to_check = HYPOTHESES["H4"]["comparisons"]

    results = {
        "hypothesis": "H4",
        "description": HYPOTHESES["H4"]["description"],
        "comparisons": [],
    }

    for comp in conditions_to_check:
        in_domain = comp["in_domain"]
        cross_domain = comp["cross_domain"]

        # In-domain improvement
        in_baseline = task_results.get_condition(in_domain["baseline"])
        in_treatment = task_results.get_condition(in_domain["treatment"])

        # Cross-domain improvement
        cross_baseline = task_results.get_condition(cross_domain["baseline"])
        cross_treatment = task_results.get_condition(cross_domain["treatment"])

        if any(c is None for c in [in_baseline, in_treatment, cross_baseline, cross_treatment]):
            continue

        in_improvement = compute_improvement(
            in_baseline.success_rate_percent,
            in_treatment.success_rate_percent
        )
        cross_improvement = compute_improvement(
            cross_baseline.success_rate_percent,
            cross_treatment.success_rate_percent
        )

        results["comparisons"].append({
            "label": comp["label"],
            "in_domain_improvement": in_improvement,
            "cross_domain_improvement": cross_improvement,
            "difference": cross_improvement - in_improvement,
            "cross_domain_larger": cross_improvement > in_improvement,
        })

    return results


def evaluate_all_hypotheses(task_results) -> Dict[str, Dict[str, Any]]:
    """Evaluate all hypotheses for a task."""
    return {
        "H1": evaluate_h1(task_results),
        "H2": evaluate_h2(task_results),
        "H3": evaluate_h3(task_results),
        "H4": evaluate_h4(task_results),
    }


# =============================================================================
# Formatting utilities
# =============================================================================

def format_percent(value: float, decimals: int = 1) -> str:
    """Format a percentage value."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_improvement(value: float, decimals: int = 1) -> str:
    """Format an improvement value with +/- sign."""
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def format_ms(value: float, decimals: int = 1) -> str:
    """Format a milliseconds value."""
    if value is None or value == 0:
        return "-"
    return f"{value:.{decimals}f}ms"


# =============================================================================
# Report generation utilities
# =============================================================================

def generate_report_header(title: str) -> str:
    """Generate a markdown report header."""
    return f"""# {title}

**Generated**: {get_timestamp()}

---

"""


def generate_task_section_header(task: str, task_description: str) -> str:
    """Generate a markdown section header for a task."""
    return f"""## {task_description} (`{task}`)

"""


def generate_condition_table_md(
    success_rates: Dict[int, float],
    include_delta: bool = True,
) -> str:
    """Generate a markdown table of condition results."""
    lines = []

    # Header
    lines.append("| Condition | Train Env | Mode | Success Rate |")
    lines.append("|-----------|-----------|------|--------------|")

    # Rows
    for cond_id in sorted(success_rates.keys()):
        rate = success_rates[cond_id]
        cond = CONDITIONS[cond_id]
        rate_str = format_percent(rate) if rate is not None else "N/A"
        lines.append(
            f"| {cond_id} | {cond['train_env']} | {cond['mode']} | {rate_str} |"
        )

    return "\n".join(lines)


def generate_hypothesis_section_md(hypothesis_results: Dict[str, Any]) -> str:
    """Generate a markdown section for hypothesis evaluation."""
    lines = []

    hypothesis = hypothesis_results["hypothesis"]
    description = hypothesis_results["description"]

    lines.append(f"### {hypothesis}: {description}")
    lines.append("")

    comparisons = hypothesis_results.get("comparisons", [])
    if not comparisons:
        lines.append("*No data available for this hypothesis.*")
        return "\n".join(lines)

    # Different format for H4
    if hypothesis == "H4":
        lines.append("| Comparison | In-Domain Δ | Cross-Domain Δ | Difference | Cross > In? |")
        lines.append("|------------|-------------|----------------|------------|-------------|")
        for comp in comparisons:
            in_delta = format_improvement(comp["in_domain_improvement"])
            cross_delta = format_improvement(comp["cross_domain_improvement"])
            diff = format_improvement(comp["difference"])
            result = "✓" if comp["cross_domain_larger"] else "✗"
            lines.append(f"| {comp['label']} | {in_delta} | {cross_delta} | {diff} | {result} |")
    else:
        lines.append("| Comparison | Baseline | Treatment | Improvement | Improved? |")
        lines.append("|------------|----------|-----------|-------------|-----------|")
        for comp in comparisons:
            baseline = format_percent(comp["baseline"])
            treatment = format_percent(comp["treatment"])
            improvement = format_improvement(comp["improvement"])
            result = "✓" if comp["improved"] else "✗"
            lines.append(f"| {comp['label']} | {baseline} | {treatment} | {improvement} | {result} |")

    lines.append("")
    return "\n".join(lines)
