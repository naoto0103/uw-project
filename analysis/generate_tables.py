#!/usr/bin/env python3
"""
Generate publication-ready tables for HAMSTER + ManiFlow evaluation.

Outputs:
- Markdown tables
- LaTeX tables
- CSV files

Usage:
    python generate_tables.py
    python generate_tables.py --format markdown
    python generate_tables.py --format latex
    python generate_tables.py --format csv
    python generate_tables.py --format all
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any

from config import (
    TASKS,
    TASK_DESCRIPTIONS,
    CONDITIONS,
    TABLES_DIR,
)
from scripts.load_results import (
    load_all_results,
    TaskResults,
    get_success_rates_table,
    get_path_stats_table,
    get_timing_stats_table,
)
from scripts.utils import (
    ensure_output_dirs,
    evaluate_all_hypotheses,
    format_percent,
    format_improvement,
)
from analyze_all_tasks import compute_average_success_rates, aggregate_hypothesis_results


# =============================================================================
# Markdown Table Generation
# =============================================================================

def generate_main_results_md(all_results: Dict[str, TaskResults]) -> str:
    """Generate main results table in Markdown format."""
    lines = []

    lines.append("# Main Results: Success Rates (%)\n")

    # Header
    header = "| Task |"
    for cond_id in range(1, 7):
        header += f" C{cond_id} |"
    lines.append(header)

    separator = "|:-----|"
    for _ in range(6):
        separator += ":-----:|"
    lines.append(separator)

    # Task rows
    success_table = get_success_rates_table(all_results)
    for task in TASKS:
        task_desc = TASK_DESCRIPTIONS.get(task, task)
        row = f"| {task_desc} |"
        for cond_id in range(1, 7):
            rate = success_table.get(task, {}).get(cond_id)
            if rate is not None:
                row += f" {rate:.1f} |"
            else:
                row += " - |"
        lines.append(row)

    # Average row
    avg_rates = compute_average_success_rates(all_results)
    row = "| **Average** |"
    for cond_id in range(1, 7):
        rate = avg_rates.get(cond_id)
        if rate is not None:
            row += f" **{rate:.1f}** |"
        else:
            row += " - |"
    lines.append(row)

    lines.append("\n")
    lines.append("*C1-C3: Trained on cluttered, C4-C6: Trained on clean. All evaluated on cluttered.*\n")
    lines.append("*C1,C4: Original ManiFlow. C2,C5: VILA current path. C3,C6: VILA initial+current path.*\n")

    return "\n".join(lines)


def generate_hypothesis_table_md(all_results: Dict[str, TaskResults]) -> str:
    """Generate hypothesis evaluation table in Markdown format."""
    lines = []

    lines.append("# Hypothesis Evaluation Results\n")

    aggregated = aggregate_hypothesis_results(all_results)

    # H1 & H2: Path guidance effect
    lines.append("## H1 & H2: Path Guidance Effect\n")
    lines.append("| Task | In-Domain (C1→C3) | Cross-Domain (C4→C6) |")
    lines.append("|:-----|:-----------------:|:--------------------:|")

    for task in TASKS:
        task_desc = TASK_DESCRIPTIONS.get(task, task)

        # Get H1 improvement (in-domain, C1 vs C3)
        h1_data = aggregated["H1"]["by_task"].get(task, {})
        h1_comps = h1_data.get("comparisons", [])
        h1_imp = None
        for comp in h1_comps:
            if "cond3" in comp["label"]:
                h1_imp = comp["improvement"]
                break

        # Get H2 improvement (cross-domain, C4 vs C6)
        h2_data = aggregated["H2"]["by_task"].get(task, {})
        h2_comps = h2_data.get("comparisons", [])
        h2_imp = None
        for comp in h2_comps:
            if "cond6" in comp["label"]:
                h2_imp = comp["improvement"]
                break

        h1_str = format_improvement(h1_imp) if h1_imp is not None else "-"
        h2_str = format_improvement(h2_imp) if h2_imp is not None else "-"

        lines.append(f"| {task_desc} | {h1_str} | {h2_str} |")

    lines.append("\n")

    # H3: Initial path benefit
    lines.append("## H3: Initial Path Benefit (Current → Initial+Current)\n")
    lines.append("| Task | In-Domain (C2→C3) | Cross-Domain (C5→C6) |")
    lines.append("|:-----|:-----------------:|:--------------------:|")

    for task in TASKS:
        task_desc = TASK_DESCRIPTIONS.get(task, task)
        h3_data = aggregated["H3"]["by_task"].get(task, {})
        h3_comps = h3_data.get("comparisons", [])

        in_domain_imp = None
        cross_domain_imp = None
        for comp in h3_comps:
            if "in-domain" in comp["label"]:
                in_domain_imp = comp["improvement"]
            elif "cross-domain" in comp["label"]:
                cross_domain_imp = comp["improvement"]

        in_str = format_improvement(in_domain_imp) if in_domain_imp is not None else "-"
        cross_str = format_improvement(cross_domain_imp) if cross_domain_imp is not None else "-"

        lines.append(f"| {task_desc} | {in_str} | {cross_str} |")

    lines.append("\n")

    return "\n".join(lines)


def generate_path_stats_md(all_results: Dict[str, TaskResults]) -> str:
    """Generate path generation statistics table in Markdown format."""
    lines = []

    lines.append("# Path Generation Statistics\n")
    lines.append("| Task | Condition | Path Success | Frame0 Success | Avg Fallbacks |")
    lines.append("|:-----|:---------:|:------------:|:--------------:|:-------------:|")

    path_table = get_path_stats_table(all_results)
    vila_conditions = [cid for cid, c in CONDITIONS.items() if c["uses_vila"]]

    for task in TASKS:
        task_desc = TASK_DESCRIPTIONS.get(task, task)
        for i, cond_id in enumerate(vila_conditions):
            stats = path_table.get(task, {}).get(cond_id)
            task_cell = task_desc if i == 0 else ""

            if stats:
                path_rate = f"{stats['path_success_rate']:.1f}%"
                frame0_rate = f"{stats['frame0_success_rate']:.1f}%"
                fallbacks = f"{stats['avg_fallbacks']:.2f}"
            else:
                path_rate = frame0_rate = fallbacks = "-"

            lines.append(f"| {task_cell} | C{cond_id} | {path_rate} | {frame0_rate} | {fallbacks} |")

    return "\n".join(lines)


def generate_timing_md(all_results: Dict[str, TaskResults]) -> str:
    """Generate timing statistics table in Markdown format."""
    lines = []

    lines.append("# Inference Timing Statistics\n")
    lines.append("| Condition | VILA (ms) | ManiFlow (ms) | Overhead |")
    lines.append("|:---------:|:---------:|:-------------:|:--------:|")

    # Get average timing across all tasks
    timing_table = get_timing_stats_table(all_results)

    for cond_id in range(1, 7):
        vila_times = []
        mf_times = []

        for task in TASKS:
            stats = timing_table.get(task, {}).get(cond_id)
            if stats:
                if stats["avg_vila_ms"] > 0:
                    vila_times.append(stats["avg_vila_ms"])
                if stats["avg_maniflow_ms"] > 0:
                    mf_times.append(stats["avg_maniflow_ms"])

        avg_vila = sum(vila_times) / len(vila_times) if vila_times else 0
        avg_mf = sum(mf_times) / len(mf_times) if mf_times else 0

        vila_str = f"{avg_vila:.1f}" if avg_vila > 0 else "-"
        mf_str = f"{avg_mf:.1f}" if avg_mf > 0 else "-"
        overhead = f"+{avg_vila:.0f}ms" if avg_vila > 0 else "-"

        lines.append(f"| C{cond_id} | {vila_str} | {mf_str} | {overhead} |")

    return "\n".join(lines)


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_main_results_latex(all_results: Dict[str, TaskResults]) -> str:
    """Generate main results table in LaTeX format."""
    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Success rates (\%) across all conditions. C1-C3: trained on cluttered, C4-C6: trained on clean. All evaluated on cluttered.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{l|ccc|ccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c|}{In-Domain} & \multicolumn{3}{c}{Cross-Domain} \\")
    lines.append(r"Task & C1 & C2 & C3 & C4 & C5 & C6 \\")
    lines.append(r"\midrule")

    success_table = get_success_rates_table(all_results)

    for task in TASKS:
        task_desc = TASK_DESCRIPTIONS.get(task, task)
        row = f"{task_desc}"
        for cond_id in range(1, 7):
            rate = success_table.get(task, {}).get(cond_id)
            if rate is not None:
                row += f" & {rate:.1f}"
            else:
                row += " & -"
        row += r" \\"
        lines.append(row)

    lines.append(r"\midrule")

    # Average row
    avg_rates = compute_average_success_rates(all_results)
    row = r"\textbf{Average}"
    for cond_id in range(1, 7):
        rate = avg_rates.get(cond_id)
        if rate is not None:
            row += f" & \\textbf{{{rate:.1f}}}"
        else:
            row += " & -"
    row += r" \\"
    lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_hypothesis_latex(all_results: Dict[str, TaskResults]) -> str:
    """Generate hypothesis evaluation table in LaTeX format."""
    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Improvement from path guidance (percentage points). Positive values indicate improvement.}")
    lines.append(r"\label{tab:hypothesis}")
    lines.append(r"\begin{tabular}{l|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c|}{H1/H2: Path Effect} & \multicolumn{2}{c}{H3: Initial Path} \\")
    lines.append(r"Task & In-Dom & Cross-Dom & In-Dom & Cross-Dom \\")
    lines.append(r"\midrule")

    aggregated = aggregate_hypothesis_results(all_results)

    for task in TASKS:
        task_desc = TASK_DESCRIPTIONS.get(task, task)

        # H1/H2 improvements
        h1_data = aggregated["H1"]["by_task"].get(task, {})
        h2_data = aggregated["H2"]["by_task"].get(task, {})

        h1_imp = None
        for comp in h1_data.get("comparisons", []):
            if "cond3" in comp["label"]:
                h1_imp = comp["improvement"]
                break

        h2_imp = None
        for comp in h2_data.get("comparisons", []):
            if "cond6" in comp["label"]:
                h2_imp = comp["improvement"]
                break

        # H3 improvements
        h3_data = aggregated["H3"]["by_task"].get(task, {})
        h3_in = h3_cross = None
        for comp in h3_data.get("comparisons", []):
            if "in-domain" in comp["label"]:
                h3_in = comp["improvement"]
            elif "cross-domain" in comp["label"]:
                h3_cross = comp["improvement"]

        def fmt(v):
            if v is None:
                return "-"
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.1f}"

        row = f"{task_desc} & {fmt(h1_imp)} & {fmt(h2_imp)} & {fmt(h3_in)} & {fmt(h3_cross)}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# =============================================================================
# CSV Generation
# =============================================================================

def generate_main_results_csv(all_results: Dict[str, TaskResults], output_path: Path):
    """Generate main results table as CSV."""
    success_table = get_success_rates_table(all_results)
    avg_rates = compute_average_success_rates(all_results)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Task", "C1", "C2", "C3", "C4", "C5", "C6"])

        # Task rows
        for task in TASKS:
            row = [task]
            for cond_id in range(1, 7):
                rate = success_table.get(task, {}).get(cond_id)
                row.append(f"{rate:.1f}" if rate is not None else "")
            writer.writerow(row)

        # Average row
        row = ["Average"]
        for cond_id in range(1, 7):
            rate = avg_rates.get(cond_id)
            row.append(f"{rate:.1f}" if rate is not None else "")
        writer.writerow(row)


def generate_hypothesis_csv(all_results: Dict[str, TaskResults], output_path: Path):
    """Generate hypothesis evaluation table as CSV."""
    aggregated = aggregate_hypothesis_results(all_results)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "Task",
            "H1_InDomain_C1toC3",
            "H2_CrossDomain_C4toC6",
            "H3_InDomain_C2toC3",
            "H3_CrossDomain_C5toC6"
        ])

        for task in TASKS:
            h1_data = aggregated["H1"]["by_task"].get(task, {})
            h2_data = aggregated["H2"]["by_task"].get(task, {})
            h3_data = aggregated["H3"]["by_task"].get(task, {})

            h1_imp = None
            for comp in h1_data.get("comparisons", []):
                if "cond3" in comp["label"]:
                    h1_imp = comp["improvement"]
                    break

            h2_imp = None
            for comp in h2_data.get("comparisons", []):
                if "cond6" in comp["label"]:
                    h2_imp = comp["improvement"]
                    break

            h3_in = h3_cross = None
            for comp in h3_data.get("comparisons", []):
                if "in-domain" in comp["label"]:
                    h3_in = comp["improvement"]
                elif "cross-domain" in comp["label"]:
                    h3_cross = comp["improvement"]

            def fmt(v):
                return f"{v:.1f}" if v is not None else ""

            writer.writerow([task, fmt(h1_imp), fmt(h2_imp), fmt(h3_in), fmt(h3_cross)])


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready tables for HAMSTER + ManiFlow evaluation."
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "latex", "csv", "all"],
        default="all",
        help="Output format",
    )

    args = parser.parse_args()

    print("Loading evaluation results...")
    all_results = load_all_results()

    ensure_output_dirs()

    formats = [args.format] if args.format != "all" else ["markdown", "latex", "csv"]

    for fmt in formats:
        print(f"\nGenerating {fmt} tables...")

        if fmt == "markdown":
            # Main results
            md_content = generate_main_results_md(all_results)
            md_path = TABLES_DIR / "main_results.md"
            with open(md_path, "w") as f:
                f.write(md_content)
            print(f"  Saved: {md_path}")

            # Hypothesis table
            md_content = generate_hypothesis_table_md(all_results)
            md_path = TABLES_DIR / "hypothesis_results.md"
            with open(md_path, "w") as f:
                f.write(md_content)
            print(f"  Saved: {md_path}")

            # Path stats
            md_content = generate_path_stats_md(all_results)
            md_path = TABLES_DIR / "path_stats.md"
            with open(md_path, "w") as f:
                f.write(md_content)
            print(f"  Saved: {md_path}")

            # Timing
            md_content = generate_timing_md(all_results)
            md_path = TABLES_DIR / "timing_stats.md"
            with open(md_path, "w") as f:
                f.write(md_content)
            print(f"  Saved: {md_path}")

        elif fmt == "latex":
            # Main results
            latex_content = generate_main_results_latex(all_results)
            latex_path = TABLES_DIR / "main_results.tex"
            with open(latex_path, "w") as f:
                f.write(latex_content)
            print(f"  Saved: {latex_path}")

            # Hypothesis table
            latex_content = generate_hypothesis_latex(all_results)
            latex_path = TABLES_DIR / "hypothesis_results.tex"
            with open(latex_path, "w") as f:
                f.write(latex_content)
            print(f"  Saved: {latex_path}")

        elif fmt == "csv":
            # Main results
            csv_path = TABLES_DIR / "main_results.csv"
            generate_main_results_csv(all_results, csv_path)
            print(f"  Saved: {csv_path}")

            # Hypothesis
            csv_path = TABLES_DIR / "hypothesis_results.csv"
            generate_hypothesis_csv(all_results, csv_path)
            print(f"  Saved: {csv_path}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
