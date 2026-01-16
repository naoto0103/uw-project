#!/usr/bin/env python3
"""
All tasks summary analysis script for HAMSTER + ManiFlow evaluation.

Usage:
    python analyze_all_tasks.py
    python analyze_all_tasks.py --output_report
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any

from config import (
    TASKS,
    TASK_DESCRIPTIONS,
    CONDITIONS,
    HYPOTHESES,
    REPORTS_DIR,
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
    format_ms,
    generate_report_header,
    get_timestamp,
)


def compute_average_success_rates(
    all_results: Dict[str, TaskResults],
) -> Dict[int, float]:
    """Compute average success rate across all tasks for each condition."""
    avg_rates = {}

    for cond_id in CONDITIONS.keys():
        rates = []
        for task, task_results in all_results.items():
            cond = task_results.get_condition(cond_id)
            if cond is not None:
                rates.append(cond.success_rate_percent)

        if rates:
            avg_rates[cond_id] = sum(rates) / len(rates)
        else:
            avg_rates[cond_id] = None

    return avg_rates


def aggregate_hypothesis_results(
    all_results: Dict[str, TaskResults],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate hypothesis evaluation results across all tasks."""
    aggregated = {}

    for h_name in HYPOTHESES.keys():
        aggregated[h_name] = {
            "hypothesis": h_name,
            "description": HYPOTHESES[h_name]["description"],
            "by_task": {},
            "summary": {},
        }

    for task, task_results in all_results.items():
        hypotheses = evaluate_all_hypotheses(task_results)
        for h_name, h_results in hypotheses.items():
            aggregated[h_name]["by_task"][task] = h_results

    # Compute summary statistics
    for h_name, h_data in aggregated.items():
        comparisons_data = HYPOTHESES[h_name]["comparisons"]

        if h_name == "H4":
            # H4 has different structure
            for i, comp_template in enumerate(comparisons_data):
                improvements_in = []
                improvements_cross = []

                for task, task_h in h_data["by_task"].items():
                    comps = task_h.get("comparisons", [])
                    if i < len(comps):
                        improvements_in.append(comps[i]["in_domain_improvement"])
                        improvements_cross.append(comps[i]["cross_domain_improvement"])

                if improvements_in and improvements_cross:
                    h_data["summary"][comp_template["label"]] = {
                        "avg_in_domain": sum(improvements_in) / len(improvements_in),
                        "avg_cross_domain": sum(improvements_cross) / len(improvements_cross),
                        "tasks_supporting": sum(
                            1 for j in range(len(improvements_in))
                            if improvements_cross[j] > improvements_in[j]
                        ),
                        "total_tasks": len(improvements_in),
                    }
        else:
            # H1, H2, H3
            for i, comp_template in enumerate(comparisons_data):
                improvements = []

                for task, task_h in h_data["by_task"].items():
                    comps = task_h.get("comparisons", [])
                    if i < len(comps):
                        improvements.append(comps[i]["improvement"])

                if improvements:
                    h_data["summary"][comp_template["label"]] = {
                        "avg_improvement": sum(improvements) / len(improvements),
                        "tasks_supporting": sum(1 for imp in improvements if imp > 0),
                        "total_tasks": len(improvements),
                    }

    return aggregated


def print_summary(all_results: Dict[str, TaskResults]):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("HAMSTER + ManiFlow Evaluation Summary")
    print("=" * 70)

    # Main results table
    print("\n## Main Results: Success Rates (%)")
    print("-" * 70)

    # Header
    header = f"{'Task':<20}"
    for cond_id in range(1, 7):
        header += f"{'C' + str(cond_id):>8}"
    print(header)
    print("-" * 70)

    # Task rows
    success_table = get_success_rates_table(all_results)
    for task in TASKS:
        row = f"{task:<20}"
        for cond_id in range(1, 7):
            rate = success_table.get(task, {}).get(cond_id)
            if rate is not None:
                row += f"{rate:>7.1f}%"
            else:
                row += f"{'N/A':>8}"
        print(row)

    # Average row
    avg_rates = compute_average_success_rates(all_results)
    row = f"{'Average':<20}"
    for cond_id in range(1, 7):
        rate = avg_rates.get(cond_id)
        if rate is not None:
            row += f"{rate:>7.1f}%"
        else:
            row += f"{'N/A':>8}"
    print("-" * 70)
    print(row)

    # Hypothesis summary
    print("\n## Hypothesis Evaluation Summary")
    print("-" * 70)

    aggregated = aggregate_hypothesis_results(all_results)

    for h_name, h_data in aggregated.items():
        print(f"\n### {h_name}: {h_data['description']}")

        for comp_label, summary in h_data["summary"].items():
            if h_name == "H4":
                avg_in = summary["avg_in_domain"]
                avg_cross = summary["avg_cross_domain"]
                support = summary["tasks_supporting"]
                total = summary["total_tasks"]
                print(f"  {comp_label}:")
                print(f"    Avg in-domain Δ: {format_improvement(avg_in)}")
                print(f"    Avg cross-domain Δ: {format_improvement(avg_cross)}")
                print(f"    Tasks supporting: {support}/{total}")
            else:
                avg_imp = summary["avg_improvement"]
                support = summary["tasks_supporting"]
                total = summary["total_tasks"]
                print(f"  {comp_label}:")
                print(f"    Avg improvement: {format_improvement(avg_imp)}")
                print(f"    Tasks supporting: {support}/{total}")

    # Path statistics
    print("\n## Path Generation Statistics (Aggregated)")
    print("-" * 70)

    path_table = get_path_stats_table(all_results)
    vila_conditions = [cid for cid, c in CONDITIONS.items() if c["uses_vila"]]

    for cond_id in vila_conditions:
        print(f"\nCondition {cond_id} ({CONDITIONS[cond_id]['short_name']}):")
        path_success_rates = []
        frame0_rates = []
        fallbacks = []

        for task in TASKS:
            stats = path_table.get(task, {}).get(cond_id)
            if stats:
                path_success_rates.append(stats["path_success_rate"])
                frame0_rates.append(stats["frame0_success_rate"])
                fallbacks.append(stats["avg_fallbacks"])

        if path_success_rates:
            print(f"  Avg path success: {sum(path_success_rates)/len(path_success_rates):.1f}%")
            print(f"  Avg frame0 success: {sum(frame0_rates)/len(frame0_rates):.1f}%")
            print(f"  Avg fallbacks/episode: {sum(fallbacks)/len(fallbacks):.2f}")


def generate_summary_report(all_results: Dict[str, TaskResults]) -> str:
    """Generate a markdown summary report."""
    lines = []

    # Header
    lines.append(generate_report_header("HAMSTER + ManiFlow Evaluation Summary"))

    # Main results table
    lines.append("## Main Results: Success Rates\n")

    # Table header
    header = "| Task |"
    for cond_id in range(1, 7):
        header += f" C{cond_id} |"
    lines.append(header)

    separator = "|------|"
    for _ in range(6):
        separator += "------|"
    lines.append(separator)

    # Task rows
    success_table = get_success_rates_table(all_results)
    for task in TASKS:
        task_desc = TASK_DESCRIPTIONS.get(task, task)
        row = f"| {task_desc} |"
        for cond_id in range(1, 7):
            rate = success_table.get(task, {}).get(cond_id)
            if rate is not None:
                row += f" {rate:.1f}% |"
            else:
                row += " N/A |"
        lines.append(row)

    # Average row
    avg_rates = compute_average_success_rates(all_results)
    row = "| **Average** |"
    for cond_id in range(1, 7):
        rate = avg_rates.get(cond_id)
        if rate is not None:
            row += f" **{rate:.1f}%** |"
        else:
            row += " N/A |"
    lines.append(row)
    lines.append("\n")

    # Condition legend
    lines.append("### Condition Legend\n")
    lines.append("| Condition | Description |")
    lines.append("|-----------|-------------|")
    for cond_id, cond in CONDITIONS.items():
        lines.append(f"| C{cond_id} | {cond['description']} |")
    lines.append("\n")

    # Hypothesis summary
    lines.append("## Hypothesis Evaluation\n")

    aggregated = aggregate_hypothesis_results(all_results)

    for h_name, h_data in aggregated.items():
        lines.append(f"### {h_name}: {h_data['description']}\n")

        if h_name == "H4":
            lines.append("| Comparison | Avg In-Domain Δ | Avg Cross-Domain Δ | Support |")
            lines.append("|------------|-----------------|---------------------|---------|")
            for comp_label, summary in h_data["summary"].items():
                avg_in = format_improvement(summary["avg_in_domain"])
                avg_cross = format_improvement(summary["avg_cross_domain"])
                support = f"{summary['tasks_supporting']}/{summary['total_tasks']}"
                lines.append(f"| {comp_label} | {avg_in} | {avg_cross} | {support} |")
        else:
            lines.append("| Comparison | Avg Improvement | Tasks Supporting |")
            lines.append("|------------|-----------------|------------------|")
            for comp_label, summary in h_data["summary"].items():
                avg_imp = format_improvement(summary["avg_improvement"])
                support = f"{summary['tasks_supporting']}/{summary['total_tasks']}"
                lines.append(f"| {comp_label} | {avg_imp} | {support} |")

        lines.append("\n")

    # Detailed results by task
    lines.append("## Detailed Results by Task\n")

    for h_name, h_data in aggregated.items():
        lines.append(f"### {h_name} by Task\n")

        if h_name == "H4":
            lines.append("| Task | In-Domain Δ | Cross-Domain Δ | Diff |")
            lines.append("|------|-------------|----------------|------|")
            for task in TASKS:
                task_h = h_data["by_task"].get(task, {})
                comps = task_h.get("comparisons", [])
                if comps:
                    comp = comps[0]  # First comparison
                    in_d = format_improvement(comp["in_domain_improvement"])
                    cross_d = format_improvement(comp["cross_domain_improvement"])
                    diff = format_improvement(comp["difference"])
                    lines.append(f"| {task} | {in_d} | {cross_d} | {diff} |")
                else:
                    lines.append(f"| {task} | N/A | N/A | N/A |")
        else:
            lines.append("| Task | Baseline | Treatment | Improvement |")
            lines.append("|------|----------|-----------|-------------|")
            for task in TASKS:
                task_h = h_data["by_task"].get(task, {})
                comps = task_h.get("comparisons", [])
                if comps:
                    comp = comps[0]  # First comparison
                    baseline = format_percent(comp["baseline"])
                    treatment = format_percent(comp["treatment"])
                    improvement = format_improvement(comp["improvement"])
                    lines.append(f"| {task} | {baseline} | {treatment} | {improvement} |")
                else:
                    lines.append(f"| {task} | N/A | N/A | N/A |")

        lines.append("\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze HAMSTER + ManiFlow evaluation results across all tasks."
    )
    parser.add_argument(
        "--output_report",
        action="store_true",
        help="Generate and save markdown report",
    )

    args = parser.parse_args()

    print("Loading all evaluation results...")
    all_results = load_all_results()

    # Print summary to console
    print_summary(all_results)

    # Generate and save report if requested
    if args.output_report:
        ensure_output_dirs()
        report = generate_summary_report(all_results)
        report_path = REPORTS_DIR / "summary_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    exit(main())
