#!/usr/bin/env python3
"""
Single task analysis script for HAMSTER + ManiFlow evaluation.

Usage:
    python analyze_task.py --task click_bell
    python analyze_task.py --task move_can_pot --output_report
    python analyze_task.py --all
"""

import argparse
from pathlib import Path

from config import (
    TASKS,
    TASK_DESCRIPTIONS,
    CONDITIONS,
    REPORTS_DIR,
)
from scripts.load_results import (
    load_task_results,
    TaskResults,
)
from scripts.utils import (
    ensure_output_dirs,
    evaluate_all_hypotheses,
    format_percent,
    format_improvement,
    format_ms,
    generate_report_header,
    generate_task_section_header,
    generate_condition_table_md,
    generate_hypothesis_section_md,
    get_timestamp,
)


def print_task_summary(task_results: TaskResults):
    """Print a summary of task results to console."""
    task = task_results.task
    task_desc = TASK_DESCRIPTIONS.get(task, task)

    print(f"\n{'=' * 60}")
    print(f"Task: {task_desc} ({task})")
    print(f"{'=' * 60}")

    # Success rates table
    print("\n## Success Rates")
    print("-" * 50)
    print(f"{'Cond':<6} {'Train':<10} {'Mode':<25} {'Rate':<10}")
    print("-" * 50)

    for cond_id in sorted(task_results.conditions.keys()):
        cond = task_results.get_condition(cond_id)
        if cond is None:
            continue
        cond_config = CONDITIONS[cond_id]
        rate_str = format_percent(cond.success_rate_percent)
        episodes_str = f"({cond.num_successes}/{cond.num_episodes})"
        print(f"{cond_id:<6} {cond_config['train_env']:<10} {cond_config['mode']:<25} {rate_str:<10} {episodes_str}")

    # Hypothesis evaluation
    print("\n## Hypothesis Evaluation")
    print("-" * 50)

    hypotheses = evaluate_all_hypotheses(task_results)

    for h_name, h_results in hypotheses.items():
        print(f"\n### {h_name}: {h_results['description']}")

        comparisons = h_results.get("comparisons", [])
        if not comparisons:
            print("  No data available")
            continue

        if h_name == "H4":
            for comp in comparisons:
                in_delta = format_improvement(comp["in_domain_improvement"])
                cross_delta = format_improvement(comp["cross_domain_improvement"])
                result = "✓" if comp["cross_domain_larger"] else "✗"
                print(f"  {comp['label']}: In-domain {in_delta}, Cross-domain {cross_delta} [{result}]")
        else:
            for comp in comparisons:
                improvement = format_improvement(comp["improvement"])
                result = "✓" if comp["improved"] else "✗"
                print(f"  {comp['label']}: {format_percent(comp['baseline'])} → {format_percent(comp['treatment'])} ({improvement}) [{result}]")

    # Path statistics (for VILA conditions)
    print("\n## Path Generation Statistics (VILA conditions)")
    print("-" * 50)

    vila_conditions = [cid for cid, c in CONDITIONS.items() if c["uses_vila"]]
    for cond_id in vila_conditions:
        cond = task_results.get_condition(cond_id)
        if cond is None:
            continue
        print(f"\n  Condition {cond_id} ({CONDITIONS[cond_id]['short_name']}):")
        print(f"    Path success rate: {format_percent(cond.path_success_rate * 100)}")
        print(f"    Frame 0 success rate: {format_percent(cond.frame0_success_rate * 100)}")
        print(f"    Avg fallbacks/episode: {cond.avg_fallbacks_per_episode:.2f}")

    # Timing statistics
    print("\n## Timing Statistics")
    print("-" * 50)
    print(f"{'Cond':<6} {'VILA (ms)':<15} {'ManiFlow (ms)':<15} {'Episode (s)':<15}")
    print("-" * 50)

    for cond_id in sorted(task_results.conditions.keys()):
        cond = task_results.get_condition(cond_id)
        if cond is None:
            continue
        vila_ms = format_ms(cond.avg_vila_inference_ms)
        mf_ms = format_ms(cond.avg_maniflow_inference_ms)
        ep_s = f"{cond.avg_episode_ms / 1000:.1f}s" if cond.avg_episode_ms > 0 else "-"
        print(f"{cond_id:<6} {vila_ms:<15} {mf_ms:<15} {ep_s:<15}")


def generate_task_report(task_results: TaskResults) -> str:
    """Generate a markdown report for a task."""
    task = task_results.task
    task_desc = TASK_DESCRIPTIONS.get(task, task)

    lines = []

    # Header
    lines.append(generate_report_header(f"Analysis Report: {task_desc}"))

    # Success rates
    lines.append("## Success Rates\n")

    success_rates = {}
    for cond_id in sorted(task_results.conditions.keys()):
        cond = task_results.get_condition(cond_id)
        if cond is not None:
            success_rates[cond_id] = cond.success_rate_percent

    lines.append(generate_condition_table_md(success_rates))
    lines.append("\n")

    # Hypothesis evaluation
    lines.append("## Hypothesis Evaluation\n")

    hypotheses = evaluate_all_hypotheses(task_results)
    for h_name, h_results in hypotheses.items():
        lines.append(generate_hypothesis_section_md(h_results))

    # Path statistics
    lines.append("## Path Generation Statistics\n")
    lines.append("*Statistics for VILA-based conditions (2, 3, 5, 6)*\n")

    lines.append("| Condition | Path Success | Frame0 Success | Avg Fallbacks |")
    lines.append("|-----------|--------------|----------------|---------------|")

    vila_conditions = [cid for cid, c in CONDITIONS.items() if c["uses_vila"]]
    for cond_id in vila_conditions:
        cond = task_results.get_condition(cond_id)
        if cond is None:
            lines.append(f"| {cond_id} | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {cond_id} | {format_percent(cond.path_success_rate * 100)} | "
            f"{format_percent(cond.frame0_success_rate * 100)} | "
            f"{cond.avg_fallbacks_per_episode:.2f} |"
        )
    lines.append("\n")

    # Timing statistics
    lines.append("## Timing Statistics\n")

    lines.append("| Condition | VILA Inference | ManiFlow Inference | Total Episode |")
    lines.append("|-----------|----------------|---------------------|---------------|")

    for cond_id in sorted(task_results.conditions.keys()):
        cond = task_results.get_condition(cond_id)
        if cond is None:
            continue
        vila_ms = format_ms(cond.avg_vila_inference_ms)
        mf_ms = format_ms(cond.avg_maniflow_inference_ms)
        ep_s = f"{cond.avg_episode_ms / 1000:.1f}s" if cond.avg_episode_ms > 0 else "-"
        lines.append(f"| {cond_id} | {vila_ms} | {mf_ms} | {ep_s} |")

    lines.append("\n")

    return "\n".join(lines)


def analyze_task(task: str, output_report: bool = False) -> TaskResults:
    """Analyze a single task."""
    print(f"\nAnalyzing task: {task}")

    # Load results
    task_results = load_task_results(task)

    if not task_results.conditions:
        print(f"  No results found for task: {task}")
        return task_results

    # Print summary to console
    print_task_summary(task_results)

    # Generate and save report if requested
    if output_report:
        ensure_output_dirs()
        report = generate_task_report(task_results)
        report_path = REPORTS_DIR / f"{task}_analysis.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

    return task_results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze HAMSTER + ManiFlow evaluation results for a single task."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=TASKS,
        help="Task to analyze",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all tasks",
    )
    parser.add_argument(
        "--output_report",
        action="store_true",
        help="Generate and save markdown report",
    )

    args = parser.parse_args()

    if args.all:
        for task in TASKS:
            analyze_task(task, output_report=args.output_report)
    elif args.task:
        analyze_task(args.task, output_report=args.output_report)
    else:
        parser.print_help()
        print("\nError: Please specify --task or --all")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
