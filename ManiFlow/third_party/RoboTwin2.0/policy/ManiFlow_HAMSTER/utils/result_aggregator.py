"""
Result Aggregator for combining multiple evaluation runs.

This module aggregates results from 5 evaluation runs (different seeds)
and computes mean and standard deviation of success rates.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


class ResultAggregator:
    """
    Aggregates results from multiple evaluation runs.

    Reads summary.json files from each run and computes
    aggregated statistics (mean, std) across runs.
    """

    def __init__(self, output_dir: str, task: str, condition: str):
        """
        Initialize the ResultAggregator.

        Args:
            output_dir: Base output directory for results
            task: Task name
            condition: Condition string
        """
        self.output_dir = Path(output_dir)
        self.task = task
        self.condition = condition

        # Directory for this task/condition
        self.condition_dir = self.output_dir / task / condition

    def find_runs(self) -> List[Path]:
        """
        Find all run directories for this task/condition.

        Returns:
            List of run directory paths
        """
        if not self.condition_dir.exists():
            return []

        runs = []
        for run_dir in self.condition_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_seed"):
                summary_path = run_dir / "summary.json"
                if summary_path.exists():
                    runs.append(run_dir)

        return sorted(runs)

    def load_summaries(self) -> List[Dict[str, Any]]:
        """
        Load summary.json from all runs.

        Returns:
            List of summary dictionaries
        """
        summaries = []
        for run_dir in self.find_runs():
            summary_path = run_dir / "summary.json"
            with open(summary_path, "r") as f:
                summary = json.load(f)
                summary["run_dir"] = str(run_dir)
                summaries.append(summary)
        return summaries

    def aggregate(self) -> Optional[Dict[str, Any]]:
        """
        Aggregate results from all runs.

        Returns:
            Aggregated statistics dictionary, or None if no runs found
        """
        summaries = self.load_summaries()

        if not summaries:
            return None

        # Extract success rates
        success_rates = [s["success_rate"] for s in summaries]
        seeds = [s["seed"] for s in summaries]

        # Extract path success rates (may not exist for original mode)
        path_success_rates = []
        for s in summaries:
            path_stats = s.get("path_stats", {})
            if "path_success_rate" in path_stats:
                path_success_rates.append(path_stats["path_success_rate"])

        # Compute aggregated statistics
        aggregated = {
            "task": self.task,
            "condition": self.condition,
            "runs": len(summaries),
            "seeds": seeds,
            "success_rate_mean": float(np.mean(success_rates)),
            "success_rate_std": float(np.std(success_rates)),
            "success_rates": success_rates,
        }

        # Add path success rate stats if available
        if path_success_rates:
            aggregated["path_success_rate_mean"] = float(np.mean(path_success_rates))
            aggregated["path_success_rate_std"] = float(np.std(path_success_rates))
            aggregated["path_success_rates"] = path_success_rates

        # Add timing aggregates
        avg_episode_times = []
        avg_vila_times = []
        avg_maniflow_times = []

        for s in summaries:
            timing = s.get("timing", {})
            if "avg_episode_ms" in timing:
                avg_episode_times.append(timing["avg_episode_ms"])
            if "avg_vila_inference_ms" in timing:
                avg_vila_times.append(timing["avg_vila_inference_ms"])
            if "avg_maniflow_inference_ms" in timing:
                avg_maniflow_times.append(timing["avg_maniflow_inference_ms"])

        if avg_episode_times:
            aggregated["timing"] = {
                "avg_episode_ms_mean": float(np.mean(avg_episode_times)),
                "avg_episode_ms_std": float(np.std(avg_episode_times)),
            }
            if avg_vila_times:
                aggregated["timing"]["avg_vila_inference_ms_mean"] = float(np.mean(avg_vila_times))
            if avg_maniflow_times:
                aggregated["timing"]["avg_maniflow_inference_ms_mean"] = float(np.mean(avg_maniflow_times))

        return aggregated

    def save_aggregated(self) -> Optional[str]:
        """
        Generate and save aggregated results.

        Returns:
            Path to aggregated.json, or None if no runs found
        """
        aggregated = self.aggregate()

        if aggregated is None:
            return None

        aggregated_path = self.condition_dir / "aggregated.json"
        with open(aggregated_path, "w") as f:
            json.dump(aggregated, f, indent=2)

        return str(aggregated_path)


def aggregate_all_conditions(
    output_dir: str,
    task: str,
    conditions: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate results for all conditions of a task.

    Args:
        output_dir: Base output directory
        task: Task name
        conditions: List of conditions to aggregate (default: all 6 conditions)

    Returns:
        Dictionary mapping condition to aggregated results
    """
    if conditions is None:
        conditions = [
            "condition1_cluttered_original",
            "condition2_cluttered_overlay_current",
            "condition3_cluttered_overlay_initial_current",
            "condition4_clean_original",
            "condition5_clean_overlay_current",
            "condition6_clean_overlay_initial_current",
        ]

    results = {}
    for condition in conditions:
        aggregator = ResultAggregator(output_dir, task, condition)
        aggregated = aggregator.aggregate()
        if aggregated is not None:
            results[condition] = aggregated
            aggregator.save_aggregated()

    return results


def generate_comparison_table(
    output_dir: str,
    tasks: Optional[List[str]] = None,
) -> str:
    """
    Generate a comparison table of success rates across tasks and conditions.

    Args:
        output_dir: Base output directory
        tasks: List of tasks (default: all 6 tasks)

    Returns:
        Markdown formatted table
    """
    if tasks is None:
        tasks = [
            "click_bell",
            "turn_switch",
            "move_can_pot",
            "open_microwave",
            "adjust_bottle",
            "beat_block_hammer",
        ]

    conditions = [
        ("condition1_cluttered_original", "Original (cluttered)"),
        ("condition2_cluttered_overlay_current", "Current (cluttered)"),
        ("condition3_cluttered_overlay_initial_current", "Init+Curr (cluttered)"),
        ("condition4_clean_original", "Original (clean)"),
        ("condition5_clean_overlay_current", "Current (clean)"),
        ("condition6_clean_overlay_initial_current", "Init+Curr (clean)"),
    ]

    # Header
    header = "| Task |"
    separator = "|------|"
    for _, label in conditions:
        header += f" {label} |"
        separator += "--------|"

    lines = [header, separator]

    # Data rows
    for task in tasks:
        row = f"| {task} |"
        for cond_id, _ in conditions:
            aggregator = ResultAggregator(output_dir, task, cond_id)
            aggregated = aggregator.aggregate()

            if aggregated is not None:
                mean = aggregated["success_rate_mean"]
                std = aggregated["success_rate_std"]
                row += f" {mean:.2f}±{std:.2f} |"
            else:
                row += " - |"

        lines.append(row)

    return "\n".join(lines)


if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 50)
    print("ResultAggregator Test")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        task = "click_bell"
        condition = "condition3_cluttered_overlay_initial_current"

        print(f"\n1. Creating test summaries in {tmpdir}...")

        for seed in [42, 43, 44, 45, 46]:
            run_dir = Path(tmpdir) / task / condition / f"run_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            summary = {
                "task": task,
                "condition": condition,
                "seed": seed,
                "total_episodes": 100,
                "successes": 80 + seed % 10,  # Vary slightly
                "failures": 20 - seed % 10,
                "success_rate": (80 + seed % 10) / 100,
                "path_stats": {
                    "total_path_calls": 823,
                    "path_success_rate": 0.94 + (seed % 5) / 100,
                    "frame0_success_rate": 0.98,
                    "avg_retries_per_episode": 0.12,
                    "avg_fallbacks_per_episode": 0.08,
                },
                "timing": {
                    "avg_vila_inference_ms": 242.5 + seed % 20,
                    "avg_maniflow_inference_ms": 11.8,
                    "avg_episode_ms": 14523.2,
                },
            }

            with open(run_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        # Test aggregation
        print("\n2. Testing ResultAggregator...")
        aggregator = ResultAggregator(tmpdir, task, condition)

        runs = aggregator.find_runs()
        print(f"   Found {len(runs)} runs")
        assert len(runs) == 5

        aggregated = aggregator.aggregate()
        print(f"   Success rate: {aggregated['success_rate_mean']:.3f} ± {aggregated['success_rate_std']:.3f}")
        print(f"   Runs: {aggregated['runs']}")

        # Save aggregated
        print("\n3. Saving aggregated.json...")
        path = aggregator.save_aggregated()
        print(f"   Saved to: {path}")

        # Verify file
        with open(path, "r") as f:
            loaded = json.load(f)
            assert loaded["runs"] == 5
            assert "success_rate_mean" in loaded
            assert "success_rate_std" in loaded

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
