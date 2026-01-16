"""
Data loading utilities for HAMSTER + ManiFlow evaluation results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_DATA_DIR,
    TASKS,
    CONDITIONS,
    EVAL_SEED,
    get_condition_dir_name,
)


@dataclass
class EpisodeResult:
    """Single episode result."""
    episode_id: int
    task: str
    condition: str
    seed: int
    success: bool
    total_steps: int
    path_stats: Dict[str, Any]
    timing: Dict[str, Any]
    failure_reason: Optional[str]

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "EpisodeResult":
        return cls(
            episode_id=data["episode_id"],
            task=data["task"],
            condition=data["condition"],
            seed=data["seed"],
            success=data["success"],
            total_steps=data["total_steps"],
            path_stats=data.get("path_stats", {}),
            timing=data.get("timing", {}),
            failure_reason=data.get("failure_reason"),
        )


@dataclass
class ConditionResults:
    """Results for a single condition."""
    task: str
    condition_id: int
    condition_name: str
    train_env: str
    eval_env: str
    mode: str
    seed: int
    episodes: List[EpisodeResult] = field(default_factory=list)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_successes(self) -> int:
        return sum(1 for ep in self.episodes if ep.success)

    @property
    def num_failures(self) -> int:
        return sum(1 for ep in self.episodes if not ep.success)

    @property
    def success_rate(self) -> float:
        if self.num_episodes == 0:
            return 0.0
        return self.num_successes / self.num_episodes

    @property
    def success_rate_percent(self) -> float:
        return self.success_rate * 100

    # Path statistics (for VILA conditions)
    @property
    def total_path_calls(self) -> int:
        return sum(ep.path_stats.get("total_path_calls", 0) for ep in self.episodes)

    @property
    def total_path_successes(self) -> int:
        return sum(ep.path_stats.get("path_successes", 0) for ep in self.episodes)

    @property
    def total_path_failures(self) -> int:
        return sum(ep.path_stats.get("path_failures", 0) for ep in self.episodes)

    @property
    def path_success_rate(self) -> float:
        if self.total_path_calls == 0:
            return 0.0
        return self.total_path_successes / self.total_path_calls

    @property
    def frame0_success_rate(self) -> float:
        """Rate of successful path generation at frame 0."""
        frame0_results = [
            ep.path_stats.get("frame0_success")
            for ep in self.episodes
            if ep.path_stats.get("frame0_success") is not None
        ]
        if not frame0_results:
            return 0.0
        return sum(1 for r in frame0_results if r) / len(frame0_results)

    @property
    def total_fallbacks(self) -> int:
        return sum(ep.path_stats.get("fallbacks_used", 0) for ep in self.episodes)

    @property
    def avg_fallbacks_per_episode(self) -> float:
        if self.num_episodes == 0:
            return 0.0
        return self.total_fallbacks / self.num_episodes

    # Timing statistics
    @property
    def avg_vila_inference_ms(self) -> float:
        """Average VILA inference time in ms."""
        all_times = []
        for ep in self.episodes:
            times = ep.timing.get("vila_inference_ms", [])
            if times:
                all_times.extend(times)
        if not all_times:
            return 0.0
        return sum(all_times) / len(all_times)

    @property
    def avg_maniflow_inference_ms(self) -> float:
        """Average ManiFlow inference time in ms."""
        all_times = []
        for ep in self.episodes:
            times = ep.timing.get("maniflow_inference_ms", [])
            if times:
                all_times.extend(times)
        if not all_times:
            return 0.0
        return sum(all_times) / len(all_times)

    @property
    def avg_episode_ms(self) -> float:
        """Average total episode time in ms."""
        times = [ep.timing.get("total_episode_ms", 0) for ep in self.episodes]
        if not times:
            return 0.0
        return sum(times) / len(times)


@dataclass
class TaskResults:
    """Results for a single task across all conditions."""
    task: str
    conditions: Dict[int, ConditionResults] = field(default_factory=dict)

    def get_condition(self, condition_id: int) -> Optional[ConditionResults]:
        return self.conditions.get(condition_id)

    def get_success_rate(self, condition_id: int) -> float:
        cond = self.get_condition(condition_id)
        if cond is None:
            return 0.0
        return cond.success_rate


def load_episodes_jsonl(jsonl_path: Path) -> List[EpisodeResult]:
    """Load episodes from a JSONL file."""
    episodes = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                episodes.append(EpisodeResult.from_json(data))
    return episodes


def find_run_directory(condition_dir: Path, seed: int = EVAL_SEED) -> Optional[Path]:
    """Find the run directory for a given seed."""
    # Look for run_seed{seed}_001, run_seed{seed}_002, etc.
    # Return the first one found (should only be one per seed now)
    pattern = f"run_seed{seed}_*"
    matches = list(condition_dir.glob(pattern))
    if matches:
        # Sort and return the first one
        return sorted(matches)[0]
    return None


def load_condition_results(
    task: str,
    condition_id: int,
    eval_env: str = "cluttered",
    seed: int = EVAL_SEED,
) -> Optional[ConditionResults]:
    """Load results for a single condition."""
    cond_config = CONDITIONS.get(condition_id)
    if cond_config is None:
        print(f"Unknown condition ID: {condition_id}")
        return None

    # Build path to condition directory
    condition_dir_name = get_condition_dir_name(condition_id, eval_env)
    condition_dir = RAW_DATA_DIR / task / condition_dir_name

    if not condition_dir.exists():
        print(f"Condition directory not found: {condition_dir}")
        return None

    # Find run directory
    run_dir = find_run_directory(condition_dir, seed)
    if run_dir is None:
        print(f"No run directory found for seed {seed} in {condition_dir}")
        return None

    # Load episodes.jsonl
    jsonl_path = run_dir / "episodes.jsonl"
    if not jsonl_path.exists():
        print(f"episodes.jsonl not found: {jsonl_path}")
        return None

    episodes = load_episodes_jsonl(jsonl_path)

    return ConditionResults(
        task=task,
        condition_id=condition_id,
        condition_name=cond_config["name"],
        train_env=cond_config["train_env"],
        eval_env=eval_env,
        mode=cond_config["mode"],
        seed=seed,
        episodes=episodes,
    )


def load_task_results(
    task: str,
    eval_env: str = "cluttered",
    seed: int = EVAL_SEED,
) -> TaskResults:
    """Load results for all conditions of a task."""
    task_results = TaskResults(task=task)

    for condition_id in CONDITIONS.keys():
        cond_results = load_condition_results(task, condition_id, eval_env, seed)
        if cond_results is not None:
            task_results.conditions[condition_id] = cond_results

    return task_results


def load_all_results(
    tasks: List[str] = None,
    eval_env: str = "cluttered",
    seed: int = EVAL_SEED,
) -> Dict[str, TaskResults]:
    """Load results for all tasks."""
    if tasks is None:
        tasks = TASKS

    all_results = {}
    for task in tasks:
        all_results[task] = load_task_results(task, eval_env, seed)

    return all_results


# =============================================================================
# Summary functions
# =============================================================================

def get_success_rates_table(
    all_results: Dict[str, TaskResults],
) -> Dict[str, Dict[int, float]]:
    """
    Get success rates for all tasks and conditions.

    Returns:
        Dict mapping task -> {condition_id -> success_rate}
    """
    table = {}
    for task, task_results in all_results.items():
        table[task] = {}
        for cond_id in CONDITIONS.keys():
            cond = task_results.get_condition(cond_id)
            if cond is not None:
                table[task][cond_id] = cond.success_rate_percent
            else:
                table[task][cond_id] = None
    return table


def get_path_stats_table(
    all_results: Dict[str, TaskResults],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Get path generation statistics for VILA conditions.

    Returns:
        Dict mapping task -> {condition_id -> {stat_name -> value}}
    """
    vila_conditions = [cid for cid, c in CONDITIONS.items() if c["uses_vila"]]

    table = {}
    for task, task_results in all_results.items():
        table[task] = {}
        for cond_id in vila_conditions:
            cond = task_results.get_condition(cond_id)
            if cond is not None:
                table[task][cond_id] = {
                    "path_success_rate": cond.path_success_rate * 100,
                    "frame0_success_rate": cond.frame0_success_rate * 100,
                    "avg_fallbacks": cond.avg_fallbacks_per_episode,
                }
            else:
                table[task][cond_id] = None
    return table


def get_timing_stats_table(
    all_results: Dict[str, TaskResults],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Get timing statistics for all conditions.

    Returns:
        Dict mapping task -> {condition_id -> {stat_name -> value}}
    """
    table = {}
    for task, task_results in all_results.items():
        table[task] = {}
        for cond_id in CONDITIONS.keys():
            cond = task_results.get_condition(cond_id)
            if cond is not None:
                table[task][cond_id] = {
                    "avg_vila_ms": cond.avg_vila_inference_ms,
                    "avg_maniflow_ms": cond.avg_maniflow_inference_ms,
                    "avg_episode_ms": cond.avg_episode_ms,
                }
            else:
                table[task][cond_id] = None
    return table


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Loading evaluation results...")
    print("=" * 60)

    all_results = load_all_results()

    for task, task_results in all_results.items():
        print(f"\n{task}:")
        print("-" * 40)
        for cond_id, cond in sorted(task_results.conditions.items()):
            print(f"  Condition {cond_id}: {cond.success_rate_percent:.1f}% "
                  f"({cond.num_successes}/{cond.num_episodes})")

    print("\n" + "=" * 60)
    print("Success rates table:")
    print("=" * 60)

    table = get_success_rates_table(all_results)

    # Print header
    header = "Task".ljust(20) + "".join(f"C{i}".rjust(8) for i in range(1, 7))
    print(header)
    print("-" * len(header))

    # Print rows
    for task, rates in table.items():
        row = task.ljust(20)
        for cond_id in range(1, 7):
            rate = rates.get(cond_id)
            if rate is not None:
                row += f"{rate:7.1f}%"
            else:
                row += "     N/A"
        print(row)
