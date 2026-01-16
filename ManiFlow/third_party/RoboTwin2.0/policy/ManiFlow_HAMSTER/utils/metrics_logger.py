"""
Metrics Logger for structured JSONL output.

This module provides:
- EpisodeMetrics: Dataclass for episode-level metrics
- MetricsLogger: JSONL writer for episode logs and summary generation
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import time


@dataclass
class EpisodeMetrics:
    """
    Metrics for a single evaluation episode.

    Attributes:
        episode_id: Episode number (0-indexed)
        task: Task name
        condition: Condition string (e.g., "condition3_cluttered_overlay_initial_current")
        seed: Evaluation seed
        success: Whether the episode succeeded
        total_steps: Total number of steps executed
        path_stats: Path generation statistics
        timing: Timing information
        failure_reason: Reason for failure (if failed)
    """
    episode_id: int
    task: str
    condition: str
    seed: int
    success: bool
    total_steps: int
    path_stats: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class MetricsLogger:
    """
    Logger for evaluation metrics in JSONL format.

    Writes episode-level metrics to episodes.jsonl and generates
    summary.json after all episodes are complete.
    """

    def __init__(
        self,
        output_dir: str,
        task: str,
        condition: str,
        seed: int,
        train_env: str = "cluttered",
        eval_env: str = "cluttered",
    ):
        """
        Initialize the MetricsLogger.

        Args:
            output_dir: Base output directory for results
            task: Task name
            condition: Condition string
            seed: Evaluation seed
            train_env: Training data environment (clean or cluttered)
            eval_env: Evaluation environment (clean or cluttered)
        """
        self.task = task
        self.condition = condition
        self.seed = seed
        self.train_env = train_env
        self.eval_env = eval_env

        # Create output directory structure with auto-incrementing run number
        base_run_dir = Path(output_dir) / task / condition
        base_run_dir.mkdir(parents=True, exist_ok=True)

        # Find next available run number
        run_number = self._get_next_run_number(base_run_dir, seed)
        self.run_dir = base_run_dir / f"run_seed{seed}_{run_number:03d}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_number = run_number

        # File paths
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.summary_path = self.run_dir / "summary.json"

        # Episode storage
        self.episodes: List[EpisodeMetrics] = []

        # Timing
        self.start_time = time.time()

    def _get_next_run_number(self, base_dir: Path, seed: int) -> int:
        """
        Find the next available run number for the given seed.

        Scans existing directories matching 'run_seed{seed}_XXX' pattern
        and returns the next available number.

        Args:
            base_dir: Base directory containing run directories
            seed: Evaluation seed

        Returns:
            Next available run number (starting from 1)
        """
        pattern = re.compile(rf"run_seed{seed}_(\d{{3}})$")
        max_num = 0

        if base_dir.exists():
            for item in base_dir.iterdir():
                if item.is_dir():
                    match = pattern.match(item.name)
                    if match:
                        num = int(match.group(1))
                        max_num = max(max_num, num)

        return max_num + 1

    def log_episode(self, metrics: EpisodeMetrics):
        """
        Log a single episode's metrics.

        Appends to episodes.jsonl immediately for fault tolerance.

        Args:
            metrics: Episode metrics to log
        """
        self.episodes.append(metrics)

        # Append to JSONL file
        with open(self.episodes_path, "a") as f:
            f.write(metrics.to_json() + "\n")

    def create_episode_metrics(
        self,
        episode_id: int,
        success: bool,
        total_steps: int,
        path_stats: Dict[str, Any],
        timing: Dict[str, Any],
        failure_reason: Optional[str] = None,
    ) -> EpisodeMetrics:
        """
        Create an EpisodeMetrics instance.

        Args:
            episode_id: Episode number
            success: Whether the episode succeeded
            total_steps: Total number of steps
            path_stats: Path generation statistics
            timing: Timing information
            failure_reason: Reason for failure

        Returns:
            EpisodeMetrics instance
        """
        return EpisodeMetrics(
            episode_id=episode_id,
            task=self.task,
            condition=self.condition,
            seed=self.seed,
            success=success,
            total_steps=total_steps,
            path_stats=path_stats,
            timing=timing,
            failure_reason=failure_reason,
        )

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics from all logged episodes.

        Returns:
            Summary dictionary
        """
        if not self.episodes:
            return {}

        total_episodes = len(self.episodes)
        successes = sum(1 for e in self.episodes if e.success)
        failures = total_episodes - successes

        # Aggregate path stats
        total_path_calls = sum(
            e.path_stats.get("total_path_calls", 0) for e in self.episodes
        )
        path_successes = sum(
            e.path_stats.get("path_successes", 0) for e in self.episodes
        )
        total_retries = sum(
            e.path_stats.get("retries", 0) for e in self.episodes
        )
        total_fallbacks = sum(
            e.path_stats.get("fallbacks_used", 0) for e in self.episodes
        )
        frame0_successes = sum(
            1 for e in self.episodes if e.path_stats.get("frame0_success", False)
        )

        # Calculate path success rate
        path_success_rate = path_successes / total_path_calls if total_path_calls > 0 else 0.0
        frame0_success_rate = frame0_successes / total_episodes if total_episodes > 0 else 0.0

        # Aggregate timing
        all_vila_times = []
        all_maniflow_times = []
        all_episode_times = []

        for e in self.episodes:
            vila_times = e.timing.get("vila_inference_ms", [])
            if isinstance(vila_times, list):
                all_vila_times.extend(vila_times)

            maniflow_times = e.timing.get("maniflow_inference_ms", [])
            if isinstance(maniflow_times, list):
                all_maniflow_times.extend(maniflow_times)

            episode_time = e.timing.get("total_episode_ms", 0)
            if episode_time > 0:
                all_episode_times.append(episode_time)

        avg_vila_time = sum(all_vila_times) / len(all_vila_times) if all_vila_times else 0.0
        avg_maniflow_time = sum(all_maniflow_times) / len(all_maniflow_times) if all_maniflow_times else 0.0
        avg_episode_time = sum(all_episode_times) / len(all_episode_times) if all_episode_times else 0.0

        summary = {
            "task": self.task,
            "condition": self.condition,
            "seed": self.seed,
            "run_number": self.run_number,
            "train_env": self.train_env,
            "eval_env": self.eval_env,
            "total_episodes": total_episodes,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total_episodes,
            "path_stats": {
                "total_path_calls": total_path_calls,
                "path_success_rate": path_success_rate,
                "frame0_success_rate": frame0_success_rate,
                "avg_retries_per_episode": total_retries / total_episodes,
                "avg_fallbacks_per_episode": total_fallbacks / total_episodes,
            },
            "timing": {
                "avg_vila_inference_ms": avg_vila_time,
                "avg_maniflow_inference_ms": avg_maniflow_time,
                "avg_episode_ms": avg_episode_time,
                "total_evaluation_time_s": time.time() - self.start_time,
            },
        }

        return summary

    def save_summary(self) -> str:
        """
        Generate and save summary to summary.json.

        Returns:
            Path to summary file
        """
        summary = self.generate_summary()

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return str(self.summary_path)

    def get_run_dir(self) -> str:
        """Get the run directory path."""
        return str(self.run_dir)


if __name__ == "__main__":
    import tempfile

    print("=" * 50)
    print("MetricsLogger Test")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Create first logger
        print("\n1. Creating first logger...")
        logger1 = MetricsLogger(
            output_dir=tmpdir,
            task="click_bell",
            condition="condition3_cluttered_overlay_initial_current",
            seed=42,
        )
        print(f"   Run directory: {logger1.get_run_dir()}")
        assert logger1.run_number == 1, f"Expected run_number=1, got {logger1.run_number}"
        assert "run_seed42_001" in logger1.get_run_dir()
        print(f"   Run number: {logger1.run_number} ✓")

        # Test 2: Create second logger with same seed - should increment
        print("\n2. Creating second logger with same seed...")
        logger2 = MetricsLogger(
            output_dir=tmpdir,
            task="click_bell",
            condition="condition3_cluttered_overlay_initial_current",
            seed=42,
        )
        print(f"   Run directory: {logger2.get_run_dir()}")
        assert logger2.run_number == 2, f"Expected run_number=2, got {logger2.run_number}"
        assert "run_seed42_002" in logger2.get_run_dir()
        print(f"   Run number: {logger2.run_number} ✓")

        # Test 3: Create logger with different seed - should start at 1
        print("\n3. Creating logger with different seed...")
        logger3 = MetricsLogger(
            output_dir=tmpdir,
            task="click_bell",
            condition="condition3_cluttered_overlay_initial_current",
            seed=43,
        )
        print(f"   Run directory: {logger3.get_run_dir()}")
        assert logger3.run_number == 1, f"Expected run_number=1, got {logger3.run_number}"
        assert "run_seed43_001" in logger3.get_run_dir()
        print(f"   Run number: {logger3.run_number} ✓")

        # Test 4: Log episodes and check summary
        print("\n4. Logging episodes...")
        for i in range(5):
            metrics = logger1.create_episode_metrics(
                episode_id=i,
                success=i % 2 == 0,
                total_steps=100 + i * 10,
                path_stats={
                    "total_path_calls": 8,
                    "path_successes": 7,
                    "path_failures": 1,
                    "retries": 1,
                    "fallbacks_used": 1,
                    "frame0_success": True,
                },
                timing={
                    "vila_inference_ms": [245, 238, 251],
                    "maniflow_inference_ms": [12, 11, 13],
                    "total_episode_ms": 15234,
                },
                failure_reason="timeout" if i % 2 == 1 else None,
            )
            logger1.log_episode(metrics)

        with open(logger1.episodes_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 5, f"Expected 5 lines, got {len(lines)}"
            print(f"   {len(lines)} episodes logged ✓")

        # Test 5: Check summary includes run_number
        print("\n5. Checking summary...")
        summary_path = logger1.save_summary()
        with open(summary_path, "r") as f:
            summary = json.load(f)
            assert summary['run_number'] == 1, f"Expected run_number=1 in summary"
            print(f"   Success rate: {summary['success_rate']:.2f}")
            print(f"   Run number in summary: {summary['run_number']} ✓")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
