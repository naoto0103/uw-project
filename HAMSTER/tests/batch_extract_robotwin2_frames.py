#!/usr/bin/env python3
"""
Batch extract frames from RoboTwin 2.0 dataset.

Phase 3.6 Stage 3 - Step 4: Extract frames from 6 tasks x 2 episodes = 12 episodes
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extract_episode_frames_robotwin2 import (
    extract_episode_from_zip,
    list_episodes_in_zip,
    SELECTED_TASKS,
    TASK_INSTRUCTIONS,
    DEFAULT_ROBOT,
    DEFAULT_CONFIG,
)


def batch_extract_frames(
    tasks: List[str],
    episodes_per_task: int = 2,
    output_base_dir: str = None,
    robot: str = DEFAULT_ROBOT,
    config: str = DEFAULT_CONFIG,
    camera: str = "head_camera",
    verbose: bool = True
) -> List[dict]:
    """
    Batch extract frames from multiple tasks and episodes.

    Args:
        tasks: List of task names
        episodes_per_task: Number of episodes to extract per task
        output_base_dir: Base output directory
        robot: Robot type
        config: Configuration
        camera: Camera to extract
        verbose: Print progress

    Returns:
        List of extraction results
    """
    if output_base_dir is None:
        script_dir = Path(__file__).parent.absolute()
        output_base_dir = script_dir.parent / "results" / "robotwin2_stage3"

    output_base_dir = Path(output_base_dir)
    results = []

    print("=" * 80)
    print("RoboTwin 2.0 Batch Frame Extraction")
    print("=" * 80)
    print(f"Tasks: {len(tasks)}")
    print(f"Episodes per task: {episodes_per_task}")
    print(f"Total episodes: {len(tasks) * episodes_per_task}")
    print(f"Robot: {robot}")
    print(f"Config: {config}")
    print(f"Camera: {camera}")
    print(f"Output base: {output_base_dir}")
    print()

    for task_idx, task_name in enumerate(tasks):
        print("-" * 60)
        print(f"Task [{task_idx + 1}/{len(tasks)}]: {task_name}")
        print("-" * 60)

        # List available episodes
        available_episodes = list_episodes_in_zip(task_name, robot, config)
        if not available_episodes:
            print(f"  ERROR: No episodes found for {task_name}")
            continue

        print(f"  Available episodes: {len(available_episodes)} ({available_episodes[:5]}...)")

        # Extract specified number of episodes
        episodes_to_extract = available_episodes[:episodes_per_task]

        for ep_idx, episode_num in enumerate(episodes_to_extract):
            print(f"\n  Episode [{ep_idx + 1}/{episodes_per_task}]: episode_{episode_num}")

            # Create output directory
            output_dir = output_base_dir / task_name / f"episode_{episode_num:02d}" / "frames"

            # Check if already extracted
            if output_dir.exists() and len(list(output_dir.glob("frame_*.png"))) > 0:
                existing_frames = len(list(output_dir.glob("frame_*.png")))
                print(f"    Already extracted ({existing_frames} frames), skipping...")
                results.append({
                    "task": task_name,
                    "episode": episode_num,
                    "status": "skipped",
                    "frames": existing_frames,
                    "output_dir": str(output_dir),
                    "instruction": TASK_INSTRUCTIONS.get(task_name, ""),
                })
                continue

            # Extract frames
            num_frames, task_instruction = extract_episode_from_zip(
                task_name=task_name,
                episode_idx=episode_num,
                output_dir=str(output_dir),
                robot=robot,
                config=config,
                camera=camera,
                verbose=False
            )

            if num_frames > 0:
                print(f"    Extracted {num_frames} frames")
                results.append({
                    "task": task_name,
                    "episode": episode_num,
                    "status": "success",
                    "frames": num_frames,
                    "output_dir": str(output_dir),
                    "instruction": task_instruction,
                })
            else:
                print(f"    ERROR: Failed to extract frames")
                results.append({
                    "task": task_name,
                    "episode": episode_num,
                    "status": "failed",
                    "frames": 0,
                    "output_dir": str(output_dir),
                    "instruction": "",
                })

    return results


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Batch extract frames from RoboTwin 2.0")
    parser.add_argument("--tasks", type=str, nargs="+", default=SELECTED_TASKS,
                        help="Tasks to process")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of episodes per task")
    parser.add_argument("--output", type=str, default=None,
                        help="Output base directory")
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT,
                        help="Robot type")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Config (clean_50 or randomized_500)")
    parser.add_argument("--camera", type=str, default="head_camera",
                        help="Camera to extract")

    args = parser.parse_args()

    # Run batch extraction
    results = batch_extract_frames(
        tasks=args.tasks,
        episodes_per_task=args.episodes,
        output_base_dir=args.output,
        robot=args.robot,
        config=args.config,
        camera=args.camera,
    )

    # Summary
    print()
    print("=" * 80)
    print("Extraction Summary")
    print("=" * 80)

    success_count = sum(1 for r in results if r["status"] in ["success", "skipped"])
    failed_count = sum(1 for r in results if r["status"] == "failed")
    total_frames = sum(r["frames"] for r in results)

    print(f"Total episodes: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total frames: {total_frames}")
    print()

    # Print results table
    print(f"{'Task':<25} {'Episode':<10} {'Status':<10} {'Frames':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['task']:<25} {r['episode']:<10} {r['status']:<10} {r['frames']:<8}")

    # Save results to JSON
    if args.output is None:
        script_dir = Path(__file__).parent.absolute()
        output_base_dir = script_dir.parent / "results" / "robotwin2_stage3"
    else:
        output_base_dir = Path(args.output)

    results_file = output_base_dir / "extraction_results.json"
    os.makedirs(output_base_dir, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
