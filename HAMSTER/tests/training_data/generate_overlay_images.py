#!/usr/bin/env python3
"""
Generate overlay images for ManiFlow training.

This script creates training data by overlaying VILA-generated paths onto
RGB frames using HAMSTER's original drawing method.

Path handling policy (consistent with evaluation):
- If early frames (frame 0, 1, 2...) have no path, use the first successful path
- If a frame after a successful path has no path, use the most recent successful path
- This ensures all training frames have path overlays and matches evaluation behavior

Output structure:
    HAMSTER/results/robotwin2_single_6tasks_vila/{task}/episode_{XX}/
    ├── frames/              # Original RGB frames (existing)
    ├── paths/               # Path coordinates (existing)
    └── overlay_images/      # Training overlay images (new)

Usage:
    python generate_overlay_images.py --tasks beat_block_hammer --episodes 50
    python generate_overlay_images.py  # Process all tasks and episodes
"""

import os
import sys
import pickle
import argparse
import json
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import cv2

# Add current directory to path for overlay_utils import
sys.path.insert(0, str(Path(__file__).parent))
from overlay_utils import draw_path_on_image_hamster_style


# Single-arm tasks
SINGLE_ARM_TASKS = [
    "beat_block_hammer",
    "click_bell",
    "move_can_pot",
    "open_microwave",
    "place_object_stand",
    "turn_switch",
]


def process_episode(
    task_name: str,
    episode_num: int,
    base_dir: Path,
    num_subdivisions: int = 100,
    overwrite: bool = False,
) -> dict:
    """
    Process a single episode: generate overlay images for all frames.

    Path handling policy (consistent with evaluation):
    - If early frames (frame 0, 1, 2...) have no path, use the first successful path
    - If a frame after a successful path has no path, use the most recent successful path
    - This ensures all training frames have path overlays and matches evaluation behavior

    Args:
        task_name: Task name
        episode_num: Episode number
        base_dir: Base results directory
        num_subdivisions: Number of subdivisions for path interpolation
        overwrite: Whether to overwrite existing overlay images

    Returns:
        Result dictionary with statistics
    """
    episode_dir = base_dir / task_name / f"episode_{episode_num:02d}"
    frames_dir = episode_dir / "frames"
    paths_dir = episode_dir / "paths"
    overlay_dir = episode_dir / "overlay_images"

    # Check directories
    if not frames_dir.exists():
        return {
            "task": task_name,
            "episode": episode_num,
            "status": "error",
            "error": "Frames directory not found",
        }
    if not paths_dir.exists():
        return {
            "task": task_name,
            "episode": episode_num,
            "status": "error",
            "error": "Paths directory not found",
        }

    # Get frame files
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        return {
            "task": task_name,
            "episode": episode_num,
            "status": "error",
            "error": "No frame files found",
        }

    # Find the first successful path in the episode
    first_successful_path = None
    first_successful_frame = None
    for frame_file in frame_files:
        frame_num = int(frame_file.stem.split('_')[1])
        path_file = paths_dir / f"path_frame_{frame_num:04d}.pkl"
        if path_file.exists():
            try:
                with open(path_file, 'rb') as f:
                    path_data = pickle.load(f)
                if path_data and len(path_data) > 0:
                    first_successful_path = path_data
                    first_successful_frame = frame_num
                    break
            except Exception:
                continue

    # If no valid path found in the entire episode, skip it
    if first_successful_path is None:
        return {
            "task": task_name,
            "episode": episode_num,
            "status": "skipped",
            "error": "No valid path found in entire episode",
        }

    # Create output directory
    os.makedirs(overlay_dir, exist_ok=True)

    # Process each frame
    success_count = 0
    skip_count = 0
    fail_count = 0
    fallback_count = 0  # Count of frames using fallback path
    early_fallback_count = 0  # Count of early frames using first successful path

    # Track the most recent successful path for fallback
    last_successful_path = None

    for frame_file in frame_files:
        frame_num = int(frame_file.stem.split('_')[1])
        output_file = overlay_dir / f"overlay_{frame_num:04d}.png"

        # Skip if exists and not overwriting
        if output_file.exists() and not overwrite:
            skip_count += 1
            continue

        # Load frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            fail_count += 1
            continue

        # Try to load path for this frame
        path_file = paths_dir / f"path_frame_{frame_num:04d}.pkl"
        current_path = None
        used_fallback = False

        if path_file.exists():
            try:
                with open(path_file, 'rb') as f:
                    current_path = pickle.load(f)
                if current_path and len(current_path) > 0:
                    last_successful_path = current_path
                else:
                    current_path = None
            except Exception:
                current_path = None

        # Use fallback if no valid path for this frame
        if current_path is None:
            if last_successful_path is not None:
                # After a successful path, use the most recent one
                current_path = last_successful_path
                used_fallback = True
            else:
                # Early frames before first success: use first successful path
                current_path = first_successful_path
                used_fallback = True
                early_fallback_count += 1

        try:
            # Generate overlay image
            overlay = draw_path_on_image_hamster_style(
                frame, current_path, num_subdivisions=num_subdivisions
            )

            # Save overlay image
            cv2.imwrite(str(output_file), overlay)
            success_count += 1
            if used_fallback:
                fallback_count += 1

        except Exception as e:
            fail_count += 1

    return {
        "task": task_name,
        "episode": episode_num,
        "status": "success",
        "total_frames": len(frame_files),
        "success": success_count,
        "skipped": skip_count,
        "fallback_used": fallback_count,
        "early_fallback_used": early_fallback_count,
        "first_successful_frame": first_successful_frame,
        "failed": fail_count,
    }


def process_episode_wrapper(args):
    """Wrapper for multiprocessing."""
    return process_episode(*args)


def main():
    parser = argparse.ArgumentParser(
        description="Generate overlay images for ManiFlow training"
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=SINGLE_ARM_TASKS,
        help="Tasks to process"
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of episodes per task (default: 50)"
    )
    parser.add_argument(
        "--base-dir", type=str, default=None,
        help="Base results directory"
    )
    parser.add_argument(
        "--num-subdivisions", type=int, default=100,
        help="Number of subdivisions for path interpolation (default: 100)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing overlay images"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Set base directory
    script_dir = Path(__file__).parent.absolute()
    if args.base_dir is None:
        base_dir = script_dir.parent.parent / "results" / "robotwin2_single_6tasks_vila"
    else:
        base_dir = Path(args.base_dir)

    # Set number of workers
    if args.num_workers is None:
        args.num_workers = min(mp.cpu_count(), 16)

    print("=" * 70)
    print("Generate Overlay Images for ManiFlow Training")
    print("=" * 70)
    print(f"Tasks: {args.tasks}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Base directory: {base_dir}")
    print(f"Num subdivisions: {args.num_subdivisions}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Num workers: {args.num_workers}")
    print()
    print("Drawing specification (HAMSTER original):")
    print("  - Line color: jet colormap (blue -> cyan -> green -> yellow -> red)")
    print("  - Gripper markers: Open=blue, Close=red (at state change points)")
    print("  - Path interpolation: 100 subdivisions")
    print()

    # Collect all episodes to process
    episodes_to_process = []
    for task_name in args.tasks:
        for ep_num in range(args.episodes):
            episodes_to_process.append((
                task_name, ep_num, base_dir, args.num_subdivisions, args.overwrite
            ))

    total_episodes = len(episodes_to_process)
    print(f"Total episodes to process: {total_episodes}")
    print("-" * 70)

    # Process episodes in parallel
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_episode_wrapper, ep): ep
            for ep in episodes_to_process
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if args.verbose or completed % 10 == 0 or completed == total_episodes:
                task = result["task"]
                ep = result["episode"]
                status = result["status"]
                if status == "success":
                    success = result["success"]
                    total = result["total_frames"]
                    print(f"  [{completed:>4}/{total_episodes}] {task}/ep{ep:02d}: "
                          f"{success}/{total} overlays created")
                else:
                    error = result.get("error", "Unknown error")
                    print(f"  [{completed:>4}/{total_episodes}] {task}/ep{ep:02d}: "
                          f"ERROR - {error}")

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    success_episodes = [r for r in results if r["status"] == "success"]
    skipped_episodes = [r for r in results if r["status"] == "skipped"]
    error_episodes = [r for r in results if r["status"] == "error"]

    total_success = sum(r["success"] for r in success_episodes)
    total_skipped = sum(r["skipped"] for r in success_episodes)
    total_fallback = sum(r["fallback_used"] for r in success_episodes)
    total_early_fallback = sum(r["early_fallback_used"] for r in success_episodes)
    total_failed = sum(r["failed"] for r in success_episodes)
    total_frames = sum(r["total_frames"] for r in success_episodes)

    # Count episodes where frame 0 failed (first_successful_frame > 0)
    episodes_with_frame0_fail = [r for r in success_episodes if r.get("first_successful_frame", 0) > 0]

    print(f"Episodes processed: {len(success_episodes)}/{total_episodes}")
    print(f"Episodes skipped (no valid path): {len(skipped_episodes)}")
    print(f"Episodes with frame 0 failure: {len(episodes_with_frame0_fail)}")
    print(f"Episodes with errors: {len(error_episodes)}")
    print()
    print(f"Total frames: {total_frames}")
    print(f"  - Overlay created: {total_success}")
    print(f"  - Skipped (existing): {total_skipped}")
    print(f"  - Used fallback path: {total_fallback}")
    print(f"  - Early fallback (before first success): {total_early_fallback}")
    print(f"  - Failed: {total_failed}")

    # Per-task summary
    print()
    print("Per-task summary:")
    print(f"{'Task':<25} {'Episodes':<10} {'Skipped':<10} {'Overlays':<10} {'Fallback':<10}")
    print("-" * 70)

    for task_name in args.tasks:
        task_success = [r for r in success_episodes if r["task"] == task_name]
        task_skipped = [r for r in skipped_episodes if r["task"] == task_name]
        task_overlays = sum(r["success"] for r in task_success)
        task_fallback = sum(r["fallback_used"] for r in task_success)
        print(f"{task_name:<25} {len(task_success):<10} {len(task_skipped):<10} {task_overlays:<10} {task_fallback:<10}")

    # List skipped episodes
    if skipped_episodes:
        print()
        print("Skipped episodes (no frame 0 path):")
        for r in skipped_episodes:
            print(f"  - {r['task']}/episode_{r['episode']:02d}: {r.get('error', 'Unknown')}")

    # Save summary
    summary = {
        "total_episodes": total_episodes,
        "success_episodes": len(success_episodes),
        "skipped_episodes": len(skipped_episodes),
        "skipped_episode_list": [{"task": r["task"], "episode": r["episode"]} for r in skipped_episodes],
        "episodes_with_frame0_fail": len(episodes_with_frame0_fail),
        "frame0_fail_episode_list": [{"task": r["task"], "episode": r["episode"], "first_successful_frame": r["first_successful_frame"]} for r in episodes_with_frame0_fail],
        "error_episodes": len(error_episodes),
        "total_frames": total_frames,
        "overlays_created": total_success,
        "skipped_frames": total_skipped,
        "fallback_used": total_fallback,
        "early_fallback_used": total_early_fallback,
        "failed": total_failed,
        "num_subdivisions": args.num_subdivisions,
        "results": results,
    }

    summary_file = base_dir / "overlay_generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Summary saved to: {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
