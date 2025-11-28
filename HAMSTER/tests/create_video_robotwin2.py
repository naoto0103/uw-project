#!/usr/bin/env python3
"""
Create visualization videos for RoboTwin 2.0 path generation results.

Phase 3.6 Stage 3 - Visualize paths and create MP4 videos
Uses HAMSTER standard drawing style:
- Green lines for path
- Red markers for gripper close
- Blue markers for gripper open
- Yellow ring for start point
"""

import os
import sys
import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_episode_frames_robotwin2 import (
    SELECTED_TASKS,
    TASK_INSTRUCTIONS,
)

# HAMSTER drawing constants
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1


def draw_bimanual_path_on_image(
    image: np.ndarray,
    paths: Dict[str, List[Tuple[float, float, int]]],
) -> np.ndarray:
    """
    Draw bimanual paths on image using HAMSTER standard style.

    Color scheme:
    - Green lines for path
    - Red markers for gripper close
    - Blue markers for gripper open
    - Yellow ring for start point

    Args:
        image: Image array (H, W, 3) in BGR format
        paths: Dictionary with 'left_arm' and/or 'right_arm' paths
               Each path is a list of (x, y, gripper_state) tuples, normalized to [0, 1]

    Returns:
        Image with paths drawn (BGR format)
    """
    if not paths:
        return image

    img = image.copy()
    h, w = img.shape[:2]

    # Calculate scale factor (HAMSTER original logic)
    scale_factor = max(min(w, h) / 512.0, 1)
    circle_radius = int(7 * scale_factor)
    line_thickness = max(1, int(3 * scale_factor))

    # Draw each arm's path using HAMSTER standard style
    for arm_key in ['left_arm', 'right_arm']:
        if arm_key not in paths:
            continue

        path = paths[arm_key]
        if len(path) == 0:
            continue

        # Convert normalized coordinates to pixel coordinates
        pixel_points = []
        gripper_status = []
        for x, y, action in path:
            px = int(x * w)
            py = int(y * h)
            pixel_points.append((px, py))
            gripper_status.append(action)

        # Draw path lines in green (HAMSTER original)
        for i in range(len(pixel_points) - 1):
            color = (0, 255, 0)  # Green in BGR
            cv2.line(img, pixel_points[i], pixel_points[i+1], color, line_thickness)

        # Draw markers at gripper state change points (HAMSTER original)
        for idx, (x, y) in enumerate(pixel_points):
            if idx == 0 or gripper_status[idx] != gripper_status[idx - 1]:
                # Red=close, Blue=open (HAMSTER original)
                circle_color = (0, 0, 255) if gripper_status[idx] == GRIPPER_CLOSE else (255, 0, 0)
                cv2.circle(img, (x, y), circle_radius, circle_color, -1)

        # Mark start point with yellow circle (HAMSTER original)
        if pixel_points:
            cv2.circle(img, pixel_points[0], circle_radius + 3, (0, 255, 255), 2)

    return img


def visualize_and_create_video(
    task_name: str,
    episode_num: int,
    base_dir: Path,
    fps: int = 10,
    verbose: bool = True
) -> dict:
    """
    Visualize paths on frames and create video for one episode.

    Args:
        task_name: Task name
        episode_num: Episode number
        base_dir: Base directory
        fps: Video frames per second
        verbose: Print progress

    Returns:
        Result dictionary
    """
    episode_dir = base_dir / task_name / f"episode_{episode_num:02d}"
    frames_dir = episode_dir / "frames"
    paths_dir = episode_dir / "paths"
    output_frames_dir = episode_dir / "frames_with_paths"
    output_video = episode_dir / "path_video.mp4"

    if verbose:
        print(f"    Frames: {frames_dir}")
        print(f"    Paths: {paths_dir}")

    # Check directories
    if not frames_dir.exists():
        return {"status": "error", "error": "Frames not found"}
    if not paths_dir.exists():
        return {"status": "error", "error": "Paths not found"}

    # Get frames and paths
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    path_files = sorted(paths_dir.glob("path_frame_*.pkl"))

    num_frames = len(frame_files)
    num_paths = len(path_files)

    if verbose:
        print(f"    Found {num_frames} frames, {num_paths} paths")

    if num_frames == 0:
        return {"status": "error", "error": "No frames"}
    if num_paths == 0:
        return {"status": "error", "error": "No paths"}

    # Create output directory
    os.makedirs(output_frames_dir, exist_ok=True)

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]

    if verbose:
        print(f"    Frame size: {width}x{height}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not video_writer.isOpened():
        return {"status": "error", "error": "Failed to create video writer"}

    # Process each frame
    success_count = 0
    fail_count = 0

    for i, frame_file in enumerate(frame_files):
        frame_num = i

        # Load frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            fail_count += 1
            continue

        # Load path
        path_file = paths_dir / f"path_frame_{frame_num:04d}.pkl"
        if path_file.exists():
            with open(path_file, 'rb') as f:
                paths = pickle.load(f)

            # Draw paths on frame
            frame_with_path = draw_bimanual_path_on_image(frame, paths)
        else:
            frame_with_path = frame
            fail_count += 1

        # Save visualized frame
        output_frame_file = output_frames_dir / f"frame_{frame_num:04d}.png"
        cv2.imwrite(str(output_frame_file), frame_with_path)

        # Write to video
        video_writer.write(frame_with_path)
        success_count += 1

        if verbose and (i + 1) % 30 == 0:
            print(f"      Processed {i + 1}/{num_frames} frames...")

    video_writer.release()

    # Check file size
    if output_video.exists():
        file_size_mb = output_video.stat().st_size / 1024 / 1024
    else:
        file_size_mb = 0

    if verbose:
        print(f"    Video created: {output_video}")
        print(f"    Video size: {file_size_mb:.2f} MB")
        print(f"    Duration: {num_frames / fps:.1f} seconds")

    return {
        "status": "success",
        "frames": num_frames,
        "paths": num_paths,
        "success": success_count,
        "failed": fail_count,
        "video": str(output_video),
        "video_size_mb": file_size_mb,
        "duration_sec": num_frames / fps,
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Create visualization videos for RoboTwin 2.0")
    parser.add_argument("--tasks", type=str, nargs="+", default=SELECTED_TASKS,
                        help="Tasks to process")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of episodes per task")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video FPS")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")

    args = parser.parse_args()

    # Set directories
    script_dir = Path(__file__).parent.absolute()
    if args.base_dir is None:
        base_dir = script_dir.parent / "results" / "robotwin2_stage3"
    else:
        base_dir = Path(args.base_dir)

    print("=" * 80)
    print("RoboTwin 2.0 Video Creation")
    print("=" * 80)
    print(f"Tasks: {len(args.tasks)}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Base directory: {base_dir}")
    print(f"FPS: {args.fps}")
    print()
    print("Color scheme (HAMSTER standard):")
    print("  Green lines = path trajectory")
    print("  Red markers = gripper close")
    print("  Blue markers = gripper open")
    print("  Yellow ring = start point")
    print()

    # Process episodes
    results = []

    for task_idx, task_name in enumerate(args.tasks):
        print("-" * 60)
        print(f"Task [{task_idx + 1}/{len(args.tasks)}]: {task_name}")
        print(f"Instruction: {TASK_INSTRUCTIONS.get(task_name, 'N/A')}")
        print("-" * 60)

        for ep_idx in range(args.episodes):
            print(f"\n  Episode [{ep_idx + 1}/{args.episodes}]: episode_{ep_idx:02d}")

            result = visualize_and_create_video(
                task_name=task_name,
                episode_num=ep_idx,
                base_dir=base_dir,
                fps=args.fps,
                verbose=True
            )
            result["task"] = task_name
            result["episode"] = ep_idx
            results.append(result)

    # Summary
    print()
    print("=" * 80)
    print("Video Creation Summary")
    print("=" * 80)

    success_count = sum(1 for r in results if r["status"] == "success")
    total_videos = len(results)

    print(f"Total episodes: {total_videos}")
    print(f"Videos created: {success_count}")
    print()

    # Print results table
    print(f"{'Task':<25} {'Ep':<4} {'Status':<10} {'Frames':<8} {'Duration':<10} {'Size':<8}")
    print("-" * 75)
    for r in results:
        duration = f"{r.get('duration_sec', 0):.1f}s" if 'duration_sec' in r else "N/A"
        size = f"{r.get('video_size_mb', 0):.1f}MB" if 'video_size_mb' in r else "N/A"
        print(f"{r['task']:<25} {r['episode']:<4} {r['status']:<10} {r.get('frames', 0):<8} {duration:<10} {size:<8}")

    # Save summary
    summary_file = base_dir / "video_creation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_episodes": total_videos,
            "videos_created": success_count,
            "fps": args.fps,
            "results": results,
        }, f, indent=2)

    print()
    print(f"Summary saved to: {summary_file}")
    print()
    print("Videos location:")
    for r in results:
        if r["status"] == "success":
            print(f"  {r['video']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
