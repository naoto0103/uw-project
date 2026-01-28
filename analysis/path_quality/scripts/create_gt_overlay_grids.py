#!/usr/bin/env python3
"""
Create grid visualization of GT (Ground Truth) path overlay images.

For each episode, creates a grid image showing overlay images at 16-frame intervals.
This is the GT path version of create_overlay_grids.py.

Key differences from VLM version:
- Data location: gt_paths_clean/gt_paths_cluttered instead of evaluation_tasks_*
- Overlay files: frame_XXXX.png instead of overlay_XXXX.png
- Path files: frame_XXXX.pkl instead of path_frame_XXXX.pkl

Usage:
    python create_gt_overlay_grids.py
    python create_gt_overlay_grids.py --tasks click_bell --conditions clean
    python create_gt_overlay_grids.py --episodes 10  # Process first 10 episodes only
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse

# Default settings
RESULTS_DIR = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results")
OUTPUT_DIR = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/analysis/path_quality/grids_gt")
TARGET_TASKS = ["click_bell", "move_can_pot", "beat_block_hammer"]
CONDITIONS = ["gt_paths_clean", "gt_paths_cluttered"]
FRAME_INTERVAL = 16
COLS = 4


def add_failure_marker(img):
    """Add red X marker to top-right corner of image"""
    h, w = img.shape[:2]
    thickness = max(3, int(min(h, w) / 30))
    margin = int(min(h, w) * 0.05)
    size = int(min(h, w) * 0.15)

    x1, y1 = w - margin - size, margin
    x2, y2 = w - margin, margin + size
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    cv2.line(img, (x2, y1), (x1, y2), (0, 0, 255), thickness)
    return img


def add_frame_label(img, frame_num, failed=False):
    """Add frame number label to top-left corner"""
    h, w = img.shape[:2]
    font_scale = max(0.5, min(h, w) / 400)
    thickness = max(1, int(font_scale * 2))

    label = f"F{frame_num}"
    color = (0, 0, 255) if failed else (255, 255, 255)

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (5, 5), (15 + text_w, 15 + text_h), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 10 + text_h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def add_gt_badge(img):
    """Add 'GT' badge to top-right corner to distinguish from VLM grids"""
    h, w = img.shape[:2]
    font_scale = max(0.4, min(h, w) / 500)
    thickness = max(1, int(font_scale * 2))

    label = "GT"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Position in top-right corner (below failure marker area)
    x = w - text_w - 15
    y = 10 + text_h

    # Green background for GT
    cv2.rectangle(img, (x - 5, 5), (x + text_w + 5, 15 + text_h), (0, 100, 0), -1)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return img


def create_episode_grid(condition, task, episode_num, frame_interval=FRAME_INTERVAL, cols=COLS):
    """Create grid image for one episode

    For GT paths:
    - overlay_images/frame_XXXX.png (instead of overlay_XXXX.png)
    - paths/frame_XXXX.pkl (instead of path_frame_XXXX.pkl)
    """
    episode_dir = RESULTS_DIR / condition / task / f"episode_{episode_num:02d}"
    overlay_dir = episode_dir / "overlay_images"
    paths_dir = episode_dir / "paths"

    if not overlay_dir.exists():
        return None, "overlay_dir not found"

    # GT paths use frame_XXXX.png naming
    overlay_files = sorted(overlay_dir.glob("frame_*.png"))
    if not overlay_files:
        return None, "no overlay files"

    total_frames = len(overlay_files)
    sample_frames = list(range(0, total_frames, frame_interval))

    images = []
    failed_frames = []

    for frame_num in sample_frames:
        # GT paths use frame_XXXX.png and frame_XXXX.pkl naming
        overlay_file = overlay_dir / f"frame_{frame_num:04d}.png"
        path_file = paths_dir / f"frame_{frame_num:04d}.pkl"

        if overlay_file.exists():
            img = cv2.imread(str(overlay_file))
            if img is not None:
                failed = not path_file.exists()
                if failed:
                    failed_frames.append(frame_num)
                    img = add_failure_marker(img)
                img = add_frame_label(img, frame_num, failed)
                img = add_gt_badge(img)
                images.append(img)

    if not images:
        return None, "no images loaded"

    h, w = images[0].shape[:2]

    while len(images) % cols != 0:
        images.append(np.zeros((h, w, 3), dtype=np.uint8))

    grid_rows = []
    for i in range(0, len(images), cols):
        row = np.hstack(images[i:i+cols])
        grid_rows.append(row)

    grid = np.vstack(grid_rows)

    return grid, {"total_frames": total_frames, "sampled": len(sample_frames), "failed": failed_frames}


def main():
    parser = argparse.ArgumentParser(description="Create GT path overlay grid visualizations")
    parser.add_argument("--tasks", nargs="+", default=TARGET_TASKS, help="Tasks to process")
    parser.add_argument("--conditions", nargs="+", default=["clean", "cluttered"], help="Conditions to process")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to process")
    parser.add_argument("--frame-interval", type=int, default=FRAME_INTERVAL, help="Frame sampling interval")
    parser.add_argument("--cols", type=int, default=COLS, help="Number of columns in grid")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating GT path quality visualization grids")
    print("=" * 60)
    print(f"Tasks: {args.tasks}")
    print(f"Conditions: {args.conditions}")
    print(f"Episodes: {args.episodes}")
    print(f"Frame interval: {args.frame_interval}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    for cond_short in args.conditions:
        condition = f"gt_paths_{cond_short}"
        print(f"\n### GT {cond_short} ###")

        for task in args.tasks:
            print(f"\n  {task}:")
            task_output_dir = OUTPUT_DIR / cond_short / task
            task_output_dir.mkdir(parents=True, exist_ok=True)

            success_count = 0
            total_failed_frames = []

            for ep_num in range(args.episodes):
                grid, info = create_episode_grid(
                    condition, task, ep_num,
                    frame_interval=args.frame_interval,
                    cols=args.cols
                )

                if grid is not None:
                    output_file = task_output_dir / f"episode_{ep_num:02d}_grid.png"
                    cv2.imwrite(str(output_file), grid)
                    success_count += 1

                    if info["failed"]:
                        total_failed_frames.extend([(ep_num, f) for f in info["failed"]])

                    if (ep_num + 1) % 10 == 0:
                        print(f"    Processed {ep_num + 1}/{args.episodes} episodes...")

            print(f"    Episodes processed: {success_count}/{args.episodes}")
            print(f"    Failed frames (no path): {len(total_failed_frames)}")
            if total_failed_frames[:5]:
                print(f"    Sample failures: {total_failed_frames[:5]}")

    print("\n" + "=" * 60)
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
