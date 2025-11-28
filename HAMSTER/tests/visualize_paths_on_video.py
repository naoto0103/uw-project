#!/usr/bin/env python3
"""
Visualize Qwen3-generated paths on video frames using HAMSTER standard drawing.

Phase 3.6 Stage 1: Visualize paths for pick_apple_messy episode 0 (159 frames)
"""

import os
import sys
import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

# HAMSTER drawing constants
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1


def draw_lines_on_image_cv(
    image: np.ndarray,
    path: List[Tuple[float, float, int]],
) -> np.ndarray:
    """
    Draw path on image using HAMSTER original style.

    This function replicates the exact drawing style from HAMSTER/test_api_client.py

    Args:
        image: Image array (H, W, 3) in RGB format
        path: List of (x, y, gripper_state) tuples, normalized to [0, 1]

    Returns:
        Image with path drawn
    """
    if len(path) == 0:
        return image

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # Calculate scale factor (HAMSTER original logic)
    scale_factor = max(min(w, h) / 512.0, 1)
    circle_radius = int(7 * scale_factor)
    line_thickness = max(1, int(3 * scale_factor))

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
        cv2.line(img_bgr, pixel_points[i], pixel_points[i+1], color, line_thickness)

    # Draw markers at gripper state change points (HAMSTER original)
    for idx, (x, y) in enumerate(pixel_points):
        if idx == 0 or gripper_status[idx] != gripper_status[idx - 1]:
            # Red=close, Blue=open (HAMSTER original)
            circle_color = (0, 0, 255) if gripper_status[idx] == GRIPPER_CLOSE else (255, 0, 0)
            cv2.circle(img_bgr, (x, y), circle_radius, circle_color, -1)

    # Mark start point with yellow circle (HAMSTER original)
    if pixel_points:
        cv2.circle(img_bgr, pixel_points[0], circle_radius + 3, (0, 255, 255), 2)

    # Convert back to RGB
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def visualize_all_frames(
    frames_dir: str,
    paths_dir: str,
    output_dir: str,
    verbose: bool = True
):
    """
    Visualize paths on all frames.

    Args:
        frames_dir: Directory containing frame images
        paths_dir: Directory containing path pickle files
        output_dir: Directory to save visualized frames
        verbose: Print progress information
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of frames
    frames_dir = Path(frames_dir)
    paths_dir = Path(paths_dir)
    output_dir = Path(output_dir)

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    num_frames = len(frame_files)

    if num_frames == 0:
        print(f"ERROR: No frames found in {frames_dir}")
        return

    if verbose:
        print(f"Processing {num_frames} frames...")

    success_count = 0
    fail_count = 0

    for i, frame_file in enumerate(frame_files):
        frame_num = i

        # Load frame
        frame_bgr = cv2.imread(str(frame_file))
        if frame_bgr is None:
            print(f"ERROR: Failed to load frame: {frame_file}")
            fail_count += 1
            continue

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Load corresponding path
        path_file = paths_dir / f"path_frame_{frame_num:04d}.pkl"

        if not path_file.exists():
            print(f"WARNING: Path file not found for frame {frame_num}")
            fail_count += 1
            continue

        with open(path_file, 'rb') as f:
            path = pickle.load(f)

        # Draw path on frame
        frame_with_path = draw_lines_on_image_cv(frame_rgb, path)

        # Save visualized frame
        output_file = output_dir / f"frame_{frame_num:04d}.png"
        frame_with_path_bgr = cv2.cvtColor(frame_with_path, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file), frame_with_path_bgr)

        success_count += 1

        if verbose and (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")

    if verbose:
        print(f"\nVisualization complete:")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {fail_count}")
        print(f"  Output directory: {output_dir}")


def main():
    # Get script directory
    script_dir = Path(__file__).parent.absolute()

    # Configuration
    FRAMES_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames"
    PATHS_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "paths"
    OUTPUT_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames_with_paths"

    # Print configuration
    print("=" * 80)
    print("Phase 3.6 Stage 1: Visualize Paths on Video Frames")
    print("=" * 80)
    print(f"Frames directory: {FRAMES_DIR}")
    print(f"Paths directory: {PATHS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Check if directories exist
    if not FRAMES_DIR.exists():
        print(f"ERROR: Frames directory not found: {FRAMES_DIR}")
        sys.exit(1)

    if not PATHS_DIR.exists():
        print(f"ERROR: Paths directory not found: {PATHS_DIR}")
        sys.exit(1)

    # Visualize all frames
    visualize_all_frames(
        frames_dir=str(FRAMES_DIR),
        paths_dir=str(PATHS_DIR),
        output_dir=str(OUTPUT_DIR),
        verbose=True
    )

    print()
    print("=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
