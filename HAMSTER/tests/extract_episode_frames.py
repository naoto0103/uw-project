#!/usr/bin/env python3
"""
Extract frames from a single RoboTwin episode for video-based path generation testing.

Phase 3.6 Stage 1: Extract frames from pick_apple_messy episode 0 (159 frames)
"""

import os
import sys
import zarr
import numpy as np
from PIL import Image
from pathlib import Path

def extract_episode_frames(
    zarr_path: str,
    episode_idx: int,
    output_dir: str,
    verbose: bool = True
):
    """
    Extract all frames from a specific episode in a zarr dataset.

    Args:
        zarr_path: Path to zarr dataset (e.g., '../ManiFlow/data/pick_apple_messy_50.zarr')
        episode_idx: Episode index to extract (e.g., 0)
        output_dir: Directory to save frames (e.g., 'results/video_path_test/episode_0/frames')
        verbose: Print progress information

    Returns:
        int: Number of frames extracted
    """
    # Load zarr dataset
    if verbose:
        print(f"Loading zarr dataset: {zarr_path}")

    root = zarr.open(zarr_path, mode='r')

    # Get episode boundaries
    episode_ends = root['meta/episode_ends'][:]

    # Calculate start and end indices for the episode
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
    end_idx = episode_ends[episode_idx]
    num_frames = end_idx - start_idx

    if verbose:
        print(f"Episode {episode_idx}: frames {start_idx} to {end_idx-1} ({num_frames} frames)")

    # Load frames for this episode
    # head_camera shape: [N_total_frames, 3, H, W]
    all_frames = root['data/head_camera']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract and save each frame
    if verbose:
        print(f"Extracting frames to: {output_dir}")

    for i in range(num_frames):
        frame_idx = start_idx + i

        # Get frame: shape (3, H, W)
        frame_chw = all_frames[frame_idx]

        # Convert from (C, H, W) to (H, W, C) for PIL
        frame_hwc = np.transpose(frame_chw, (1, 2, 0))

        # Convert to uint8 if needed
        if frame_hwc.dtype != np.uint8:
            # Assume values are in [0, 1] range
            if frame_hwc.max() <= 1.0:
                frame_hwc = (frame_hwc * 255).astype(np.uint8)
            else:
                frame_hwc = frame_hwc.astype(np.uint8)

        # Save as PNG
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        img = Image.fromarray(frame_hwc)
        img.save(output_path)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Extracted {i + 1}/{num_frames} frames...")

    if verbose:
        print(f"Successfully extracted {num_frames} frames")

    return num_frames


def main():
    # Get script directory (HAMSTER/tests/)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent  # HAMSTER-ManiFlow-Integration/

    # Configuration for Phase 3.6 Stage 1
    ZARR_PATH = project_root / "ManiFlow" / "data" / "pick_apple_messy_50.zarr"
    EPISODE_IDX = 0
    OUTPUT_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames"

    # Print configuration
    print("=" * 80)
    print("Phase 3.6 Stage 1: Extract Episode Frames")
    print("=" * 80)
    print(f"Task: pick_apple_messy")
    print(f"Episode: {EPISODE_IDX}")
    print(f"Zarr path: {ZARR_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Check if zarr file exists
    if not ZARR_PATH.exists():
        print(f"ERROR: Zarr file not found: {ZARR_PATH}")
        sys.exit(1)

    # Extract frames
    num_frames = extract_episode_frames(
        zarr_path=str(ZARR_PATH),
        episode_idx=EPISODE_IDX,
        output_dir=str(OUTPUT_DIR),
        verbose=True
    )

    print()
    print("=" * 80)
    print(f"Extraction complete: {num_frames} frames saved to {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
