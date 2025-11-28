#!/usr/bin/env python3
"""
Create MP4 video from frames with visualized paths.

Phase 3.6 Stage 1: Create video for pick_apple_messy episode 0 (159 frames)
"""

import os
import sys
import cv2
from pathlib import Path


def create_video(
    frames_dir: str,
    output_video: str,
    fps: int = 10,
    verbose: bool = True
):
    """
    Create MP4 video from frame images.

    Args:
        frames_dir: Directory containing frame images
        output_video: Output video file path
        fps: Frames per second
        verbose: Print progress information
    """
    frames_dir = Path(frames_dir)

    # Get list of frames
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    num_frames = len(frame_files)

    if num_frames == 0:
        print(f"ERROR: No frames found in {frames_dir}")
        return

    if verbose:
        print(f"Found {num_frames} frames")
        print(f"FPS: {fps}")
        print(f"Video duration: {num_frames / fps:.2f} seconds")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"ERROR: Failed to load first frame: {frame_files[0]}")
        return

    height, width = first_frame.shape[:2]

    if verbose:
        print(f"Frame size: {width}x{height}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    if not video_writer.isOpened():
        print(f"ERROR: Failed to create video writer for {output_video}")
        return

    # Write frames to video
    if verbose:
        print(f"\nWriting frames to video...")

    for i, frame_file in enumerate(frame_files):
        # Load frame
        frame = cv2.imread(str(frame_file))

        if frame is None:
            print(f"WARNING: Failed to load frame: {frame_file}")
            continue

        # Write frame
        video_writer.write(frame)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Written {i + 1}/{num_frames} frames...")

    # Release video writer
    video_writer.release()

    if verbose:
        print(f"\nVideo created successfully!")
        print(f"  Frames: {num_frames}")
        print(f"  Duration: {num_frames / fps:.2f} seconds")
        print(f"  Output: {output_video}")

        # Check file size
        file_size = os.path.getsize(output_video)
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")


def main():
    # Get script directory
    script_dir = Path(__file__).parent.absolute()

    # Configuration
    FRAMES_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames_with_paths"
    OUTPUT_VIDEO = script_dir.parent / "results" / "video_path_test" / "episode_0" / "qwen3_path_video.mp4"
    FPS = 50

    # Print configuration
    print("=" * 80)
    print("Phase 3.6 Stage 1: Create Path Video")
    print("=" * 80)
    print(f"Frames directory: {FRAMES_DIR}")
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"FPS: {FPS}")
    print()

    # Check if frames directory exists
    if not FRAMES_DIR.exists():
        print(f"ERROR: Frames directory not found: {FRAMES_DIR}")
        sys.exit(1)

    # Create output directory
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    # Create video
    create_video(
        frames_dir=str(FRAMES_DIR),
        output_video=str(OUTPUT_VIDEO),
        fps=FPS,
        verbose=True
    )

    print()
    print("=" * 80)
    print("Video creation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
