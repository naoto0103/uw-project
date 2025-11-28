#!/usr/bin/env python3
"""
Visualize HAMSTER 2D path overlaid on RoboTwin episode video.

This script creates a video showing the HAMSTER-generated 2D path
overlaid on the simulation frames from a RoboTwin episode.
"""

import pickle
import zarr
import numpy as np
import cv2
from pathlib import Path


def create_overlay_video(
    zarr_path: str,
    hamster_paths_pkl: str,
    episode_idx: int = 0,
    output_path: str = "hamster_path_overlay.mp4",
    fps: int = 10
):
    """
    Create a video with HAMSTER path overlaid on episode frames.

    Args:
        zarr_path: Path to .zarr dataset
        hamster_paths_pkl: Path to HAMSTER paths .pkl file
        episode_idx: Episode index to visualize
        output_path: Output video path
        fps: Frames per second for output video
    """
    # Load dataset
    print(f"Loading dataset from {zarr_path}")
    group = zarr.open(zarr_path, mode='r')

    # Get episode boundaries
    episode_ends = group['meta']['episode_ends'][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    start_idx = episode_starts[episode_idx]
    end_idx = episode_ends[episode_idx]

    print(f"Episode {episode_idx}: frames {start_idx} to {end_idx} ({end_idx - start_idx} frames)")

    # Load camera frames for this episode
    camera_data = group['data']['head_camera'][start_idx:end_idx]  # (T, C, H, W)

    # Load HAMSTER paths
    print(f"Loading HAMSTER paths from {hamster_paths_pkl}")
    with open(hamster_paths_pkl, 'rb') as f:
        path_data = pickle.load(f)

    path_points = path_data['paths'][episode_idx]
    print(f"Path has {len(path_points)} points:")
    for i, pt in enumerate(path_points):
        gripper_state = "CLOSE" if pt[2] == 1 else "OPEN"
        print(f"  {i}: ({pt[0]:.2f}, {pt[1]:.2f}) - Gripper: {gripper_state}")

    # Get frame dimensions
    _, H, W = camera_data[0].shape
    print(f"Frame size: {W}x{H}")

    # Convert normalized coordinates to pixel coordinates
    path_pixels = []
    for pt in path_points:
        x_pixel = int(pt[0] * W)
        y_pixel = int(pt[1] * H)
        gripper = int(pt[2])
        path_pixels.append((x_pixel, y_pixel, gripper))

    print(f"Path in pixels: {path_pixels}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Calculate scale factor for drawing (HAMSTER style)
    scale_factor = max(min(W, H) / 512.0, 1)
    circle_radius = int(7 * scale_factor)
    line_thickness = max(1, int(3 * scale_factor))

    # Process each frame
    print(f"Creating video with {len(camera_data)} frames...")
    for frame_idx in range(len(camera_data)):
        # Get frame (C, H, W) -> (H, W, C)
        frame = camera_data[frame_idx]
        frame = np.transpose(frame, (1, 2, 0))

        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw path on frame (HAMSTER default style)
        # Draw green lines connecting path points
        for i in range(len(path_pixels) - 1):
            pt1 = (path_pixels[i][0], path_pixels[i][1])
            pt2 = (path_pixels[i+1][0], path_pixels[i+1][1])
            cv2.line(frame_bgr, pt1, pt2, (0, 255, 0), line_thickness)  # Green line

        # Draw markers only at gripper state change points
        for i, (x, y, gripper) in enumerate(path_pixels):
            if i == 0 or gripper != path_pixels[i-1][2]:
                # Red=Close, Blue=Open (HAMSTER convention)
                circle_color = (0, 0, 255) if gripper == 1 else (255, 0, 0)
                cv2.circle(frame_bgr, (x, y), circle_radius, circle_color, -1)

        # Mark start point with yellow circle outline
        if path_pixels:
            cv2.circle(frame_bgr, (path_pixels[0][0], path_pixels[0][1]),
                      circle_radius + 3, (0, 255, 255), 2)

        # Write frame
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")

    # Also save first frame as image for quick preview
    preview_path = output_path.replace('.mp4', '_preview.png')
    first_frame = camera_data[0]
    first_frame = np.transpose(first_frame, (1, 2, 0))
    if first_frame.dtype != np.uint8:
        if first_frame.max() <= 1.0:
            first_frame = (first_frame * 255).astype(np.uint8)
        else:
            first_frame = first_frame.astype(np.uint8)
    first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)

    # Draw path on preview (HAMSTER default style)
    for i in range(len(path_pixels) - 1):
        pt1 = (path_pixels[i][0], path_pixels[i][1])
        pt2 = (path_pixels[i+1][0], path_pixels[i+1][1])
        cv2.line(first_frame_bgr, pt1, pt2, (0, 255, 0), line_thickness)  # Green line

    for i, (x, y, gripper) in enumerate(path_pixels):
        if i == 0 or gripper != path_pixels[i-1][2]:
            # Red=Close, Blue=Open
            circle_color = (0, 0, 255) if gripper == 1 else (255, 0, 0)
            cv2.circle(first_frame_bgr, (x, y), circle_radius, circle_color, -1)

    # Mark start point with yellow circle outline
    if path_pixels:
        cv2.circle(first_frame_bgr, (path_pixels[0][0], path_pixels[0][1]),
                  circle_radius + 3, (0, 255, 255), 2)

    cv2.imwrite(preview_path, first_frame_bgr)
    print(f"Preview image saved to {preview_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize HAMSTER path on episode video")
    parser.add_argument("--zarr-path", type=str,
                       default="../data/pick_apple_messy_50.zarr",
                       help="Path to .zarr dataset")
    parser.add_argument("--hamster-paths", type=str,
                       default="../data/pick_apple_messy_50/hamster_paths.pkl",
                       help="Path to HAMSTER paths .pkl file")
    parser.add_argument("--episode", type=int, default=0,
                       help="Episode index to visualize")
    parser.add_argument("--output", type=str, default="hamster_path_overlay.mp4",
                       help="Output video path")
    parser.add_argument("--fps", type=int, default=10,
                       help="Frames per second")

    args = parser.parse_args()

    create_overlay_video(
        zarr_path=args.zarr_path,
        hamster_paths_pkl=args.hamster_paths,
        episode_idx=args.episode,
        output_path=args.output,
        fps=args.fps
    )
