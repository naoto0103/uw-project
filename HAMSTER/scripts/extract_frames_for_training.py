#!/usr/bin/env python3
"""
Extract frames from RoboTwin 2.0 HDF5 dataset for HAMSTER path generation.

This script extracts head_camera RGB images from HDF5 files and saves them
as PNG files in the format expected by HAMSTER path generation scripts.

Output structure:
    {output_dir}/{task}/episode_{XX}/frames/frame_{XXXX}.png
"""

import os
import h5py
import numpy as np
from PIL import Image
import io
from pathlib import Path
from tqdm import tqdm
import argparse


def decode_image(image_bytes):
    """Decode image from bytes (JPEG/PNG compressed)."""
    if isinstance(image_bytes, bytes):
        img = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_bytes, np.ndarray):
        if image_bytes.dtype == np.uint8 and len(image_bytes.shape) == 1:
            img = Image.open(io.BytesIO(image_bytes.tobytes()))
        else:
            img = Image.fromarray(image_bytes)
    else:
        # Try converting to bytes
        img = Image.open(io.BytesIO(bytes(image_bytes)))
    return img


def extract_task_episodes(
    dataset_base_dir: str,
    output_base_dir: str,
    task_name: str,
    condition: str,  # 'cluttered' or 'clean'
    num_episodes: int = 50,
):
    """Extract frames from a single task's episodes."""

    # Source directory
    config_name = f"aloha-agilex_{condition}_50"
    data_dir = Path(dataset_base_dir) / task_name / config_name / "data"

    if not data_dir.exists():
        print(f"[WARNING] Data directory not found: {data_dir}")
        return 0

    # Output directory
    output_task_dir = Path(output_base_dir) / task_name

    total_frames = 0

    for ep_idx in tqdm(range(num_episodes), desc=f"{task_name}"):
        # Source file
        hdf5_path = data_dir / f"episode{ep_idx}.hdf5"
        if not hdf5_path.exists():
            print(f"[WARNING] Episode file not found: {hdf5_path}")
            continue

        # Output directory for this episode
        episode_dir = output_task_dir / f"episode_{ep_idx:02d}" / "frames"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames
        with h5py.File(hdf5_path, 'r') as f:
            rgb_data = f['observation']['head_camera']['rgb']
            num_frames = len(rgb_data)

            for frame_idx in range(num_frames):
                output_path = episode_dir / f"frame_{frame_idx:04d}.png"

                # Skip if already exists
                if output_path.exists():
                    continue

                # Decode and save
                try:
                    img_bytes = rgb_data[frame_idx]
                    img = decode_image(img_bytes)
                    img.save(output_path)
                    total_frames += 1
                except Exception as e:
                    print(f"[ERROR] Failed to decode frame {frame_idx} in {hdf5_path}: {e}")

    return total_frames


def main():
    parser = argparse.ArgumentParser(description="Extract frames from RoboTwin 2.0 HDF5 dataset")
    parser.add_argument(
        "--condition",
        type=str,
        choices=["cluttered", "clean"],
        required=True,
        help="Table condition (cluttered or clean)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["beat_block_hammer", "click_bell", "move_can_pot", "open_microwave", "turn_switch", "hanging_mug"],
        help="Tasks to extract",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--dataset_base_dir",
        type=str,
        default="/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/dataset/dataset",
        help="Base directory of RoboTwin 2.0 dataset",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Output base directory (default: HAMSTER/results/evaluation_tasks_{condition})",
    )
    args = parser.parse_args()

    # Set default output directory
    if args.output_base_dir is None:
        args.output_base_dir = f"/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_{args.condition}"

    print(f"=" * 60)
    print(f"Extracting frames for {args.condition} condition")
    print(f"=" * 60)
    print(f"Tasks: {args.tasks}")
    print(f"Episodes per task: {args.num_episodes}")
    print(f"Dataset base: {args.dataset_base_dir}")
    print(f"Output base: {args.output_base_dir}")
    print(f"=" * 60)

    total_frames = 0
    for task in args.tasks:
        print(f"\nProcessing {task}...")
        frames = extract_task_episodes(
            dataset_base_dir=args.dataset_base_dir,
            output_base_dir=args.output_base_dir,
            task_name=task,
            condition=args.condition,
            num_episodes=args.num_episodes,
        )
        total_frames += frames
        print(f"  Extracted {frames} frames")

    print(f"\n{'=' * 60}")
    print(f"Total frames extracted: {total_frames}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
