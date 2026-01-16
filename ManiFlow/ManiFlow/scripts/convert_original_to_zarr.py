#!/usr/bin/env python3
"""
Convert RoboTwin 2.0 original RGB images and actions to Zarr format for fast training.

This script converts:
- Original RGB images (PNG files from frames/) → Zarr array
- Actions from HDF5 → Zarr array
- States from HDF5 → Zarr array

Output structure matches ManiFlow's ReplayBuffer format:
    robotwin2_original.zarr/
    ├── data/
    │   ├── image            # (N, 3, 224, 224) float32, normalized [0,1]
    │   ├── action           # (N, 14) float32
    │   └── state            # (N, 14) float32
    └── meta/
        └── episode_ends     # (num_episodes,) int64

Usage:
    python convert_original_to_zarr.py --input-dir /path/to/evaluation_tasks_clean --output data/zarr/clean_original.zarr
    python convert_original_to_zarr.py --input-dir /path/to/evaluation_tasks_cluttered --output data/zarr/cluttered_original.zarr
    python convert_original_to_zarr.py --dry-run  # Preview without writing
"""

import os
import sys
import argparse
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm
from termcolor import cprint


# Default paths
DEFAULT_INPUT_BASE = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_clean")
DEFAULT_ROBOTWIN2_BASE = Path("/mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/dataset/dataset")
DEFAULT_OUTPUT_PATH = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/zarr/clean_original.zarr")

# Single-arm tasks
SINGLE_ARM_TASKS = [
    "beat_block_hammer",
    "click_bell",
]

# Image settings
IMAGE_SIZE = (224, 224)  # Target size for training


def load_rgb_image(img_path: Path, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Load and preprocess an RGB image from frames/ directory.

    Args:
        img_path: Path to the PNG file
        target_size: Target (H, W) for resizing

    Returns:
        Image array (3, H, W) float32, normalized to [0, 1]
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if needed
    if img.shape[:2] != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))

    # Normalize to [0, 1] and transpose to (C, H, W)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)

    return img


def load_actions_from_hdf5(
    robotwin2_base: Path,
    task: str,
    episode_idx: int,
    robot: str = "aloha-agilex",
    config: str = "clean_50"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load action and state data from RoboTwin 2.0 HDF5 file.

    Returns:
        Tuple of (actions, states) arrays, both (T, 14) float32
    """
    inner_dir = f"{robot}_{config}"
    hdf5_name = f"episode{episode_idx}.hdf5"

    # Try extracted HDF5 first
    extracted_hdf5 = robotwin2_base / task / inner_dir / "data" / hdf5_name

    if extracted_hdf5.exists():
        with h5py.File(extracted_hdf5, 'r') as f:
            actions = f['joint_action/vector'][:].astype(np.float32)
            states = actions.copy()  # Use actions as states for now
        return actions, states

    # Fallback to ZIP
    zip_path = robotwin2_base / task / f"{robot}_{config}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Neither extracted HDF5 nor ZIP found for {task}/episode{episode_idx}")

    hdf5_inner_path = f"{inner_dir}/data/{hdf5_name}"

    with zipfile.ZipFile(zip_path, 'r') as zf:
        with tempfile.TemporaryDirectory() as temp_dir:
            zf.extract(hdf5_inner_path, temp_dir)
            temp_hdf5 = os.path.join(temp_dir, hdf5_inner_path)

            with h5py.File(temp_hdf5, 'r') as f:
                actions = f['joint_action/vector'][:].astype(np.float32)
                states = actions.copy()

    return actions, states


def process_episode(
    input_base: Path,
    robotwin2_base: Path,
    task: str,
    episode_idx: int,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    hdf5_config: str = "clean_50"
) -> Optional[dict]:
    """
    Process a single episode: load all RGB images and actions.

    Returns:
        Dict with 'image', 'action', 'state' arrays, or None if failed
    """
    episode_dir = input_base / task / f"episode_{episode_idx:02d}"
    frames_dir = episode_dir / "frames"

    if not frames_dir.exists():
        return None

    # Get sorted frame files
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if len(frame_files) == 0:
        return None

    # Extract frame indices
    frame_indices = []
    for f in frame_files:
        idx = int(f.stem.split("_")[1])
        frame_indices.append(idx)
    frame_indices = sorted(frame_indices)

    # Load all RGB images
    images = []
    for frame_idx in frame_indices:
        img_path = frames_dir / f"frame_{frame_idx:04d}.png"
        try:
            img = load_rgb_image(img_path, target_size)
            images.append(img)
        except Exception as e:
            cprint(f"Warning: Failed to load {img_path}: {e}", "yellow")
            return None

    images = np.stack(images, axis=0)  # (T, 3, H, W)

    # Load actions
    try:
        all_actions, all_states = load_actions_from_hdf5(
            robotwin2_base, task, episode_idx, config=hdf5_config
        )
    except Exception as e:
        cprint(f"Warning: Failed to load actions for {task}/ep{episode_idx}: {e}", "yellow")
        return None

    # Align actions with frame indices
    max_frame = max(frame_indices)
    if max_frame >= len(all_actions):
        cprint(f"Warning: Frame index {max_frame} >= action length {len(all_actions)} for {task}/ep{episode_idx}", "yellow")
        # Truncate frame indices to available actions
        valid_frames = [f for f in frame_indices if f < len(all_actions)]
        if len(valid_frames) == 0:
            return None
        frame_indices = valid_frames
        images = images[:len(valid_frames)]

    # Select actions/states corresponding to frame indices
    actions = all_actions[frame_indices].astype(np.float32)
    states = all_states[frame_indices].astype(np.float32)

    return {
        'image': images,
        'action': actions,
        'state': states,
    }


def convert_to_zarr(
    input_base: Path,
    robotwin2_base: Path,
    output_path: Path,
    tasks: List[str],
    n_episodes: int = 50,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    hdf5_config: str = "clean_50",
    dry_run: bool = False
):
    """
    Convert all original RGB data to Zarr format.
    """
    cprint("=" * 70, "cyan")
    cprint("Convert RoboTwin 2.0 Original RGB Data to Zarr", "cyan")
    cprint("=" * 70, "cyan")
    cprint(f"Input: {input_base}", "yellow")
    cprint(f"Tasks: {tasks}", "yellow")
    cprint(f"Episodes per task: {n_episodes}", "yellow")
    cprint(f"Image size: {target_size}", "yellow")
    cprint(f"HDF5 config: {hdf5_config}", "yellow")
    cprint(f"Output: {output_path}", "yellow")
    cprint(f"Dry run: {dry_run}", "yellow")
    print()

    # Collect all episodes to process
    episodes_to_process = []
    for task in tasks:
        for ep_idx in range(n_episodes):
            episodes_to_process.append((task, ep_idx))

    cprint(f"Total episodes to process: {len(episodes_to_process)}", "green")

    # Process episodes
    all_data = {
        'image': [],
        'action': [],
        'state': [],
    }
    episode_ends = []
    current_length = 0

    success_count = 0
    fail_count = 0

    for task, ep_idx in tqdm(episodes_to_process, desc="Processing episodes"):
        result = process_episode(
            input_base, robotwin2_base, task, ep_idx, target_size, hdf5_config
        )

        if result is None:
            fail_count += 1
            continue

        n_frames = len(result['action'])

        all_data['image'].append(result['image'])
        all_data['action'].append(result['action'])
        all_data['state'].append(result['state'])

        current_length += n_frames
        episode_ends.append(current_length)
        success_count += 1

    cprint(f"\nProcessed: {success_count} episodes, Failed: {fail_count} episodes", "green")

    if success_count == 0:
        cprint("No episodes processed successfully!", "red")
        return

    # Concatenate all data
    cprint("Concatenating data...", "cyan")
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key], axis=0)
        cprint(f"  {key}: {all_data[key].shape}, dtype={all_data[key].dtype}", "cyan")

    episode_ends = np.array(episode_ends, dtype=np.int64)
    cprint(f"  episode_ends: {episode_ends.shape}", "cyan")

    # Calculate total size
    total_bytes = sum(arr.nbytes for arr in all_data.values())
    cprint(f"\nTotal data size: {total_bytes / 1e9:.2f} GB", "yellow")

    if dry_run:
        cprint("\n[DRY RUN] Would write to:", "magenta")
        cprint(f"  {output_path}", "magenta")
        return

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create Zarr store
    cprint(f"\nWriting to Zarr: {output_path}", "green")

    # Remove existing if present
    if output_path.exists():
        import shutil
        cprint(f"Removing existing: {output_path}", "yellow")
        shutil.rmtree(output_path)

    # Create Zarr structure
    compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.NOSHUFFLE)

    root = zarr.open(str(output_path), mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    # Write data arrays with optimal chunking
    # For images: chunk by single frames to allow random access
    image_chunks = (1, 3, target_size[0], target_size[1])
    data_group.create_dataset(
        'image',
        data=all_data['image'],
        chunks=image_chunks,
        compressor=compressor,
        dtype=np.float32
    )
    cprint(f"  Written image: {all_data['image'].shape}", "green")

    # For actions/states: chunk by ~1000 frames
    action_chunks = (min(1000, len(all_data['action'])), all_data['action'].shape[1])
    data_group.create_dataset(
        'action',
        data=all_data['action'],
        chunks=action_chunks,
        compressor=compressor,
        dtype=np.float32
    )
    cprint(f"  Written action: {all_data['action'].shape}", "green")

    data_group.create_dataset(
        'state',
        data=all_data['state'],
        chunks=action_chunks,
        compressor=compressor,
        dtype=np.float32
    )
    cprint(f"  Written state: {all_data['state'].shape}", "green")

    # Write episode_ends
    meta_group.create_dataset(
        'episode_ends',
        data=episode_ends,
        chunks=(len(episode_ends),),
        compressor=None,
        dtype=np.int64
    )
    cprint(f"  Written episode_ends: {episode_ends.shape}", "green")

    # Print summary
    cprint("\n" + "=" * 70, "cyan")
    cprint("Conversion Complete!", "green")
    cprint("=" * 70, "cyan")
    cprint(f"Output: {output_path}", "green")
    cprint(f"Total frames: {current_length}", "green")
    cprint(f"Total episodes: {len(episode_ends)}", "green")

    # Verify
    cprint("\nVerifying Zarr structure:", "cyan")
    root = zarr.open(str(output_path), mode='r')
    print(root.tree())


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin 2.0 original RGB images to Zarr format"
    )
    parser.add_argument(
        "--input-dir", type=str, default=str(DEFAULT_INPUT_BASE),
        help="Base directory containing frames/ subdirectories"
    )
    parser.add_argument(
        "--robotwin2-base", type=str, default=str(DEFAULT_ROBOTWIN2_BASE),
        help="Base directory containing RoboTwin 2.0 dataset (for HDF5)"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT_PATH),
        help="Output Zarr path"
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=SINGLE_ARM_TASKS,
        help="Tasks to include"
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of episodes per task"
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=[224, 224],
        help="Target image size (H W)"
    )
    parser.add_argument(
        "--hdf5-config", type=str, default="clean_50",
        help="HDF5 config name (e.g., clean_50, cluttered_50)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without writing"
    )

    args = parser.parse_args()

    convert_to_zarr(
        input_base=Path(args.input_dir),
        robotwin2_base=Path(args.robotwin2_base),
        output_path=Path(args.output),
        tasks=args.tasks,
        n_episodes=args.episodes,
        target_size=tuple(args.image_size),
        hdf5_config=args.hdf5_config,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
