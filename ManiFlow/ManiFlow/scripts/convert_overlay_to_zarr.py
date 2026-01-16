#!/usr/bin/env python3
"""
Convert RoboTwin 2.0 overlay images and actions to Zarr format for fast training.

This script converts:
- Overlay images (PNG files) → Zarr array
- Actions from HDF5 → Zarr array
- States from HDF5 → Zarr array

Supports two data sources:
- VILA path overlays: evaluation_tasks_{clean,cluttered}/ (default)
- GT path overlays: gt_paths_{clean,cluttered}/ (with --gt-path flag)

Output structure matches ManiFlow's ReplayBuffer format:
    robotwin2_overlay_single_arm.zarr/
    ├── data/
    │   ├── overlay_image    # (N, 3, 224, 224) float32, normalized [0,1]
    │   ├── action           # (N, 14) float32
    │   └── state            # (N, 14) float32
    └── meta/
        └── episode_ends     # (num_episodes,) int64

Usage:
    # VILA path overlays (default)
    python convert_overlay_to_zarr.py --overlay-base .../evaluation_tasks_clean --output clean_overlay.zarr

    # GT path overlays
    python convert_overlay_to_zarr.py --gt-path --overlay-base .../gt_paths_clean --output clean_gt_overlay.zarr

    python convert_overlay_to_zarr.py --tasks beat_block_hammer click_bell --episodes 10
    python convert_overlay_to_zarr.py --dry-run  # Preview without writing
"""

import os
import sys
import argparse
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm
from termcolor import cprint


# Default paths
DEFAULT_OVERLAY_BASE = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/robotwin2_single_6tasks_vila")
DEFAULT_ROBOTWIN2_BASE = Path("/mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/dataset/dataset")
DEFAULT_OUTPUT_PATH = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/robotwin2_overlay_single_arm.zarr")

# Single-arm tasks
SINGLE_ARM_TASKS = [
    "beat_block_hammer",
    "click_bell",
    "move_can_pot",
    # "open_microwave",      # Excluded: low path generation success rate
    # "place_object_stand",  # Can add later
    # "turn_switch",         # Can add later
]

# Image settings
IMAGE_SIZE = (224, 224)  # Target size for training


def load_overlay_image(img_path: Path, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Load and preprocess an overlay image.

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
    overlay_base: Path,
    robotwin2_base: Path,
    task: str,
    episode_idx: int,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    gt_path: bool = False
) -> Optional[dict]:
    """
    Process a single episode: load all overlay images and actions.

    Args:
        overlay_base: Base directory containing overlay data
        robotwin2_base: Base directory containing RoboTwin 2.0 HDF5 data
        task: Task name
        episode_idx: Episode index
        target_size: Target image size (H, W)
        gt_path: If True, use GT path naming convention (frame_XXXX.png)
                 If False, use VILA path naming convention (overlay_XXXX.png)

    Returns:
        Dict with 'overlay_image', 'action', 'state' arrays, or None if failed
    """
    episode_dir = overlay_base / task / f"episode_{episode_idx:02d}"
    overlay_dir = episode_dir / "overlay_images"

    if not overlay_dir.exists():
        return None

    # Get sorted overlay files based on data source
    # GT path: frame_XXXX.png, VILA path: overlay_XXXX.png
    if gt_path:
        overlay_files = sorted(overlay_dir.glob("frame_*.png"))
    else:
        overlay_files = sorted(overlay_dir.glob("overlay_*.png"))

    if len(overlay_files) == 0:
        return None

    # Extract frame indices
    # GT path: "frame_0000" -> 0, VILA path: "overlay_0000" -> 0
    frame_indices = []
    for f in overlay_files:
        # Both formats: split by "_" and take the last numeric part
        idx = int(f.stem.split("_")[-1])
        frame_indices.append(idx)
    frame_indices = sorted(frame_indices)

    # Load all overlay images
    overlay_images = []
    for frame_idx in frame_indices:
        if gt_path:
            img_path = overlay_dir / f"frame_{frame_idx:04d}.png"
        else:
            img_path = overlay_dir / f"overlay_{frame_idx:04d}.png"
        try:
            img = load_overlay_image(img_path, target_size)
            overlay_images.append(img)
        except Exception as e:
            cprint(f"Warning: Failed to load {img_path}: {e}", "yellow")
            return None

    overlay_images = np.stack(overlay_images, axis=0)  # (T, 3, H, W)

    # Load actions
    try:
        all_actions, all_states = load_actions_from_hdf5(
            robotwin2_base, task, episode_idx
        )
    except Exception as e:
        cprint(f"Warning: Failed to load actions for {task}/ep{episode_idx}: {e}", "yellow")
        return None

    # Align actions with frame indices
    # frame_indices are the actual frame numbers in the episode
    max_frame = max(frame_indices)
    if max_frame >= len(all_actions):
        cprint(f"Warning: Frame index {max_frame} >= action length {len(all_actions)} for {task}/ep{episode_idx}", "yellow")
        # Truncate frame indices to available actions
        valid_frames = [f for f in frame_indices if f < len(all_actions)]
        if len(valid_frames) == 0:
            return None
        frame_indices = valid_frames
        overlay_images = overlay_images[:len(valid_frames)]

    # Select actions/states corresponding to frame indices
    actions = all_actions[frame_indices].astype(np.float32)
    states = all_states[frame_indices].astype(np.float32)

    return {
        'overlay_image': overlay_images,
        'action': actions,
        'state': states,
    }


def convert_to_zarr(
    overlay_base: Path,
    robotwin2_base: Path,
    output_path: Path,
    tasks: List[str],
    n_episodes: int = 50,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    num_workers: int = 8,
    dry_run: bool = False,
    gt_path: bool = False
):
    """
    Convert all overlay data to Zarr format.

    Args:
        overlay_base: Base directory containing overlay data
        robotwin2_base: Base directory containing RoboTwin 2.0 HDF5 data
        output_path: Output Zarr path
        tasks: List of task names
        n_episodes: Number of episodes per task
        target_size: Target image size (H, W)
        num_workers: Number of parallel workers (unused currently)
        dry_run: If True, preview without writing
        gt_path: If True, use GT path naming convention
    """
    cprint("=" * 70, "cyan")
    cprint("Convert RoboTwin 2.0 Overlay Data to Zarr", "cyan")
    cprint("=" * 70, "cyan")
    cprint(f"Data source: {'GT path' if gt_path else 'VILA path'}", "yellow")
    cprint(f"Tasks: {tasks}", "yellow")
    cprint(f"Episodes per task: {n_episodes}", "yellow")
    cprint(f"Image size: {target_size}", "yellow")
    cprint(f"Output: {output_path}", "yellow")
    cprint(f"Dry run: {dry_run}", "yellow")
    print()

    # Collect all episodes to process
    episodes_to_process = []
    for task in tasks:
        for ep_idx in range(n_episodes):
            episodes_to_process.append((task, ep_idx))

    cprint(f"Total episodes to process: {len(episodes_to_process)}", "green")

    # Process episodes (can parallelize image loading)
    all_data = {
        'overlay_image': [],
        'action': [],
        'state': [],
    }
    episode_ends = []
    current_length = 0

    success_count = 0
    fail_count = 0

    for task, ep_idx in tqdm(episodes_to_process, desc="Processing episodes"):
        result = process_episode(
            overlay_base, robotwin2_base, task, ep_idx, target_size, gt_path=gt_path
        )

        if result is None:
            fail_count += 1
            continue

        n_frames = len(result['action'])

        all_data['overlay_image'].append(result['overlay_image'])
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
    overlay_chunks = (1, 3, target_size[0], target_size[1])
    data_group.create_dataset(
        'overlay_image',
        data=all_data['overlay_image'],
        chunks=overlay_chunks,
        compressor=compressor,
        dtype=np.float32
    )
    cprint(f"  Written overlay_image: {all_data['overlay_image'].shape}", "green")

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
        description="Convert RoboTwin 2.0 overlay images to Zarr format"
    )
    parser.add_argument(
        "--overlay-base", type=str, default=str(DEFAULT_OVERLAY_BASE),
        help="Base directory containing overlay images"
    )
    parser.add_argument(
        "--robotwin2-base", type=str, default=str(DEFAULT_ROBOTWIN2_BASE),
        help="Base directory containing RoboTwin 2.0 dataset"
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
        "--num-workers", type=int, default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without writing"
    )
    parser.add_argument(
        "--gt-path", action="store_true",
        help="Use GT path data (frame_XXXX.png) instead of VILA path data (overlay_XXXX.png)"
    )

    args = parser.parse_args()

    convert_to_zarr(
        overlay_base=Path(args.overlay_base),
        robotwin2_base=Path(args.robotwin2_base),
        output_path=Path(args.output),
        tasks=args.tasks,
        n_episodes=args.episodes,
        target_size=tuple(args.image_size),
        num_workers=args.num_workers,
        dry_run=args.dry_run,
        gt_path=args.gt_path
    )


if __name__ == "__main__":
    main()
