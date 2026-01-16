#!/usr/bin/env python3
"""
Extract frames from RoboTwin 2.0 HDF5 files.

Phase 3.6 Stage 3: Extract head_camera RGB frames from RoboTwin 2.0 dataset.
Supports extracting from zip files without full extraction.
"""

import os
import sys
import zipfile
import tempfile
import shutil
import h5py
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


# RoboTwin 2.0 dataset location (Hyak HPC)
ROBOTWIN2_DATASET_DIR = Path("/mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/dataset/dataset")

# Default robot and config
DEFAULT_ROBOT = "aloha-agilex"
DEFAULT_CONFIG = "clean_50"

# Selected tasks for Stage 3 (original 6 tasks)
SELECTED_TASKS = [
    "beat_block_hammer",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_shoe",
    "place_empty_cup",
    "hanging_mug",
]

# Single-arm tasks (newly downloaded)
SINGLE_ARM_TASKS = [
    "lift_pot",
    "open_laptop",
    "put_object_cabinet",
    "stack_blocks_two",
    # Additional single-arm tasks (2024-12-04)
    "click_bell",
    "move_can_pot",
    "place_object_stand",
    "open_microwave",
    "turn_switch",
]

# Task instructions for Qwen3 path generation
TASK_INSTRUCTIONS = {
    # Original bimanual tasks
    "beat_block_hammer": "Pick up the hammer and beat the block",
    "pick_diverse_bottles": "Pick up the bottles from the table",
    "pick_dual_bottles": "Pick up two bottles with both hands",
    "place_shoe": "Place the shoe on the target location",
    "place_empty_cup": "Place the empty cup on the target location",
    "hanging_mug": "Hang the mug on the hook",
    # Single-arm tasks
    "lift_pot": "Lift the pot from the table",
    "open_laptop": "Open the laptop",
    "put_object_cabinet": "Put the object into the cabinet",
    "stack_blocks_two": "Stack two blocks on top of each other",
    # Additional single-arm tasks (2024-12-04)
    "click_bell": "click the <bell's top center> on the table",
    "move_can_pot": "there is a can and a pot on the table, use one arm to <pick up the can> and <move it to beside the pot>",
    "place_object_stand": "use appropriate arm to place the object on the stand",
    "open_microwave": "Use one arm to open the microwave.",
    "turn_switch": "use the robotic arm to click the switch",
}


def get_zip_path(task_name: str, robot: str = DEFAULT_ROBOT, config: str = DEFAULT_CONFIG) -> Path:
    """Get the path to the zip file for a task/robot/config combination."""
    zip_name = f"{robot}_{config}.zip"
    return ROBOTWIN2_DATASET_DIR / task_name / zip_name


def extract_frames_from_hdf5(
    hdf5_path: str,
    output_dir: str,
    camera: str = "head_camera",
    verbose: bool = True
) -> int:
    """
    Extract RGB frames from an HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        output_dir: Directory to save frames
        camera: Camera to extract (head_camera, front_camera, left_camera, right_camera)
        verbose: Print progress

    Returns:
        Number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        rgb_key = f'observation/{camera}/rgb'
        if rgb_key not in f:
            print(f"ERROR: {rgb_key} not found in {hdf5_path}")
            return 0

        rgb_data = f[rgb_key]
        num_frames = len(rgb_data)

        if verbose:
            print(f"  Extracting {num_frames} frames from {camera}...")

        for i in range(num_frames):
            # Decode image from byte stream
            img_bytes = rgb_data[i]
            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if image is None:
                print(f"  WARNING: Failed to decode frame {i}")
                continue

            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, image)

        if verbose:
            print(f"  Saved {num_frames} frames to {output_dir}")

    return num_frames


def extract_episode_from_zip(
    task_name: str,
    episode_idx: int,
    output_dir: str,
    robot: str = DEFAULT_ROBOT,
    config: str = DEFAULT_CONFIG,
    camera: str = "head_camera",
    verbose: bool = True
) -> Tuple[int, str]:
    """
    Extract frames from a specific episode within a zip file.

    Args:
        task_name: Task name (e.g., "beat_block_hammer")
        episode_idx: Episode index (0-49 for clean_50)
        output_dir: Directory to save frames
        robot: Robot type
        config: Configuration (clean_50 or randomized_500)
        camera: Camera to extract
        verbose: Print progress

    Returns:
        Tuple of (num_frames, task_instruction)
    """
    zip_path = get_zip_path(task_name, robot, config)

    if not zip_path.exists():
        print(f"ERROR: Zip file not found: {zip_path}")
        return 0, ""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Construct path within zip
    inner_dir = f"{robot}_{config}"
    hdf5_name = f"episode{episode_idx}.hdf5"
    hdf5_inner_path = f"{inner_dir}/data/{hdf5_name}"

    if verbose:
        print(f"Extracting {task_name}/{hdf5_name}...")
        print(f"  Zip: {zip_path}")
        print(f"  Inner path: {hdf5_inner_path}")

    # Extract HDF5 to temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Check if file exists in zip
            if hdf5_inner_path not in zf.namelist():
                print(f"ERROR: {hdf5_inner_path} not found in zip")
                return 0, ""

            # Extract HDF5 file
            zf.extract(hdf5_inner_path, temp_dir)
            temp_hdf5_path = os.path.join(temp_dir, hdf5_inner_path)

            # Extract frames from HDF5
            num_frames = extract_frames_from_hdf5(
                temp_hdf5_path,
                output_dir,
                camera=camera,
                verbose=verbose
            )

    task_instruction = TASK_INSTRUCTIONS.get(task_name, f"Complete the {task_name} task")

    return num_frames, task_instruction


def list_episodes_in_zip(
    task_name: str,
    robot: str = DEFAULT_ROBOT,
    config: str = DEFAULT_CONFIG
) -> List[int]:
    """List available episode indices in a zip file."""
    zip_path = get_zip_path(task_name, robot, config)

    if not zip_path.exists():
        print(f"ERROR: Zip file not found: {zip_path}")
        return []

    episodes = []
    inner_dir = f"{robot}_{config}"

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.startswith(f"{inner_dir}/data/episode") and name.endswith(".hdf5"):
                # Extract episode number
                basename = os.path.basename(name)
                ep_str = basename.replace("episode", "").replace(".hdf5", "")
                try:
                    episodes.append(int(ep_str))
                except ValueError:
                    pass

    return sorted(episodes)


def main():
    """Main function for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from RoboTwin 2.0 dataset")
    parser.add_argument("--task", type=str, default="beat_block_hammer",
                        choices=SELECTED_TASKS + SINGLE_ARM_TASKS, help="Task name")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--robot", type=str, default=DEFAULT_ROBOT, help="Robot type")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Config (clean_50 or randomized_500)")
    parser.add_argument("--camera", type=str, default="head_camera", help="Camera to extract")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available episodes")

    args = parser.parse_args()

    if args.list:
        print(f"Available episodes for {args.task} ({args.robot}_{args.config}):")
        episodes = list_episodes_in_zip(args.task, args.robot, args.config)
        print(f"  Episodes: {episodes}")
        return

    # Set output directory
    if args.output is None:
        script_dir = Path(__file__).parent.absolute()
        output_dir = script_dir.parent / "results" / "video_path_test" / "robotwin2" / args.task / f"episode_{args.episode:02d}" / "frames"
    else:
        output_dir = Path(args.output)

    print("=" * 60)
    print("RoboTwin 2.0 Frame Extraction")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Episode: {args.episode}")
    print(f"Robot: {args.robot}")
    print(f"Config: {args.config}")
    print(f"Camera: {args.camera}")
    print(f"Output: {output_dir}")
    print()

    num_frames, task_instruction = extract_episode_from_zip(
        task_name=args.task,
        episode_idx=args.episode,
        output_dir=str(output_dir),
        robot=args.robot,
        config=args.config,
        camera=args.camera,
        verbose=True
    )

    print()
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"  Frames: {num_frames}")
    print(f"  Task instruction: {task_instruction}")
    print("=" * 60)


if __name__ == "__main__":
    main()
