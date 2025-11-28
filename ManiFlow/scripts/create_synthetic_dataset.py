#!/usr/bin/env python3
"""
Create a synthetic RoboTwin-like dataset for testing HAMSTER integration.

This script generates a small synthetic dataset with:
- Point cloud data (random 3D points)
- Head camera images (simple colored backgrounds with shapes)
- State data (robot joint positions)
- Action data (robot actions)

The dataset is saved in .zarr format compatible with ManiFlow.
"""

import argparse
import os
import numpy as np
import zarr
from pathlib import Path
from termcolor import cprint
import cv2


def create_synthetic_image(
    height: int = 224,
    width: int = 224,
    episode_idx: int = 0,
    timestep: int = 0
) -> np.ndarray:
    """
    Create a synthetic RGB image with geometric shapes.

    Args:
        height: Image height
        width: Image width
        episode_idx: Episode index (affects color/shape)
        timestep: Timestep index

    Returns:
        RGB image as numpy array (H, W, 3)
    """
    # Create background
    bg_color = np.array([
        (episode_idx * 50 + 100) % 255,
        (episode_idx * 30 + 150) % 255,
        (episode_idx * 70 + 200) % 255
    ], dtype=np.uint8)
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Add some objects (circles, rectangles)
    # Object 1: Circle (target object)
    center_x = int(width * (0.3 + 0.1 * np.sin(episode_idx)))
    center_y = int(height * (0.4 + 0.05 * np.cos(episode_idx)))
    radius = int(min(height, width) * 0.1)
    color1 = (255, 0, 0)  # Red
    cv2.circle(image, (center_x, center_y), radius, color1, -1)

    # Object 2: Rectangle (table/surface)
    rect_x1 = int(width * 0.1)
    rect_y1 = int(height * 0.7)
    rect_x2 = int(width * 0.9)
    rect_y2 = int(height * 0.95)
    color2 = (139, 69, 19)  # Brown
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), color2, -1)

    # Object 3: Small square (gripper indicator)
    gripper_x = int(width * (0.6 + 0.3 * timestep / 50))
    gripper_y = int(height * (0.3 + 0.4 * timestep / 50))
    gripper_size = 15
    color3 = (0, 255, 0)  # Green
    cv2.rectangle(
        image,
        (gripper_x - gripper_size, gripper_y - gripper_size),
        (gripper_x + gripper_size, gripper_y + gripper_size),
        color3,
        -1
    )

    return image


def create_synthetic_point_cloud(
    n_points: int = 1024,
    episode_idx: int = 0,
    timestep: int = 0
) -> np.ndarray:
    """
    Create a synthetic point cloud.

    Args:
        n_points: Number of points
        episode_idx: Episode index
        timestep: Timestep index

    Returns:
        Point cloud as numpy array (N, 6) for XYZRGB
    """
    # Generate random 3D points (simulating scene)
    np.random.seed(episode_idx * 100 + timestep)

    # Table surface points
    n_table = n_points // 2
    table_xyz = np.random.randn(n_table, 3) * 0.1
    table_xyz[:, 2] = 0  # Flat surface
    table_rgb = np.tile([139 / 255, 69 / 255, 19 / 255], (n_table, 1))

    # Object points
    n_obj = n_points // 4
    obj_center = np.array([0.3, 0.4, 0.1])
    obj_xyz = np.random.randn(n_obj, 3) * 0.05 + obj_center
    obj_rgb = np.tile([1.0, 0.0, 0.0], (n_obj, 1))

    # Random background points
    n_bg = n_points - n_table - n_obj
    bg_xyz = np.random.randn(n_bg, 3) * 0.5
    bg_rgb = np.random.rand(n_bg, 3)

    # Combine
    xyz = np.vstack([table_xyz, obj_xyz, bg_xyz])
    rgb = np.vstack([table_rgb, obj_rgb, bg_rgb])

    # Combine XYZ and RGB
    point_cloud = np.hstack([xyz, rgb]).astype(np.float32)

    return point_cloud


def create_synthetic_dataset(
    output_path: str,
    n_episodes: int = 10,
    episode_length: int = 50,
    n_points: int = 1024,
    img_height: int = 224,
    img_width: int = 224,
    state_dim: int = 14,
    action_dim: int = 14
):
    """
    Create a complete synthetic dataset in zarr format.

    Args:
        output_path: Path to save .zarr directory
        n_episodes: Number of episodes
        episode_length: Length of each episode
        n_points: Number of points per point cloud
        img_height: Image height
        img_width: Image width
        state_dim: State dimension
        action_dim: Action dimension
    """
    cprint(f"Creating synthetic dataset at {output_path}", "green")
    cprint(f"Episodes: {n_episodes}, Length: {episode_length}", "cyan")

    # Calculate total timesteps
    total_timesteps = n_episodes * episode_length

    # Create zarr group
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)

    # Create data group
    data = root.create_group('data')

    # Create arrays
    cprint("Creating point cloud array...", "yellow")
    point_cloud = data.zeros(
        'point_cloud',
        shape=(total_timesteps, n_points, 6),  # XYZRGB
        chunks=(1, n_points, 6),
        dtype=np.float32
    )

    cprint("Creating head camera array...", "yellow")
    head_camera = data.zeros(
        'head_camera',
        shape=(total_timesteps, 3, img_height, img_width),  # CHW format
        chunks=(1, 3, img_height, img_width),
        dtype=np.uint8
    )

    cprint("Creating state array...", "yellow")
    state = data.zeros(
        'state',
        shape=(total_timesteps, state_dim),
        chunks=(episode_length, state_dim),
        dtype=np.float32
    )

    cprint("Creating action array...", "yellow")
    action = data.zeros(
        'action',
        shape=(total_timesteps, action_dim),
        chunks=(episode_length, action_dim),
        dtype=np.float32
    )

    # Fill with synthetic data
    cprint("Generating synthetic data...", "yellow")
    episode_ends = []

    for ep_idx in range(n_episodes):
        ep_start = ep_idx * episode_length

        for t in range(episode_length):
            global_t = ep_start + t

            # Generate point cloud
            pc = create_synthetic_point_cloud(n_points, ep_idx, t)
            point_cloud[global_t] = pc

            # Generate image (HWC -> CHW)
            img = create_synthetic_image(img_height, img_width, ep_idx, t)
            img_chw = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            head_camera[global_t] = img_chw

            # Generate state (sinusoidal motion)
            s = np.sin(np.linspace(0, 2 * np.pi, state_dim) + t * 0.1 + ep_idx)
            state[global_t] = s.astype(np.float32)

            # Generate action (similar to state with offset)
            a = np.sin(np.linspace(0, 2 * np.pi, action_dim) + (t + 1) * 0.1 + ep_idx)
            action[global_t] = a.astype(np.float32)

        episode_ends.append(ep_start + episode_length)
        cprint(f"Episode {ep_idx + 1}/{n_episodes} generated", "cyan")

    # Create metadata
    meta = root.create_group('meta')
    episode_ends_arr = meta.array('episode_ends', data=np.array(episode_ends, dtype=np.int64))

    cprint(f"Dataset created successfully!", "green")
    cprint(f"Total timesteps: {total_timesteps}", "green")
    cprint(f"Point cloud shape: {point_cloud.shape}", "green")
    cprint(f"Head camera shape: {head_camera.shape}", "green")
    cprint(f"State shape: {state.shape}", "green")
    cprint(f"Action shape: {action.shape}", "green")
    cprint(f"Episode ends: {episode_ends}", "green")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic RoboTwin dataset")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/synthetic_test_50/replay_buffer.zarr",
        help="Output path for zarr dataset"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes (default: 10)"
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=50,
        help="Length of each episode (default: 50)"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=1024,
        help="Number of points in point cloud (default: 1024)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size (default: 224)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    create_synthetic_dataset(
        output_path=args.output_path,
        n_episodes=args.n_episodes,
        episode_length=args.episode_length,
        n_points=args.n_points,
        img_height=args.img_size,
        img_width=args.img_size
    )


if __name__ == "__main__":
    main()
