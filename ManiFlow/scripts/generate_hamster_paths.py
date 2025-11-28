#!/usr/bin/env python3
"""
Generate HAMSTER paths for RoboTwin dataset episodes.

This script processes a RoboTwin .zarr dataset and generates HAMSTER 2D paths
for each episode. The paths are cached to a .pkl file for use with
HAMSTERRoboTwinDataset.

Usage:
    python generate_hamster_paths.py \
        --zarr-path /path/to/dataset.zarr \
        --output-path /path/to/hamster_paths.pkl \
        --task-description "Pick up the apple" \
        --server-ip 127.0.0.1 \
        --server-port 8000
"""

import argparse
import base64
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import zarr
from openai import OpenAI
from termcolor import cprint
from tqdm import tqdm


# Constants
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1
MODEL_NAME = "HAMSTER_dev"


def load_zarr_dataset(zarr_path: str) -> zarr.Group:
    """
    Load a zarr dataset.

    Args:
        zarr_path: Path to .zarr directory

    Returns:
        zarr.Group containing the dataset
    """
    cprint(f"Loading dataset from {zarr_path}", "green")
    group = zarr.open(zarr_path, mode='r')
    return group


def get_episode_boundaries(group: zarr.Group) -> np.ndarray:
    """
    Get episode boundaries from zarr metadata.

    Args:
        group: zarr.Group containing the dataset

    Returns:
        Array of episode end indices
    """
    episode_ends = group['meta']['episode_ends'][:]
    return episode_ends


def get_first_frame_per_episode(
    group: zarr.Group,
    episode_ends: np.ndarray,
    camera_key: str = 'head_camera'
) -> List[np.ndarray]:
    """
    Extract the first frame from each episode.

    Args:
        group: zarr.Group containing the dataset
        episode_ends: Array of episode end indices
        camera_key: Key for camera data in the dataset

    Returns:
        List of images (numpy arrays) for each episode
    """
    if camera_key not in group['data']:
        available_keys = list(group['data'].keys())
        raise KeyError(
            f"Camera key '{camera_key}' not found in dataset. "
            f"Available keys: {available_keys}"
        )

    camera_data = group['data'][camera_key]
    cprint(f"Camera data shape: {camera_data.shape}", "cyan")

    images = []
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    for ep_idx, start_idx in enumerate(episode_starts):
        # Get first frame of this episode
        img = camera_data[start_idx]

        # Handle different image formats
        if img.shape[0] == 3:
            # (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))

        # Ensure uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        images.append(img)

    return images


def process_hamster_response(input_str: str) -> List[List[float]]:
    """
    Parse HAMSTER model response to extract path points.

    Args:
        input_str: Raw response string from HAMSTER

    Returns:
        List of [x, y, gripper_state] points
    """
    # Replace action tags with special coordinates
    input_str = input_str.replace(
        '<action>Close Gripper</action>', '(1000.0, 1000.0)'
    ).replace(
        '<action>Open Gripper</action>', '(1001.0, 1001.0)'
    )

    # Extract points using eval (safe since we control the input format)
    try:
        keypoints = eval(input_str)
    except Exception as e:
        raise ValueError(f"Failed to parse HAMSTER response: {e}")

    # Process points and track gripper state
    processed_points = []
    action_flag = GRIPPER_OPEN  # Default to open

    for point in keypoints:
        x, y = point
        if x == y and x == 1000.0:
            # Close gripper command
            action_flag = GRIPPER_CLOSE
            if processed_points:
                processed_points[-1][2] = action_flag
            continue
        elif x == y and x == 1001.0:
            # Open gripper command
            action_flag = GRIPPER_OPEN
            if processed_points:
                processed_points[-1][2] = action_flag
            continue

        processed_points.append([float(x), float(y), float(action_flag)])

    return processed_points


def generate_path_for_image(
    client: OpenAI,
    image: np.ndarray,
    task_description: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95
) -> List[List[float]]:
    """
    Generate HAMSTER path for a single image.

    Args:
        client: OpenAI API client
        image: RGB image (H, W, 3)
        task_description: Natural language task description
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        List of path points [[x, y, gripper_state], ...]
    """
    # Convert image to BGR for OpenCV encoding
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Encode image to base64
    _, encoded_image_array = cv2.imencode('.jpg', image_bgr)
    encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode('utf-8')

    # Construct prompt
    prompt = (
        f"\nIn the image, please execute the command described in "
        f"<quest>{task_description}</quest>.\n"
        "Provide a sequence of points denoting the trajectory of a robot gripper "
        "to achieve the goal.\n"
        "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. "
        "For example:\n"
        "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), "
        "<action>Open Gripper</action>, (0.74, 0.21), "
        "<action>Close Gripper</action>, ...]</ans>\n"
        "The tuple denotes point x and y location of the end effector in the image. "
        "The action tags indicate gripper actions.\n"
        "Coordinates should be floats between 0 and 1, representing relative positions.\n"
        "Remember to provide points between <ans> and </ans> tags and think step by step."
    )

    # Send request
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=int(max_tokens),
        model=MODEL_NAME,
        extra_body={
            "num_beams": 1,
            "use_cache": False,
            "temperature": float(temperature),
            "top_p": float(top_p)
        },
    )

    # Extract response text
    response_text = response.choices[0].message.content
    if isinstance(response_text, list):
        response_text = response_text[0].get('text', '')

    # Parse the answer
    match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL)
    if match is None:
        raise ValueError(f"No <ans> tags found in response: {response_text[:200]}...")

    ans_content = match.group(1).strip()
    path_points = process_hamster_response(ans_content)

    return path_points


def generate_all_paths(
    zarr_path: str,
    task_description: str,
    server_ip: str = "127.0.0.1",
    server_port: int = 8000,
    camera_key: str = "head_camera",
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95,
    resume_from: Optional[str] = None,
    save_interval: int = 10
) -> dict:
    """
    Generate HAMSTER paths for all episodes in a dataset.

    Args:
        zarr_path: Path to .zarr dataset
        task_description: Task description for HAMSTER
        server_ip: HAMSTER server IP address
        server_port: HAMSTER server port
        camera_key: Key for camera data
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        resume_from: Path to partial results to resume from
        save_interval: Save checkpoint every N episodes

    Returns:
        Dictionary containing paths and metadata
    """
    # Load dataset
    group = load_zarr_dataset(zarr_path)
    episode_ends = get_episode_boundaries(group)
    n_episodes = len(episode_ends)

    cprint(f"Found {n_episodes} episodes", "green")

    # Get first frames
    cprint("Extracting first frames from episodes...", "cyan")
    images = get_first_frame_per_episode(group, episode_ends, camera_key)
    cprint(f"Extracted {len(images)} images", "cyan")

    # Initialize client
    cprint(f"Connecting to HAMSTER server at {server_ip}:{server_port}", "yellow")
    client = OpenAI(base_url=f"http://{server_ip}:{server_port}", api_key="fake-key")

    # Initialize or resume paths
    if resume_from and Path(resume_from).exists():
        cprint(f"Resuming from {resume_from}", "yellow")
        with open(resume_from, 'rb') as f:
            cached_data = pickle.load(f)
        paths = cached_data.get('paths', [])
        start_idx = len(paths)
        cprint(f"Resuming from episode {start_idx}", "yellow")
    else:
        paths = []
        start_idx = 0

    # Generate paths
    failed_episodes = []
    output_path = resume_from if resume_from else None

    for ep_idx in tqdm(range(start_idx, n_episodes), desc="Generating paths", initial=start_idx, total=n_episodes):
        image = images[ep_idx]

        try:
            path = generate_path_for_image(
                client=client,
                image=image,
                task_description=task_description,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            paths.append(path)
            cprint(f"Episode {ep_idx}: Generated {len(path)} points", "green")

        except Exception as e:
            cprint(f"Episode {ep_idx}: Failed - {str(e)}", "red")
            # Add empty path for failed episode
            paths.append([])
            failed_episodes.append(ep_idx)

        # Periodic checkpoint
        if output_path and (ep_idx + 1) % save_interval == 0:
            checkpoint_data = {
                'paths': paths,
                'task_description': task_description,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'zarr_path': zarr_path,
                    'n_episodes': n_episodes,
                    'n_completed': len(paths),
                    'failed_episodes': failed_episodes,
                    'server_ip': server_ip,
                    'server_port': server_port,
                }
            }
            with open(output_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            cprint(f"Checkpoint saved at episode {ep_idx + 1}", "yellow")

        # Small delay to avoid overwhelming the server
        time.sleep(0.1)

    # Final result
    result = {
        'paths': paths,
        'task_description': task_description,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'zarr_path': zarr_path,
            'n_episodes': n_episodes,
            'n_successful': n_episodes - len(failed_episodes),
            'n_failed': len(failed_episodes),
            'failed_episodes': failed_episodes,
            'server_ip': server_ip,
            'server_port': server_port,
            'model_name': MODEL_NAME,
        }
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate HAMSTER paths for RoboTwin dataset"
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        required=True,
        help="Path to the .zarr dataset directory"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the generated paths (.pkl file)"
    )
    parser.add_argument(
        "--task-description",
        type=str,
        required=True,
        help="Natural language task description for HAMSTER"
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="127.0.0.1",
        help="HAMSTER server IP address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="HAMSTER server port (default: 8000)"
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="head_camera",
        help="Key for camera data in zarr (default: head_camera)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for HAMSTER generation (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N episodes (default: 10)"
    )

    args = parser.parse_args()

    cprint("=" * 60, "cyan")
    cprint("HAMSTER Path Generation Script", "cyan")
    cprint("=" * 60, "cyan")
    cprint(f"Dataset: {args.zarr_path}", "cyan")
    cprint(f"Output: {args.output_path}", "cyan")
    cprint(f"Task: {args.task_description}", "cyan")
    cprint(f"Server: {args.server_ip}:{args.server_port}", "cyan")
    cprint("=" * 60, "cyan")

    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate paths
    resume_from = args.output_path if args.resume else None

    result = generate_all_paths(
        zarr_path=args.zarr_path,
        task_description=args.task_description,
        server_ip=args.server_ip,
        server_port=args.server_port,
        camera_key=args.camera_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        resume_from=resume_from,
        save_interval=args.save_interval
    )

    # Save final result
    with open(args.output_path, 'wb') as f:
        pickle.dump(result, f)

    cprint("=" * 60, "green")
    cprint("Path generation completed!", "green")
    cprint(f"Results saved to: {args.output_path}", "green")
    cprint(f"Total episodes: {result['metadata']['n_episodes']}", "green")
    cprint(f"Successful: {result['metadata']['n_successful']}", "green")
    cprint(f"Failed: {result['metadata']['n_failed']}", "green")

    if result['metadata']['failed_episodes']:
        cprint(f"Failed episodes: {result['metadata']['failed_episodes']}", "yellow")

    cprint("=" * 60, "green")

    # Return error code if any failures
    if result['metadata']['n_failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
