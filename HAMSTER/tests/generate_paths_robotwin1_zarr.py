#!/usr/bin/env python3
"""
Generate Qwen3-VL paths for RoboTwin 1.0 pick_apple_messy dataset (Zarr format).

Uses the same parameters as RoboTwin 2.0 path generation:
- VERSION 18 prompt
- temperature=0.7
- presence_penalty=1.1
- top_p=0.8
- max_tokens=1024
"""

import os
import sys
import pickle
import time
import re
import base64
import zarr
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from openai import OpenAI
import json

# Qwen3 server configuration
QWEN3_SERVER_URL = "http://localhost:8001/v1"
QWEN3_MODEL = "Qwen3-VL-8B-Instruct"

# Task instruction for pick_apple_messy
TASK_INSTRUCTION = "Pick up the apple and put it behind the hammer"

def encode_image_array(image_array: np.ndarray) -> str:
    """
    Encode numpy image array to base64 string for Qwen3 server.

    No resizing - use original resolution directly.

    Args:
        image_array: Image as numpy array (H, W, C) in RGB format

    Returns:
        Base64 encoded string
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Encode as JPEG (no resizing - use original resolution)
    _, buffer = cv2.imencode('.jpg', image_bgr)

    # Convert to base64
    return base64.b64encode(buffer).decode('utf-8')


def parse_hamster_path(response_text: str, image_width: int = 1000, image_height: int = 1000) -> Optional[List[Tuple[float, float, int]]]:
    """
    Parse HAMSTER-style path from Qwen3 response.

    Args:
        response_text: Raw response from Qwen3
        image_width: Image width for normalization (default: 1000)
        image_height: Image height for normalization (default: 1000)

    Returns:
        List of (x, y, gripper_state) tuples, normalized to [0, 1]
        gripper_state: 0=CLOSE, 1=OPEN
        Returns None if parsing fails
    """
    # Extract content from <ans> tags
    ans_match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL)
    if not ans_match:
        return None

    ans_content = ans_match.group(1).strip()

    # Parse waypoints and actions
    path = []
    current_gripper_state = 1  # Start with OPEN (default)

    # Split by commas, handling nested structures
    tokens = []
    current_token = ""
    paren_depth = 0

    for char in ans_content:
        if char == '(':
            paren_depth += 1
            current_token += char
        elif char == ')':
            paren_depth -= 1
            current_token += char
        elif char == ',' and paren_depth == 0:
            tokens.append(current_token.strip())
            current_token = ""
        else:
            current_token += char

    if current_token.strip():
        tokens.append(current_token.strip())

    # Process tokens
    for token in tokens:
        # Check if this is an action
        if '<action>' in token.lower():
            action_match = re.search(r'<action>(.*?)</action>', token, re.IGNORECASE)
            if action_match:
                action_text = action_match.group(1).strip().lower()
                if 'close' in action_text:
                    current_gripper_state = 0  # CLOSE
                elif 'open' in action_text:
                    current_gripper_state = 1  # OPEN
        else:
            # Try to parse as coordinate
            coord_match = re.search(r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)', token)
            if coord_match:
                x = float(coord_match.group(1))
                y = float(coord_match.group(2))

                # Normalize to [0, 1]
                x_norm = x / image_width
                y_norm = y / image_height

                path.append((x_norm, y_norm, current_gripper_state))

    if len(path) == 0:
        return None

    return path


def generate_path_for_frame(
    client: OpenAI,
    image_array: np.ndarray,
    task_instruction: str,
    verbose: bool = False
) -> Tuple[Optional[List[Tuple[float, float, int]]], str]:
    """
    Generate path for a single frame using Qwen3-VL with VERSION 18 prompt.

    Args:
        client: OpenAI client configured for Qwen3 server
        image_array: Frame image as numpy array (H, W, C) in RGB format
        task_instruction: Task description
        verbose: Print detailed information

    Returns:
        Tuple of (path, raw_response):
        - path: List of (x, y, gripper_state) tuples, or None if generation fails
        - raw_response: Raw response text from the model
    """
    # VERSION 18 prompt (same as RoboTwin 2.0)
    prompt = f"""Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: {task_instruction}

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(534, 439), (675, 306), <action>Close Gripper</action>, (190, 711), <action>Open Gripper</action>]
</ans>"""

    # Encode image to base64
    image_base64 = encode_image_array(image_array)

    # Create message with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    try:
        # Call Qwen3-VL API with same parameters as RoboTwin 2.0
        response = client.chat.completions.create(
            model=QWEN3_MODEL,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.1,
            max_tokens=1024
        )

        # Extract response text
        response_content = response.choices[0].message.content

        # Handle both string and list formats (Qwen3 returns list format)
        if isinstance(response_content, list):
            response_text = ""
            for item in response_content:
                if hasattr(item, 'text'):
                    response_text += item.text
                elif isinstance(item, dict) and 'text' in item:
                    response_text += item['text']
        else:
            response_text = response_content

        if verbose:
            print(f"Raw response: {response_text[:200]}...")

        # Parse path
        path = parse_hamster_path(response_text)

        return path, response_text

    except Exception as e:
        print(f"ERROR generating path: {e}")
        return None, str(e)


def main():
    # Get script directory
    script_dir = Path(__file__).parent.absolute()

    # Configuration
    ZARR_PATH = "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/data/robotwin_data/pick_apple_messy.zarr"
    OUTPUT_DIR = script_dir.parent / "results" / "robotwin1_pick_apple_messy" / "episode_00"
    EPISODE_IDX = 0  # Process episode 0

    # Create output directories
    paths_dir = OUTPUT_DIR / "paths"
    raw_outputs_dir = OUTPUT_DIR / "raw_outputs"
    os.makedirs(paths_dir, exist_ok=True)
    os.makedirs(raw_outputs_dir, exist_ok=True)

    # Print configuration
    print("=" * 80)
    print("RoboTwin 1.0 Path Generation (pick_apple_messy)")
    print("=" * 80)
    print(f"Task: {TASK_INSTRUCTION}")
    print(f"Zarr path: {ZARR_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Episode: {EPISODE_IDX}")
    print(f"Qwen3 server: {QWEN3_SERVER_URL}")
    print(f"Model: {QWEN3_MODEL}")
    print(f"Parameters: temp=0.7, presence_penalty=1.1, top_p=0.8, max_tokens=1024")
    print(f"Prompt: VERSION 18")
    print()

    # Load Zarr dataset
    print("Loading Zarr dataset...")
    store = zarr.open(ZARR_PATH, mode='r')

    # Get episode boundaries
    episode_ends = store['meta']['episode_ends'][:]
    start_idx = 0 if EPISODE_IDX == 0 else episode_ends[EPISODE_IDX - 1]
    end_idx = episode_ends[EPISODE_IDX]
    num_frames = end_idx - start_idx

    print(f"Episode {EPISODE_IDX}: {num_frames} frames (index {start_idx} to {end_idx - 1})")
    print()

    # Get head camera data
    head_camera = store['data']['head_camera']
    print(f"Head camera shape: {head_camera.shape}")
    print(f"Head camera dtype: {head_camera.dtype}")
    print()

    # Initialize OpenAI client
    client = OpenAI(
        base_url=QWEN3_SERVER_URL,
        api_key="dummy"  # Not needed for local server
    )

    # Test server connection
    print("Testing Qwen3 server connection...")
    try:
        import requests
        health_response = requests.get("http://localhost:8001/health", timeout=5)
        if health_response.status_code == 200:
            print(f"Server connected: {health_response.json()}")
        else:
            raise Exception(f"Health check failed with status {health_response.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Qwen3 server at {QWEN3_SERVER_URL}")
        print(f"Error: {e}")
        print("\nPlease start the Qwen3 server first")
        sys.exit(1)

    print()
    print("=" * 80)
    print(f"Starting path generation for {num_frames} frames...")
    print("=" * 80)

    # Generate paths for each frame
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for frame_idx in range(num_frames):
        global_idx = start_idx + frame_idx

        # Check if path already exists
        output_path_file = paths_dir / f"path_frame_{frame_idx:04d}.pkl"
        if output_path_file.exists():
            print(f"  [{frame_idx + 1}/{num_frames}] Already exists, skipping")
            success_count += 1
            continue

        # Load frame image
        # Zarr stores as (N, C, H, W), so we need to transpose to (H, W, C)
        image_chw = head_camera[global_idx]  # Shape: (3, 240, 320)
        image_hwc = np.transpose(image_chw, (1, 2, 0))  # Shape: (240, 320, 3)

        # Generate path
        frame_start = time.time()
        path, raw_response = generate_path_for_frame(
            client=client,
            image_array=image_hwc,
            task_instruction=TASK_INSTRUCTION,
            verbose=False
        )
        frame_time = time.time() - frame_start

        # Save raw output
        raw_output_file = raw_outputs_dir / f"raw_frame_{frame_idx:04d}.txt"
        with open(raw_output_file, 'w') as f:
            f.write(raw_response)

        if path is not None:
            # Save path
            with open(output_path_file, 'wb') as f:
                pickle.dump(path, f)

            success_count += 1

            # Progress update every 10 frames
            if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (frame_idx + 1)
                remaining = avg_time * (num_frames - frame_idx - 1)
                print(f"  [{frame_idx + 1}/{num_frames}] {len(path)} WP, {frame_time:.1f}s/frame, ETA: {remaining/60:.1f}min")
        else:
            fail_count += 1
            print(f"  [{frame_idx + 1}/{num_frames}] Failed (raw: {len(raw_response)} chars)")

    # Final summary
    total_time = time.time() - start_time
    print()
    print("=" * 80)
    print("Path Generation Complete")
    print("=" * 80)
    print(f"Total frames: {num_frames}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success rate: {success_count/num_frames*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per frame: {total_time/num_frames:.2f}s")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    # Save summary
    summary = {
        "task": "pick_apple_messy",
        "instruction": TASK_INSTRUCTION,
        "episode": EPISODE_IDX,
        "total_frames": num_frames,
        "success_count": success_count,
        "fail_count": fail_count,
        "success_rate": success_count / num_frames,
        "total_time_minutes": total_time / 60,
        "parameters": {
            "temperature": 0.7,
            "presence_penalty": 1.1,
            "top_p": 0.8,
            "max_tokens": 1024,
            "prompt_version": "VERSION 18"
        }
    }

    summary_file = OUTPUT_DIR / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
