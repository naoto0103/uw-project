#!/usr/bin/env python3
"""
Generate Qwen3-VL paths for each frame in a video sequence.

Phase 3.6 Stage 1: Generate paths for pick_apple_messy episode 0 (159 frames)
Uses VERSION 18 prompt (production version)
"""

import os
import sys
import pickle
import time
import re
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from openai import OpenAI

# Qwen3 server configuration
QWEN3_SERVER_URL = "http://localhost:8001/v1"
QWEN3_MODEL = "Qwen3-VL-8B-Instruct"

# Path generation task
TASK_INSTRUCTION = "Pick up the apple and put it behind the hammer"


def encode_image(image_path: str) -> str:
    """
    Encode image file to base64 string for Qwen3 server.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string
    """
    # Load image
    image = cv2.imread(image_path)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', image)

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
        print(f"WARNING: No <ans> tags found in response")
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
        print(f"WARNING: No valid coordinates parsed from response")
        return None

    return path


def generate_path_for_frame(
    client: OpenAI,
    frame_path: str,
    task_instruction: str,
    verbose: bool = False
) -> Optional[List[Tuple[float, float, int]]]:
    """
    Generate path for a single frame using Qwen3-VL with VERSION 18 prompt.

    Args:
        client: OpenAI client configured for Qwen3 server
        frame_path: Path to frame image
        task_instruction: Task description
        verbose: Print detailed information

    Returns:
        List of (x, y, gripper_state) tuples, or None if generation fails
    """
    # VERSION 18 prompt (production version)
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
    image_base64 = encode_image(frame_path)

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
        # Call Qwen3-VL API
        response = client.chat.completions.create(
            model=QWEN3_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=512
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

        return path

    except Exception as e:
        print(f"ERROR generating path: {e}")
        return None


def main():
    # Get script directory
    script_dir = Path(__file__).parent.absolute()

    # Configuration
    FRAMES_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames"
    OUTPUT_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "paths"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Print configuration
    print("=" * 80)
    print("Phase 3.6 Stage 1: Generate Paths for Video Frames")
    print("=" * 80)
    print(f"Task: {TASK_INSTRUCTION}")
    print(f"Frames directory: {FRAMES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Qwen3 server: {QWEN3_SERVER_URL}")
    print(f"Model: {QWEN3_MODEL}")
    print(f"Prompt version: VERSION 18 (production)")
    print()

    # Check if frames directory exists
    if not FRAMES_DIR.exists():
        print(f"ERROR: Frames directory not found: {FRAMES_DIR}")
        sys.exit(1)

    # Get list of frames
    frame_files = sorted(FRAMES_DIR.glob("frame_*.png"))
    num_frames = len(frame_files)

    if num_frames == 0:
        print(f"ERROR: No frames found in {FRAMES_DIR}")
        sys.exit(1)

    print(f"Found {num_frames} frames")
    print()

    # Initialize OpenAI client
    client = OpenAI(
        base_url=QWEN3_SERVER_URL,
        api_key="dummy"  # Not needed for local server
    )

    # Test server connection using health endpoint
    print("Testing Qwen3 server connection...")
    try:
        import requests
        health_response = requests.get("http://localhost:8001/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✓ Server connected: {health_data}")
        else:
            raise Exception(f"Health check failed with status {health_response.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Qwen3 server at {QWEN3_SERVER_URL}")
        print(f"Error: {e}")
        print("\nPlease start the Qwen3 server first:")
        print("  cd HAMSTER")
        print("  conda activate qwen3")
        print("  python server_qwen3.py")
        sys.exit(1)

    print()
    print("=" * 80)
    print(f"Starting path generation for {num_frames} frames...")
    print("=" * 80)

    # Generate paths for each frame
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for i, frame_file in enumerate(frame_files):
        frame_num = i
        frame_name = frame_file.name

        print(f"\n[{i+1}/{num_frames}] Processing {frame_name}...")

        # Check if path already exists
        output_file = OUTPUT_DIR / f"path_frame_{frame_num:04d}.pkl"
        if output_file.exists():
            print(f"  ✓ Path already exists, skipping")
            success_count += 1
            continue

        # Generate path
        frame_start = time.time()
        path = generate_path_for_frame(
            client=client,
            frame_path=str(frame_file),
            task_instruction=TASK_INSTRUCTION,
            verbose=False
        )
        frame_time = time.time() - frame_start

        if path is not None:
            # Save path
            with open(output_file, 'wb') as f:
                pickle.dump(path, f)

            print(f"  ✓ Generated path with {len(path)} waypoints in {frame_time:.2f}s")
            print(f"    Path: {path}")
            success_count += 1
        else:
            print(f"  ✗ Failed to generate path")
            fail_count += 1

        # Progress estimate
        elapsed = time.time() - start_time
        avg_time_per_frame = elapsed / (i + 1)
        remaining_frames = num_frames - (i + 1)
        est_remaining_time = avg_time_per_frame * remaining_frames

        print(f"  Progress: {success_count} success, {fail_count} failed")
        print(f"  Time: {frame_time:.2f}s/frame, Est. remaining: {est_remaining_time/60:.1f} min")

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


if __name__ == "__main__":
    main()
