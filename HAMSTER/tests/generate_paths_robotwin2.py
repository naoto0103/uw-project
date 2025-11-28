#!/usr/bin/env python3
"""
Generate Qwen3-VL paths for RoboTwin 2.0 episodes.

Phase 3.6 Stage 3 - Step 5: Generate bimanual paths using VERSION 19 prompt
for 6 tasks x 2 episodes = 12 episodes from RoboTwin 2.0 dataset.
"""

import os
import sys
import json
import pickle
import time
import re
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from openai import OpenAI

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_episode_frames_robotwin2 import (
    SELECTED_TASKS,
    TASK_INSTRUCTIONS,
)

# Qwen3 server configuration
QWEN3_SERVER_URL = "http://localhost:8001/v1"
QWEN3_MODEL = "Qwen3-VL-8B-Instruct"


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def parse_arm_path(
    content: str,
    image_width: int = 1000,
    image_height: int = 1000
) -> Optional[List[Tuple[float, float, int]]]:
    """
    Parse path for a single arm from content string.

    Returns:
        List of (x, y, gripper_state) tuples normalized to [0, 1]
        gripper_state: 0=CLOSE, 1=OPEN
    """
    path = []
    current_gripper_state = 1  # Start with OPEN

    # Tokenize by commas
    tokens = []
    current_token = ""
    paren_depth = 0

    for char in content:
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
        if '<action>' in token.lower():
            action_match = re.search(r'<action>(.*?)</action>', token, re.IGNORECASE)
            if action_match:
                action_text = action_match.group(1).strip().lower()
                if 'close' in action_text:
                    current_gripper_state = 0
                elif 'open' in action_text:
                    current_gripper_state = 1
        else:
            coord_match = re.search(r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)', token)
            if coord_match:
                x = float(coord_match.group(1))
                y = float(coord_match.group(2))
                x_norm = x / image_width
                y_norm = y / image_height
                path.append((x_norm, y_norm, current_gripper_state))

    return path if path else None


def parse_bimanual_path(
    response_text: str,
    image_width: int = 1000,
    image_height: int = 1000
) -> Optional[Dict[str, List[Tuple[float, float, int]]]]:
    """Parse bimanual path from Qwen3 response."""
    result = {}

    left_match = re.search(r'<left_arm>(.*?)</left_arm>', response_text, re.DOTALL | re.IGNORECASE)
    if left_match:
        left_path = parse_arm_path(left_match.group(1), image_width, image_height)
        if left_path:
            result['left_arm'] = left_path

    right_match = re.search(r'<right_arm>(.*?)</right_arm>', response_text, re.DOTALL | re.IGNORECASE)
    if right_match:
        right_path = parse_arm_path(right_match.group(1), image_width, image_height)
        if right_path:
            result['right_arm'] = right_path

    return result if result else None


def generate_bimanual_path(
    client: OpenAI,
    initial_frame_path: str,
    final_frame_path: str,
    task_instruction: str,
    verbose: bool = False
) -> Tuple[Optional[Dict], str]:
    """
    Generate bimanual path using VERSION 19 prompt with initial and final frames.

    Args:
        client: OpenAI client
        initial_frame_path: Path to initial frame
        final_frame_path: Path to final frame
        task_instruction: Task description
        verbose: Print detailed info

    Returns:
        Tuple of (path_dict, raw_response)
    """
    # VERSION 19 prompt (bimanual version)
    prompt = f"""Generate the spatial trajectories in 2D images as [(x, y), ...] for the following task: {task_instruction}

This task requires two robot arms (LEFT and RIGHT). Generate separate trajectories for each arm.

The first image shows the initial state and the second image shows the goal state.
Plan the trajectories to move objects from the initial state to achieve the goal state.

Generate an output in <left_arm>X</left_arm> and <right_arm>X</right_arm> blocks, where X is the trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<left_arm>
[(252, 422), <action>Close Gripper</action>, (307, 353), (424, 557), <action>Open Gripper</action>]
</left_arm>

<right_arm>
[(754, 429), <action>Close Gripper</action>, (719, 356), (633, 593), <action>Open Gripper</action>]
</right_arm>"""

    # Encode images
    initial_base64 = encode_image(initial_frame_path)
    final_base64 = encode_image(final_frame_path)

    # Create message with both images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{initial_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{final_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=QWEN3_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1024
        )

        response_content = response.choices[0].message.content

        # Handle list format
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
            print(f"Raw response: {response_text[:500]}...")

        path = parse_bimanual_path(response_text)
        return path, response_text

    except Exception as e:
        print(f"ERROR generating path: {e}")
        return None, str(e)


def process_episode(
    client: OpenAI,
    task_name: str,
    episode_num: int,
    frames_dir: Path,
    output_dir: Path,
    verbose: bool = False
) -> dict:
    """
    Process a single episode: generate path from initial to final frame.

    Returns:
        Result dictionary
    """
    task_instruction = TASK_INSTRUCTIONS.get(task_name, f"Complete the {task_name} task")

    # Find initial and final frames
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if len(frame_files) < 2:
        return {
            "task": task_name,
            "episode": episode_num,
            "status": "error",
            "error": f"Not enough frames ({len(frame_files)})",
        }

    initial_frame = frame_files[0]
    final_frame = frame_files[-1]

    print(f"    Initial frame: {initial_frame.name}")
    print(f"    Final frame: {final_frame.name}")
    print(f"    Total frames: {len(frame_files)}")

    # Check if already processed
    output_file = output_dir / "path.pkl"
    if output_file.exists():
        print(f"    Already processed, loading existing...")
        with open(output_file, 'rb') as f:
            existing_path = pickle.load(f)
        return {
            "task": task_name,
            "episode": episode_num,
            "status": "skipped",
            "path": existing_path,
            "instruction": task_instruction,
        }

    # Generate path
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    path, raw_response = generate_bimanual_path(
        client=client,
        initial_frame_path=str(initial_frame),
        final_frame_path=str(final_frame),
        task_instruction=task_instruction,
        verbose=verbose
    )
    gen_time = time.time() - start_time

    # Save results
    if path:
        with open(output_file, 'wb') as f:
            pickle.dump(path, f)

        # Also save JSON for readability
        json_file = output_dir / "path.json"
        with open(json_file, 'w') as f:
            # Convert tuples to lists for JSON
            json_path = {
                arm: [(float(x), float(y), int(g)) for x, y, g in waypoints]
                for arm, waypoints in path.items()
            }
            json.dump({
                "task": task_name,
                "episode": episode_num,
                "instruction": task_instruction,
                "path": json_path,
            }, f, indent=2)

        # Save raw response
        response_file = output_dir / "raw_response.txt"
        with open(response_file, 'w') as f:
            f.write(raw_response)

        left_wp = len(path.get('left_arm', []))
        right_wp = len(path.get('right_arm', []))
        print(f"    Generated path (L:{left_wp}, R:{right_wp} waypoints) in {gen_time:.2f}s")

        return {
            "task": task_name,
            "episode": episode_num,
            "status": "success",
            "path": path,
            "instruction": task_instruction,
            "generation_time": gen_time,
        }
    else:
        # Save error response
        error_file = output_dir / "error_response.txt"
        with open(error_file, 'w') as f:
            f.write(raw_response)

        print(f"    Failed to generate path")
        return {
            "task": task_name,
            "episode": episode_num,
            "status": "failed",
            "error": "Failed to parse path",
            "instruction": task_instruction,
        }


def check_qwen3_server() -> bool:
    """Check if Qwen3 server is running."""
    try:
        import requests
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print(f"Qwen3 server connected: {response.json()}")
            return True
    except Exception as e:
        print(f"Cannot connect to Qwen3 server: {e}")
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate paths for RoboTwin 2.0 episodes")
    parser.add_argument("--tasks", type=str, nargs="+", default=SELECTED_TASKS,
                        help="Tasks to process")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of episodes per task")
    parser.add_argument("--input-base", type=str, default=None,
                        help="Input base directory (frames)")
    parser.add_argument("--output-base", type=str, default=None,
                        help="Output base directory (paths)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")

    args = parser.parse_args()

    # Set directories
    script_dir = Path(__file__).parent.absolute()
    if args.input_base is None:
        input_base_dir = script_dir.parent / "results" / "robotwin2_stage3"
    else:
        input_base_dir = Path(args.input_base)

    if args.output_base is None:
        output_base_dir = script_dir.parent / "results" / "robotwin2_stage3"
    else:
        output_base_dir = Path(args.output_base)

    print("=" * 80)
    print("RoboTwin 2.0 Path Generation (VERSION 19 Bimanual)")
    print("=" * 80)
    print(f"Tasks: {len(args.tasks)}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Total episodes: {len(args.tasks) * args.episodes}")
    print(f"Input base: {input_base_dir}")
    print(f"Output base: {output_base_dir}")
    print(f"Qwen3 server: {QWEN3_SERVER_URL}")
    print(f"Model: {QWEN3_MODEL}")
    print()

    # Check server
    print("Checking Qwen3 server...")
    if not check_qwen3_server():
        print("\nERROR: Qwen3 server is not running!")
        print("Please start the server first:")
        print("  cd HAMSTER")
        print("  conda activate qwen3")
        print("  python server_qwen3.py --port 8001")
        sys.exit(1)
    print()

    # Initialize client
    client = OpenAI(
        base_url=QWEN3_SERVER_URL,
        api_key="dummy"
    )

    # Process episodes
    results = []
    start_time = time.time()

    for task_idx, task_name in enumerate(args.tasks):
        print("-" * 60)
        print(f"Task [{task_idx + 1}/{len(args.tasks)}]: {task_name}")
        print(f"Instruction: {TASK_INSTRUCTIONS.get(task_name, 'N/A')}")
        print("-" * 60)

        for ep_idx in range(args.episodes):
            print(f"\n  Episode [{ep_idx + 1}/{args.episodes}]: episode_{ep_idx:02d}")

            frames_dir = input_base_dir / task_name / f"episode_{ep_idx:02d}" / "frames"
            output_dir = output_base_dir / task_name / f"episode_{ep_idx:02d}" / "paths"

            if not frames_dir.exists():
                print(f"    ERROR: Frames directory not found: {frames_dir}")
                results.append({
                    "task": task_name,
                    "episode": ep_idx,
                    "status": "error",
                    "error": "Frames not found",
                })
                continue

            result = process_episode(
                client=client,
                task_name=task_name,
                episode_num=ep_idx,
                frames_dir=frames_dir,
                output_dir=output_dir,
                verbose=args.verbose
            )
            results.append(result)

    total_time = time.time() - start_time

    # Summary
    print()
    print("=" * 80)
    print("Path Generation Summary")
    print("=" * 80)

    success_count = sum(1 for r in results if r["status"] in ["success", "skipped"])
    failed_count = sum(1 for r in results if r["status"] == "failed")
    error_count = sum(1 for r in results if r["status"] == "error")

    print(f"Total episodes: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Errors: {error_count}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()

    # Print results table
    print(f"{'Task':<25} {'Episode':<10} {'Status':<10} {'L-WP':<6} {'R-WP':<6}")
    print("-" * 65)
    for r in results:
        path = r.get("path", {})
        left_wp = len(path.get('left_arm', [])) if path else 0
        right_wp = len(path.get('right_arm', [])) if path else 0
        print(f"{r['task']:<25} {r['episode']:<10} {r['status']:<10} {left_wp:<6} {right_wp:<6}")

    # Save summary
    summary_file = output_base_dir / "generation_summary.json"
    os.makedirs(output_base_dir, exist_ok=True)

    # Convert results for JSON (remove non-serializable items)
    json_results = []
    for r in results:
        json_r = {k: v for k, v in r.items() if k != 'path'}
        if 'path' in r and r['path']:
            json_r['path'] = {
                arm: [(float(x), float(y), int(g)) for x, y, g in waypoints]
                for arm, waypoints in r['path'].items()
            }
        json_results.append(json_r)

    with open(summary_file, 'w') as f:
        json.dump({
            "total_episodes": len(results),
            "success": success_count,
            "failed": failed_count,
            "errors": error_count,
            "total_time_minutes": total_time / 60,
            "results": json_results,
        }, f, indent=2)

    print()
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
