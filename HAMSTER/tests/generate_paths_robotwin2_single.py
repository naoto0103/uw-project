#!/usr/bin/env python3
"""
Generate Qwen3-VL paths for all frames in RoboTwin 2.0 episodes (Single-Arm).

Phase 3.6 Stage 3 - Full frame path generation for single-arm tasks.
Uses VERSION 20 prompt with <ans> tag output format (anti-repetition + edge cases).

Suitable tasks: lift_pot, open_laptop, put_object_cabinet
(NOT for bimanual tasks like pick_dual_bottles - use generate_paths_robotwin2_full.py instead)
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
from typing import List, Tuple, Optional
from openai import OpenAI

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_episode_frames_robotwin2 import (
    TASK_INSTRUCTIONS,
)

# Qwen3 server configuration
QWEN3_SERVER_URL = "http://localhost:8001/v1"
QWEN3_MODEL = "Qwen3-VL-8B-Instruct"

# Single-arm tasks (default)
SINGLE_ARM_TASKS = [
    "lift_pot",
    "open_laptop",
    "put_object_cabinet",
    "stack_blocks_two",
    # Additional single-arm tasks (2024-12-04)
    "beat_block_hammer",
    "click_bell",
    "move_can_pot",
    "place_object_stand",
    "open_microwave",
    "turn_switch",
]

# Task instructions for single-arm tasks
SINGLE_ARM_INSTRUCTIONS = {
    "lift_pot": "Lift the pot from the table",
    "open_laptop": "Open the laptop",
    "put_object_cabinet": "Put the object into the cabinet",
    "stack_blocks_two": "Stack two blocks on top of each other",
    # Additional single-arm tasks (2024-12-04)
    "beat_block_hammer": "Pick up the hammer",
    "click_bell": "click the <bell's top center> on the table",
    "move_can_pot": "there is a can and a pot on the table, use one arm to <pick up the can> and <move it to beside the pot>",
    "place_object_stand": "use appropriate arm to place the object on the stand",
    "open_microwave": "Use one arm to open the microwave.",
    "turn_switch": "use the robotic arm to click the switch",
}


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def parse_single_path(
    response_text: str,
    image_width: int = 1000,
    image_height: int = 1000
) -> Optional[List[Tuple[float, float, int]]]:
    """
    Parse single-arm path from Qwen3 response with <ans> tag.

    Args:
        response_text: Raw response from Qwen3
        image_width: Image width for normalization (default: 1000)
        image_height: Image height for normalization (default: 1000)

    Returns:
        List of (x, y, gripper_state) tuples normalized to [0, 1]
        gripper_state: 0=CLOSE, 1=OPEN
        Returns None if parsing fails
    """
    # Extract content from <ans> tags
    ans_match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL | re.IGNORECASE)
    if not ans_match:
        print(f"WARNING: No <ans> tags found in response")
        return None

    ans_content = ans_match.group(1).strip()

    # Parse waypoints and actions
    path = []
    current_gripper_state = 1  # Start with OPEN (default)

    # Tokenize by commas, handling nested structures
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


def generate_single_path_for_frame(
    client: OpenAI,
    frame_path: str,
    task_instruction: str,
    verbose: bool = False
) -> Tuple[Optional[List[Tuple[float, float, int]]], str]:
    """
    Generate single-arm path for a frame using VERSION 18 prompt.

    Args:
        client: OpenAI client
        frame_path: Path to frame image
        task_instruction: Task description
        verbose: Print detailed info

    Returns:
        Tuple of (path, raw_response_text)
        - path: List of (x, y, gripper_state) tuples, or None if failed
        - raw_response_text: Raw model output for debugging
    """
    # VERSION 18 prompt (single-arm)
    prompt = f"""Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: {task_instruction}

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(534, 439), (675, 306), <action>Close Gripper</action>, (190, 711), <action>Open Gripper</action>]
</ans>"""

    # Encode image
    image_base64 = encode_image(frame_path)

    # Create message
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
        response = client.chat.completions.create(
            model=QWEN3_MODEL,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.1,
            max_tokens=1024
        )

        response_content = response.choices[0].message.content

        # Handle list format (some versions return list)
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
            print(f"Raw response: {response_text[:300]}...")

        path = parse_single_path(response_text)
        return path, response_text

    except Exception as e:
        print(f"ERROR generating path: {e}")
        return None, f"ERROR: {e}"


def check_qwen3_server(port: int = 8001) -> bool:
    """Check if Qwen3 server is running."""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if response.status_code == 200:
            print(f"Qwen3 server connected on port {port} (status: {response.status_code})")
            return True
    except Exception as e:
        print(f"Cannot connect to Qwen3 server on port {port}: {e}")
    return False


def process_episode(
    client: OpenAI,
    task_name: str,
    episode_num: int,
    frames_dir: Path,
    output_dir: Path,
    verbose: bool = False
) -> dict:
    """
    Process all frames in an episode for single-arm path generation.

    Returns:
        Result dictionary
    """
    # Get task instruction
    task_instruction = SINGLE_ARM_INSTRUCTIONS.get(
        task_name,
        TASK_INSTRUCTIONS.get(task_name, f"Complete the {task_name} task")
    )

    # Get all frames
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    total_frames = len(frame_files)

    if total_frames == 0:
        return {
            "task": task_name,
            "episode": episode_num,
            "status": "error",
            "error": "No frames found",
        }

    print(f"    Total frames: {total_frames}")

    num_frames = len(frame_files)
    print(f"    Instruction: {task_instruction}")

    # Create output directories
    paths_dir = output_dir / "paths"
    raw_outputs_dir = output_dir / "raw_outputs"
    os.makedirs(paths_dir, exist_ok=True)
    os.makedirs(raw_outputs_dir, exist_ok=True)

    # Process each frame
    success_count = 0
    fail_count = 0
    start_time_proc = time.time()

    for i, frame_file in enumerate(frame_files):
        frame_num = i
        output_file = paths_dir / f"path_frame_{frame_num:04d}.pkl"
        raw_output_file = raw_outputs_dir / f"raw_frame_{frame_num:04d}.txt"

        # Check if already processed (both path and raw output exist)
        if output_file.exists() and raw_output_file.exists():
            success_count += 1
            if verbose and (i + 1) % 20 == 0:
                print(f"      [{i + 1}/{num_frames}] Already exists, skipping...")
            continue

        # Generate path
        frame_start = time.time()
        path, raw_response = generate_single_path_for_frame(
            client=client,
            frame_path=str(frame_file),
            task_instruction=task_instruction,
            verbose=False
        )
        frame_time = time.time() - frame_start

        # Always save raw response
        with open(raw_output_file, 'w') as f:
            f.write(raw_response)

        if path is not None:
            with open(output_file, 'wb') as f:
                pickle.dump(path, f)
            success_count += 1

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time_proc
                remaining = (num_frames - i - 1) * (elapsed / (i + 1))
                print(f"      [{frame_num}/{num_frames - 1}] {len(path)} WP, {frame_time:.1f}s/frame, ETA: {remaining/60:.1f}min")
        else:
            fail_count += 1
            print(f"      [{i + 1}/{num_frames}] Failed to generate path (raw: {len(raw_response)} chars)")

    total_time = time.time() - start_time_proc

    return {
        "task": task_name,
        "episode": episode_num,
        "status": "success" if fail_count == 0 else "partial",
        "total_frames": num_frames,
        "success": success_count,
        "failed": fail_count,
        "time_minutes": total_time / 60,
        "instruction": task_instruction,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate single-arm paths for RoboTwin 2.0 (VERSION 18)"
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=SINGLE_ARM_TASKS,
                        help="Tasks to process (default: lift_pot, open_laptop, put_object_cabinet)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes per task")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory for input/output")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--port", type=int, default=8001,
                        help="Qwen3 server port (default: 8001)")

    args = parser.parse_args()

    # Set directories
    script_dir = Path(__file__).parent.absolute()
    if args.base_dir is None:
        base_dir = script_dir.parent / "results" / "robotwin2_single"
    else:
        base_dir = Path(args.base_dir)

    # Dynamic server URL based on port
    server_url = f"http://localhost:{args.port}/v1"

    print("=" * 80)
    print("RoboTwin 2.0 Single-Arm Path Generation (VERSION 18)")
    print("=" * 80)
    print(f"Tasks: {args.tasks}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Base directory: {base_dir}")
    print(f"Qwen3 server: {server_url}")
    print(f"Model: {QWEN3_MODEL}")
    print(f"Prompt: VERSION 20 (single-arm, anti-repetition + edge cases)")
    print()

    # Check server
    print("Checking Qwen3 server...")
    if not check_qwen3_server(args.port):
        print("\nERROR: Qwen3 server is not running!")
        print("Please start the server first:")
        print(f"  cd HAMSTER")
        print(f"  CUDA_VISIBLE_DEVICES=0 singularity exec --nv <sif> python server_qwen3.py --port {args.port}")
        sys.exit(1)
    print()

    # Initialize client (default settings, same as RoboTwin 1.0)
    client = OpenAI(
        base_url=server_url,
        api_key="dummy"
    )

    # Process episodes
    results = []
    total_start = time.time()

    for task_idx, task_name in enumerate(args.tasks):
        print("-" * 60)
        print(f"Task [{task_idx + 1}/{len(args.tasks)}]: {task_name}")
        print("-" * 60)

        for ep_idx in range(args.episodes):
            print(f"\n  Episode [{ep_idx + 1}/{args.episodes}]: episode_{ep_idx:02d}")

            frames_dir = base_dir / task_name / f"episode_{ep_idx:02d}" / "frames"
            output_dir = base_dir / task_name / f"episode_{ep_idx:02d}"

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

    total_time = time.time() - total_start

    # Summary
    print()
    print("=" * 80)
    print("Single-Arm Path Generation Summary (VERSION 20)")
    print("=" * 80)

    total_frames = sum(r.get("total_frames", 0) for r in results)
    total_success = sum(r.get("success", 0) for r in results)
    total_failed = sum(r.get("failed", 0) for r in results)

    print(f"Total episodes: {len(results)}")
    print(f"Total frames processed: {total_frames}")
    print(f"Successful paths: {total_success}")
    print(f"Failed paths: {total_failed}")
    print(f"Success rate: {total_success/total_frames*100:.1f}%" if total_frames > 0 else "N/A")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()

    # Print results table
    print(f"{'Task':<25} {'Ep':<4} {'Status':<10} {'Frames':<8} {'Success':<8} {'Time':<8}")
    print("-" * 70)
    for r in results:
        time_str = f"{r.get('time_minutes', 0):.1f}min" if 'time_minutes' in r else "N/A"
        print(f"{r['task']:<25} {r['episode']:<4} {r['status']:<10} {r.get('total_frames', 0):<8} {r.get('success', 0):<8} {time_str:<8}")

    # Save summary
    os.makedirs(base_dir, exist_ok=True)
    summary_file = base_dir / "single_generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "prompt_version": "VERSION 20",
            "arm_type": "single",
            "total_episodes": len(results),
            "total_frames": total_frames,
            "total_success": total_success,
            "total_failed": total_failed,
            "total_time_minutes": total_time / 60,
            "results": results,
        }, f, indent=2)

    print()
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
