#!/usr/bin/env python3
"""
Generate VILA (HAMSTER finetuned) paths for all frames in RoboTwin 2.0 episodes (Single-Arm).

This script uses the VILA-1.5-13B model finetuned for HAMSTER to generate 2D paths.
Results are saved separately from Qwen3 results for comparison.

Key differences from Qwen3 version:
- Server port: 8000 (VILA) vs 8001 (Qwen3)
- Model: HAMSTER_dev (VILA finetuned)
- Coordinates: [0, 1] normalized vs [0, 1000]
- Prompt: VILA format with <quest> tags
- Parameters: temperature=0.0, top_p=0.95
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

# VILA server configuration
VILA_SERVER_URL = "http://localhost:8000/v1"
VILA_MODEL = "HAMSTER_dev"

# Single-arm tasks
SINGLE_ARM_TASKS = [
    "beat_block_hammer",
    "click_bell",
    "move_can_pot",
    "place_object_stand",
    "open_microwave",
    "turn_switch",
]

# Task instructions (same as Qwen3 version)
SINGLE_ARM_INSTRUCTIONS = {
    "beat_block_hammer": "Pick up the hammer and beat the block",
    "click_bell": "click the bell's top center on the table",
    "move_can_pot": "pick up the can and move it to beside the pot",
    "place_object_stand": "place the object on the stand",
    "open_microwave": "open the microwave",
    "turn_switch": "click the switch",
}

# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def parse_vila_path(response_text: str) -> Optional[List[Tuple[float, float, int]]]:
    """
    Parse VILA path from response with <ans> tag.

    VILA outputs coordinates in [0, 1] range (already normalized).

    Args:
        response_text: Raw response from VILA

    Returns:
        List of (x, y, gripper_state) tuples in [0, 1] range
        gripper_state: 0=CLOSE, 1=OPEN
        Returns None if parsing fails
    """
    # Extract content from <ans> tags
    ans_match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL | re.IGNORECASE)
    if not ans_match:
        print(f"WARNING: No <ans> tags found in response")
        return None

    ans_content = ans_match.group(1).strip()

    # Replace action tags with special coordinates for parsing
    ans_content = ans_content.replace('<action>Close Gripper</action>', '(1000.0, 1000.0)')
    ans_content = ans_content.replace('<action>Open Gripper</action>', '(1001.0, 1001.0)')

    try:
        # Parse as Python list
        keypoints = eval(ans_content)
    except Exception as e:
        print(f"WARNING: Failed to parse keypoints: {e}")
        return None

    # Process keypoints
    path = []
    current_gripper_state = GRIPPER_OPEN  # Start with OPEN (default for VILA)

    for point in keypoints:
        x, y = point

        # Check for action markers
        if x == y and x == 1000.0:
            current_gripper_state = GRIPPER_CLOSE
            if path:
                path[-1] = (path[-1][0], path[-1][1], current_gripper_state)
            continue
        elif x == y and x == 1001.0:
            current_gripper_state = GRIPPER_OPEN
            if path:
                path[-1] = (path[-1][0], path[-1][1], current_gripper_state)
            continue

        # Regular waypoint - VILA outputs [0, 1] coordinates
        path.append((float(x), float(y), current_gripper_state))

    return path if path else None


def generate_vila_path_for_frame(
    client: OpenAI,
    frame_path: str,
    task_instruction: str,
    verbose: bool = False
) -> Tuple[Optional[List[Tuple[float, float, int]]], str]:
    """
    Generate single-arm path for a frame using VILA with HAMSTER prompt format.

    Args:
        client: OpenAI client configured for VILA server
        frame_path: Path to frame image
        task_instruction: Task description
        verbose: Print detailed info

    Returns:
        Tuple of (path, raw_response_text)
        - path: List of (x, y, gripper_state) tuples, or None if failed
        - raw_response_text: Raw model output for debugging
    """
    # VILA/HAMSTER prompt format - EXACT original format from test_api_client.py
    prompt = (
        f"\nIn the image, please execute the command described in <quest>{task_instruction}</quest>.\n"
        "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
        "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
        "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\n"
        "The tuple denotes point x and y location of the end effector in the image. The action tags indicate gripper actions.\n"
        "Coordinates should be floats between 0 and 1, representing relative positions.\n"
        "Remember to provide points between <ans> and </ans> tags and think step by step."
    )

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
            model=VILA_MODEL,
            messages=messages,
            temperature=0.0,  # VILA default
            top_p=0.95,       # VILA default
            max_tokens=256,   # VILA paths are usually shorter
            extra_body={"num_beams": 1, "use_cache": False}
        )

        response_content = response.choices[0].message.content

        # Handle list format (VILA returns list format)
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

        path = parse_vila_path(response_text)
        return path, response_text

    except Exception as e:
        print(f"ERROR generating path: {e}")
        return None, f"ERROR: {e}"


def check_vila_server(port: int = 8000) -> bool:
    """Check if VILA server is running."""
    try:
        import requests
        # Try /health first, then fall back to /v1/models or just a connection check
        for endpoint in ["/health", "/v1/models", "/"]:
            try:
                response = requests.get(f"http://localhost:{port}{endpoint}", timeout=5)
                # Any response (even 404) means server is running
                print(f"VILA server connected on port {port} (endpoint: {endpoint}, status: {response.status_code})")
                return True
            except:
                continue
    except Exception as e:
        print(f"Cannot connect to VILA server on port {port}: {e}")
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
    Process all frames in an episode for VILA path generation.

    Returns:
        Result dictionary
    """
    # Get task instruction
    task_instruction = SINGLE_ARM_INSTRUCTIONS.get(
        task_name,
        f"Complete the {task_name} task"
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

        # Check if already processed
        if output_file.exists() and raw_output_file.exists():
            success_count += 1
            if verbose and (i + 1) % 20 == 0:
                print(f"      [{i + 1}/{total_frames}] Already exists, skipping...")
            continue

        # Generate path
        frame_start = time.time()
        path, raw_response = generate_vila_path_for_frame(
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
                remaining = (total_frames - i - 1) * (elapsed / (i + 1))
                print(f"      [{frame_num}/{total_frames - 1}] {len(path)} WP, {frame_time:.1f}s/frame, ETA: {remaining/60:.1f}min")
        else:
            fail_count += 1
            print(f"      [{i + 1}/{total_frames}] Failed to generate path (raw: {len(raw_response)} chars)")

    total_time = time.time() - start_time_proc

    return {
        "task": task_name,
        "episode": episode_num,
        "status": "success" if fail_count == 0 else "partial",
        "total_frames": total_frames,
        "success": success_count,
        "failed": fail_count,
        "time_minutes": total_time / 60,
        "instruction": task_instruction,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate VILA paths for RoboTwin 2.0 single-arm tasks"
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=SINGLE_ARM_TASKS,
                        help="Tasks to process")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes per task")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory for input/output")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--port", type=int, default=8000,
                        help="VILA server port (default: 8000)")

    args = parser.parse_args()

    # Set directories - use existing robotwin2_single_6tasks_vila directly
    script_dir = Path(__file__).parent.absolute()
    if args.base_dir is None:
        base_dir = script_dir.parent / "results" / "robotwin2_single_6tasks_vila"
    else:
        base_dir = Path(args.base_dir)

    # Dynamic server URL based on port
    server_url = f"http://localhost:{args.port}/v1"

    print("=" * 80)
    print("RoboTwin 2.0 Single-Arm Path Generation (VILA / HAMSTER finetuned)")
    print("=" * 80)
    print(f"Tasks: {args.tasks}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Base directory: {base_dir}")
    print(f"VILA server: {server_url}")
    print(f"Model: {VILA_MODEL}")
    print(f"Parameters: temp=0.0, top_p=0.95, max_tokens=256")
    print(f"Prompt: VILA/HAMSTER format with <quest> tags")
    print()

    # Check server
    print("Checking VILA server...")
    if not check_vila_server(args.port):
        print("\nERROR: VILA server is not running!")
        print("Please start the server first:")
        print(f"  cd HAMSTER")
        print(f"  ./setup_server.sh")
        print(f"  # Or: python server.py --port {args.port} --model-path Hamster_dev/VILA1.5-13b-... --conv-mode vicuna_v1")
        sys.exit(1)
    print()

    # Initialize client
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

            # Use existing directory structure directly
            output_dir = base_dir / task_name / f"episode_{ep_idx:02d}"
            frames_dir = output_dir / "frames"

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
    print("VILA Path Generation Summary")
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
    summary_file = base_dir / "vila_generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "model": "VILA-1.5-13B (HAMSTER finetuned)",
            "prompt_format": "VILA/HAMSTER with <quest> tags",
            "parameters": {
                "temperature": 0.0,
                "top_p": 0.95,
                "max_tokens": 256
            },
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
