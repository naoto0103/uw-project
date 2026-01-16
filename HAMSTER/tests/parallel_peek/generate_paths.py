#!/usr/bin/env python3
"""
Parallel PEEK (VILA-3B) Path Generation using Dynamic Dispatch across Multiple GPUs.

This script distributes frame processing across multiple PEEK servers (one per GPU)
using a dynamic task queue for optimal load balancing.

Key differences from VILA-13B (HAMSTER):
- Uses PEEK's path_mask prompt format
- Outputs both TRAJECTORY and MASK (no gripper state)
- Saves trajectory.pkl and mask.pkl separately

Usage:
    # First, start the servers
    ./start_servers.sh

    # Then run path generation
    python generate_paths.py --episodes 50

    # Stop servers when done
    ./stop_servers.sh
"""

import os
import sys

# Clear SSL environment variables to avoid certificate errors in Singularity
for key in ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"]:
    if key in os.environ:
        del os.environ[key]

import json
import pickle
import time
import re
import base64
import cv2
import argparse
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from queue import Empty

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NUM_GPUS, BASE_PORT, PEEK_MODEL, RESULTS_DIR,
    SINGLE_ARM_TASKS, SINGLE_ARM_INSTRUCTIONS,
    PEEK_TEMPERATURE, PEEK_TOP_P, PEEK_MAX_TOKENS,
    get_peek_prompt
)


@dataclass
class FrameTask:
    """Represents a single frame to process."""
    task_name: str
    episode_num: int
    frame_num: int
    frame_path: str
    traj_output_path: str
    mask_output_path: str
    raw_output_path: str
    instruction: str


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    task: FrameTask
    success: bool
    trajectory: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    raw_response: str
    processing_time: float
    worker_id: int


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def parse_peek_response(response_text: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse PEEK path_mask response to extract trajectory and mask.

    PEEK output format:
    TRAJECTORY: <ans>[(x, y), ...]</ans> MASK: <ans>[(x, y), ...]</ans>

    Returns:
        Tuple of (trajectory, mask) as numpy arrays, or (None, None) if parsing fails
    """
    def extract_points(text: str) -> Optional[np.ndarray]:
        """Extract points from text containing <ans>...</ans> tags."""
        # Remove <ans> tags
        text = re.sub(r'</?ans>', '', text)
        try:
            # Parse as Python list of tuples
            elements = re.findall(r'\((\d*\.?\d+),\s*(\d*\.?\d+)\)', text)
            if not elements:
                return None
            points = np.array([(float(x), float(y)) for x, y in elements])
            # Clip to [0, 1] range
            points = np.clip(points, 0, 1)
            return points
        except Exception:
            return None

    try:
        # Split by TRAJECTORY: and MASK:
        # Format: "TRAJECTORY: <ans>[...]</ans> MASK: <ans>[...]</ans>"
        response_text = response_text.replace("TRAJECTORY: ", "")
        parts = response_text.split(" MASK:")

        if len(parts) != 2:
            # Try alternative split
            parts = response_text.split("MASK:")

        if len(parts) != 2:
            return None, None

        traj_text = parts[0].strip()
        mask_text = parts[1].strip()

        trajectory = extract_points(traj_text)
        mask = extract_points(mask_text)

        return trajectory, mask

    except Exception as e:
        return None, None


def worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event
):
    """
    Worker process that processes frames using a dedicated PEEK server.
    """
    # Clear SSL environment variables in worker process
    import os
    for key in ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"]:
        if key in os.environ:
            del os.environ[key]

    from openai import OpenAI

    port = BASE_PORT + worker_id
    # PEEK server uses /chat/completions (no /v1 prefix)
    server_url = f"http://localhost:{port}"

    # Initialize client
    client = OpenAI(base_url=server_url, api_key="dummy")

    while not shutdown_event.is_set():
        try:
            # Get task from queue (with timeout to check shutdown)
            try:
                task = task_queue.get(timeout=1.0)
            except Empty:
                continue

            # Check for poison pill
            if task is None:
                break

            start_time = time.time()

            # Check if already processed (skip if raw output exists)
            if os.path.exists(task.raw_output_path):
                # Load existing results if available
                trajectory = None
                mask = None
                if os.path.exists(task.traj_output_path):
                    with open(task.traj_output_path, 'rb') as f:
                        trajectory = pickle.load(f)
                if os.path.exists(task.mask_output_path):
                    with open(task.mask_output_path, 'rb') as f:
                        mask = pickle.load(f)

                result = FrameResult(
                    task=task,
                    success=(trajectory is not None and mask is not None),
                    trajectory=trajectory,
                    mask=mask,
                    raw_response="[CACHED]",
                    processing_time=0.0,
                    worker_id=worker_id
                )
                result_queue.put(result)
                continue

            # Generate path using PEEK prompt
            try:
                image_base64 = encode_image(task.frame_path)
                prompt = get_peek_prompt(task.instruction)
                # Remove <image> tag since we're sending image separately
                prompt = prompt.replace("<image>\n", "")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                response = client.chat.completions.create(
                    model=PEEK_MODEL,
                    messages=messages,
                    temperature=PEEK_TEMPERATURE,
                    top_p=PEEK_TOP_P,
                    max_tokens=PEEK_MAX_TOKENS,
                    extra_body={"num_beams": 1, "use_cache": False, "prompt_type": "path_mask"}
                )

                response_content = response.choices[0].message.content

                # Handle list format
                if isinstance(response_content, list):
                    raw_response = ""
                    for item in response_content:
                        if hasattr(item, 'text'):
                            raw_response += item.text
                        elif isinstance(item, dict) and 'text' in item:
                            raw_response += item['text']
                else:
                    raw_response = response_content

                # Parse trajectory and mask
                trajectory, mask = parse_peek_response(raw_response)

                # Save results
                os.makedirs(os.path.dirname(task.traj_output_path), exist_ok=True)
                os.makedirs(os.path.dirname(task.mask_output_path), exist_ok=True)
                os.makedirs(os.path.dirname(task.raw_output_path), exist_ok=True)

                # Always save raw response
                with open(task.raw_output_path, 'w') as f:
                    f.write(raw_response)

                # Save trajectory and mask if successful
                success = trajectory is not None and mask is not None
                if trajectory is not None:
                    with open(task.traj_output_path, 'wb') as f:
                        pickle.dump(trajectory, f)
                if mask is not None:
                    with open(task.mask_output_path, 'wb') as f:
                        pickle.dump(mask, f)

                processing_time = time.time() - start_time

                result = FrameResult(
                    task=task,
                    success=success,
                    trajectory=trajectory,
                    mask=mask,
                    raw_response=raw_response[:200],
                    processing_time=processing_time,
                    worker_id=worker_id
                )

            except Exception as e:
                processing_time = time.time() - start_time
                result = FrameResult(
                    task=task,
                    success=False,
                    trajectory=None,
                    mask=None,
                    raw_response=f"ERROR: {str(e)}",
                    processing_time=processing_time,
                    worker_id=worker_id
                )

            result_queue.put(result)

        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            continue


def check_servers(num_gpus: int = NUM_GPUS) -> List[bool]:
    """Check which servers are available."""
    import requests

    status = []
    for i in range(num_gpus):
        port = BASE_PORT + i
        try:
            response = requests.get(f"http://localhost:{port}/", timeout=5)
            status.append(True)
        except:
            status.append(False)
    return status


def collect_frames(
    tasks: List[str],
    episodes: int,
    results_dir: Path
) -> List[FrameTask]:
    """Collect all frames to process."""
    frames = []

    for task_name in tasks:
        instruction = SINGLE_ARM_INSTRUCTIONS.get(
            task_name,
            f"Complete the {task_name} task"
        )

        for ep_num in range(episodes):
            episode_dir = results_dir / task_name / f"episode_{ep_num:02d}"
            frames_dir = episode_dir / "frames"
            paths_dir = episode_dir / "paths"
            masks_dir = episode_dir / "masks"
            raw_dir = episode_dir / "raw_outputs"

            if not frames_dir.exists():
                print(f"WARNING: Frames not found for {task_name}/episode_{ep_num:02d}")
                continue

            frame_files = sorted(frames_dir.glob("frame_*.png"))

            for frame_file in frame_files:
                frame_num = int(frame_file.stem.split('_')[1])

                frame_task = FrameTask(
                    task_name=task_name,
                    episode_num=ep_num,
                    frame_num=frame_num,
                    frame_path=str(frame_file),
                    traj_output_path=str(paths_dir / f"trajectory_frame_{frame_num:04d}.pkl"),
                    mask_output_path=str(masks_dir / f"mask_frame_{frame_num:04d}.pkl"),
                    raw_output_path=str(raw_dir / f"raw_frame_{frame_num:04d}.txt"),
                    instruction=instruction
                )
                frames.append(frame_task)

    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Parallel PEEK (VILA-3B) path generation across multiple GPUs"
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=SINGLE_ARM_TASKS,
                        help="Tasks to process")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes per task (default: 50)")
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS,
                        help=f"Number of GPUs/servers to use (default: {NUM_GPUS})")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                        help="Results directory")
    parser.add_argument("--skip-server-check", action="store_true",
                        help="Skip server availability check")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print("=" * 70)
    print("Parallel PEEK (VILA-3B) Path Generation")
    print("=" * 70)
    print(f"Tasks: {args.tasks}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Results directory: {results_dir}")
    print(f"Model: {PEEK_MODEL}")
    print(f"Prompt type: path_mask (TRAJECTORY + MASK)")
    print()

    # Check servers
    if not args.skip_server_check:
        print("Checking PEEK servers...")
        server_status = check_servers(args.num_gpus)
        ready_count = sum(server_status)

        for i, status in enumerate(server_status):
            port = BASE_PORT + i
            status_str = "READY" if status else "NOT READY"
            print(f"  GPU {i} (port {port}): {status_str}")

        if ready_count == 0:
            print("\nERROR: No servers available!")
            print("Please start servers first: ./start_servers.sh")
            sys.exit(1)

        if ready_count < args.num_gpus:
            print(f"\nWARNING: Only {ready_count}/{args.num_gpus} servers ready")
            args.num_gpus = ready_count
        print()

    # Collect frames
    print("Collecting frames...")
    all_frames = collect_frames(args.tasks, args.episodes, results_dir)
    total_frames = len(all_frames)
    print(f"Total frames to process: {total_frames}")

    if total_frames == 0:
        print("No frames found!")
        sys.exit(1)

    # Count already processed
    already_processed = sum(
        1 for f in all_frames
        if (os.path.exists(f.traj_output_path) and
            os.path.exists(f.mask_output_path) and
            os.path.exists(f.raw_output_path))
    )
    print(f"Already processed: {already_processed}")
    print(f"Remaining: {total_frames - already_processed}")
    print()

    # Create queues and events
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    shutdown_event = mp.Event()

    # Start workers
    print(f"Starting {args.num_gpus} worker processes...")
    workers = []
    for i in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(i, task_queue, result_queue, shutdown_event)
        )
        p.start()
        workers.append(p)
        print(f"  Worker {i} started (PID: {p.pid})")
    print()

    # Enqueue all frames
    print("Enqueueing frames...")
    for frame in all_frames:
        task_queue.put(frame)

    # Add poison pills for each worker
    for _ in range(args.num_gpus):
        task_queue.put(None)

    # Process results
    print("Processing...")
    print("-" * 70)

    start_time = time.time()
    processed = 0
    success_count = 0
    fail_count = 0
    cached_count = 0
    worker_stats = {i: {"processed": 0, "time": 0.0} for i in range(args.num_gpus)}

    try:
        while processed < total_frames:
            try:
                result = result_queue.get(timeout=300)  # 5 min timeout
            except Empty:
                print("WARNING: Timeout waiting for results")
                break

            processed += 1

            if result.raw_response == "[CACHED]":
                cached_count += 1
            elif result.success:
                success_count += 1
            else:
                fail_count += 1

            # Update worker stats
            worker_stats[result.worker_id]["processed"] += 1
            worker_stats[result.worker_id]["time"] += result.processing_time

            # Progress display
            if processed % 50 == 0 or processed == total_frames:
                elapsed = time.time() - start_time
                rate = (processed - cached_count) / elapsed if elapsed > 0 and processed > cached_count else 0
                remaining = (total_frames - processed) / rate / 60 if rate > 0 else 0

                print(f"  [{processed:>6}/{total_frames}] "
                      f"Success: {success_count}, Fail: {fail_count}, Cached: {cached_count} | "
                      f"Rate: {rate:.1f} frames/s | ETA: {remaining:.1f} min")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        shutdown_event.set()

    # Wait for workers to finish
    print("\nWaiting for workers to finish...")
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # Summary
    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total frames: {total_frames}")
    print(f"Processed: {processed}")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {fail_count}")
    print(f"  - Cached: {cached_count}")
    print(f"Success rate: {success_count/(success_count+fail_count)*100:.1f}%" if success_count + fail_count > 0 else "N/A")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()

    print("Worker Statistics:")
    for i in range(args.num_gpus):
        stats = worker_stats[i]
        avg_time = stats["time"] / stats["processed"] if stats["processed"] > 0 else 0
        print(f"  Worker {i} (GPU {i}): {stats['processed']} frames, avg {avg_time:.2f}s/frame")

    # Save summary
    summary = {
        "model": "PEEK VILA-3B (path_mask)",
        "num_gpus": args.num_gpus,
        "total_frames": total_frames,
        "processed": processed,
        "success": success_count,
        "failed": fail_count,
        "cached": cached_count,
        "total_time_minutes": total_time / 60,
        "worker_stats": worker_stats,
        "tasks": args.tasks,
        "episodes": args.episodes,
    }

    summary_file = results_dir / "peek_generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    print("=" * 70)


if __name__ == "__main__":
    main()
