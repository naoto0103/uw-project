#!/usr/bin/env python3
"""
Batch process episodes to generate path videos for all RoboTwin 1.0 tasks.

Phase 3.6 Stage 2: Process 12 episodes (6 tasks × 2 episodes)
Uses actual RoboTwin 1.0 task instructions.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# RoboTwin 1.0 task instructions (actual instructions from dataset)
TASK_INSTRUCTIONS = {
    'pick_apple_messy': 'Pick up the apple from the messy table',
    'diverse_bottles_pick': 'Pick up the bottles from the table',
    'dual_bottles_pick_hard': 'Pick up two bottles with both hands',
    'empty_cup_place': 'Place the empty cup on the target location',
    'block_hammer_beat': 'Pick up the hammer and beat the block',
    'shoe_place': 'Place the shoe on the target location',
}

# ManiFlow dataset paths
MANIFLOW_DATA_DIR = Path(__file__).parent.parent.parent / "ManiFlow" / "data"

# Output base directory
OUTPUT_BASE_DIR = Path(__file__).parent.parent / "results" / "video_path_test"

# Progress tracking file
PROGRESS_FILE = OUTPUT_BASE_DIR / "batch_progress.json"

# Number of episodes per task
EPISODES_PER_TASK = 2

# Qwen3 server configuration
QWEN3_SERVER_URL = "http://localhost:8001/v1"


def load_progress() -> Dict:
    """
    Load progress from JSON file.

    Returns:
        Dictionary with progress information
    """
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    else:
        return {
            'completed_episodes': [],
            'failed_episodes': [],
            'last_updated': None
        }


def save_progress(progress: Dict):
    """
    Save progress to JSON file.

    Args:
        progress: Dictionary with progress information
    """
    progress['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def check_qwen3_server() -> bool:
    """
    Check if Qwen3 server is running.

    Returns:
        True if server is running, False otherwise
    """
    try:
        import requests
        response = requests.get(f"{QWEN3_SERVER_URL.replace('/v1', '')}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def process_single_episode(
    task_name: str,
    episode_idx: int,
    task_instruction: str,
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Process a single episode to generate path video.

    Args:
        task_name: Name of the task (e.g., 'pick_apple_messy')
        episode_idx: Episode index (0-9)
        task_instruction: Task instruction text
        verbose: Print detailed information

    Returns:
        Tuple of (success, error_message)
    """
    script_dir = Path(__file__).parent.absolute()

    # Input: ManiFlow dataset
    zarr_file = MANIFLOW_DATA_DIR / f"{task_name}_50.zarr"

    # Output: results directory
    episode_output_dir = OUTPUT_BASE_DIR / task_name / f"episode_{episode_idx:02d}"
    frames_dir = episode_output_dir / "frames"
    paths_dir = episode_output_dir / "paths"
    frames_with_paths_dir = episode_output_dir / "frames_with_paths"
    output_video = episode_output_dir / "qwen3_path_video.mp4"

    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing: {task_name} / episode_{episode_idx:02d}")
        print(f"{'='*80}")
        print(f"Task instruction: {task_instruction}")
        print(f"Input dataset: {zarr_file}")
        print(f"Output directory: {episode_output_dir}")
        print()

    # Check if input exists
    if not zarr_file.exists():
        error_msg = f"Dataset not found: {zarr_file}"
        if verbose:
            print(f"ERROR: {error_msg}")
        return False, error_msg

    # Create output directories
    episode_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Extract frames
        if verbose:
            print(f"[Step 1/4] Extracting frames from episode {episode_idx}...")

        # Create temporary script for frame extraction
        temp_script = episode_output_dir / "temp_extract_frames.py"

        # Read the template script
        template_script = script_dir / "extract_episode_frames.py"
        with open(template_script, 'r') as f:
            script_content = f.read()

        # Modify the script for this episode
        # Note: project_root is defined in main(), so we need to keep that line
        script_content = script_content.replace(
            'ZARR_PATH = project_root / "ManiFlow" / "data" / "pick_apple_messy_50.zarr"',
            f'ZARR_PATH = Path("{str(zarr_file)}")'
        )
        script_content = script_content.replace(
            'EPISODE_IDX = 0',
            f'EPISODE_IDX = {episode_idx}'
        )
        script_content = script_content.replace(
            'OUTPUT_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames"',
            f'OUTPUT_DIR = Path("{str(frames_dir)}")'
        )

        # Write temporary script
        with open(temp_script, 'w') as f:
            f.write(script_content)

        # Run frame extraction
        result = subprocess.run([sys.executable, str(temp_script)], capture_output=True, text=True)

        # Clean up temporary script
        temp_script.unlink()

        if result.returncode != 0:
            error_msg = f"Frame extraction failed: {result.stderr}"
            if verbose:
                print(f"ERROR: {error_msg}")
            return False, error_msg

        if verbose:
            print(f"  ✓ Frames extracted to {frames_dir}")

        # Step 2: Generate paths
        if verbose:
            print(f"[Step 2/4] Generating paths using Qwen3-VL...")

        # Create temporary script for this episode
        temp_script = episode_output_dir / "temp_generate_paths.py"

        # Read the template script
        template_script = script_dir / "generate_paths_for_video.py"
        with open(template_script, 'r') as f:
            script_content = f.read()

        # Modify the script for this episode
        script_content = script_content.replace(
            'TASK_INSTRUCTION = "Pick up the apple and put it behind the hammer"',
            f'TASK_INSTRUCTION = "{task_instruction}"'
        )
        script_content = script_content.replace(
            'FRAMES_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames"',
            f'FRAMES_DIR = Path("{str(frames_dir)}")'
        )
        script_content = script_content.replace(
            'OUTPUT_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "paths"',
            f'OUTPUT_DIR = Path("{str(paths_dir)}")'
        )

        # Write temporary script
        with open(temp_script, 'w') as f:
            f.write(script_content)

        # Run path generation
        result = subprocess.run([sys.executable, str(temp_script)], capture_output=True, text=True)

        if result.returncode != 0:
            error_msg = f"Path generation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            if verbose:
                print(f"ERROR: {error_msg}")
            # Keep temp script for debugging on error
            if verbose:
                print(f"  Temporary script saved for debugging: {temp_script}")
            return False, error_msg

        # Clean up temporary script on success
        temp_script.unlink()

        if verbose:
            print(f"  ✓ Paths generated to {paths_dir}")

        # Step 3: Visualize paths on frames
        if verbose:
            print(f"[Step 3/4] Visualizing paths on frames...")

        # Create temporary script for visualization
        temp_script = episode_output_dir / "temp_visualize_paths.py"

        # Read the template script
        template_script = script_dir / "visualize_paths_on_video.py"
        with open(template_script, 'r') as f:
            script_content = f.read()

        # Modify the script for this episode
        script_content = script_content.replace(
            'FRAMES_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames"',
            f'FRAMES_DIR = Path("{str(frames_dir)}")'
        )
        script_content = script_content.replace(
            'PATHS_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "paths"',
            f'PATHS_DIR = Path("{str(paths_dir)}")'
        )
        script_content = script_content.replace(
            'OUTPUT_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames_with_paths"',
            f'OUTPUT_DIR = Path("{str(frames_with_paths_dir)}")'
        )

        # Write temporary script
        with open(temp_script, 'w') as f:
            f.write(script_content)

        # Run visualization
        result = subprocess.run([sys.executable, str(temp_script)], capture_output=True, text=True)

        # Clean up temporary script
        temp_script.unlink()

        if result.returncode != 0:
            error_msg = f"Path visualization failed: {result.stderr}"
            if verbose:
                print(f"ERROR: {error_msg}")
            return False, error_msg

        if verbose:
            print(f"  ✓ Visualized frames saved to {frames_with_paths_dir}")

        # Step 4: Create video
        if verbose:
            print(f"[Step 4/4] Creating video...")

        # Create temporary script for video creation
        temp_script = episode_output_dir / "temp_create_video.py"

        # Read the template script
        template_script = script_dir / "create_path_video.py"
        with open(template_script, 'r') as f:
            script_content = f.read()

        # Modify the script for this episode
        script_content = script_content.replace(
            'FRAMES_DIR = script_dir.parent / "results" / "video_path_test" / "episode_0" / "frames_with_paths"',
            f'FRAMES_DIR = Path("{str(frames_with_paths_dir)}")'
        )
        script_content = script_content.replace(
            'OUTPUT_VIDEO = script_dir.parent / "results" / "video_path_test" / "episode_0" / "qwen3_path_video.mp4"',
            f'OUTPUT_VIDEO = Path("{str(output_video)}")'
        )

        # Write temporary script
        with open(temp_script, 'w') as f:
            f.write(script_content)

        # Run video creation
        result = subprocess.run([sys.executable, str(temp_script)], capture_output=True, text=True)

        # Clean up temporary script
        temp_script.unlink()

        if result.returncode != 0:
            error_msg = f"Video creation failed: {result.stderr}"
            if verbose:
                print(f"ERROR: {error_msg}")
            return False, error_msg

        if verbose:
            print(f"  ✓ Video created: {output_video}")
            print(f"\n{'='*80}")
            print(f"Episode processing complete!")
            print(f"{'='*80}")

        return True, ""

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if verbose:
            print(f"ERROR: {error_msg}")
        return False, error_msg


def main():
    print("="*80)
    print("Phase 3.6 Stage 2: Batch Process Episodes")
    print("="*80)
    print(f"Tasks: {len(TASK_INSTRUCTIONS)}")
    print(f"Episodes per task: {EPISODES_PER_TASK}")
    print(f"Total episodes: {len(TASK_INSTRUCTIONS) * EPISODES_PER_TASK}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print()

    # Check Qwen3 server
    print("Checking Qwen3 server...")
    if not check_qwen3_server():
        print("ERROR: Qwen3 server is not running!")
        print("\nPlease start the server first:")
        print("  cd HAMSTER")
        print("  conda activate qwen3")
        print("  python server_qwen3.py")
        sys.exit(1)
    print("✓ Qwen3 server is running")
    print()

    # Load progress
    progress = load_progress()
    completed = set(progress['completed_episodes'])
    failed = set(progress['failed_episodes'])

    if completed:
        print(f"Resuming from previous run:")
        print(f"  Completed: {len(completed)} episodes")
        print(f"  Failed: {len(failed)} episodes")
        print()

    # Create list of all episodes to process
    all_episodes = []
    for task_name in sorted(TASK_INSTRUCTIONS.keys()):
        for episode_idx in range(EPISODES_PER_TASK):
            episode_id = f"{task_name}/episode_{episode_idx:02d}"
            all_episodes.append((task_name, episode_idx, episode_id))

    # Filter out already completed episodes
    episodes_to_process = [
        (task_name, episode_idx, episode_id)
        for task_name, episode_idx, episode_id in all_episodes
        if episode_id not in completed
    ]

    total_episodes = len(all_episodes)
    remaining_episodes = len(episodes_to_process)

    print(f"Processing plan:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Already completed: {len(completed)}")
    print(f"  To process: {remaining_episodes}")
    print()

    if remaining_episodes == 0:
        print("All episodes already completed!")
        return

    # Estimate processing time (assuming 2 seconds per frame, ~150 frames per episode)
    estimated_time_per_episode = 150 * 2  # 300 seconds = 5 minutes
    estimated_total_time = remaining_episodes * estimated_time_per_episode

    print(f"Estimated processing time:")
    print(f"  Per episode: ~{estimated_time_per_episode/60:.1f} minutes")
    print(f"  Total: ~{estimated_total_time/3600:.1f} hours")
    print()

    # Process episodes
    print("="*80)
    print("Starting batch processing...")
    print("="*80)

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for i, (task_name, episode_idx, episode_id) in enumerate(episodes_to_process):
        task_instruction = TASK_INSTRUCTIONS[task_name]

        print(f"\n[{i+1}/{remaining_episodes}] Processing {episode_id}...")

        episode_start = time.time()
        success, error_msg = process_single_episode(
            task_name=task_name,
            episode_idx=episode_idx,
            task_instruction=task_instruction,
            verbose=True
        )
        episode_time = time.time() - episode_start

        if success:
            completed.add(episode_id)
            success_count += 1
            print(f"✓ Episode completed in {episode_time/60:.1f} minutes")
        else:
            failed.add(episode_id)
            fail_count += 1
            print(f"✗ Episode failed: {error_msg}")

        # Update progress
        progress['completed_episodes'] = list(completed)
        progress['failed_episodes'] = list(failed)
        save_progress(progress)

        # Print progress summary
        elapsed = time.time() - start_time
        avg_time_per_episode = elapsed / (i + 1)
        remaining = remaining_episodes - (i + 1)
        est_remaining_time = avg_time_per_episode * remaining

        print(f"\nProgress summary:")
        print(f"  Completed: {success_count}/{remaining_episodes}")
        print(f"  Failed: {fail_count}/{remaining_episodes}")
        print(f"  Elapsed time: {elapsed/3600:.1f} hours")
        print(f"  Estimated remaining: {est_remaining_time/3600:.1f} hours")

    # Final summary
    total_time = time.time() - start_time
    print()
    print("="*80)
    print("Batch Processing Complete")
    print("="*80)
    print(f"Total episodes processed: {remaining_episodes}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success rate: {success_count/remaining_episodes*100:.1f}%")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Average time per episode: {total_time/remaining_episodes/60:.1f} minutes")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print("="*80)

    if failed:
        print("\nFailed episodes:")
        for episode_id in sorted(failed):
            print(f"  - {episode_id}")


if __name__ == "__main__":
    main()
