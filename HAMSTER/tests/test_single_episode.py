#!/usr/bin/env python3
"""
Test script to process a single episode.

Test: diverse_bottles_pick task, episode 1
"""

import sys
from pathlib import Path

# Import the batch processing module
sys.path.insert(0, str(Path(__file__).parent))
from batch_process_episodes import process_single_episode, check_qwen3_server

def main():
    # Test configuration
    task_name = "diverse_bottles_pick"
    episode_idx = 1
    task_instruction = "Pick up the bottles from the table"

    print("="*80)
    print("Test: Process Single Episode")
    print("="*80)
    print(f"Task: {task_name}")
    print(f"Episode: {episode_idx}")
    print(f"Instruction: {task_instruction}")
    print()

    # Check server
    print("Checking Qwen3 server...")
    if not check_qwen3_server():
        print("ERROR: Qwen3 server is not running!")
        sys.exit(1)
    print("✓ Server is running")
    print()

    # Process episode
    success, error_msg = process_single_episode(
        task_name=task_name,
        episode_idx=episode_idx,
        task_instruction=task_instruction,
        verbose=True
    )

    print()
    print("="*80)
    if success:
        print("Test PASSED: Episode processed successfully!")
    else:
        print(f"Test FAILED: {error_msg}")
    print("="*80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
