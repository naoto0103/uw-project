#!/usr/bin/env python3
"""
Restore prompts for past experiments
"""
import pickle

# VERSION 1: Initial improved prompt with examples
prompt_v1 = """In the image, execute the task described in <quest>Pick up the apple and put it behind the hammer</quest>.

Generate a sequence of waypoints representing the robot gripper's trajectory to achieve the goal.

Output in JSON format:
[
  {"point_2d": [253, 768], "gripper": "close"},
  {"point_2d": [489, 326], "gripper": "close"},
  {"point_2d": [757, 836], "gripper": "open"},
  ...
]

Format:
- point_2d: [x, y] coordinates in range [0, 1000] representing relative positions in the image
- gripper: "open" or "close" at each waypoint. When the state changes (e.g., "close" → "open"), the gripper actuates at that point. When the state remains the same (e.g., "open" → "open"), the gripper maintains its current state.

Think step by step and output the complete trajectory in JSON format."""

# VERSION 2: Trajectory-focused prompt with end-effector path emphasis
prompt_v2 = """In the image, execute the task described in <quest>Pick up the apple and put it behind the hammer</quest>.

You are controlling a robot gripper's end-effector. Generate a trajectory showing the complete path the gripper should follow to accomplish the task.

The trajectory should include:
1. Starting position (where the gripper begins)
2. Intermediate waypoints (positions along the path to the goal)
3. Goal position (where the task is completed)
4. Gripper actions (open/close) only if the task requires grasping or releasing objects

Output in JSON format:
[
  {"point_2d": [150, 400], "gripper": "open"},
  {"point_2d": [200, 450], "gripper": "open"},
  {"point_2d": [253, 500], "gripper": "open"},
  {"point_2d": [253, 500], "gripper": "close"},
  {"point_2d": [320, 480], "gripper": "close"},
  {"point_2d": [450, 520], "gripper": "close"},
  {"point_2d": [550, 600], "gripper": "close"},
  {"point_2d": [550, 600], "gripper": "open"},
  ...
]

Format:
- point_2d: [x, y] coordinates in range [0, 1000] representing the gripper's position in the image
- gripper: "open" or "close" state at each waypoint
- Generate multiple waypoints along the path, not just start and end positions
- The gripper follows a smooth trajectory through all waypoints in sequence

Think step by step and output the complete trajectory in JSON format."""


def restore_prompts():
    """Restore prompts to existing pickle files"""

    files_to_update = [
        {
            'path': '/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_improved_prompt_path.pkl',
            'prompt': prompt_v1,
            'prompt_type': 'improved_with_examples'
        },
        {
            'path': '/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_trajectory_prompt_path.pkl',
            'prompt': prompt_v2,
            'prompt_type': 'trajectory_focused'
        }
    ]

    for file_info in files_to_update:
        try:
            # Load existing data
            with open(file_info['path'], 'rb') as f:
                data = pickle.load(f)

            # Add prompt if not present
            if 'prompt' not in data:
                data['prompt'] = file_info['prompt']

                # Save updated data
                with open(file_info['path'], 'wb') as f:
                    pickle.dump(data, f)

                print(f"✅ Updated: {file_info['path']}")
                print(f"   Added prompt for: {file_info['prompt_type']}")
            else:
                print(f"⏭️  Skipped: {file_info['path']}")
                print(f"   Prompt already exists")

        except FileNotFoundError:
            print(f"❌ Not found: {file_info['path']}")
        except Exception as e:
            print(f"❌ Error processing {file_info['path']}: {e}")

        print()


if __name__ == "__main__":
    print("="*60)
    print("Restoring Prompts to Past Experiments")
    print("="*60)
    print()

    restore_prompts()

    print("="*60)
    print("Done!")
    print("="*60)
