#!/usr/bin/env python3
"""
Generate Qwen3-VL path with improved prompt
Task: "Pick up the apple and put it behind the hammer"
"""
import base64
import cv2
import numpy as np
from openai import OpenAI
import json
import pickle
import zarr

# Server configuration
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8001

print(f"Connecting to Qwen3 server: {SERVER_IP}:{SERVER_PORT}")

# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1


def encode_image(image_array):
    """Encode numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def convert_qwen3_to_hamster_format(qwen3_response):
    """
    Convert Qwen3-VL JSON format to HAMSTER format

    Qwen3 format: [{"point_2d": [x, y], "gripper": "open/close"}, ...]
    HAMSTER format: [[x, y, gripper_state], ...] where x,y in [0,1]

    Returns:
        List of [x, y, gripper_state] where gripper_state is 0 (close) or 1 (open)
    """
    try:
        # Parse JSON response
        # Remove markdown fencing if present
        text = qwen3_response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text.strip())

        if not isinstance(data, list):
            print(f"ERROR: Expected list, got {type(data)}")
            return None

        # Convert to HAMSTER format
        hamster_path = []
        for waypoint in data:
            if "point_2d" not in waypoint or "gripper" not in waypoint:
                print(f"WARNING: Skipping malformed waypoint: {waypoint}")
                continue

            # Qwen3-VL uses [0, 1000] range, convert to [0, 1]
            x, y = waypoint["point_2d"]
            x_norm = x / 1000.0
            y_norm = y / 1000.0

            # Convert gripper state
            gripper_str = waypoint["gripper"].lower()
            if "unchanged" in gripper_str:
                # Keep previous gripper state
                if hamster_path:
                    gripper_state = hamster_path[-1][2]
                else:
                    gripper_state = GRIPPER_OPEN  # Default to open if first point
            elif "close" in gripper_str:
                gripper_state = GRIPPER_CLOSE
            elif "open" in gripper_str:
                gripper_state = GRIPPER_OPEN
            else:
                print(f"WARNING: Unknown gripper state '{gripper_str}', defaulting to CLOSE")
                gripper_state = GRIPPER_CLOSE

            hamster_path.append([x_norm, y_norm, gripper_state])

        return hamster_path

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response text: {text[:200]}...")
        return None
    except Exception as e:
        print(f"Conversion error: {e}")
        return None


def test_qwen3_improved_prompt():
    """Test Qwen3-VL with improved prompt"""

    # Load dataset
    zarr_path = "/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/data/pick_apple_messy_50.zarr"
    dataset = zarr.open(zarr_path, mode='r')

    # Get first frame
    frame_chw = dataset['data']['head_camera'][0]
    first_frame = np.transpose(frame_chw, (1, 2, 0))  # (H, W, C)

    print(f"\nLoading first frame from dataset")
    print(f"  Frame shape: {first_frame.shape}")

    # Encode image
    image_base64 = encode_image(first_frame)

    # Task instruction
    task_description = "Pick up the apple and put it behind the hammer"

    # VERSION 1: Initial improved prompt with examples
    # prompt = f"""In the image, execute the task described in <quest>{task_description}</quest>.
    #
    # Generate a sequence of waypoints representing the robot gripper's trajectory to achieve the goal.
    #
    # Output in JSON format:
    # [
    #   {{"point_2d": [253, 768], "gripper": "close"}},
    #   {{"point_2d": [489, 326], "gripper": "close"}},
    #   {{"point_2d": [757, 836], "gripper": "open"}},
    #   ...
    # ]
    #
    # Format:
    # - point_2d: [x, y] coordinates in range [0, 1000] representing relative positions in the image
    # - gripper: "open" or "close" at each waypoint. When the state changes (e.g., "close" → "open"), the gripper actuates at that point. When the state remains the same (e.g., "open" → "open"), the gripper maintains its current state.
    #
    # Think step by step and output the complete trajectory in JSON format."""

    # VERSION 2: Trajectory-focused prompt with end-effector path emphasis
    # prompt = f"""In the image, execute the task described in <quest>{task_description}</quest>.
    #
    # You are controlling a robot gripper's end-effector. Generate a trajectory showing the complete path the gripper should follow to accomplish the task.
    #
    # The trajectory should include:
    # 1. Starting position (where the gripper begins)
    # 2. Intermediate waypoints (positions along the path to the goal)
    # 3. Goal position (where the task is completed)
    # 4. Gripper actions (open/close) only if the task requires grasping or releasing objects
    #
    # Output in JSON format:
    # [
    #   {{"point_2d": [150, 400], "gripper": "open"}},
    #   {{"point_2d": [200, 450], "gripper": "open"}},
    #   {{"point_2d": [253, 500], "gripper": "open"}},
    #   {{"point_2d": [253, 500], "gripper": "close"}},
    #   {{"point_2d": [320, 480], "gripper": "close"}},
    #   {{"point_2d": [450, 520], "gripper": "close"}},
    #   {{"point_2d": [550, 600], "gripper": "close"}},
    #   {{"point_2d": [550, 600], "gripper": "open"}},
    #   ...
    # ]
    #
    # Format:
    # - point_2d: [x, y] coordinates in range [0, 1000] representing the gripper's position in the image
    # - gripper: "open" or "close" state at each waypoint
    # - Generate multiple waypoints along the path, not just start and end positions
    # - The gripper follows a smooth trajectory through all waypoints in sequence
    #
    # Think step by step and output the complete trajectory in JSON format."""

    # VERSION 3: Variable coordinates (x, y) instead of concrete examples
    # prompt = f"""In the image, execute the task described in <quest>{task_description}</quest>.
    #
    # You are controlling a robot gripper's end-effector. Generate a trajectory showing the complete path the gripper should follow to accomplish the task.
    #
    # The trajectory should include:
    # 1. Starting position (where the gripper begins)
    # 2. Intermediate waypoints (positions along the path to the goal)
    # 3. Goal position (where the task is completed)
    # 4. Gripper actions (open/close) only if the task requires grasping or releasing objects
    #
    # Output in JSON format:
    # [
    #   {{"point_2d": [x1, y1], "gripper": "open"}},
    #   {{"point_2d": [x2, y2], "gripper": "open"}},
    #   {{"point_2d": [x3, y3], "gripper": "close"}},
    #   {{"point_2d": [x4, y4], "gripper": "close"}},
    #   {{"point_2d": [x5, y5], "gripper": "close"}},
    #   {{"point_2d": [x6, y6], "gripper": "open"}},
    #   ...
    # ]
    #
    # Format:
    # - point_2d: [x, y] coordinates in range [0, 1000] representing the gripper's position in the image
    # - gripper: "open" or "close" state at each waypoint
    # - Generate multiple waypoints along the path, not just start and end positions
    # - The gripper follows a smooth trajectory through all waypoints in sequence
    #
    # Think step by step and output the complete trajectory in JSON format."""

    # VERSION 4: Updated concrete examples with diverse coordinates
    # prompt = f"""In the image, execute the task described in <quest>{task_description}</quest>.
    #
    # You are controlling a robot gripper's end-effector. Generate a trajectory showing the complete path the gripper should follow to accomplish the task.
    #
    # The trajectory should include:
    # 1. Starting position (where the gripper begins)
    # 2. Intermediate waypoints (positions along the path to the goal)
    # 3. Goal position (where the task is completed)
    # 4. Gripper actions (open/close) only if the task requires grasping or releasing objects
    #
    # Output in JSON format:
    # [
    #   {{"point_2d": [153, 442], "gripper": "open"}},
    #   {{"point_2d": [203, 454], "gripper": "open"}},
    #   {{"point_2d": [253, 569], "gripper": "close"}},
    #   {{"point_2d": [320, 481], "gripper": "close"}},
    #   {{"point_2d": [455, 523], "gripper": "close"}},
    #   {{"point_2d": [554, 692], "gripper": "open"}},
    #   ...
    # ]
    #
    # Format:
    # - point_2d: [x, y] coordinates in range [0, 1000] representing the gripper's position in the image
    # - gripper: "open" or "close" state at each waypoint
    # - Generate multiple waypoints along the path, not just start and end positions
    # - The gripper follows a smooth trajectory through all waypoints in sequence
    #
    # Think step by step and output the complete trajectory in JSON format."""

    # VERSION 5: Removed two redundant format lines + explicit example clarification
    # prompt = f"""In the image, execute the task described in <quest>{task_description}</quest>.
    #
    # You are controlling a robot gripper's end-effector. Generate a trajectory showing the complete path the gripper should follow to accomplish the task.
    #
    # The trajectory should include:
    # 1. Starting position (where the gripper begins)
    # 2. Intermediate waypoints (positions along the path to the goal)
    # 3. Goal position (where the task is completed)
    # 4. Gripper actions (open/close) only if the task requires grasping or releasing objects
    #
    # Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
    # [
    #   {{"point_2d": [153, 442], "gripper": "open"}},
    #   {{"point_2d": [203, 454], "gripper": "open"}},
    #   {{"point_2d": [253, 569], "gripper": "close"}},
    #   {{"point_2d": [320, 481], "gripper": "close"}},
    #   {{"point_2d": [455, 523], "gripper": "close"}},
    #   {{"point_2d": [554, 692], "gripper": "open"}}
    # ]
    #
    # Format:
    # - point_2d: [x, y] coordinates in range [0, 1000] representing the gripper's position in the image
    # - gripper: "open" or "close" state at each waypoint
    #
    # Think step by step and output the complete trajectory in JSON format."""

    # VERSION 6: Added "unchanged" gripper state and initial gripper condition
    # prompt = f"""In the image, execute the task described in <quest>{task_description}</quest>.
    #
    # You are controlling a robot gripper's end-effector. Generate a trajectory showing the complete path the gripper should follow to accomplish the task.
    #
    # Initial condition: The gripper starts in the "open" state.
    #
    # The trajectory should include:
    # 1. Starting position (where the gripper begins)
    # 2. Multiple intermediate waypoints (positions along the path to the goal)
    # 3. Goal position (where the task is completed)
    # 4. Gripper actions (open/close) only if the task requires grasping or releasing objects
    #
    # Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
    # [
    #   {{"point_2d": [153, 442], "gripper": "close"}},
    #   {{"point_2d": [203, 454], "gripper": "unchanged"}},
    #   {{"point_2d": [253, 569], "gripper": "unchanged"}},
    #   {{"point_2d": [554, 692], "gripper": "open"}}
    # ]
    #
    # Format:
    # - point_2d: [x, y] coordinates in range [0, 1000] representing the gripper's position in the image
    # - gripper: "open", "close", or "unchanged" state at each waypoint
    # - Each waypoint should have DIFFERENT coordinates to show the movement path.
    #
    # Think step by step and output the complete trajectory in JSON format."""

    # VERSION 12: Simplified examples (only 2 waypoints in example)
    # prompt = f"""In the image, execute the task described in <quest>{task_description}</quest>.
    #
    # You are controlling a robot gripper's end-effector. Generate a trajectory showing the complete path the gripper should follow to accomplish the task.
    #
    # The trajectory should include:
    # 1. Starting position (where the gripper begins)
    # 2. Intermediate waypoints (positions along the path to the goal)
    # 3. Goal position (where the task is completed)
    # 4. Gripper actions (open/close) only if the task requires grasping or releasing objects
    #
    # Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
    # [
    #   {{"point_2d": [153, 442], "gripper": "close"}},
    #   {{"point_2d": [554, 692], "gripper": "open"}}
    # ]
    #
    # Format:
    # - point_2d: [x, y] coordinates in range [0, 1000] representing the gripper's position in the image
    # - gripper: "open" or "close" state at each waypoint
    #
    # Think step by step and output the complete trajectory in JSON format."""

    # VERSION 13: VILA-style expression (concise, no trajectory requirements list)
    # prompt = f"""In the image, please execute the command described in <quest>{task_description}</quest>.
    #
    # Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.
    #
    # Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
    # [
    #   {{"point_2d": [153, 442], "gripper": "close"}},
    #   {{"point_2d": [554, 692], "gripper": "open"}}
    # ]
    #
    # The point_2d denotes x and y location of the end effector in the image. The gripper field indicates gripper actions.
    #
    # Coordinates should be integers between 0 and 1000, representing relative positions.
    #
    # Remember to output the complete trajectory in JSON format and think step by step."""

    # VERSION 14: VILA-style expression + trajectory requirements list
    # prompt = f"""In the image, please execute the command described in <quest>{task_description}</quest>.
    #
    # Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.
    #
    # The trajectory should include:
    # 1. Starting position (where the gripper begins)
    # 2. Intermediate waypoints (positions along the path to the goal)
    # 3. Goal position (where the task is completed)
    # 4. Gripper actions (open/close) only if the task requires grasping or releasing objects
    #
    # Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
    # [
    #   {{"point_2d": [153, 442], "gripper": "close"}},
    #   {{"point_2d": [554, 692], "gripper": "open"}}
    # ]
    #
    # The point_2d denotes x and y location of the end effector in the image. The gripper field indicates gripper actions.
    #
    # Coordinates should be integers between 0 and 1000, representing relative positions.
    #
    # Remember to output the complete trajectory in JSON format and think step by step."""

    # VERSION 15: VILA-style, no trajectory requirements list, gripper description integrated
    # prompt = f"""In the image, please execute the command described in <quest>{task_description}</quest>.
    #
    # Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.
    #
    # Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
    # [
    #   {{"point_2d": [153, 442], "gripper": "close"}},
    #   {{"point_2d": [554, 692], "gripper": "open"}}
    # ]
    #
    # The point_2d denotes x and y location of the end effector in the image. The gripper field indicates gripper actions (open/close) only if the task requires grasping or releasing objects.
    #
    # Coordinates should be integers between 0 and 1000, representing relative positions.
    #
    # Remember to output the complete trajectory in JSON format and think step by step."""

    # VERSION 16: VERSION 10 prompt (HAMSTER format) without system prompt
    prompt = f"""Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: {task_description}

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(500, 400), (600, 300), <action>Close Gripper</action>, (700, 500), <action>Open Gripper</action>]
</ans>"""

    print(f"\nTask: {task_description}")
    print(f"Prompt length: {len(prompt)} chars")
    print("\n" + "="*60)
    print("IMPROVED PROMPT:")
    print("="*60)
    print(prompt)
    print("="*60)

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{SERVER_IP}:{SERVER_PORT}/v1"
    )

    # Send request
    print("\nSending request to Qwen3-VL server...")

    response = client.chat.completions.create(
        model="Qwen3-VL-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=1024,  # Increased for longer trajectories
        temperature=0.0
    )

    # Extract response
    response_content = response.choices[0].message.content

    # Handle both string and list formats
    if isinstance(response_content, list):
        response_text = ""
        for item in response_content:
            if hasattr(item, 'text'):
                response_text += item.text
            elif isinstance(item, dict) and 'text' in item:
                response_text += item['text']
    else:
        response_text = response_content

    print("\n" + "="*60)
    print("QWEN3-VL RESPONSE (Improved Prompt):")
    print("="*60)
    print(response_text)
    print("="*60)

    # Convert to HAMSTER format
    print("\nConverting to HAMSTER format...")
    hamster_path = convert_qwen3_to_hamster_format(response_text)

    if hamster_path:
        print(f"\nProcessed path: {len(hamster_path)} waypoints")
        for i, wp in enumerate(hamster_path):
            gripper_str = "CLOSE" if wp[2] == GRIPPER_CLOSE else "OPEN"
            print(f"  {i}: ({wp[0]:.3f}, {wp[1]:.3f}) - {gripper_str}")

        # Count unique positions
        unique_positions = len(set((wp[0], wp[1]) for wp in hamster_path))
        print(f"\n  Unique positions: {unique_positions}")

        # Count gripper actions
        close_count = sum(1 for p in hamster_path if p[2] == GRIPPER_CLOSE)
        open_count = sum(1 for p in hamster_path if p[2] == GRIPPER_OPEN)
        print(f"  Gripper states: {close_count} CLOSE, {open_count} OPEN")

        # Count gripper state transitions
        transitions = 0
        for i in range(len(hamster_path) - 1):
            if hamster_path[i][2] != hamster_path[i+1][2]:
                transitions += 1
        print(f"  Gripper transitions: {transitions}")

        # Calculate path length
        path_length = 0.0
        for i in range(len(hamster_path) - 1):
            dx = hamster_path[i+1][0] - hamster_path[i][0]
            dy = hamster_path[i+1][1] - hamster_path[i][1]
            path_length += (dx**2 + dy**2)**0.5
        print(f"  Total path length: {path_length:.4f}")

        # Save path
        # VERSION 1: output_path = ".../qwen3_improved_prompt_path.pkl"  # prompt_type: 'improved_with_examples'
        # VERSION 2: output_path = ".../qwen3_trajectory_prompt_path.pkl"  # prompt_type: 'trajectory_focused'
        # VERSION 3: output_path = ".../qwen3_variable_coords_path.pkl"  # prompt_type: 'variable_coordinates'
        # VERSION 4: output_path = ".../qwen3_diverse_coords_path.pkl"  # prompt_type: 'diverse_coordinates'
        # VERSION 5: output_path = ".../qwen3_simplified_format_path.pkl"  # prompt_type: 'simplified_format'
        # VERSION 6: output_path = ".../qwen3_unchanged_state_path.pkl"  # prompt_type: 'unchanged_state'
        # VERSION 12: output_path = ".../qwen3_v12_simplified_examples_path.pkl"  # prompt_type: 'simplified_examples'
        # VERSION 13: output_path = ".../qwen3_v13_vila_style_concise_path.pkl"  # prompt_type: 'vila_style_concise'
        # VERSION 14: output_path = ".../qwen3_v14_vila_style_with_requirements_path.pkl"  # prompt_type: 'vila_style_with_requirements'
        # VERSION 15: output_path = ".../qwen3_v15_vila_style_gripper_only_path.pkl"  # prompt_type: 'vila_style_gripper_only'
        output_path = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_v15_vila_style_gripper_only_path.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'episode_0': hamster_path,
                'instruction': task_description,
                'model': 'Qwen3-VL-8B-Instruct',
                'prompt_type': 'vila_style_gripper_only',
                'prompt': prompt,  # Full prompt text
                'raw_response': response_text
            }, f)
        print(f"\nSaved path to: {output_path}")

        return hamster_path
    else:
        print("\nERROR: Failed to convert response to HAMSTER format")
        return None


if __name__ == "__main__":
    test_qwen3_improved_prompt()
