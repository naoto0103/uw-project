#!/usr/bin/env python3
"""
Generate Qwen3-VL path for new task: "Pick up the apple and put it behind the hammer"
Using JSON optimized prompt format
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
            if "close" in gripper_str:
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


def test_qwen3_new_task():
    """Test Qwen3-VL with new task instruction"""

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

    # NEW TASK INSTRUCTION
    task_description = "Pick up the apple and put it behind the hammer"

    # Optimized prompt based on Qwen3-VL cookbook
    prompt = f"""In the image, execute the task described in <quest>{task_description}</quest>.

Generate a sequence of waypoints representing the robot gripper's trajectory to achieve the goal.

For each waypoint, report the point coordinates and gripper action in JSON format like this:
[
  {{"point_2d": [x, y], "gripper": "open"}},
  {{"point_2d": [x, y], "gripper": "open"}},
  {{"point_2d": [x, y], "gripper": "close"}},
  ...
]

Where:
- point_2d: [x, y] coordinates in range [0, 1000] representing relative positions in the image
- gripper: "open" or "close"

Think step by step and output the complete trajectory in JSON format."""

    print(f"\nTask: {task_description}")
    print(f"Prompt length: {len(prompt)} chars")

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
        max_tokens=512,
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
    print("QWEN3-VL RESPONSE (New Task):")
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

        # Count gripper actions
        close_count = sum(1 for p in hamster_path if p[2] == GRIPPER_CLOSE)
        open_count = sum(1 for p in hamster_path if p[2] == GRIPPER_OPEN)
        print(f"\n  Gripper states: {close_count} CLOSE, {open_count} OPEN")

        # Count gripper state transitions
        transitions = 0
        for i in range(len(hamster_path) - 1):
            if hamster_path[i][2] != hamster_path[i+1][2]:
                transitions += 1
        print(f"  Gripper transitions: {transitions}")

        # Save path
        output_path = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_new_task_path.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'episode_0': hamster_path,
                'instruction': task_description,
                'model': 'Qwen3-VL-8B-Instruct',
                'prompt_type': 'optimized_json',
                'raw_response': response_text
            }, f)
        print(f"\nSaved path to: {output_path}")

        return hamster_path
    else:
        print("\nERROR: Failed to convert response to HAMSTER format")
        return None


if __name__ == "__main__":
    test_qwen3_new_task()
