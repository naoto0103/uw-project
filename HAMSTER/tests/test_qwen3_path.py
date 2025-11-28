#!/usr/bin/env python3
"""
Qwen3-VL server test: Generate 2D path for pick_apple_messy task
"""
import base64
import cv2
import numpy as np
from openai import OpenAI
import re
import os
import pickle
import zarr

# Server configuration
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8001

print(f"Connecting to Qwen3 server: {SERVER_IP}:{SERVER_PORT}")

# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1

def process_answer(input_str):
    """
    Extract waypoints from model output
    Example: [(150, 350), (450, 350), <action>Open Gripper</action>, ...]
    Returns: [[x, y, gripper_state], ...] with normalized coordinates [0, 1]
    """
    pattern = r'\(([0-9.]+),\s*([0-9.]+)\)|<action>(.*?)</action>'
    matches = re.findall(pattern, input_str)

    processed_points = []
    action_flag = GRIPPER_CLOSE  # Default: closed

    for match in matches:
        x, y, action = match
        if action:  # Action instruction
            action_lower = action.lower()
            if 'close' in action_lower:
                action_flag = GRIPPER_CLOSE
                if processed_points:
                    processed_points[-1][-1] = action_flag
            elif 'open' in action_lower:
                action_flag = GRIPPER_OPEN
                if processed_points:
                    processed_points[-1][-1] = action_flag
        else:  # Coordinate
            # Convert from [0, 1000] to [0, 1] range
            x_val = float(x) / 1000.0
            y_val = float(y) / 1000.0
            processed_points.append([x_val, y_val, action_flag])

    return processed_points


def encode_image(image_array):
    """Encode numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def test_qwen3_path_generation():
    """Test Qwen3-VL path generation on pick_apple_messy task"""

    # Load dataset
    zarr_path = "/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/data/pick_apple_messy_50.zarr"
    dataset = zarr.open(zarr_path, mode='r')

    # Use first frame from head_camera
    # Shape: (N, C, H, W) -> need to transpose to (H, W, C)
    frame_chw = dataset['data']['head_camera'][0]  # (C, H, W)
    first_frame = np.transpose(frame_chw, (1, 2, 0))  # (H, W, C)

    print(f"\nLoading first frame from dataset")
    print(f"  Frame shape: {first_frame.shape}")
    print(f"  Total frames in dataset: {dataset['data']['head_camera'].shape[0]}")

    # Encode image
    image_base64 = encode_image(first_frame)

    # Task instruction (same as used in VILA generation)
    instruction = "Pick up the apple and put it behind the hammer"

    # Prepare prompt
    prompt = f"""Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: {instruction}

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

The generated path should include not only the start and end points, but also intermediate waypoints as needed to represent actions such as lifting objects.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(500, 400), (600, 300), <action>Close Gripper</action>, (700, 500), <action>Open Gripper</action>]
</ans>
"""

    print(f"\nTask: {instruction}")
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
                "role": "system",
                "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant is specialized in generating precise robot gripper trajectories for manipulation tasks, providing clear waypoint sequences with appropriate gripper states. The trajectories represent the path of the gripper tip."
            },
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
    print("QWEN3-VL RESPONSE:")
    print("="*60)
    print(response_text)
    print("="*60)

    # Extract path from <ans> tags
    ans_match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL)

    if ans_match:
        ans_content = ans_match.group(1).strip()
        print(f"\nExtracted answer: {ans_content[:200]}...")

        # Process answer
        path = process_answer(ans_content)

        print(f"\nProcessed path: {len(path)} waypoints")
        if path:
            print(f"  First point: {path[0]}")
            print(f"  Last point: {path[-1]}")

            # Count gripper actions
            close_count = sum(1 for p in path if p[2] == GRIPPER_CLOSE)
            open_count = sum(1 for p in path if p[2] == GRIPPER_OPEN)
            print(f"  Gripper close: {close_count}, open: {open_count}")

            # Save path
            output_path = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_test_path.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'episode_0': path,
                    'instruction': instruction,
                    'model': 'Qwen3-VL-8B-Instruct'
                }, f)
            print(f"\nSaved path to: {output_path}")

            return path
        else:
            print("\nERROR: Failed to process path from answer")
            return None
    else:
        print("\nERROR: No <ans> tags found in response")
        return None


if __name__ == "__main__":
    test_qwen3_path_generation()
