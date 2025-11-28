#!/usr/bin/env python3
"""
Generate VILA path for new task: "Pick up the apple and put it behind the hammer"
"""
import base64
import cv2
import numpy as np
from openai import OpenAI
import re
import pickle
import zarr

# Server configuration
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8000

print(f"Connecting to VILA server: {SERVER_IP}:{SERVER_PORT}")

# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1

def encode_image(image_array):
    """Encode numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def process_answer(input_str):
    """
    Extract waypoints from VILA output
    Format: [(0.25, 0.32), ..., <action>Open Gripper</action>, ...]
    Returns: [[x, y, gripper_state], ...]
    """
    # Replace action tags with special coordinates
    input_str = input_str.replace('<action>Close Gripper</action>', '(1000.0, 1000.0)')
    input_str = input_str.replace('<action>Open Gripper</action>', '(1001.0, 1001.0)')

    # Evaluate as Python list
    keypoints = eval(input_str)

    processed_points = []
    action_flag = GRIPPER_CLOSE  # Default: closed

    for point in keypoints:
        x, y = point

        # Check for action markers
        if x == y and x == 1000.0:
            action_flag = GRIPPER_CLOSE
            if processed_points:
                processed_points[-1][-1] = action_flag
            continue
        elif x == y and x == 1001.0:
            action_flag = GRIPPER_OPEN
            if processed_points:
                processed_points[-1][-1] = action_flag
            continue

        # Regular waypoint
        processed_points.append([x, y, action_flag])

    return processed_points


def test_vila_new_task():
    """Test VILA with new task instruction"""

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
    task_instruction = "Pick up the apple and put it behind the hammer"

    # VILA prompt (from gradio_server_example.py)
    prompt = f"""In the image, please execute the command described in <quest>{task_instruction}</quest>.

Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.

Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:
<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>

The tuple denotes point x and y location of the end effector in the image. The action tags indicate gripper actions.

Coordinates should be floats between 0 and 1, representing relative positions.

Remember to provide points between <ans> and </ans> tags and think step by step.
"""

    print(f"\nTask: {task_instruction}")
    print(f"Prompt length: {len(prompt)} chars")

    # Initialize OpenAI client
    client = OpenAI(
        api_key="fake-key",
        base_url=f"http://{SERVER_IP}:{SERVER_PORT}"
    )

    # Send request
    print("\nSending request to VILA server...")

    response = client.chat.completions.create(
        model="HAMSTER_dev",
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
        max_tokens=128,
        temperature=0.0,
        top_p=0.95
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
    print("VILA-1.5-13B RESPONSE:")
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
            for i, wp in enumerate(path):
                gripper_str = "CLOSE" if wp[2] == GRIPPER_CLOSE else "OPEN"
                print(f"  {i}: ({wp[0]:.3f}, {wp[1]:.3f}) - {gripper_str}")

            # Count gripper actions
            close_count = sum(1 for p in path if p[2] == GRIPPER_CLOSE)
            open_count = sum(1 for p in path if p[2] == GRIPPER_OPEN)
            print(f"\n  Gripper states: {close_count} CLOSE, {open_count} OPEN")

            # Save path
            output_path = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/vila_new_task_path.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'episode_0': path,
                    'instruction': task_instruction,
                    'model': 'VILA-1.5-13B',
                    'raw_response': response_text
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
    test_vila_new_task()
