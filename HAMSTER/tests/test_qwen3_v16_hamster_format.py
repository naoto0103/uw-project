#!/usr/bin/env python3
"""
Generate Qwen3 VERSION 18 path: HAMSTER format with final example coordinate changed to (190, 711)
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
SERVER_PORT = 8001

print(f"Connecting to Qwen3 server: {SERVER_IP}:{SERVER_PORT}")

# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1

def encode_image(image_array):
    """Encode numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def process_answer(input_str):
    """
    Extract waypoints from Qwen3 HAMSTER format output
    Format: [(500, 400), ..., <action>Open Gripper</action>, ...]
    Coordinates are in [0, 1000] range, will be normalized to [0, 1]
    Returns: [[x, y, gripper_state], ...]
    """
    # Replace action tags with special coordinates
    input_str = input_str.replace('<action>Close Gripper</action>', '(10000.0, 10000.0)')
    input_str = input_str.replace('<action>Open Gripper</action>', '(10001.0, 10001.0)')

    # Evaluate as Python list
    keypoints = eval(input_str)

    processed_points = []
    action_flag = GRIPPER_CLOSE  # Default: closed

    for point in keypoints:
        x, y = point

        # Check for action markers
        if x == y and x == 10000.0:
            action_flag = GRIPPER_CLOSE
            if processed_points:
                processed_points[-1][-1] = action_flag
            continue
        elif x == y and x == 10001.0:
            action_flag = GRIPPER_OPEN
            if processed_points:
                processed_points[-1][-1] = action_flag
            continue

        # Regular waypoint - normalize from [0, 1000] to [0, 1]
        x_norm = x / 1000.0
        y_norm = y / 1000.0
        processed_points.append([x_norm, y_norm, action_flag])

    return processed_points


def test_qwen3_v18():
    """Test Qwen3 VERSION 18: HAMSTER format with final example coordinate (190, 711)"""

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
    task_instruction = "Pick up the apple and put it behind the hammer"

    # VERSION 18 prompt: HAMSTER format with final example coordinate changed to (190, 711)
    prompt = f"""Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: {task_instruction}

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(534, 439), (675, 306), <action>Close Gripper</action>, (190, 711), <action>Open Gripper</action>]
</ans>"""

    print(f"\nTask: {task_instruction}")
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
        max_tokens=1024,
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
    print("QWEN3-VL VERSION 18 RESPONSE:")
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
            output_path = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_v18_hamster_final_coord_190_711_path.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'episode_0': path,
                    'instruction': task_instruction,
                    'model': 'Qwen3-VL-8B-Instruct',
                    'prompt_type': 'hamster_final_coord_190_711',
                    'prompt': prompt,
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
    test_qwen3_v18()
