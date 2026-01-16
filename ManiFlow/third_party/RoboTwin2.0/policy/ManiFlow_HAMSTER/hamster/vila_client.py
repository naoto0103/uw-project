"""
VILA (HAMSTER finetuned) client for path generation during evaluation.

This module provides a client to communicate with a single VILA-13B server
for generating HAMSTER-style 2D paths.

Reference: HAMSTER/tests/parallel_vila/generate_paths.py
"""

import os
import re
import base64
import cv2
import numpy as np
from typing import List, Tuple, Optional

# Clear SSL environment variables to avoid certificate errors in Singularity
for key in ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"]:
    if key in os.environ:
        del os.environ[key]

from openai import OpenAI


# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1

# VILA prompt template (exact format from parallel_vila)
VILA_PROMPT_TEMPLATE = (
    "\nIn the image, please execute the command described in <quest>{instruction}</quest>.\n"
    "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
    "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
    "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\n"
    "The tuple denotes point x and y location of the end effector in the image. The action tags indicate gripper actions.\n"
    "Coordinates should be floats between 0 and 1, representing relative positions.\n"
    "Remember to provide points between <ans> and </ans> tags and think step by step."
)


def parse_vila_path(response_text: str) -> Optional[List[Tuple[float, float, int]]]:
    """
    Parse VILA path from response with <ans> tag.

    Args:
        response_text: Raw response from VILA

    Returns:
        List of (x, y, gripper_state) tuples in [0, 1] range
        Returns None if parsing fails
    """
    # Extract content from <ans> tags
    ans_match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL | re.IGNORECASE)
    if not ans_match:
        return None

    ans_content = ans_match.group(1).strip()

    # Replace action tags with special coordinates
    ans_content = ans_content.replace('<action>Close Gripper</action>', '(1000.0, 1000.0)')
    ans_content = ans_content.replace('<action>Open Gripper</action>', '(1001.0, 1001.0)')

    try:
        keypoints = eval(ans_content)
    except Exception:
        return None

    # Process keypoints
    path = []
    current_gripper_state = GRIPPER_OPEN

    for point in keypoints:
        x, y = point

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

        path.append((float(x), float(y), current_gripper_state))

    return path if path else None


class VILAClient:
    """
    Client for VILA (HAMSTER finetuned) path generation.

    Communicates with a single VILA-13B server to generate 2D paths
    for robot manipulation tasks during evaluation.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000/v1",
        model_name: str = "HAMSTER_dev",
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 256,
    ):
        """
        Initialize the VILA client.

        Args:
            server_url: URL of the VILA server (with /v1 endpoint)
            model_name: Model name to use
            temperature: Sampling temperature (0.0 for deterministic)
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens in response
        """
        self.server_url = server_url
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key="dummy",
            base_url=server_url,
        )

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode numpy image to base64 string.

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            Base64 encoded JPEG string
        """
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_image)
        return base64.b64encode(buffer).decode('utf-8')

    def predict_path(
        self,
        image: np.ndarray,
        task_instruction: str,
        verbose: bool = False,
    ) -> Optional[List[Tuple[float, float, int]]]:
        """
        Generate path prediction for an image.

        Args:
            image: RGB image array (H, W, 3)
            task_instruction: Task description
            verbose: Print detailed info

        Returns:
            List of (x, y, gripper_state) tuples, normalized to [0, 1]
            Returns None if prediction fails
        """
        # Build prompt
        prompt = VILA_PROMPT_TEMPLATE.format(instruction=task_instruction)

        # Encode image
        image_base64 = self._encode_image(image)

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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={"num_beams": 1, "use_cache": False}
            )

            response_content = response.choices[0].message.content

            # Handle list format (VILA may return list format)
            if isinstance(response_content, list):
                raw_response = ""
                for item in response_content:
                    if hasattr(item, 'text'):
                        raw_response += item.text
                    elif isinstance(item, dict) and 'text' in item:
                        raw_response += item['text']
            else:
                raw_response = response_content

            if verbose:
                print(f"[VILAClient] Raw response: {raw_response[:200]}...")

            path = parse_vila_path(raw_response)

            if path is None and verbose:
                print(f"[VILAClient] WARNING: Failed to parse path")

            return path

        except Exception as e:
            print(f"[VILAClient] ERROR: {e}")
            return None

    def check_server(self) -> bool:
        """
        Check if VILA server is running.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            import requests
            # Extract base URL
            base_url = self.server_url.rsplit('/v1', 1)[0]

            # Just check if server is responding (404 is OK, means server is up)
            response = requests.get(f"{base_url}/", timeout=5)
            print(f"[VILAClient] Server connected (status: {response.status_code})")
            return True
        except Exception as e:
            print(f"[VILAClient] Cannot connect to server: {e}")
            return False


# Task instructions for RoboTwin 2.0 single-arm tasks
SINGLE_ARM_INSTRUCTIONS = {
    "beat_block_hammer": "Pick up the hammer and beat the block",
    "click_bell": "click the bell's top center on the table",
    "move_can_pot": "pick up the can and move it to beside the pot",
    "place_object_stand": "place the object on the stand",
    "open_microwave": "open the microwave",
    "turn_switch": "click the switch",
    "adjust_bottle": "adjust the bottle to the correct orientation",
}


def get_task_instruction(task_name: str) -> str:
    """
    Get task instruction for a given task name.

    Args:
        task_name: Name of the task

    Returns:
        Task instruction string
    """
    return SINGLE_ARM_INSTRUCTIONS.get(
        task_name,
        f"Complete the {task_name} task"
    )


if __name__ == "__main__":
    print("=" * 50)
    print("VILAClient Test")
    print("=" * 50)

    client = VILAClient()

    print("\n1. Checking server...")
    if client.check_server():
        print("   Server is running!")

        print("\n2. Testing with dummy image...")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_image[:] = (128, 128, 128)

        path = client.predict_path(
            dummy_image,
            "Pick up the object",
            verbose=True
        )

        if path:
            print(f"   Path generated: {len(path)} points")
        else:
            print("   No path generated (expected with dummy image)")
    else:
        print("   Server not running. Start with: ./start_servers.sh")

    print("\n" + "=" * 50)
