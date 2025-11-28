import base64
import gradio as gr
from openai import OpenAI
import re
import cv2
import numpy as np
from matplotlib import cm
import os

from datetime import datetime

GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1
MODEL = "HAMSTER_dev"

# Attempt to load the server IP from file at the beginning.
server_ip_file = "./ip_eth0.txt"
try:
    with open(os.path.expanduser(f"{server_ip_file}"), "r") as f:
        SERVER_IP = f.read().strip()
except Exception:
    SERVER_IP = "127.0.0.1"  # Fallback to default IP
print("connection to server: ", SERVER_IP)
def preprocess_image(image, crop_type):
    """Process the image by either stretching or center cropping."""
    height, width, _ = image.shape
    if crop_type == "Center Crop":
        crop_size = min(height, width)
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    return image

def annotate_image(image, quest):
    """
    Annotate the given image by overlaying the quest (prompt) text in the top-left corner,
    then save the image with a timestamp in the filename.
    
    The image is assumed to be in BGR color space.
    """
    # Choose font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Get size of the text box
    (text_w, text_h), baseline = cv2.getTextSize(quest, font, font_scale, thickness)
    # Draw a filled rectangle as background for the text for better visibility
    cv2.rectangle(image, (5, 5), (5 + text_w + 10, 5 + text_h + 10), (0, 0, 0), -1)
    # Overlay the quest text on top of the rectangle
    cv2.putText(image, quest, (10, 5 + text_h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return image

def draw_lines_on_image_cv(image, points, draw_action=False, num_subdivisions=100):
    height, width, _ = image.shape

    # Calculate a scale factor relative to a 512x512 image
    scale_factor = max(min(width, height) / 512.0, 1)
    circle_radius = int(7 * scale_factor)
    circle_thickness = max(1, int(2 * scale_factor))
    line_thickness = max(1, int(2 * scale_factor))
    font_scale = 0.5 * scale_factor
    font_thickness = max(1, int(1 * scale_factor))
    text_color = (255, 255, 255)  # White color

    # Convert normalized coordinates to pixel coordinates
    pixel_points = []
    gripper_status = []
    for point in points:
        x = int(point[0] * width)
        y = int(point[1] * height)
        action = int(point[2])
        pixel_points.append((x, y))
        gripper_status.append(action)

    # Draw optional markers or numbers at the predicted points
    for idx, (x, y) in enumerate(pixel_points):
        if draw_action:
            if idx == 0 or gripper_status[idx] != gripper_status[idx - 1]:
                circle_color = (0, 0, 255) if gripper_status[idx] == GRIPPER_CLOSE else (255, 0, 0)
                cv2.circle(image, (x, y), circle_radius, circle_color, circle_thickness)

    # Convert list to NumPy array for interpolation
    pixel_points = np.array(pixel_points, dtype=np.float32)

    # Calculate cumulative distances along the path
    distances = [0]
    for i in range(1, len(pixel_points)):
        dist = np.linalg.norm(pixel_points[i] - pixel_points[i - 1])
        distances.append(distances[-1] + dist)
    total_distance = distances[-1]

    # Generate equally spaced distances along the path
    num_samples = num_subdivisions
    sample_distances = np.linspace(0, total_distance, num_samples)

    # Interpolate points along the path
    interpolated_points = []
    idx = 0
    for sd in sample_distances:
        while sd > distances[idx + 1] and idx < len(distances) - 2:
            idx += 1
        t = (sd - distances[idx]) / (distances[idx + 1] - distances[idx])
        point = (1 - t) * pixel_points[idx] + t * pixel_points[idx + 1]
        interpolated_points.append(point)
    interpolated_points = np.array(interpolated_points, dtype=np.int32)

    # Map positions along the path to colors using the jet colormap
    cmap = cm.get_cmap('jet')
    colors = (cmap(np.linspace(0, 1, len(interpolated_points)))[:, :3] * 255).astype(np.uint8)

    # Draw line segments with varying colors using the scaled line thickness
    for i in range(len(interpolated_points) - 1):
        pt1 = tuple(interpolated_points[i])
        pt2 = tuple(interpolated_points[i + 1])
        color = tuple(int(c) for c in colors[i])
        cv2.line(image, pt1, pt2, color=color, thickness=line_thickness)

    return image

def process_answer(input_str):
    """Extract keypoints from the model response."""
    input_str = input_str.replace('<action>Close Gripper</action>', '(1000.0, 1000.0)').replace('<action>Open Gripper</action>', '(1001.0, 1001.0)')
    keypoints = eval(input_str)
    processed_points = []
    action_flag = 0
    for point in keypoints:
        x, y = point
        if x == y and x == 1000.0:
            action_flag = GRIPPER_CLOSE
            processed_points[-1][-1] = action_flag
            continue
        elif x == y and x == 1001.0:
            action_flag = GRIPPER_OPEN
            processed_points[-1][-1] = action_flag
            continue
        processed_points.append([x, y, action_flag])
    return processed_points

def send_request(image, quest, max_tokens, temperature, top_p):
    """Send image and quest to OpenAI model and get response."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encoded_image_array = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode('utf-8')
    print(quest)
    # Try to send request using the IP loaded at startup.
    with open(os.path.expanduser(f"{server_ip_file}"), "r") as f:
        SERVER_IP = f.read().strip()
    try:
        client = OpenAI(base_url=f"http://{SERVER_IP}:8000", api_key="fake-key")
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                        {"type": "text", "text": 
                            f"\nIn the image, please execute the command described in <quest>{quest}</quest>.\n"
                            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
                            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
                            "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\n"
                            "The tuple denotes point x and y location of the end effector in the image. The action tags indicate gripper actions.\n"
                            "Coordinates should be floats between 0 and 1, representing relative positions.\n"
                            "Remember to provide points between <ans> and </ans> tags and think step by step."
                        },
                    ],
                }
            ],
            max_tokens=int(max_tokens),
            model=MODEL,
            extra_body={"num_beams": 1, "use_cache": False, "temperature": float(temperature), "top_p": float(top_p)},
        )
    except Exception as inner_e:
        raise inner_e
    response_text = response.choices[0].message.content[0]['text']
    return response_text

def process_image_and_quest(image, quest, max_tokens, temperature, top_p, crop_type):
    image = preprocess_image(image, crop_type)
    response_text = send_request(image, quest, max_tokens, temperature, top_p)
    try:
        response_text_strip = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL).group(1)
        points = process_answer(response_text_strip)
        output_image = draw_lines_on_image_cv(image.copy(), points, draw_action=True)
    except:
        output_image = image
    annotated_image = annotate_image(output_image.copy(), quest)
    return annotated_image, response_text

# Define examples as a list of inputs.
examples = [
    ["examples/ocr_reasoning.jpg", "Move the S to the plate the arrow is pointing at"],
    ["examples/non_prehensile.jpg", "open the top drawer"],
    ["examples/spatial_world_knowledge.jpg", "Have the middle block on Jensen Huang"]
]

with gr.Blocks() as demo:
    gr.Markdown("## HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation")
    with gr.Row():
        # Left Column: Inputs
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Input Image")
            quest_input = gr.Textbox(lines=2, label="Quest")
            with gr.Row():
                max_tokens_input = gr.Number(value=128, label="Max Tokens", precision=0)
                temperature_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="Temperature")
                top_p_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.95, label="Top P")
                crop_type_input = gr.Radio(["Stretch", "Center Crop"], value="Stretch", label="Image Processing Type")
            submit_btn = gr.Button("Submit")
            gr.Markdown("**Troubleshooting:**\n- If you encounter a `Connection Error`, simply resubmit the request or reupload the image; this is likely a network issue.\n- If you see an `Error`, please wait up to 5 minutes as the backend server is automatically relaunching.")
        # Right Column: Outputs and Examples
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Output Image")
            response_text = gr.Textbox(label="Response")
            gr.Markdown("#### Example (click the image for instructions)")
            gr.Examples(
                examples=examples,
                inputs=[image_input, quest_input],
                label="Example"
            )
            
    submit_btn.click(
        fn=process_image_and_quest, 
        inputs=[image_input, quest_input, max_tokens_input, temperature_input, top_p_input, crop_type_input], 
        outputs=[output_image, response_text]
    )

demo.launch(share=True)
