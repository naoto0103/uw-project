#!/usr/bin/env python3
"""
Visualize comparison: VILA vs Qwen3 (improved prompt)
Task: "Pick up the apple and put it behind the hammer"
"""
import cv2
import numpy as np
import pickle
import zarr
from matplotlib import cm

# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1


def draw_lines_on_image_cv(image, points, draw_action=False, num_subdivisions=100):
    """
    HAMSTER's original drawing method
    """
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

        # Handle division by zero (when consecutive points are identical)
        denominator = distances[idx + 1] - distances[idx]
        if denominator == 0 or np.isnan(denominator):
            # Use the current point without interpolation
            point = pixel_points[idx]
        else:
            t = (sd - distances[idx]) / denominator
            # Clamp t to [0, 1] to avoid extrapolation
            t = np.clip(t, 0, 1)
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


def draw_path_on_frame(frame, path, title="", subtitle=""):
    """
    Draw path on frame with HAMSTER style

    Args:
        frame: RGB image (H, W, 3)
        path: List of [x, y, gripper_state] where x,y in [0, 1]
        title: Title text
        subtitle: Subtitle text
    """
    # Convert to BGR for cv2
    frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

    # Use HAMSTER's original drawing method
    frame_bgr = draw_lines_on_image_cv(frame_bgr, path, draw_action=True)

    # Add title
    if title:
        cv2.putText(frame_bgr, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Add subtitle
    if subtitle:
        cv2.putText(frame_bgr, subtitle, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Convert back to RGB
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def main():
    print("="*60)
    print("Creating VILA vs Qwen3 (Improved Prompt) Comparison")
    print("="*60)

    # Load dataset
    zarr_path = "/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/data/pick_apple_messy_50.zarr"
    dataset = zarr.open(zarr_path, mode='r')

    # Get first frame
    frame_chw = dataset['data']['head_camera'][0]
    first_frame = np.transpose(frame_chw, (1, 2, 0))  # (H, W, C)

    print(f"\nLoaded frame: {first_frame.shape}")

    # Load VILA path
    vila_path_file = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/vila_new_task_path.pkl"
    with open(vila_path_file, 'rb') as f:
        vila_data = pickle.load(f)
        vila_path = vila_data['episode_0']

    print(f"\nVILA path loaded: {len(vila_path)} waypoints")

    # Load Qwen3 path
    # VERSION 1: qwen3_path_file = ".../qwen3_improved_prompt_path.pkl"
    # VERSION 2: qwen3_path_file = ".../qwen3_trajectory_prompt_path.pkl"
    # VERSION 3: qwen3_path_file = ".../qwen3_variable_coords_path.pkl"
    # VERSION 4: qwen3_path_file = ".../qwen3_diverse_coords_path.pkl"
    # VERSION 5: qwen3_path_file = ".../qwen3_simplified_format_path.pkl"
    qwen3_path_file = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_simplified_format_path.pkl"
    with open(qwen3_path_file, 'rb') as f:
        qwen3_data = pickle.load(f)
        qwen3_path = qwen3_data['episode_0']
        qwen3_prompt_type = qwen3_data.get('prompt_type', 'unknown')

    print(f"Qwen3 ({qwen3_prompt_type}) path loaded: {len(qwen3_path)} waypoints")

    # Calculate statistics
    def calc_stats(path, name):
        unique_pos = len(set((wp[0], wp[1]) for wp in path))
        path_length = sum(
            ((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)**0.5
            for i in range(len(path) - 1)
        )
        transitions = sum(
            1 for i in range(len(path) - 1)
            if path[i][2] != path[i+1][2]
        )
        print(f"\n{name}:")
        print(f"  Waypoints: {len(path)}")
        print(f"  Unique positions: {unique_pos}")
        print(f"  Path length: {path_length:.4f}")
        print(f"  Gripper transitions: {transitions}")
        return unique_pos, path_length, transitions

    vila_stats = calc_stats(vila_path, "VILA")
    qwen3_stats = calc_stats(qwen3_path, "Qwen3 (Improved)")

    # Create visualization
    vila_frame = draw_path_on_frame(
        first_frame,
        vila_path,
        "VILA-1.5-13B",
        f"{len(vila_path)} waypoints, {vila_stats[0]} positions, len={vila_stats[1]:.3f}"
    )

    qwen3_frame = draw_path_on_frame(
        first_frame,
        qwen3_path,
        f"Qwen3-VL-8B ({qwen3_prompt_type})",
        f"{len(qwen3_path)} waypoints, {qwen3_stats[0]} positions, len={qwen3_stats[1]:.3f}"
    )

    # Combine side by side
    comparison = np.hstack([vila_frame, qwen3_frame])

    # Add task instruction at top
    task_text = "Task: Pick up the apple and put it behind the hammer"
    comparison_with_title = np.zeros((comparison.shape[0] + 50, comparison.shape[1], 3), dtype=np.uint8)
    comparison_with_title[50:] = comparison

    # Convert to BGR for text
    comparison_bgr = cv2.cvtColor(comparison_with_title, cv2.COLOR_RGB2BGR)
    cv2.putText(comparison_bgr, task_text, (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Save
    # VERSION 1: output_path = ".../improved_prompt_comparison.png"
    # VERSION 2: output_path = ".../trajectory_prompt_comparison.png"
    # VERSION 3: output_path = ".../variable_coords_comparison.png"
    # VERSION 4: output_path = ".../diverse_coords_comparison.png"
    # VERSION 5: output_path = ".../simplified_format_comparison.png"
    output_path = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/visualizations/simplified_format_comparison.png"
    cv2.imwrite(output_path, comparison_bgr)

    print(f"\n{'='*60}")
    print(f"Comparison saved to: {output_path}")
    print(f"Image size: {comparison_bgr.shape}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
