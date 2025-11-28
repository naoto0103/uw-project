#!/usr/bin/env python3
"""
Create side-by-side visualization comparing VILA and Qwen3 paths
"""
import cv2
import numpy as np
import pickle
import zarr

# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1

# Load dataset
zarr_path = "/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/data/pick_apple_messy_50.zarr"
dataset = zarr.open(zarr_path, mode='r')

# Get first frame
frame_chw = dataset['data']['head_camera'][0]
frame = np.transpose(frame_chw, (1, 2, 0))
H, W = frame.shape[:2]

print(f"Frame shape: {frame.shape}")
print(f"Frame size: {W}x{H}")

# Load paths
vila_data = pickle.load(open('/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/data/pick_apple_messy_50/hamster_paths.pkl', 'rb'))
vila_path = vila_data['paths'][0]

qwen3_orig_data = pickle.load(open('/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_test_path.pkl', 'rb'))
qwen3_orig_path = qwen3_orig_data['episode_0']

qwen3_opt_data = pickle.load(open('/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_optimized_path.pkl', 'rb'))
qwen3_opt_path = qwen3_opt_data['episode_0']

print(f"\nVILA path: {len(vila_path)} waypoints")
print(f"Qwen3 (original) path: {len(qwen3_orig_path)} waypoints")
print(f"Qwen3 (optimized) path: {len(qwen3_opt_path)} waypoints")


def draw_path_on_frame(frame, path, title=""):
    """Draw path on frame with HAMSTER style"""
    img = frame.copy()
    img_h, img_w = img.shape[:2]

    # Convert normalized coords to pixels
    points_px = []
    for x, y, gripper in path:
        px_x = int(x * img_w)
        px_y = int(y * img_h)
        points_px.append((px_x, px_y, gripper))

    # Draw lines (green)
    for i in range(len(points_px) - 1):
        pt1 = points_px[i][:2]
        pt2 = points_px[i + 1][:2]
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    # Draw points
    scale = max(img_w, img_h) / 800.0
    base_radius = int(8 * scale)

    for i, (px_x, px_y, gripper) in enumerate(points_px):
        # Color based on gripper state
        if gripper == GRIPPER_CLOSE:
            color = (0, 0, 255)  # Red for close
        else:
            color = (255, 0, 0)  # Blue for open

        # Draw circle
        cv2.circle(img, (px_x, px_y), base_radius, color, -1)

        # Yellow outline for first point
        if i == 0:
            cv2.circle(img, (px_x, px_y), base_radius + 2, (0, 255, 255), 2)

        # Add waypoint number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_color = (255, 255, 255)
        cv2.putText(img, str(i), (px_x - 5, px_y + 5), font, font_scale, text_color, 1)

    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, font_thickness)
    cv2.rectangle(img, (5, 5), (text_w + 15, text_h + 15), bg_color, -1)
    cv2.putText(img, title, (10, text_h + 10), font, font_scale, text_color, font_thickness)

    # Add waypoint count
    count_text = f"{len(path)} waypoints"
    (count_w, count_h), _ = cv2.getTextSize(count_text, font, 0.5, 1)
    cv2.rectangle(img, (5, text_h + 20), (count_w + 15, text_h + count_h + 30), bg_color, -1)
    cv2.putText(img, count_text, (10, text_h + count_h + 25), font, 0.5, text_color, 1)

    return img


# Create visualizations
print("\nCreating visualizations...")

vila_vis = draw_path_on_frame(frame, vila_path, "VILA-1.5-13B (Fine-tuned)")
qwen3_orig_vis = draw_path_on_frame(frame, qwen3_orig_path, "Qwen3-VL-8B (Original)")
qwen3_opt_vis = draw_path_on_frame(frame, qwen3_opt_path, "Qwen3-VL-8B (Optimized)")

# Create horizontal comparison (all three)
comparison_h = np.hstack([vila_vis, qwen3_orig_vis, qwen3_opt_vis])

# Add separators
separator_x1 = W
separator_x2 = W * 2
cv2.line(comparison_h, (separator_x1, 0), (separator_x1, H), (255, 255, 255), 2)
cv2.line(comparison_h, (separator_x2, 0), (separator_x2, H), (255, 255, 255), 2)

# Create vertical comparison (stacked)
comparison_v = np.vstack([vila_vis, qwen3_orig_vis, qwen3_opt_vis])

# Add separators
separator_y1 = H
separator_y2 = H * 2
cv2.line(comparison_v, (0, separator_y1), (W, separator_y1), (255, 255, 255), 2)
cv2.line(comparison_v, (0, separator_y2), (W, separator_y2), (255, 255, 255), 2)

# Save images
output_dir = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/visualizations"

cv2.imwrite(f'{output_dir}/path_vila.png',
            cv2.cvtColor(vila_vis, cv2.COLOR_RGB2BGR))
cv2.imwrite(f'{output_dir}/path_qwen3_original.png',
            cv2.cvtColor(qwen3_orig_vis, cv2.COLOR_RGB2BGR))
cv2.imwrite(f'{output_dir}/path_qwen3_optimized.png',
            cv2.cvtColor(qwen3_opt_vis, cv2.COLOR_RGB2BGR))
cv2.imwrite(f'{output_dir}/comparison_horizontal.png',
            cv2.cvtColor(comparison_h, cv2.COLOR_RGB2BGR))
cv2.imwrite(f'{output_dir}/comparison_vertical.png',
            cv2.cvtColor(comparison_v, cv2.COLOR_RGB2BGR))

print("\nSaved visualizations:")
print(f"  - {output_dir}/path_vila.png")
print(f"  - {output_dir}/path_qwen3_original.png")
print(f"  - {output_dir}/path_qwen3_optimized.png")
print(f"  - {output_dir}/comparison_horizontal.png (side-by-side)")
print(f"  - {output_dir}/comparison_vertical.png (stacked)")

print("\nVisualization complete!")
print("\nLegend:")
print("  - Green lines: trajectory path")
print("  - Red circles: gripper CLOSED")
print("  - Blue circles: gripper OPEN")
print("  - Yellow outline: starting waypoint")
print("  - White numbers: waypoint index")
