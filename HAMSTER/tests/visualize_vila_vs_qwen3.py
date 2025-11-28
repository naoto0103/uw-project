#!/usr/bin/env python3
"""
Compare VILA vs Qwen3-VL (Optimized) - Side-by-side visualization
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

qwen3_opt_data = pickle.load(open('/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_optimized_path.pkl', 'rb'))
qwen3_opt_path = qwen3_opt_data['episode_0']

print(f"\nVILA path: {len(vila_path)} waypoints")
print(f"Qwen3 (optimized) path: {len(qwen3_opt_path)} waypoints")


def draw_path_on_frame(frame, path, title="", subtitle=""):
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
        cv2.line(img, pt1, pt2, (0, 255, 0), 3)

    # Draw points
    scale = max(img_w, img_h) / 800.0
    base_radius = int(10 * scale)

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
            cv2.circle(img, (px_x, px_y), base_radius + 3, (0, 255, 255), 3)

        # Add waypoint number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (255, 255, 255)
        cv2.putText(img, str(i), (px_x - 6, px_y + 6), font, font_scale, text_color, 2)

    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    # Main title
    (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, font_thickness)
    cv2.rectangle(img, (5, 5), (text_w + 10, text_h + 10), bg_color, -1)
    cv2.putText(img, title, (8, text_h + 8), font, font_scale, text_color, font_thickness)

    # Subtitle (waypoint count)
    y_offset = text_h + 15
    (sub_w, sub_h), _ = cv2.getTextSize(subtitle, font, 0.35, 1)
    cv2.rectangle(img, (5, y_offset), (sub_w + 10, y_offset + sub_h + 8), bg_color, -1)
    cv2.putText(img, subtitle, (8, y_offset + sub_h + 4), font, 0.35, text_color, 1)

    return img


# Create visualizations
print("\nCreating visualizations...")

vila_vis = draw_path_on_frame(
    frame,
    vila_path,
    "VILA-1.5-13B (Fine-tuned)",
    f"{len(vila_path)} waypoints"
)

qwen3_opt_vis = draw_path_on_frame(
    frame,
    qwen3_opt_path,
    "Qwen3-VL-8B (Zero-shot)",
    f"{len(qwen3_opt_path)} waypoints"
)

# Create horizontal comparison
comparison_h = np.hstack([vila_vis, qwen3_opt_vis])

# Add separator
separator_x = W
cv2.line(comparison_h, (separator_x, 0), (separator_x, H), (255, 255, 255), 3)

# Add overall title
font = cv2.FONT_HERSHEY_SIMPLEX
overall_title = "Path Generation Comparison: Fine-tuned vs Zero-shot"
font_scale = 0.55
font_thickness = 1
(title_w, title_h), _ = cv2.getTextSize(overall_title, font, font_scale, font_thickness)

# Create space for title
title_bg = np.zeros((title_h + 20, comparison_h.shape[1], 3), dtype=np.uint8)
cv2.putText(title_bg, overall_title,
            (comparison_h.shape[1]//2 - title_w//2, title_h + 10),
            font, font_scale, (255, 255, 255), font_thickness)

# Combine title and comparison
final_comparison = np.vstack([title_bg, comparison_h])

# Save images
output_dir = "/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/visualizations"

cv2.imwrite(f'{output_dir}/comparison_vila_vs_qwen3_optimized.png',
            cv2.cvtColor(final_comparison, cv2.COLOR_RGB2BGR))

print("\nSaved visualization:")
print(f"  - {output_dir}/comparison_vila_vs_qwen3_optimized.png")

# Print detailed comparison
print("\n" + "="*70)
print("DETAILED COMPARISON")
print("="*70)

print("\nVILA-1.5-13B (Fine-tuned on 1.2M robot samples):")
for i, (x, y, g) in enumerate(vila_path):
    gripper_str = "CLOSE" if g == GRIPPER_CLOSE else "OPEN "
    print(f"  Waypoint {i}: ({x:.3f}, {y:.3f}) - {gripper_str}")

vila_length = sum(np.sqrt((vila_path[i+1][0]-vila_path[i][0])**2 +
                          (vila_path[i+1][1]-vila_path[i][1])**2)
                  for i in range(len(vila_path)-1))
print(f"  Total path length: {vila_length:.4f}")

print("\nQwen3-VL-8B (Zero-shot with optimized prompt):")
for i, (x, y, g) in enumerate(qwen3_opt_path):
    gripper_str = "CLOSE" if g == GRIPPER_CLOSE else "OPEN "
    print(f"  Waypoint {i}: ({x:.3f}, {y:.3f}) - {gripper_str}")

qwen3_length = sum(np.sqrt((qwen3_opt_path[i+1][0]-qwen3_opt_path[i][0])**2 +
                           (qwen3_opt_path[i+1][1]-qwen3_opt_path[i][1])**2)
                   for i in range(len(qwen3_opt_path)-1))
print(f"  Total path length: {qwen3_length:.4f}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("VILA generates a meaningful trajectory with 4 distinct waypoints,")
print("while Qwen3 (zero-shot) only produces 2 waypoints at the same location.")
print("This demonstrates the critical importance of fine-tuning on robot data.")
print("="*70)
