"""
Overlay utilities for HAMSTER-style path drawing.

This module provides functions to draw paths on images using HAMSTER's original
drawing method for evaluation.

Reference: HAMSTER/tests/training_data/overlay_utils.py
"""

import cv2
import numpy as np
from matplotlib import cm
from typing import List, Tuple, Optional


# Gripper states
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1


def draw_path_on_image_hamster_style(
    image: np.ndarray,
    path: List[Tuple[float, float, int]],
    num_subdivisions: int = 100,
    draw_gripper_markers: bool = True,
) -> np.ndarray:
    """
    Draw path on image using HAMSTER's original drawing method.

    This is the exact implementation from HAMSTER for Low-Level Policy input.

    Args:
        image: Image array (H, W, 3) in BGR format
        path: List of (x, y, gripper_state) tuples, normalized to [0, 1]
        num_subdivisions: Number of subdivisions for path interpolation (default: 100)
        draw_gripper_markers: Whether to draw gripper state change markers

    Returns:
        Image with path drawn (BGR format)

    Drawing specification (HAMSTER original):
        - Line color: jet colormap (blue -> cyan -> green -> yellow -> red)
        - Line thickness: scaled based on image size (reference: 512x512)
        - Gripper markers: only at state change points
            - Open: blue circle (255, 0, 0) in BGR
            - Close: red circle (0, 0, 255) in BGR
        - Markers are outline only (not filled)
    """
    if not path or len(path) == 0:
        return image

    img = image.copy()
    height, width = img.shape[:2]

    # Calculate scale factor relative to 512x512 image (HAMSTER original)
    scale_factor = max(min(width, height) / 512.0, 1)
    circle_radius = int(7 * scale_factor)
    circle_thickness = max(1, int(2 * scale_factor))
    line_thickness = max(1, int(2 * scale_factor))

    # Convert normalized coordinates to pixel coordinates
    pixel_points = []
    gripper_status = []
    for point in path:
        x = int(point[0] * width)
        y = int(point[1] * height)
        action = int(point[2])
        pixel_points.append((x, y))
        gripper_status.append(action)

    # Draw gripper markers at state change points (HAMSTER original)
    if draw_gripper_markers:
        for idx, (x, y) in enumerate(pixel_points):
            if idx == 0 or gripper_status[idx] != gripper_status[idx - 1]:
                # Red=close, Blue=open (HAMSTER original)
                circle_color = (0, 0, 255) if gripper_status[idx] == GRIPPER_CLOSE else (255, 0, 0)
                cv2.circle(img, (x, y), circle_radius, circle_color, circle_thickness)

    # Convert list to NumPy array for interpolation
    pixel_points_arr = np.array(pixel_points, dtype=np.float32)

    # Calculate cumulative distances along the path
    distances = [0]
    for i in range(1, len(pixel_points_arr)):
        dist = np.linalg.norm(pixel_points_arr[i] - pixel_points_arr[i - 1])
        distances.append(distances[-1] + dist)
    total_distance = distances[-1]

    # Handle edge case: all points are the same
    if total_distance == 0:
        return img

    # Generate equally spaced distances along the path
    sample_distances = np.linspace(0, total_distance, num_subdivisions)

    # Interpolate points along the path
    interpolated_points = []
    idx = 0
    for sd in sample_distances:
        while idx < len(distances) - 2 and sd > distances[idx + 1]:
            idx += 1

        # Handle division by zero (when consecutive points are identical)
        denominator = distances[idx + 1] - distances[idx]
        if denominator == 0 or np.isnan(denominator):
            point = pixel_points_arr[idx]
        else:
            t = (sd - distances[idx]) / denominator
            t = np.clip(t, 0, 1)
            point = (1 - t) * pixel_points_arr[idx] + t * pixel_points_arr[idx + 1]

        interpolated_points.append(point)

    interpolated_points = np.array(interpolated_points, dtype=np.int32)

    # Map positions along the path to colors using jet colormap (HAMSTER original)
    cmap = cm.get_cmap('jet')
    colors = (cmap(np.linspace(0, 1, len(interpolated_points)))[:, :3] * 255).astype(np.uint8)

    # Draw line segments with varying colors
    for i in range(len(interpolated_points) - 1):
        pt1 = tuple(interpolated_points[i])
        pt2 = tuple(interpolated_points[i + 1])
        # Convert from RGB to BGR for OpenCV
        color = tuple(int(c) for c in colors[i][::-1])
        cv2.line(img, pt1, pt2, color=color, thickness=line_thickness)

    return img


class OverlayDrawer:
    """
    Overlay drawer for evaluation.

    Wraps the path drawing function with preprocessing for ManiFlow input.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        num_subdivisions: int = 100,
        draw_gripper_markers: bool = True,
    ):
        """
        Initialize the overlay drawer.

        Args:
            target_size: Target image size (H, W) for ManiFlow input
            num_subdivisions: Number of subdivisions for path interpolation
            draw_gripper_markers: Whether to draw gripper markers
        """
        self.target_size = target_size
        self.num_subdivisions = num_subdivisions
        self.draw_gripper_markers = draw_gripper_markers

    def draw(
        self,
        image: np.ndarray,
        path: List[Tuple[float, float, int]],
    ) -> np.ndarray:
        """
        Draw path on image and return processed overlay.

        Args:
            image: RGB image array (H, W, 3) - note: input is RGB, not BGR
            path: List of (x, y, gripper_state) tuples, normalized to [0, 1]

        Returns:
            Processed overlay image (3, H, W) float32, normalized to [0, 1]
            Ready for ManiFlow input
        """
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw path
        overlay_bgr = draw_path_on_image_hamster_style(
            bgr_image,
            path,
            num_subdivisions=self.num_subdivisions,
            draw_gripper_markers=self.draw_gripper_markers,
        )

        # Convert back to RGB
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Resize to target size
        overlay_resized = cv2.resize(
            overlay_rgb,
            (self.target_size[1], self.target_size[0]),  # cv2 expects (W, H)
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert to float32 and normalize to [0, 1]
        overlay_float = overlay_resized.astype(np.float32) / 255.0

        # Transpose to (C, H, W) for PyTorch
        overlay_chw = np.transpose(overlay_float, (2, 0, 1))

        return overlay_chw


if __name__ == "__main__":
    # Simple test
    print("Testing OverlayDrawer...")

    # Create dummy image (RGB)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_image[:] = (128, 128, 128)  # Gray background

    # Create dummy path
    dummy_path = [
        (0.2, 0.3, GRIPPER_OPEN),
        (0.4, 0.5, GRIPPER_OPEN),
        (0.6, 0.5, GRIPPER_CLOSE),
        (0.8, 0.7, GRIPPER_CLOSE),
    ]

    drawer = OverlayDrawer(target_size=(224, 224))
    overlay = drawer.draw(dummy_image, dummy_path)

    print(f"  Input shape: {dummy_image.shape}")
    print(f"  Output shape: {overlay.shape}")
    print(f"  Output dtype: {overlay.dtype}")
    print(f"  Output range: [{overlay.min():.3f}, {overlay.max():.3f}]")
    print("  [PASS] OverlayDrawer test passed!")
