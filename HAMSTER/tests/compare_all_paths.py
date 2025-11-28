#!/usr/bin/env python3
"""
Compare VILA vs Qwen3 (original) vs Qwen3 (optimized) path generation
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

print("="*70)
print("COMPARISON: VILA vs Qwen3 (Original) vs Qwen3 (Optimized)")
print("="*70)
print(f"\nFrame size: {W}x{H}")

# Load VILA path
vila_data = pickle.load(open('/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/data/pick_apple_messy_50/hamster_paths.pkl', 'rb'))
vila_path = vila_data['paths'][0]

# Load Qwen3 original path
qwen3_orig_data = pickle.load(open('/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_test_path.pkl', 'rb'))
qwen3_orig_path = qwen3_orig_data['episode_0']

# Load Qwen3 optimized path
qwen3_opt_data = pickle.load(open('/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/results/qwen3_optimized_path.pkl', 'rb'))
qwen3_opt_path = qwen3_opt_data['episode_0']

# Print detailed comparison
print("\n" + "="*70)
print("PATH DETAILS")
print("="*70)

def print_path_details(name, path, color="white"):
    print(f"\n{name}:")
    print(f"  Waypoints: {len(path)}")

    # Count gripper states
    close_count = sum(1 for p in path if p[2] == GRIPPER_CLOSE)
    open_count = sum(1 for p in path if p[2] == GRIPPER_OPEN)

    # Count transitions
    transitions = 0
    for i in range(len(path) - 1):
        if path[i][2] != path[i+1][2]:
            transitions += 1

    print(f"  Gripper states: {close_count} CLOSE, {open_count} OPEN")
    print(f"  Gripper transitions: {transitions}")

    # Calculate path length
    length = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i][:2]
        x2, y2 = path[i+1][:2]
        length += np.sqrt((x2-x1)**2 + (y2-y1)**2)

    print(f"  Total path length: {length:.4f}")

    # Unique positions
    unique_pos = len(set([(x, y) for x, y, _ in path]))
    print(f"  Unique positions: {unique_pos}")

    # Show waypoints
    print(f"  Waypoints:")
    for i, (x, y, g) in enumerate(path):
        gripper_str = "CLOSE" if g == GRIPPER_CLOSE else "OPEN "
        print(f"    {i}: ({x:.3f}, {y:.3f}) - {gripper_str}")

print_path_details("VILA-1.5-13B (Fine-tuned)", vila_path)
print_path_details("Qwen3-VL-8B (Original Prompt)", qwen3_orig_path)
print_path_details("Qwen3-VL-8B (Optimized Prompt)", qwen3_opt_path)

# Summary comparison table
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)
print(f"{'Model':<35} {'Waypoints':<12} {'Unique Pos':<12} {'Path Length':<12}")
print("-"*70)

def calc_metrics(path):
    unique = len(set([(x, y) for x, y, _ in path]))
    length = sum(np.sqrt((path[i+1][0]-path[i][0])**2 + (path[i+1][1]-path[i][1])**2)
                 for i in range(len(path)-1))
    return len(path), unique, length

vila_metrics = calc_metrics(vila_path)
qwen3_orig_metrics = calc_metrics(qwen3_orig_path)
qwen3_opt_metrics = calc_metrics(qwen3_opt_path)

print(f"{'VILA-1.5-13B (Fine-tuned)':<35} {vila_metrics[0]:<12} {vila_metrics[1]:<12} {vila_metrics[2]:<12.4f}")
print(f"{'Qwen3-VL-8B (Original)':<35} {qwen3_orig_metrics[0]:<12} {qwen3_orig_metrics[1]:<12} {qwen3_orig_metrics[2]:<12.4f}")
print(f"{'Qwen3-VL-8B (Optimized)':<35} {qwen3_opt_metrics[0]:<12} {qwen3_opt_metrics[1]:<12} {qwen3_opt_metrics[2]:<12.4f}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print("""
1. VILA (Fine-tuned):
   - Generates 4 distinct waypoints with actual movement trajectory
   - All waypoints start with gripper CLOSED, final waypoint OPEN
   - Shows clear pick-and-place behavior

2. Qwen3 (Original Prompt):
   - Generates 2 waypoints at same position (0.15, 0.35)
   - Simple CLOSE -> OPEN action without movement
   - Uses HAMSTER-style <ans> tag format

3. Qwen3 (Optimized Prompt):
   - Generates 2 waypoints at same position (0.126, 0.500)
   - Uses Qwen3-native JSON format with [0, 1000] coordinates
   - Still minimal trajectory, no movement between waypoints

Conclusion:
- VILA's fine-tuning on robotics data is crucial for trajectory generation
- Qwen3-VL (zero-shot) struggles to generate meaningful trajectories
- Both Qwen3 prompts produce similar results: minimal waypoints, no movement
- Prompt optimization alone is insufficient without task-specific training
""")

print("="*70)
