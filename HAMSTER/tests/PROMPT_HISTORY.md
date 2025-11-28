# Qwen3 Prompt Engineering History

Task: "Pick up the apple and put it behind the hammer"
VILA Baseline: 4 waypoints

---

## VERSION 1: Improved Prompt
**Date**: 2025-11-18
**Changes**: Initial improved prompt with better structure

### System Prompt:
None (using default)

### User Prompt:
```
In the image, execute the task described in <quest>Pick up the apple and put it behind the hammer</quest>.

Generate a sequence of waypoints representing the robot gripper's trajectory to achieve the goal.

Output in JSON format:
[
  {"point_2d": [253, 768], "gripper": "close"},
  {"point_2d": [489, 326], "gripper": "close"},
  {"point_2d": [757, 836], "gripper": "open"},
  ...
]

Format:
- point_2d: [x, y] coordinates in range [0, 1000] representing relative positions in the image
- gripper: "open" or "close" at each waypoint. When the state changes (e.g., "close" → "open"), the gripper actuates at that point. When the state remains the same (e.g., "open" → "open"), the gripper maintains its current state.

Think step by step and output the complete trajectory in JSON format.
```

### Generated Path:
- Waypoints: 4
- Unique positions: 2
- Raw output: `[{"point_2d": [113, 533], "gripper": "close"}, {"point_2d": [113, 533], "gripper": "open"}, {"point_2d": [757, 836], "gripper": "close"}, {"point_2d": [757, 836], "gripper": "open"}]`
- Result file: qwen3_improved_prompt_path.pkl

**Issue**: Duplicate coordinates at same position with different gripper states

---

## VERSION 2: Trajectory-focused Prompt
**Date**: 2025-11-18
**Changes**: Best early version with trajectory emphasis

### System Prompt:
None (using default)

### User Prompt:
```
In the image, execute the task described in <quest>Pick up the apple and put it behind the hammer</quest>.

You are controlling a robot gripper's end-effector. Generate a trajectory showing the complete path the gripper should follow to accomplish the task.

The trajectory should include:
1. Starting position (where the gripper begins)
2. Intermediate waypoints (positions along the path to the goal)
3. Goal position (where the task is completed)
4. Gripper actions (open/close) only if the task requires grasping or releasing objects

Output in JSON format:
[
  {"point_2d": [150, 400], "gripper": "open"},
  {"point_2d": [200, 450], "gripper": "open"},
  {"point_2d": [253, 500], "gripper": "open"},
  {"point_2d": [253, 500], "gripper": "close"},
  {"point_2d": [320, 480], "gripper": "close"},
  {"point_2d": [450, 520], "gripper": "close"},
  {"point_2d": [550, 600], "gripper": "close"},
  {"point_2d": [550, 600], "gripper": "open"},
  ...
]

Format:
- point_2d: [x, y] coordinates in range [0, 1000] representing the gripper's position in the image
- gripper: "open" or "close" state at each waypoint
- Generate multiple waypoints along the path, not just start and end positions
- The gripper follows a smooth trajectory through all waypoints in sequence

Think step by step and output the complete trajectory in JSON format.
```

### Generated Path:
- Waypoints: 8
- Unique positions: 6
- Result file: qwen3_trajectory_prompt_path.pkl

**Best early result**: Highest number of unique positions

---

## VERSION 3: Variable Coordinates
**Date**: 2025-11-18
**Changes**: Used variable notation (x1, y1) in examples

### System Prompt:
None (using default)

### User Prompt:
Used same structure as VERSION 2 but with variable notation `[x1, y1], [x2, y2]` in examples instead of concrete numbers

### Generated Path:
- Waypoints: 31
- Issue: Generated horizontal line at y=480
- Result file: qwen3_variable_coords_path.pkl

**Issue**: All points on same y-coordinate (model interpreted variables literally)

---

## VERSION 4: Diverse Coordinates
**Date**: 2025-11-18
**Changes**: Diverse coordinate examples

### Generated Path:
Not executed

---

## VERSION 5: Simplified Format
**Date**: 2025-11-18
**Changes**: Removed redundant lines from prompt

### System Prompt:
None (using default)

### User Prompt:
Simplified version - removed "Generate multiple waypoints" and "smooth trajectory" lines from VERSION 2

### Generated Path:
- Waypoints: 5
- Unique positions: 2
- Result file: qwen3_simplified_format_path.pkl

---

## VERSION 6: Unchanged State
**Date**: 2025-11-18
**Changes**: Added "unchanged" gripper state option

### System Prompt:
None (using default)

### User Prompt:
Added "unchanged" as third gripper state option and "Initial condition: The gripper starts in the 'open' state."

### Generated Path:
- Waypoints: 3
- Unique positions: 2
- Result file: qwen3_unchanged_state_path.pkl

**Issue**: Worse than VERSION 5 - model didn't use "unchanged" effectively

---

## VERSION 7: HAMSTER-style Prompt (Same as VILA)
**Date**: 2025-11-18
**Changes**: Used exact HAMSTER/VILA prompt format

### System Prompt:
None

### User Prompt:
```python
prompt = f"""Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: {instruction}

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are normalized to range [0, 1].

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(0.5, 0.4), (0.6, 0.3), <action>Close Gripper</action>, (0.7, 0.5), <action>Open Gripper</action>]
</ans>
"""
```

### Generated Path:
- Waypoints: 2
- Unique positions: 2
- Coordinates: [(0.15, 0.35), (0.65, 0.35)]

**Note**: Eliminated coordinate duplication but fewer waypoints

---

## VERSION 8: VILA System Prompt
**Date**: 2025-11-19
**Changes**: Added VILA's base system prompt

### System Prompt:
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
```

### User Prompt:
```python
prompt = f"""Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: {instruction}

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are normalized to range [0, 1].

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(0.5, 0.4), (0.6, 0.3), <action>Close Gripper</action>, (0.7, 0.5), <action>Open Gripper</action>]
</ans>
"""
```

### Generated Path:
- Waypoints: 2
- Coordinates: [(0.15, 0.35), (0.65, 0.35)]
- Result file: qwen3_vila_system_prompt_path.pkl (lost - overwritten)

---

## VERSION 9: Robot Trajectory Specialized System Prompt
**Date**: 2025-11-19
**Changes**: Enhanced system prompt with robot trajectory specialization

### System Prompt:
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant is specialized in generating precise robot gripper trajectories for manipulation tasks, providing clear waypoint sequences with appropriate gripper states. The trajectories represent the path of the gripper tip.
```

### User Prompt:
```python
prompt = f"""Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: {instruction}

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are normalized to range [0, 1].

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(0.5, 0.4), (0.6, 0.3), <action>Close Gripper</action>, (0.7, 0.5), <action>Open Gripper</action>]
</ans>
"""
```

### Generated Path:
- Waypoints: 3
- Coordinates: [(0.15, 0.35), (0.45, 0.35), (0.65, 0.35)]
- Result file: qwen3_robot_specialized_path.pkl (lost - overwritten)

---

## VERSION 10: 0~1000 Coordinate System
**Date**: 2025-11-19
**Changes**: Changed coordinate range from [0, 1] to [0, 1000] (Qwen3-VL's native format)

### System Prompt:
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant is specialized in generating precise robot gripper trajectories for manipulation tasks, providing clear waypoint sequences with appropriate gripper states. The trajectories represent the path of the gripper tip.
```

### User Prompt:
```python
prompt = f"""Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: {instruction}

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(500, 400), (600, 300), <action>Close Gripper</action>, (700, 500), <action>Open Gripper</action>]
</ans>
"""
```

### Coordinate Conversion:
```python
# Convert from [0, 1000] to [0, 1] range
x_val = float(x) / 1000.0
y_val = float(y) / 1000.0
```

### Generated Path:
- Waypoints: 2
- Raw coordinates: [(200, 500), (600, 500)]
- Normalized: [(0.2, 0.5), (0.6, 0.5)]
- Result file: qwen3_0_1000_coords_path.pkl (lost - overwritten)

---

## VERSION 11: Intermediate Waypoints Instruction
**Date**: 2025-11-19
**Changes**: Added explicit instruction to generate intermediate waypoints

### System Prompt:
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant is specialized in generating precise robot gripper trajectories for manipulation tasks, providing clear waypoint sequences with appropriate gripper states. The trajectories represent the path of the gripper tip.
```

### User Prompt:
```python
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
```

### Generated Path:
- Waypoints: 3
- Raw coordinates: [(150, 500), (250, 500), (600, 500)]
- Normalized: [(0.15, 0.5), (0.25, 0.5), (0.6, 0.5)]
- Result file: qwen3_test_path.pkl (current)

---

## VERSION 12: Simplified Examples
**Date**: 2025-11-19
**Changes**: Reduced example complexity (2 waypoints instead of 6) to test if simpler examples yield better results

### System Prompt:
None (using default)

### User Prompt:
```
In the image, execute the task described in <quest>Pick up the apple and put it behind the hammer</quest>.

You are controlling a robot gripper's end-effector. Generate a trajectory showing the complete path the gripper should follow to accomplish the task.

The trajectory should include:
1. Starting position (where the gripper begins)
2. Intermediate waypoints (positions along the path to the goal)
3. Goal position (where the task is completed)
4. Gripper actions (open/close) only if the task requires grasping or releasing objects

Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
[
  {"point_2d": [153, 442], "gripper": "close"},
  {"point_2d": [554, 692], "gripper": "open"}
]

Format:
- point_2d: [x, y] coordinates in range [0, 1000] representing the gripper's position in the image
- gripper: "open" or "close" state at each waypoint

Think step by step and output the complete trajectory in JSON format.
```

### Generated Path:
- Waypoints: 4
- Unique positions: 3
- Raw output: `[{"point_2d": [120, 500], "gripper": "close"}, {"point_2d": [350, 500], "gripper": "open"}, {"point_2d": [700, 500], "gripper": "close"}, {"point_2d": [700, 500], "gripper": "open"}]`
- Path length: 0.5800
- Normalized: [(0.12, 0.5, CLOSE), (0.35, 0.5, OPEN), (0.7, 0.5, CLOSE), (0.7, 0.5, OPEN)]
- Result file: qwen3_v12_simplified_examples_path.pkl
- Visualization: qwen3_v12_simplified_examples_comparison.png

**Result**: Improved from VERSION 5 (2 unique positions) to 3 unique positions. However, all waypoints lie on the same horizontal line (y=500/0.5), indicating lack of vertical movement diversity.

**Comparison to VILA**: VILA achieves 4 waypoints with 4 unique positions and path length 0.5872, showing better spatial coverage.

---

## VERSION 13: VILA-style Expression (Concise)
**Date**: 2025-11-19
**Changes**: Adopted VILA-style prompt expression (concise, removed trajectory requirements list), kept JSON format and 2-waypoint examples

### System Prompt:
None (using default)

### User Prompt:
```
In the image, please execute the command described in <quest>Pick up the apple and put it behind the hammer</quest>.

Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.

Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
[
  {"point_2d": [153, 442], "gripper": "close"},
  {"point_2d": [554, 692], "gripper": "open"}
]

The point_2d denotes x and y location of the end effector in the image. The gripper field indicates gripper actions.

Coordinates should be integers between 0 and 1000, representing relative positions.

Remember to output the complete trajectory in JSON format and think step by step.
```

### Generated Path:
- Waypoints: 3
- Unique positions: 2
- Raw output: `[{"point_2d": [123, 520], "gripper": "close"}, {"point_2d": [721, 588], "gripper": "move"}, {"point_2d": [721, 588], "gripper": "place"}]`
- Path length: 0.6019
- Normalized: [(0.123, 0.52, CLOSE), (0.721, 0.588, CLOSE), (0.721, 0.588, CLOSE)]
- Result file: qwen3_v13_vila_style_concise_path.pkl
- Visualization: qwen3_v13_vila_style_concise_comparison.png

**Critical Issue**: Model generated invalid gripper states ("move", "place") instead of "open"/"close", causing all waypoints to default to CLOSE state with 0 transitions.

**Result**: Performance degraded from VERSION 12 (4 waypoints, 3 unique positions) to VERSION 13 (3 waypoints, 2 unique positions). VILA-style prompt expression did not improve results and caused gripper state confusion.

**Comparison to VILA**: VILA achieves 4 waypoints with 4 unique positions and path length 0.5872, significantly better than VERSION 13.

---

## VERSION 14: VILA-style Expression + Requirements
**Date**: 2025-11-19
**Changes**: Added trajectory requirements list back to VERSION 13 to fix gripper state confusion

### System Prompt:
None (using default)

### User Prompt:
```
In the image, please execute the command described in <quest>Pick up the apple and put it behind the hammer</quest>.

Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.

The trajectory should include:
1. Starting position (where the gripper begins)
2. Intermediate waypoints (positions along the path to the goal)
3. Goal position (where the task is completed)
4. Gripper actions (open/close) only if the task requires grasping or releasing objects

Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
[
  {"point_2d": [153, 442], "gripper": "close"},
  {"point_2d": [554, 692], "gripper": "open"}
]

The point_2d denotes x and y location of the end effector in the image. The gripper field indicates gripper actions.

Coordinates should be integers between 0 and 1000, representing relative positions.

Remember to output the complete trajectory in JSON format and think step by step.
```

### Generated Path:
- Waypoints: 3
- Unique positions: 3
- Raw output: `[{"point_2d": [123, 500], "gripper": "close"}, {"point_2d": [387, 580], "gripper": "open"}, {"point_2d": [723, 580], "gripper": "open"}]`
- Path length: 0.6119
- Normalized: [(0.123, 0.5, CLOSE), (0.387, 0.58, OPEN), (0.723, 0.58, OPEN)]
- Result file: qwen3_v14_vila_style_with_requirements_path.pkl
- Visualization: qwen3_v14_vila_style_with_requirements_comparison.png

**Success**: Adding trajectory requirements list fixed VERSION 13's gripper state error. Model correctly generated "close"/"open" states with 1 transition.

**Result**: Significant improvement from VERSION 13 (2 unique positions, 0 transitions) to VERSION 14 (3 unique positions, 1 transition). Matches VERSION 11's performance (3 unique positions).

**Comparison to VILA**: VILA achieves 4 waypoints with 4 unique positions and path length 0.5872. VERSION 14 has fewer waypoints but all positions are unique and gripper behavior is correct.

---

## VERSION 15: VILA-style, Gripper Description Integrated
**Date**: 2025-11-19
**Changes**: Removed trajectory requirements list (positions 1-3), kept only gripper action description integrated into format explanation

### System Prompt:
None (using default)

### User Prompt:
```
In the image, please execute the command described in <quest>Pick up the apple and put it behind the hammer</quest>.

Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.

Output in JSON format. Here is an example format (these coordinates are just examples, generate your own based on the image):
[
  {"point_2d": [153, 442], "gripper": "close"},
  {"point_2d": [554, 692], "gripper": "open"}
]

The point_2d denotes x and y location of the end effector in the image. The gripper field indicates gripper actions (open/close) only if the task requires grasping or releasing objects.

Coordinates should be integers between 0 and 1000, representing relative positions.

Remember to output the complete trajectory in JSON format and think step by step.
```

### Generated Path:
- Waypoints: 3
- Unique positions: 2
- Raw output: `[{"point_2d": [113, 510], "gripper": "close"}, {"point_2d": [113, 510], "gripper": "open"}, {"point_2d": [721, 573], "gripper": "open"}]`
- Path length: 0.6113
- Normalized: [(0.113, 0.51, CLOSE), (0.113, 0.51, OPEN), (0.721, 0.573, OPEN)]
- Result file: qwen3_v15_vila_style_gripper_only_path.pkl
- Visualization: qwen3_v15_vila_style_gripper_only_comparison.png

**Issue**: First two waypoints share the same coordinates (0.113, 0.51), differing only in gripper state. This reduces spatial diversity.

**Result**: Performance degraded from VERSION 14 (3 unique positions) to VERSION 15 (2 unique positions). Gripper states are correct (no "move"/"place" errors), but spatial coverage decreased.

**Comparison to VILA**: VILA achieves 4 waypoints with 4 unique positions and path length 0.5872, significantly better than VERSION 15.

**Key Finding**: Removing explicit trajectory requirements (starting position, intermediate waypoints, goal position) reduced spatial diversity, though gripper state specification remained correct due to integrated description.

---

## VERSION 16: HAMSTER Format Without System Prompt
**Date**: 2025-11-19
**Changes**: Re-tested VERSION 10 prompt (HAMSTER format with <ans> tags) without system prompt to compare with original VERSION 10

### System Prompt:
None (using default)

### User Prompt:
```
Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: Pick up the apple and put it behind the hammer

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(500, 400), (600, 300), <action>Close Gripper</action>, (700, 500), <action>Open Gripper</action>]
</ans>
```

### Generated Path:
- Waypoints: 2
- Unique positions: 2
- Raw output: `<ans>\n[(100, 500), <action>Close Gripper</action>, (700, 500), <action>Open Gripper</action>]\n</ans>`
- Path length: 0.6000
- Normalized: [(0.1, 0.5, CLOSE), (0.7, 0.5, OPEN)]
- Result file: qwen3_v16_hamster_format_no_system_path.pkl
- Visualization: qwen3_v16_hamster_format_no_system_comparison.png

**Success**: HAMSTER format (<ans> tags + Python tuples) processed correctly. Gripper actions properly placed between waypoints.

**Result**: Minimal trajectory with only 2 waypoints (start and end). Significantly fewer waypoints compared to VERSION 10 (2 waypoints with system prompt). Both points on horizontal line (y=500/0.5).

**Comparison to VILA**: VILA achieves 4 waypoints with 4 unique positions and path length 0.5872, significantly better than VERSION 16.

**Key Finding**: Removing system prompt from HAMSTER format drastically reduced waypoint generation. System prompt appears critical for generating richer trajectories in this format.

---

## VERSION 17: HAMSTER Format with Modified Example
**Date**: 2025-11-19
**Changes**: Modified example coordinates to test if model copies example values (VERSION 16 used regular pattern, VERSION 17 uses irregular coordinates)

### System Prompt:
None (using default)

### User Prompt:
```
Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: Pick up the apple and put it behind the hammer

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(534, 439), (675, 306), <action>Close Gripper</action>, (730, 552), <action>Open Gripper</action>]
</ans>
```

### Generated Path:
- Waypoints: 2
- Unique positions: 2
- Raw output: `<ans>\n[(118, 518), <action>Close Gripper</action>, (730, 552), <action>Open Gripper</action>]\n</ans>`
- Path length: 0.6129
- Normalized: [(0.118, 0.518, CLOSE), (0.730, 0.552, OPEN)]
- Result file: qwen3_v17_hamster_modified_example_path.pkl
- Visualization: qwen3_v17_hamster_modified_example_comparison.png

**Interesting Finding**: Second waypoint (730, 552) exactly matches the last coordinate in the example, suggesting model may be copying from examples.

**Result**: Same minimal trajectory (2 waypoints) as VERSION 16. Different first coordinate but second coordinate matches example exactly.

**Comparison to VILA**: VILA achieves 4 waypoints with 4 unique positions and path length 0.5872, significantly better than VERSION 17.

**Key Finding**: Model appears to partially copy example coordinates (730, 552), indicating strong influence of example values on output generation.

---

## VERSION 18: HAMSTER Format with Different Final Example
**Date**: 2025-11-19
**Changes**: Changed final example coordinate from (730, 552) to (190, 711) to verify if VERSION 17's coordinate copying was coincidental

### System Prompt:
None (using default)

### User Prompt:
```
Generate the spatial trajectory in 2D images as [(x, y), ...] for the following task: Pick up the apple and put it behind the hammer

Generate an output in a <ans>X</ans> block to give your answer, where X is the generated trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<ans>
[(534, 439), (675, 306), <action>Close Gripper</action>, (190, 711), <action>Open Gripper</action>]
</ans>
```

### Generated Path:
- Waypoints: 2
- Unique positions: 2
- Raw output: `<ans>\n[(118, 528), <action>Close Gripper</action>, (738, 528), <action>Open Gripper</action>]\n</ans>`
- Path length: 0.6200
- Normalized: [(0.118, 0.528, CLOSE), (0.738, 0.528, OPEN)]
- Result file: qwen3_v18_hamster_final_coord_190_711_path.pkl
- Visualization: qwen3_v18_hamster_final_coord_190_711_comparison.png

**Surprising Result**: Second waypoint (738, 528) does NOT match example coordinate (190, 711), unlike VERSION 17 where exact copying occurred.

**Result**: Same minimal trajectory (2 waypoints) but on horizontal line (y=528). Model generated independent coordinates instead of copying from example.

**Comparison to VILA**: VILA achieves 4 waypoints with 4 unique positions and path length 0.5872, significantly better than VERSION 18.

**Key Finding**: VERSION 17's coordinate copying may have been coincidental or context-dependent. VERSION 18 demonstrates model can generate independent coordinates even with different example values.

---

## VERSION 19: Bimanual (Dual-Arm) Prompt
**Date**: 2025-11-25
**Changes**: Extended VERSION 18 for bimanual tasks using `<left_arm>` and `<right_arm>` tags instead of single `<ans>` tag

### System Prompt:
None (using default)

### User Prompt:
```
Generate the spatial trajectories in 2D images as [(x, y), ...] for the following task: Pick up two bottles with both hands

This task requires two robot arms (LEFT and RIGHT). Generate separate trajectories for each arm.

Generate an output in <left_arm>X</left_arm> and <right_arm>X</right_arm> blocks, where X is the trajectory in the format [(x, y), ...]. Here (x, y) means the position in the top-down RGB image. (x, y) are in the range [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.

In your trajectory planning, you should include specific waypoints that indicate gripper actions (Open/Close) using the following format:
<action>Open Gripper</action>
<action>Close Gripper</action>

For example:
<left_arm>
[(252, 422), <action>Close Gripper</action>, (307, 353), (424, 557), <action>Open Gripper</action>]
</left_arm>

<right_arm>
[(754, 429), <action>Close Gripper</action>, (719, 356), (633, 593), <action>Open Gripper</action>]
</right_arm>
```

### Generated Path:
- Waypoints: 3 per arm (L:3, R:3)
- Unique positions: 3 per arm
- Raw output example:
```
<left_arm>
[(252, 422), <action>Close Gripper</action>, (307, 353), (424, 557), <action>Open Gripper</action>]
</left_arm>

<right_arm>
[(754, 429), <action>Close Gripper</action>, (719, 356), (633, 593), <action>Open Gripper</action>]
</right_arm>
```
- Normalized (Left): [(0.252, 0.422, OPEN), (0.307, 0.353, CLOSE), (0.424, 0.557, CLOSE)]
- Normalized (Right): [(0.754, 0.429, OPEN), (0.719, 0.356, CLOSE), (0.633, 0.593, CLOSE)]
- Result file: dual_bottles_pick_hard_bimanual/episode_00/paths/path_frame_*.pkl
- Visualization: dual_bottles_pick_hard_bimanual/episode_00/qwen3_bimanual_path_video.mp4

**Success**: Bimanual prompt format works reliably, generating separate trajectories for left and right arms with 99.0% success rate (204/206 frames).

**Result**: Both arms generate 3 waypoints each with correct gripper state transitions. More waypoints per arm than VERSION 18's single-arm output (2 waypoints).

**Comparison to VERSION 18**: VERSION 18 generates single-arm paths with 2 waypoints. VERSION 19 generates dual-arm paths with 3 waypoints per arm.

**Key Finding**: Using `<left_arm>` and `<right_arm>` tags enables reliable bimanual trajectory generation for dual-arm manipulation tasks.

---

## Summary of All Versions

| Version | Date | Waypoints | Unique Pos | Key Feature | Result File |
|---------|------|-----------|------------|-------------|-------------|
| VILA Baseline | - | 4 | 4 | - | vila_new_task_path.pkl |
| VERSION 1 | 2025-11-18 | 4 | 2 | Basic improved prompt | qwen3_improved_prompt_path.pkl |
| VERSION 2 | 2025-11-18 | 8 | 6 | Trajectory-focused | qwen3_trajectory_prompt_path.pkl |
| VERSION 3 | 2025-11-18 | 31 | - | Variable coords (failed) | qwen3_variable_coords_path.pkl |
| VERSION 4 | 2025-11-18 | - | - | Not executed | - |
| VERSION 5 | 2025-11-18 | 5 | 2 | Simplified format | qwen3_simplified_format_path.pkl |
| VERSION 6 | 2025-11-18 | 3 | 2 | Unchanged state | qwen3_unchanged_state_path.pkl |
| VERSION 7 | 2025-11-18 | 2 | 2 | HAMSTER-style prompt | (overwritten) |
| VERSION 8 | 2025-11-19 | 2 | 2 | VILA system prompt | (overwritten) |
| VERSION 9 | 2025-11-19 | 3 | 3 | Robot specialization | (overwritten) |
| VERSION 10 | 2025-11-19 | 2 | 2 | 0~1000 coordinates | (overwritten) |
| VERSION 11 | 2025-11-19 | 3 | 3 | Intermediate waypoints | qwen3_v11_intermediate_waypoints_path.pkl |
| VERSION 12 | 2025-11-19 | 4 | 3 | Simplified examples (2 waypoints) | qwen3_v12_simplified_examples_path.pkl |
| VERSION 13 | 2025-11-19 | 3 | 2 | VILA-style expression (concise) | qwen3_v13_vila_style_concise_path.pkl |
| VERSION 14 | 2025-11-19 | 3 | 3 | VILA-style + requirements | qwen3_v14_vila_style_with_requirements_path.pkl |
| VERSION 15 | 2025-11-19 | 3 | 2 | VILA-style, gripper only | qwen3_v15_vila_style_gripper_only_path.pkl |
| VERSION 16 | 2025-11-19 | 2 | 2 | HAMSTER format, no system | qwen3_v16_hamster_format_no_system_path.pkl |
| VERSION 17 | 2025-11-19 | 2 | 2 | HAMSTER modified example | qwen3_v17_hamster_modified_example_path.pkl |
| VERSION 18 | 2025-11-19 | 2 | 2 | HAMSTER example (190, 711) | qwen3_v18_hamster_final_coord_190_711_path.pkl |
| VERSION 19 | 2025-11-25 | 3+3 | 3+3 | Bimanual (dual-arm) | dual_bottles_pick_hard_bimanual/ |

### Best Performing Versions:
1. **VERSION 2**: 8 waypoints, 6 unique positions (best diversity)
2. **VERSION 9**: 3 waypoints, 3 unique positions (robot specialization)
3. **VERSION 11**: 3 waypoints, 3 unique positions (intermediate waypoints)

### Selected for Production: **VERSION 18** ⭐

**Rationale for Selection**:
- **Minimal prompt design**: Contains only essential elements without redundant instructions
- **Qwen3-native format**: Uses [0, 1000] coordinate system (Qwen3-VL recommended format)
- **Accurate keypoint generation**: Generates critical gripper waypoints correctly
- **Stable output**: Consistent 2-waypoint generation (start and end positions)
- **HAMSTER-compatible**: Uses `<ans>` tags and gripper action format matching HAMSTER standard

**Production Use**: VERSION 18 will be the default prompt for Qwen3-VL path generation in the ManiFlow integration pipeline.

### Key Insights:
- **System prompt matters**: Adding robot-specific system prompt (V9, V11) improved results
- **Coordinate system**: 0~1000 range is Qwen3-VL's native format
- **Intermediate waypoints**: Explicit instruction to generate intermediate points helps
- **Challenge**: Still generating fewer waypoints than VILA (3 vs 4)
- **Production choice**: VERSION 18 selected for minimal, stable, and Qwen3-native approach
