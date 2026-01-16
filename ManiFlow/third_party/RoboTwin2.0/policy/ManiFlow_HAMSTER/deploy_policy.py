"""
Deploy policy interface for ManiFlow + HAMSTER evaluation on RoboTwin 2.0.

This module provides the required interface functions for RoboTwin 2.0's
eval_policy.py script:
- get_model(): Load the model
- eval(): Run one step of evaluation
- reset_model(): Reset state for new episode
- encode_obs(): Convert observation to model input format

Supports 3 evaluation modes:
- original: Raw RGB images (no VILA, conditions 1,4)
- current: Current overlay only (conditions 2,5)
- initial_current: Initial + current overlay (conditions 3,6)

Key design:
- Path generation every 16 steps (action horizon)
- initial_overlay: Fixed at episode start (initial_current mode only)
- current_overlay: Updated every 16 steps with new path
- Retry up to 2 times on path generation failure
- Fallback to last successful path or no-path image
"""

import os
import sys
import numpy as np
import time
import cv2
from pathlib import Path
from typing import Optional, Dict, Any

# Add paths for imports
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, 'ManiFlow'))
sys.path.append(os.path.join(parent_directory, 'hamster'))
sys.path.append(os.path.join(parent_directory, 'utils'))

# Add ManiFlow root for imports
MANIFLOW_ROOT = Path(parent_directory).parent.parent.parent.parent / "ManiFlow" / "ManiFlow"
sys.path.insert(0, str(MANIFLOW_ROOT))

from maniflow_policy import ManiFlowPolicy
from vila_client import VILAClient, get_task_instruction
from overlay_utils import OverlayDrawer
from path_manager import PathManager, PathManagerOriginal
from metrics_logger import MetricsLogger


# Evaluation modes
MODE_ORIGINAL = "original"
MODE_CURRENT = "current"
MODE_INITIAL_CURRENT = "initial_current"

VALID_MODES = [MODE_ORIGINAL, MODE_CURRENT, MODE_INITIAL_CURRENT]

# Task instruction mapping (updated with adjust_bottle)
SINGLE_ARM_INSTRUCTIONS = {
    "beat_block_hammer": "Pick up the hammer and beat the block",
    "click_bell": "click the bell's top center on the table",
    "move_can_pot": "pick up the can and move it to beside the pot",
    "open_microwave": "open the microwave",
    "turn_switch": "click the switch",
    "adjust_bottle": "adjust the bottle to the correct orientation",
}


def get_task_instruction_extended(task_name: str) -> str:
    """Get task instruction for a given task name."""
    return SINGLE_ARM_INSTRUCTIONS.get(
        task_name,
        f"Complete the {task_name} task"
    )


class ObsEncoder:
    """
    Observation encoder that manages path generation and overlay creation.

    Supports 3 modes:
    - original: No path generation, raw RGB
    - current: Only current overlay
    - initial_current: Initial + current overlay
    """

    def __init__(
        self,
        mode: str,
        path_manager,
        overlay_drawer: Optional[OverlayDrawer],
        action_horizon: int = 16,
    ):
        """
        Initialize the observation encoder.

        Args:
            mode: Evaluation mode (original, current, initial_current)
            path_manager: PathManager or PathManagerOriginal instance
            overlay_drawer: OverlayDrawer instance (None for original mode)
            action_horizon: Number of steps between path generations
        """
        assert mode in VALID_MODES, f"Invalid mode: {mode}"

        self.mode = mode
        self.path_manager = path_manager
        self.overlay_drawer = overlay_drawer
        self.action_horizon = action_horizon

        # State
        self.initial_overlay = None
        self.current_path = None
        self.step_count = 0

        # Timing
        self.maniflow_inference_times_ms = []

    def reset(self):
        """Reset state for new episode."""
        self.initial_overlay = None
        self.current_path = None
        self.step_count = 0
        self.maniflow_inference_times_ms = []
        self.path_manager.reset()

    def get_path_stats(self) -> Dict[str, Any]:
        """Get path generation statistics."""
        return self.path_manager.get_stats()

    def get_timing(self) -> Dict[str, Any]:
        """Get timing information."""
        stats = self.path_manager.get_stats()
        return {
            "vila_inference_ms": stats.get("vila_inference_times_ms", []),
            "maniflow_inference_ms": self.maniflow_inference_times_ms,
        }

    def _normalize_image(self, rgb: np.ndarray) -> np.ndarray:
        """Normalize RGB image to (C, H, W) float32 [0, 1]."""
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        return np.transpose(normalized, (2, 0, 1))

    def encode_obs(self, observation: dict, task_instruction: str) -> dict:
        """
        Convert RoboTwin observation to ManiFlow input format.

        Args:
            observation: RoboTwin observation dict
            task_instruction: Task instruction for path generation

        Returns:
            Dict with observation tensors based on mode:
            - original: {'image', 'agent_pos'}
            - current: {'current_overlay', 'agent_pos'}
            - initial_current: {'initial_overlay', 'current_overlay', 'agent_pos'}
        """
        # Get RGB image from observation
        rgb = observation['observation']['head_camera']['rgb']  # (H, W, 3) uint8

        # Get agent position
        agent_pos = observation['joint_action']['vector'].astype(np.float32)

        # Original mode: just normalize the image
        if self.mode == MODE_ORIGINAL:
            self.step_count += 1
            return {
                'image': self._normalize_image(rgb),
                'agent_pos': agent_pos,
            }

        # Overlay modes: generate path at start of each action sequence
        if self.step_count % self.action_horizon == 0:
            is_frame0 = (self.step_count == 0)

            path, success, used_fallback = self.path_manager.generate_path(
                rgb,
                task_instruction,
                is_frame0=is_frame0,
            )

            if path is not None:
                self.current_path = path
            else:
                # No path - will use no-path image
                self.current_path = None

            # Initial overlay: set at episode start (initial_current mode)
            # or when first successful path is generated
            if self.mode == MODE_INITIAL_CURRENT:
                if self.initial_overlay is None:
                    if self.current_path:
                        self.initial_overlay = self.overlay_drawer.draw(rgb, self.current_path)
                    else:
                        # No path available at frame 0 - use no-path image
                        self.initial_overlay = self._normalize_image(rgb)

        # Current overlay: current RGB with current path
        if self.current_path:
            current_overlay = self.overlay_drawer.draw(rgb, self.current_path)
        else:
            current_overlay = self._normalize_image(rgb)

        # Update initial_overlay if path becomes available later
        if self.mode == MODE_INITIAL_CURRENT:
            if self.path_manager.has_initial_path() and self.initial_overlay is None:
                initial_path = self.path_manager.get_initial_path()
                if initial_path:
                    # Re-draw initial overlay with first successful path
                    # Note: We use current rgb as approximation since we don't store frame 0
                    self.initial_overlay = self.overlay_drawer.draw(rgb, initial_path)

        self.step_count += 1

        # Return based on mode
        if self.mode == MODE_CURRENT:
            return {
                'current_overlay': current_overlay,
                'agent_pos': agent_pos,
            }
        else:  # MODE_INITIAL_CURRENT
            return {
                'initial_overlay': self.initial_overlay,
                'current_overlay': current_overlay,
                'agent_pos': agent_pos,
            }


class ManiFlowModel:
    """
    Combined model wrapper for evaluation.

    Holds ManiFlow policy, observation encoder, and metrics logger.
    """

    def __init__(
        self,
        policy: ManiFlowPolicy,
        obs_encoder: ObsEncoder,
        task_name: str,
        mode: str,
        metrics_logger: Optional[MetricsLogger] = None,
    ):
        self.policy = policy
        self.obs_encoder = obs_encoder
        self.task_name = task_name
        self.mode = mode
        self.task_instruction = get_task_instruction_extended(task_name)
        self.metrics_logger = metrics_logger

        # For compatibility with original interface
        self.env_runner = policy.env_runner

        # Episode tracking
        self.episode_id = 0
        self.episode_start_time = None
        self.episode_steps = 0

        # For eval_policy.py video saving compatibility
        self.run_dir = metrics_logger.get_run_dir() if metrics_logger else None
        self.epoch = 500  # We use epoch 500 checkpoints

    def start_episode(self):
        """Mark the start of an episode."""
        self.episode_start_time = time.time()
        self.episode_steps = 0

    def update_obs(self, obs: dict):
        """Update observation history."""
        self.policy.update_obs(obs)
        self.episode_steps += 1

    def get_action(self) -> np.ndarray:
        """Get action sequence from policy."""
        start_time = time.time()
        actions = self.policy.get_action()
        inference_time_ms = (time.time() - start_time) * 1000
        self.obs_encoder.maniflow_inference_times_ms.append(inference_time_ms)
        return actions

    def end_episode(self, success: bool, failure_reason: Optional[str] = None):
        """
        End the current episode and log metrics.

        Args:
            success: Whether the episode succeeded
            failure_reason: Reason for failure (if failed)
        """
        if self.metrics_logger is None:
            self.episode_id += 1
            return

        total_episode_ms = (time.time() - self.episode_start_time) * 1000

        # Get timing info
        timing = self.obs_encoder.get_timing()
        timing["total_episode_ms"] = total_episode_ms

        # Get path stats
        path_stats = self.obs_encoder.get_path_stats()

        # Create and log metrics
        metrics = self.metrics_logger.create_episode_metrics(
            episode_id=self.episode_id,
            success=success,
            total_steps=self.episode_steps,
            path_stats=path_stats,
            timing=timing,
            failure_reason=failure_reason,
        )
        self.metrics_logger.log_episode(metrics)

        self.episode_id += 1

        # Mark episode as ended to prevent duplicate logging in reset_model()
        self.episode_start_time = None


# =============================================================================
# Required interface functions for RoboTwin 2.0
# =============================================================================

def get_model(usr_args: dict) -> ManiFlowModel:
    """
    Load model from checkpoint and initialize components.

    Args:
        usr_args: User arguments from deploy_policy.yml

    Returns:
        ManiFlowModel wrapper
    """
    # Get parameters from usr_args
    task_name = usr_args['task_name']
    mode = usr_args.get('mode', MODE_INITIAL_CURRENT)
    env = usr_args.get('env', 'cluttered')  # Training data environment
    eval_env = usr_args.get('eval_env', 'cluttered')  # Evaluation environment
    checkpoint_path = usr_args.get('checkpoint_path', None)
    seed = usr_args.get('seed', 42)

    # VILA server configuration (only needed for overlay modes)
    vila_server_url = usr_args.get('vila_server_url', 'http://localhost:8000/v1')
    vila_model = usr_args.get('vila_model', 'HAMSTER_dev')

    # Output configuration
    output_dir = usr_args.get('output_dir', './eval_results')

    # Validate mode
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {VALID_MODES}")

    # Construct condition string
    if mode == MODE_ORIGINAL:
        mode_str = "original"
    elif mode == MODE_CURRENT:
        mode_str = "overlay_current"
    else:
        mode_str = "overlay_initial_current"

    condition_num = {
        ("cluttered", MODE_ORIGINAL): 1,
        ("cluttered", MODE_CURRENT): 2,
        ("cluttered", MODE_INITIAL_CURRENT): 3,
        ("clean", MODE_ORIGINAL): 4,
        ("clean", MODE_CURRENT): 5,
        ("clean", MODE_INITIAL_CURRENT): 6,
    }.get((env, mode), 0)

    condition = f"condition{condition_num}_{env}_{mode_str}_eval{eval_env}"

    # If checkpoint path not specified, try to find it
    if checkpoint_path is None:
        base_dir = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs")
        checkpoint_path = find_checkpoint(base_dir, task_name, env, mode)

    print(f"[get_model] Loading checkpoint: {checkpoint_path}")
    print(f"[get_model] Task: {task_name}")
    print(f"[get_model] Mode: {mode}")
    print(f"[get_model] Train Env: {env}")
    print(f"[get_model] Eval Env: {eval_env}")
    print(f"[get_model] Condition: {condition}")

    # Load ManiFlow policy
    policy = ManiFlowPolicy(checkpoint_path, mode=mode)

    # Initialize components based on mode
    if mode == MODE_ORIGINAL:
        # No VILA, no path generation
        path_manager = PathManagerOriginal()
        overlay_drawer = None
        print("[get_model] Original mode: VILA not required")
    else:
        # Initialize VILA client
        vila_client = VILAClient(
            server_url=vila_server_url,
            model_name=vila_model,
        )

        # Check VILA server
        if not vila_client.check_server():
            print("[get_model] WARNING: VILA server not responding!")
            print("[get_model] Make sure to start the server before evaluation.")

        # Initialize path manager with retry
        path_manager = PathManager(
            vila_client=vila_client,
            max_retries=2,
            verbose=True,
        )

        # Initialize overlay drawer
        overlay_drawer = OverlayDrawer(target_size=(224, 224))

        print(f"[get_model] VILA server: {vila_server_url}")

    # Initialize observation encoder
    obs_encoder = ObsEncoder(
        mode=mode,
        path_manager=path_manager,
        overlay_drawer=overlay_drawer,
        action_horizon=policy.n_action_steps,  # 16
    )

    # Initialize metrics logger
    metrics_logger = MetricsLogger(
        output_dir=output_dir,
        task=task_name,
        condition=condition,
        seed=seed,
        train_env=env,
        eval_env=eval_env,
    )
    print(f"[get_model] Results will be saved to: {metrics_logger.get_run_dir()}")

    # Create combined model
    model = ManiFlowModel(
        policy=policy,
        obs_encoder=obs_encoder,
        task_name=task_name,
        mode=mode,
        metrics_logger=metrics_logger,
    )

    return model


def eval(TASK_ENV, model: ManiFlowModel, observation: dict):
    """
    Run one evaluation step: generate actions and execute.

    This function is called in a loop by eval_policy.py.
    Generates 16 actions at once and executes them sequentially.

    Args:
        TASK_ENV: RoboTwin environment
        model: ManiFlowModel wrapper
        observation: Current observation from environment
    """
    # Start episode timing on first call
    if model.episode_start_time is None:
        model.start_episode()

    # Encode observation (may generate new path)
    obs = model.obs_encoder.encode_obs(observation, model.task_instruction)

    # Initialize observation history if empty
    if len(model.env_runner.obs) == 0:
        model.update_obs(obs)

    # Get action sequence (16 steps)
    actions = model.get_action()

    # Execute each action
    for action in actions:
        TASK_ENV.take_action(action)

        # Get new observation
        observation = TASK_ENV.get_obs()

        # Encode and update (may generate new path every 16 steps)
        obs = model.obs_encoder.encode_obs(observation, model.task_instruction)
        model.update_obs(obs)

        # Check for success
        if TASK_ENV.eval_success:
            model.end_episode(success=True)
            break


def reset_model(model: ManiFlowModel):
    """
    Reset model state for new episode.

    Args:
        model: ManiFlowModel wrapper
    """
    # End previous episode if not already ended
    if model.episode_start_time is not None:
        # Episode ended without success
        model.end_episode(success=False, failure_reason="timeout_or_failure")

    # Reset policy state
    model.policy.reset()

    # Reset observation encoder (clears initial_overlay and path)
    model.obs_encoder.reset()

    # Prepare for next episode
    model.episode_start_time = None
    model.episode_steps = 0

    print(f"[reset_model] Model state reset for episode {model.episode_id}")


def encode_obs(observation: dict) -> dict:
    """
    Encode observation (standalone version).

    Note: This is provided for compatibility but the main encoding
    logic is in ObsEncoder which requires mode-specific handling.

    Args:
        observation: RoboTwin observation

    Returns:
        Partially encoded observation
    """
    rgb = observation['observation']['head_camera']['rgb']
    agent_pos = observation['joint_action']['vector'].astype(np.float32)

    return {
        'rgb': rgb,
        'agent_pos': agent_pos,
    }


def finalize_evaluation(model: ManiFlowModel):
    """
    Finalize evaluation and save summary.

    Should be called after all episodes are complete.

    Args:
        model: ManiFlowModel wrapper
    """
    if model.metrics_logger is not None:
        summary_path = model.metrics_logger.save_summary()
        print(f"[finalize_evaluation] Summary saved to: {summary_path}")


# =============================================================================
# Utility functions
# =============================================================================

def find_checkpoint(
    base_dir: Path,
    task_name: str,
    env: str,
    mode: str,
) -> str:
    """
    Find the epoch 500 checkpoint for a task/env/mode combination.

    Args:
        base_dir: Base directory for outputs
        task_name: Task name
        env: Training environment (clean or cluttered)
        mode: Evaluation mode

    Returns:
        Path to checkpoint
    """
    import glob

    # Map mode to directory name
    mode_dir_map = {
        MODE_ORIGINAL: "original",
        MODE_CURRENT: "overlay_current",
        MODE_INITIAL_CURRENT: "overlay_initial_current",
    }
    mode_dir = mode_dir_map.get(mode, mode)

    # Search pattern for epoch 500
    pattern = str(base_dir / f"{env}_{task_name}" / f"{mode_dir}_seed*" / "checkpoints" / "epoch=0500-*.ckpt")
    matches = glob.glob(pattern)

    if matches:
        return matches[0]

    # Fallback: try latest.ckpt
    pattern = str(base_dir / f"{env}_{task_name}" / f"{mode_dir}_seed*" / "checkpoints" / "latest.ckpt")
    matches = glob.glob(pattern)

    if matches:
        print(f"[find_checkpoint] WARNING: epoch=0500 not found, using latest.ckpt")
        return matches[0]

    # Fallback: broader search
    pattern = str(base_dir / "*" / f"*{task_name}*" / "checkpoints" / "*.ckpt")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for task={task_name}, env={env}, mode={mode}"
        )

    matches.sort(key=os.path.getmtime, reverse=True)
    print(f"[find_checkpoint] WARNING: Using fallback checkpoint: {matches[0]}")
    return matches[0]


if __name__ == "__main__":
    print("=" * 60)
    print("ManiFlow Deploy Policy Test")
    print("=" * 60)

    # Test configuration for each mode
    test_configs = [
        {"mode": MODE_ORIGINAL, "needs_vila": False},
        {"mode": MODE_CURRENT, "needs_vila": True},
        {"mode": MODE_INITIAL_CURRENT, "needs_vila": True},
    ]

    for config in test_configs:
        mode = config["mode"]
        print(f"\n--- Testing {mode} mode ---")

        test_args = {
            'task_name': 'click_bell',
            'mode': mode,
            'env': 'cluttered',
            'vila_server_url': 'http://localhost:8000/v1',
            'vila_model': 'HAMSTER_dev',
            'output_dir': '/tmp/eval_test',
            'seed': 42,
        }

        try:
            if config["needs_vila"]:
                # Check if VILA is running
                from vila_client import VILAClient
                client = VILAClient()
                if not client.check_server():
                    print(f"   [SKIP] VILA server not running")
                    continue

            model = get_model(test_args)
            print(f"   Model loaded successfully!")
            print(f"   - n_obs_steps: {model.policy.n_obs_steps}")
            print(f"   - n_action_steps: {model.policy.n_action_steps}")

            reset_model(model)
            print(f"   Reset successful!")
            print(f"   [PASS] {mode} mode works!")

        except FileNotFoundError as e:
            print(f"   [SKIP] {e}")
        except Exception as e:
            print(f"   [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
