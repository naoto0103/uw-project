"""
ManiFlow Policy wrapper for evaluation.

This module provides a wrapper class for loading and running ManiFlow policies
with support for 3 evaluation modes:
- original: Raw RGB images (conditions 1,4)
- current: Current overlay only (conditions 2,5)
- initial_current: Initial + current overlay (conditions 3,6)

Reference: ManiFlow_Policy_robotwin2.0/RoboTwin/policy/ManiFlow/ManiFlow/maniflow_policy.py
"""

import os
import sys
import pathlib
import copy

# Add ManiFlow root to sys.path for hydra to find maniflow package
MANIFLOW_ROOT = pathlib.Path(__file__).parent.parent.parent / "ManiFlow" / "ManiFlow"
if str(MANIFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(MANIFLOW_ROOT))

import torch
import numpy as np
import dill
from collections import deque
from omegaconf import OmegaConf

# Register eval resolver for OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)


# Evaluation modes
MODE_ORIGINAL = "original"
MODE_CURRENT = "current"
MODE_INITIAL_CURRENT = "initial_current"


class ObsRunner:
    """
    Observation runner for ManiFlow policy.

    Manages observation history and converts observations to policy input format.
    Supports 3 modes with different observation keys.
    """

    def __init__(self, n_obs_steps: int = 2, mode: str = MODE_INITIAL_CURRENT, task_name: str = None):
        """
        Initialize the observation runner.

        Args:
            n_obs_steps: Number of observation steps to stack
            mode: Evaluation mode (original, current, initial_current)
            task_name: Name of the task
        """
        self.n_obs_steps = n_obs_steps
        self.mode = mode
        self.task_name = task_name
        self.obs = deque(maxlen=n_obs_steps + 1)

    def reset_obs(self):
        """Clear observation history."""
        self.obs.clear()

    def update_obs(self, current_obs: dict):
        """
        Add new observation to history.

        Args:
            current_obs: Dictionary with observation tensors based on mode
        """
        self.obs.append(current_obs)

    def stack_last_n_obs(self, all_obs: list, n_steps: int) -> np.ndarray:
        """
        Stack last n observations, padding with first observation if needed.

        Args:
            all_obs: List of observations
            n_steps: Number of steps to stack

        Returns:
            Stacked observations array
        """
        assert len(all_obs) > 0, "No observations to stack"
        all_obs = list(all_obs)

        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # Pad with first observation
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type: {type(all_obs[0])}")

        return result

    def get_n_steps_obs(self) -> dict:
        """
        Get stacked observations for policy input.

        Returns:
            Dictionary with stacked observations
        """
        assert len(self.obs) > 0, "No observation recorded. Call update_obs first."

        result = {}
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )
        return result

    @torch.no_grad()
    def get_action(self, policy, observation: dict = None) -> np.ndarray:
        """
        Get action from policy given observation.

        Args:
            policy: ManiFlow policy
            observation: Current observation dict (optional, uses history if None)

        Returns:
            Action array of shape (n_action_steps, action_dim)
        """
        device = policy.device
        dtype = policy.dtype

        if observation is not None:
            self.obs.append(observation)

        obs = self.get_n_steps_obs()

        # Convert to torch tensors
        obs_dict = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs_dict[key] = torch.from_numpy(val).to(device=device)
            else:
                obs_dict[key] = val.to(device=device)

        # Build input dict based on mode
        obs_dict_input = {}
        for key, val in obs_dict.items():
            obs_dict_input[key] = val.unsqueeze(0)

        # Run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict_input)

        # Convert to numpy
        action = action_dict['action'].detach().cpu().numpy().squeeze(0)
        return action


class ManiFlowPolicy:
    """
    ManiFlow policy wrapper for evaluation.

    Loads a trained ManiFlow checkpoint and provides interface for
    evaluation with RoboTwin 2.0. Supports 3 evaluation modes.
    """

    def __init__(self, checkpoint_path: str, mode: str = MODE_INITIAL_CURRENT, device: str = "cuda"):
        """
        Initialize the ManiFlow policy.

        Args:
            checkpoint_path: Path to the trained checkpoint (.ckpt)
            mode: Evaluation mode (original, current, initial_current)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.mode = mode

        # Load checkpoint and create policy
        self.policy, self.cfg = self._load_checkpoint(checkpoint_path)
        self.policy.eval()
        self.policy.to(self.device)

        # Get model parameters from config
        self.n_obs_steps = self.cfg.n_obs_steps
        self.n_action_steps = self.cfg.n_action_steps
        self.horizon = self.cfg.horizon

        # Create observation runner
        self.env_runner = ObsRunner(
            n_obs_steps=self.n_obs_steps,
            mode=mode,
            task_name=self.cfg.get('task_name', 'unknown')
        )

        # Timestep counter
        self.t = 0

        print(f"[ManiFlowPolicy] Loaded checkpoint: {checkpoint_path}")
        print(f"[ManiFlowPolicy] Mode: {mode}")
        print(f"[ManiFlowPolicy] n_obs_steps={self.n_obs_steps}, "
              f"n_action_steps={self.n_action_steps}, horizon={self.horizon}")

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and create policy.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Tuple of (policy, cfg)
        """
        # Load checkpoint
        payload = torch.load(checkpoint_path, pickle_module=dill, map_location='cpu')
        cfg = payload['cfg']

        # Import and instantiate policy
        import hydra
        policy = hydra.utils.instantiate(cfg.policy)

        # Load state dict (handle both old and new checkpoint formats)
        state_dicts = payload.get('state_dicts', payload)

        if cfg.training.use_ema and 'ema_model' in state_dicts:
            policy.load_state_dict(state_dicts['ema_model'])
            print("[ManiFlowPolicy] Loaded EMA model weights")
        elif 'model' in state_dicts:
            policy.load_state_dict(state_dicts['model'])
            print("[ManiFlowPolicy] Loaded model weights")
        elif 'state_dict' in state_dicts:
            policy.load_state_dict(state_dicts['state_dict'])
            print("[ManiFlowPolicy] Loaded model weights (legacy format)")
        else:
            raise KeyError(f"Cannot find model weights in checkpoint. Keys: {list(state_dicts.keys())}")

        # Set normalizer if available (check both locations)
        normalizer = payload.get('normalizer') or payload.get('pickles', {}).get('normalizer')
        if normalizer is not None:
            policy.set_normalizer(normalizer)

        return policy, cfg

    def reset(self):
        """Reset state for new episode."""
        self.env_runner.reset_obs()
        self.t = 0

    def update_obs(self, observation: dict):
        """
        Update observation history.

        Args:
            observation: Dict with observation tensors based on mode
        """
        self.env_runner.update_obs(observation)

    def get_action(self, observation: dict = None) -> np.ndarray:
        """
        Get action sequence from policy.

        Args:
            observation: Current observation (optional)

        Returns:
            Action array of shape (n_action_steps, action_dim)
        """
        actions = self.env_runner.get_action(self.policy, observation)
        self.t += 1
        return actions


def load_maniflow_policy(
    checkpoint_path: str,
    mode: str = MODE_INITIAL_CURRENT,
    device: str = "cuda"
) -> ManiFlowPolicy:
    """
    Convenience function to load ManiFlow policy.

    Args:
        checkpoint_path: Path to checkpoint
        mode: Evaluation mode
        device: Device to run on

    Returns:
        ManiFlowPolicy wrapper
    """
    return ManiFlowPolicy(checkpoint_path, mode, device)


if __name__ == "__main__":
    print("=" * 50)
    print("ManiFlowPolicy Test")
    print("=" * 50)

    # Test for each mode
    test_modes = [
        (MODE_ORIGINAL, {'image': (3, 224, 224), 'agent_pos': (14,)}),
        (MODE_CURRENT, {'current_overlay': (3, 224, 224), 'agent_pos': (14,)}),
        (MODE_INITIAL_CURRENT, {'initial_overlay': (3, 224, 224), 'current_overlay': (3, 224, 224), 'agent_pos': (14,)}),
    ]

    # Test checkpoint path (update this)
    test_ckpt = "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/2025.12.14/22.38.04_train_maniflow_overlay_image_policy_zarr_move_can_pot/checkpoints/latest.ckpt"

    if os.path.exists(test_ckpt):
        for mode, obs_shapes in test_modes:
            print(f"\n--- Testing {mode} mode ---")

            try:
                model = load_maniflow_policy(test_ckpt, mode=mode)

                print(f"1. Creating dummy observation...")
                dummy_obs = {}
                for key, shape in obs_shapes.items():
                    dummy_obs[key] = np.random.randn(*shape).astype(np.float32)

                # First observation (will be padded)
                model.update_obs(dummy_obs)
                actions = model.get_action()

                print(f"   Action shape: {actions.shape}")
                print(f"   Action range: [{actions.min():.3f}, {actions.max():.3f}]")
                print(f"   [PASS] {mode} mode test passed!")

            except Exception as e:
                print(f"   [ERROR] {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"\nCheckpoint not found: {test_ckpt}")
        print("Skip loading test. Run with valid checkpoint path.")

    print("\n" + "=" * 50)
