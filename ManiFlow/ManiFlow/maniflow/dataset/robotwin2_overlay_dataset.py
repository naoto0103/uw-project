"""
RoboTwin 2.0 Overlay Dataset for ManiFlow training.

This dataset loads pre-generated overlay images (RGB + HAMSTER-style path)
along with action data from RoboTwin 2.0 HDF5 files.

The key insight is that we use TWO overlay images per sample:
1. initial_overlay: The overlay from frame 0 (provides task goal/memory)
2. current_overlay: The overlay from the current timestep

This dual-image approach provides implicit memory for handling occlusions,
as the initial overlay contains the full intended path even when objects
become occluded during execution.
"""

from typing import Dict, List, Optional, Tuple
import os
import zipfile
import tempfile
import copy

import torch
import numpy as np
import cv2
import h5py
from pathlib import Path

from maniflow.common.pytorch_util import dict_apply
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset
from termcolor import cprint


# Default paths for Hyak HPC
DEFAULT_OVERLAY_BASE = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/robotwin2_single_6tasks_vila")
DEFAULT_ROBOTWIN2_BASE = Path("/mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/dataset/dataset")

# Single-arm tasks
SINGLE_ARM_TASKS = [
    "beat_block_hammer",
    "click_bell",
    "move_can_pot",
    "open_microwave",
    "place_object_stand",
    "turn_switch",
]

# Task instructions for language conditioning
TASK_INSTRUCTIONS = {
    "beat_block_hammer": "Pick up the hammer and beat the block",
    "click_bell": "Click the bell's top center on the table",
    "move_can_pot": "Pick up the can and move it beside the pot",
    "open_microwave": "Open the microwave door",
    "place_object_stand": "Place the object on the stand",
    "turn_switch": "Click the switch",
}


class RoboTwin2OverlayDataset(BaseDataset):
    """
    Dataset for ManiFlow training with overlay images from RoboTwin 2.0.

    This dataset combines:
    - Overlay images (RGB + path) from pre-generated files
    - Action data from RoboTwin 2.0 HDF5 files

    Args:
        overlay_base_dir: Base directory containing overlay images
        robotwin2_base_dir: Base directory containing RoboTwin 2.0 dataset
        tasks: List of task names to include
        n_episodes: Number of episodes per task
        horizon: Sequence length for training
        pad_before: Padding before sequence
        pad_after: Padding after sequence
        seed: Random seed
        val_ratio: Validation set ratio
        max_train_episodes: Maximum training episodes (None = all)
        image_size: Target image size (H, W)
        robot: Robot type (default: aloha-agilex)
        config: Dataset config (default: clean_50)
    """

    def __init__(
        self,
        overlay_base_dir: str = None,
        robotwin2_base_dir: str = None,
        tasks: List[str] = None,
        n_episodes: int = 50,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 15,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        robot: str = "aloha-agilex",
        config: str = "clean_50",
        **kwargs
    ):
        super().__init__()

        # Set default paths
        self.overlay_base_dir = Path(overlay_base_dir) if overlay_base_dir else DEFAULT_OVERLAY_BASE
        self.robotwin2_base_dir = Path(robotwin2_base_dir) if robotwin2_base_dir else DEFAULT_ROBOTWIN2_BASE
        self.tasks = tasks if tasks else SINGLE_ARM_TASKS
        self.n_episodes = n_episodes
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.image_size = image_size
        self.robot = robot
        self.config = config

        cprint(f"Loading RoboTwin2 Overlay Dataset", "green")
        cprint(f"  Tasks: {self.tasks}", "cyan")
        cprint(f"  Episodes per task: {n_episodes}", "cyan")
        cprint(f"  Overlay base: {self.overlay_base_dir}", "cyan")

        # Build episode index
        # Each entry: (task_name, episode_idx, frame_indices, n_frames)
        self.episodes = []
        self._build_episode_index()

        # Create train/val split
        np.random.seed(seed)
        n_total = len(self.episodes)
        n_val = int(n_total * val_ratio)

        indices = np.random.permutation(n_total)
        val_indices = set(indices[:n_val])
        train_indices = set(indices[n_val:])

        if max_train_episodes is not None and len(train_indices) > max_train_episodes:
            train_indices = set(list(train_indices)[:max_train_episodes])

        self.train_mask = np.array([i in train_indices for i in range(n_total)])
        self.val_mask = np.array([i in val_indices for i in range(n_total)])

        # Build sample indices for training
        self._build_sample_indices(self.train_mask)

        cprint(f"  Total episodes: {n_total}", "green")
        cprint(f"  Train episodes: {np.sum(self.train_mask)}", "green")
        cprint(f"  Val episodes: {np.sum(self.val_mask)}", "green")
        cprint(f"  Total samples: {len(self.sample_indices)}", "green")

    def _build_episode_index(self):
        """Build index of all available episodes."""
        for task in self.tasks:
            for ep_idx in range(self.n_episodes):
                overlay_dir = self.overlay_base_dir / task / f"episode_{ep_idx:02d}" / "overlay_images"

                if not overlay_dir.exists():
                    cprint(f"  Warning: Missing overlay dir: {overlay_dir}", "yellow")
                    continue

                # Count overlay files
                overlay_files = sorted(overlay_dir.glob("overlay_*.png"))
                n_frames = len(overlay_files)

                if n_frames == 0:
                    cprint(f"  Warning: No overlay files in {overlay_dir}", "yellow")
                    continue

                # Extract frame indices from filenames
                frame_indices = []
                for f in overlay_files:
                    idx = int(f.stem.split("_")[1])
                    frame_indices.append(idx)

                self.episodes.append({
                    "task": task,
                    "episode_idx": ep_idx,
                    "frame_indices": sorted(frame_indices),
                    "n_frames": n_frames,
                    "overlay_dir": overlay_dir,
                })

    def _build_sample_indices(self, episode_mask: np.ndarray):
        """Build sample indices for sequence sampling."""
        self.sample_indices = []

        for ep_id, (episode, mask) in enumerate(zip(self.episodes, episode_mask)):
            if not mask:
                continue

            n_frames = episode["n_frames"]

            # Create samples with padding consideration
            # We need at least horizon frames for a valid sample
            for start_idx in range(n_frames - self.horizon + 1):
                self.sample_indices.append({
                    "episode_id": ep_id,
                    "start_idx": start_idx,
                })

    def _load_overlay_image(self, overlay_dir: Path, frame_idx: int) -> np.ndarray:
        """Load and preprocess an overlay image."""
        img_path = overlay_dir / f"overlay_{frame_idx:04d}.png"

        if not img_path.exists():
            # Fallback: try to find closest available frame
            available = sorted([int(f.stem.split("_")[1]) for f in overlay_dir.glob("overlay_*.png")])
            if available:
                closest = min(available, key=lambda x: abs(x - frame_idx))
                img_path = overlay_dir / f"overlay_{closest:04d}.png"

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if img.shape[:2] != self.image_size:
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

        # Normalize to [0, 1] and transpose to (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        return img

    def _load_action_from_hdf5(
        self,
        task: str,
        episode_idx: int,
        start_frame: int,
        n_frames: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load action and state data from RoboTwin 2.0 HDF5 file.

        Supports both:
        1. Extracted HDF5 files (faster, recommended)
        2. HDF5 files inside ZIP archives (fallback)

        Returns:
            Tuple of (actions, states) arrays
        """
        inner_dir = f"{self.robot}_{self.config}"
        hdf5_name = f"episode{episode_idx}.hdf5"

        # Try extracted HDF5 first (faster)
        extracted_hdf5 = self.robotwin2_base_dir / task / inner_dir / "data" / hdf5_name

        if extracted_hdf5.exists():
            # Direct read from extracted file (fast path)
            with h5py.File(extracted_hdf5, 'r') as f:
                all_actions = f['joint_action/vector'][:]
                all_states = all_actions.copy()

                end_frame = min(start_frame + n_frames, len(all_actions))
                actions = all_actions[start_frame:end_frame].astype(np.float32)
                states = all_states[start_frame:end_frame].astype(np.float32)

                if len(actions) < n_frames:
                    pad_len = n_frames - len(actions)
                    actions = np.pad(actions, ((0, pad_len), (0, 0)), mode='edge')
                    states = np.pad(states, ((0, pad_len), (0, 0)), mode='edge')

            return actions, states

        # Fallback to ZIP extraction (slow path)
        zip_path = self.robotwin2_base_dir / task / f"{self.robot}_{self.config}.zip"

        if not zip_path.exists():
            raise FileNotFoundError(f"Neither extracted HDF5 nor ZIP found for {task}/episode{episode_idx}")

        hdf5_inner_path = f"{inner_dir}/data/{hdf5_name}"

        with zipfile.ZipFile(zip_path, 'r') as zf:
            with tempfile.TemporaryDirectory() as temp_dir:
                zf.extract(hdf5_inner_path, temp_dir)
                temp_hdf5 = os.path.join(temp_dir, hdf5_inner_path)

                with h5py.File(temp_hdf5, 'r') as f:
                    all_actions = f['joint_action/vector'][:]
                    all_states = all_actions.copy()

                    end_frame = min(start_frame + n_frames, len(all_actions))
                    actions = all_actions[start_frame:end_frame].astype(np.float32)
                    states = all_states[start_frame:end_frame].astype(np.float32)

                    if len(actions) < n_frames:
                        pad_len = n_frames - len(actions)
                        actions = np.pad(actions, ((0, pad_len), (0, 0)), mode='edge')
                        states = np.pad(states, ((0, pad_len), (0, 0)), mode='edge')

        return actions, states

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.sample_indices[idx]
        episode_id = sample_info["episode_id"]
        start_idx = sample_info["start_idx"]

        episode = self.episodes[episode_id]
        task = episode["task"]
        ep_idx = episode["episode_idx"]
        overlay_dir = episode["overlay_dir"]
        frame_indices = episode["frame_indices"]

        # Load overlay images for the sequence
        initial_overlays = []
        current_overlays = []

        for t in range(self.horizon):
            frame_idx = frame_indices[start_idx + t] if (start_idx + t) < len(frame_indices) else frame_indices[-1]

            # Initial overlay is always frame 0
            initial_img = self._load_overlay_image(overlay_dir, frame_indices[0])
            initial_overlays.append(initial_img)

            # Current overlay is the actual frame
            current_img = self._load_overlay_image(overlay_dir, frame_idx)
            current_overlays.append(current_img)

        initial_overlays = np.stack(initial_overlays, axis=0)  # (T, C, H, W)
        current_overlays = np.stack(current_overlays, axis=0)  # (T, C, H, W)

        # Load actions and states from HDF5
        start_frame = frame_indices[start_idx]
        actions, states = self._load_action_from_hdf5(task, ep_idx, start_frame, self.horizon)

        # Build output dict
        data = {
            'obs': {
                'initial_overlay': initial_overlays,
                'current_overlay': current_overlays,
                'agent_pos': states,
            },
            'action': actions,
        }

        # Convert to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data

    def get_validation_dataset(self) -> 'RoboTwin2OverlayDataset':
        """Return validation dataset."""
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask.copy()
        val_set._build_sample_indices(val_set.train_mask)
        return val_set

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        """Get normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # Collect all actions for normalization
        all_actions = self.get_all_actions()
        normalizer.fit(data={'action': all_actions}, last_n_dims=1, mode=mode, **kwargs)

        # Identity normalization for images (already [0, 1])
        normalizer['initial_overlay'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['current_overlay'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Get all actions from training episodes for normalization."""
        all_actions = []

        for ep_id, (episode, mask) in enumerate(zip(self.episodes, self.train_mask)):
            if not mask:
                continue

            task = episode["task"]
            ep_idx = episode["episode_idx"]
            n_frames = episode["n_frames"]

            try:
                actions, _ = self._load_action_from_hdf5(task, ep_idx, 0, n_frames)
                all_actions.append(actions)
            except Exception as e:
                cprint(f"Warning: Failed to load actions for {task}/ep{ep_idx}: {e}", "yellow")

        if not all_actions:
            # Return dummy actions if none loaded
            return torch.zeros((1, 14), dtype=torch.float32)

        all_actions = np.concatenate(all_actions, axis=0)
        return torch.from_numpy(all_actions).float()


class RoboTwin2OverlayMultiTaskDataset(RoboTwin2OverlayDataset):
    """
    Multi-task version of the overlay dataset with language conditioning.

    This dataset adds task_name to each sample for language-conditioned training.
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get base data
        data = super().__getitem__(idx)

        # Add task name for language conditioning
        sample_info = self.sample_indices[idx]
        episode_id = sample_info["episode_id"]
        episode = self.episodes[episode_id]
        task = episode["task"]

        # Add task instruction
        data['task_name'] = task
        data['task_instruction'] = TASK_INSTRUCTIONS.get(task, f"Complete the {task} task")

        return data


if __name__ == "__main__":
    # Test the dataset
    print("=" * 60)
    print("RoboTwin2 Overlay Dataset Test")
    print("=" * 60)

    # Test with one task
    dataset = RoboTwin2OverlayDataset(
        tasks=["beat_block_hammer"],
        n_episodes=2,
        horizon=16,
    )

    print(f"\nDataset length: {len(dataset)}")

    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]
        print("\nSample keys:", sample.keys())
        print("Obs keys:", sample['obs'].keys())

        for key, val in sample['obs'].items():
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        print(f"  action: shape={sample['action'].shape}, dtype={sample['action'].dtype}")

        # Test normalizer
        normalizer = dataset.get_normalizer()
        print(f"\nNormalizer created successfully")

        # Test validation dataset
        val_dataset = dataset.get_validation_dataset()
        print(f"Validation dataset length: {len(val_dataset)}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
