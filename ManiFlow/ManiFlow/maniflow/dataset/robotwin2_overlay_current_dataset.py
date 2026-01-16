"""
RoboTwin 2.0 Overlay Dataset (Current Only, Zarr-based) for ManiFlow training.

This dataset loads pre-converted Zarr data containing overlay images
and actions. Unlike the initial+current version, this only uses the
current frame's overlay image (no Memory Function).

The Zarr file should be created using convert_overlay_to_zarr.py

Expected Zarr structure:
    robotwin2_overlay.zarr/
    ├── data/
    │   ├── overlay_image    # (N, 3, 224, 224) float32
    │   ├── action           # (N, 14) float32
    │   └── state            # (N, 14) float32
    └── meta/
        └── episode_ends     # (num_episodes,) int64
"""

from typing import Dict
import torch
import numpy as np
import copy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset
from termcolor import cprint


class RoboTwin2OverlayCurrentDataset(BaseDataset):
    """
    Zarr-based dataset for RoboTwin 2.0 overlay images (current only).

    This dataset provides:
    - current_overlay: Current frame overlay image
    - agent_pos: Robot state

    Unlike RoboTwin2OverlayZarrDataset, this does NOT include initial_overlay.
    This is for testing the effect of path guidance without Memory Function.

    Args:
        zarr_path: Path to the Zarr file
        horizon: Sequence length for training
        pad_before: Padding before sequence
        pad_after: Padding after sequence
        seed: Random seed
        val_ratio: Validation set ratio
        max_train_episodes: Maximum training episodes (None = all)
        task_name: Task name for logging
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: int = None,
        task_name: str = None,
        **kwargs
    ):
        super().__init__()
        self.task_name = task_name
        self.zarr_path = zarr_path

        cprint(f'Loading RoboTwin2 Overlay Current Dataset from {zarr_path}', 'green')

        # Load data using ReplayBuffer (copies to memory for fast access)
        buffer_keys = ['overlay_image', 'state', 'action']
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=buffer_keys
        )

        # Get episode ends
        self.episode_ends = self.replay_buffer.episode_ends[:]

        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        if max_train_episodes is None:
            max_train_episodes = self.replay_buffer.n_episodes - np.sum(val_mask)

        cprint(f'Maximum training episodes: {max_train_episodes}', 'yellow')
        cprint(f'Validation ratio: {val_ratio}', 'yellow')

        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

        cprint(f'Train episodes: {self.train_episodes_num}', 'green')
        cprint(f'Val episodes: {self.val_episodes_num}', 'green')
        cprint(f'Total samples: {len(self.sampler)}', 'green')

    def get_validation_dataset(self):
        """Return validation dataset."""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """Get normalizer for the dataset."""
        data = {
            'action': self.replay_buffer['action']
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # Identity normalization for images and states (already normalized)
        normalizer['current_overlay'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        Convert sampled sequence to training data format.

        Args:
            sample: Dict from replay buffer with 'overlay_image', 'state', 'action'
        """
        # Current overlay images (T, 3, H, W)
        current_overlay = sample['overlay_image'].astype(np.float32)

        # Agent state (T, 14)
        agent_pos = sample['state'].astype(np.float32)

        data = {
            'obs': {
                'current_overlay': current_overlay,
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32)
        }

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


if __name__ == "__main__":
    # Test the dataset
    print("=" * 60)
    print("RoboTwin2 Overlay Current Dataset Test")
    print("=" * 60)

    zarr_path = "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/zarr/clean_overlay.zarr"

    try:
        dataset = RoboTwin2OverlayCurrentDataset(
            zarr_path=zarr_path,
            horizon=16,
            pad_before=1,
            pad_after=15,
            val_ratio=0.02,
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

    except FileNotFoundError as e:
        print(f"\nZarr file not found: {zarr_path}")
        print("Run convert_overlay_to_zarr.py first to create the Zarr file.")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
