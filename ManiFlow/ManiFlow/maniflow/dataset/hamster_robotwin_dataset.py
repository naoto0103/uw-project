"""
HAMSTERRoboTwinDataset: RoboTwin dataset extended with HAMSTER path information.

This dataset class loads pre-generated HAMSTER 2D paths and adds them to the
standard RoboTwin point cloud dataset for training ManiFlow policies with
hierarchical action guidance.
"""

from typing import Dict, Optional
import torch
import numpy as np
import pickle
import os
from maniflow.dataset.robotwin_dataset import RoboTwinDataset
from maniflow.common.pytorch_util import dict_apply
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from termcolor import cprint


class HAMSTERRoboTwinDataset(RoboTwinDataset):
    """
    RoboTwin dataset extended with HAMSTER path information.

    This dataset extends the base RoboTwinDataset by adding pre-generated
    HAMSTER 2D paths as additional observation features. The paths are loaded
    from a cache file and padded to a fixed length.

    Args:
        zarr_path: Path to the .zarr dataset file
        hamster_path_cache: Path to the .pkl file containing pre-generated paths
        max_path_length: Maximum number of points in the path (default: 50)
        path_dim: Dimension of each path point (default: 3 for x, y, gripper_state)
        horizon: Length of the observation/action sequence
        pad_before: Padding before the sequence
        pad_after: Padding after the sequence
        seed: Random seed for train/val split
        val_ratio: Ratio of validation episodes
        max_train_episodes: Maximum number of training episodes
        task_name: Name of the task
        use_pc_color: Whether to use point cloud color
        pointcloud_color_aug_cfg: Point cloud color augmentation config
        **kwargs: Additional arguments passed to parent class
    """

    def __init__(self,
            zarr_path: str,
            hamster_path_cache: str,
            max_path_length: int = 50,
            path_dim: int = 3,
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
            task_name: Optional[str] = None,
            use_pc_color: bool = False,
            pointcloud_color_aug_cfg: Optional[dict] = None,
            **kwargs
            ):
        # Initialize parent class
        super().__init__(
            zarr_path=zarr_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            task_name=task_name,
            use_pc_color=use_pc_color,
            pointcloud_color_aug_cfg=pointcloud_color_aug_cfg,
            **kwargs
        )

        self.max_path_length = max_path_length
        self.path_dim = path_dim
        self.hamster_path_cache = hamster_path_cache

        # Load HAMSTER paths from cache
        cprint(f'Loading HAMSTER paths from {hamster_path_cache}', 'cyan')
        if not os.path.exists(hamster_path_cache):
            raise FileNotFoundError(
                f"HAMSTER path cache not found: {hamster_path_cache}\n"
                f"Please run the path generation script first."
            )

        with open(hamster_path_cache, 'rb') as f:
            path_data = pickle.load(f)

        # Expected format: {
        #   'paths': List[List[Tuple[float, float, int]]], # paths for each episode
        #   'task_description': str,
        #   'metadata': dict
        # }
        if isinstance(path_data, dict):
            self.hamster_paths = path_data['paths']
            self.task_description = path_data.get('task_description', '')
            self.path_metadata = path_data.get('metadata', {})
        else:
            # Legacy format: just a list of paths
            self.hamster_paths = path_data
            self.task_description = ''
            self.path_metadata = {}

        # Verify path count matches episode count
        n_episodes = self.replay_buffer.n_episodes
        n_paths = len(self.hamster_paths)
        if n_paths != n_episodes:
            raise ValueError(
                f"Number of HAMSTER paths ({n_paths}) does not match "
                f"number of episodes ({n_episodes})"
            )

        # Preprocess all paths to fixed length
        self.processed_paths = self._preprocess_all_paths()

        cprint(f'Loaded {n_paths} HAMSTER paths', 'cyan')
        cprint(f'Path shape per episode: ({self.max_path_length}, {self.path_dim})', 'cyan')
        if self.task_description:
            cprint(f'Task description: {self.task_description}', 'cyan')

    def _preprocess_all_paths(self) -> np.ndarray:
        """
        Preprocess all paths to fixed length arrays.

        Returns:
            np.ndarray: Shape (n_episodes, max_path_length, path_dim)
        """
        n_episodes = len(self.hamster_paths)
        processed = np.zeros((n_episodes, self.max_path_length, self.path_dim), dtype=np.float32)

        for ep_idx, path in enumerate(self.hamster_paths):
            if len(path) == 0:
                cprint(f'Warning: Episode {ep_idx} has empty path', 'yellow')
                continue

            # Convert path to numpy array
            path_array = np.array(path, dtype=np.float32)

            # Ensure path has correct dimension
            if path_array.ndim == 1:
                # Single point, reshape to (1, path_dim)
                path_array = path_array.reshape(1, -1)

            if path_array.shape[1] != self.path_dim:
                raise ValueError(
                    f"Episode {ep_idx} path has wrong dimension: "
                    f"{path_array.shape[1]} vs expected {self.path_dim}"
                )

            # Pad or truncate to max_path_length
            n_points = path_array.shape[0]
            if n_points >= self.max_path_length:
                # Truncate
                processed[ep_idx] = path_array[:self.max_path_length]
            else:
                # Pad with zeros (or repeat last point)
                processed[ep_idx, :n_points] = path_array
                # Pad remaining with last valid point
                if n_points > 0:
                    processed[ep_idx, n_points:] = path_array[-1]

        return processed

    def _get_episode_index(self, sample_idx: int) -> int:
        """
        Get the episode index for a given sample index.

        The sampler maps sample indices to episode indices internally.
        We need to extract this information.

        Args:
            sample_idx: Index of the sample in the dataset

        Returns:
            Episode index
        """
        # The sampler's indices array contains (buffer_start_idx, buffer_end_idx, ...)
        # We need to find which episode this buffer index belongs to
        idx_info = self.sampler.indices[sample_idx]
        buffer_start_idx = idx_info[0]

        # Get episode boundaries
        episode_ends = self.replay_buffer.episode_ends[:]

        # Find which episode this buffer index belongs to
        for ep_idx, ep_end in enumerate(episode_ends):
            if buffer_start_idx < ep_end:
                return ep_idx

        # Should not reach here
        return len(episode_ends) - 1

    def _sample_to_data(self, sample):
        """
        Convert a raw sample to the data format expected by the model.

        Extends parent method to add HAMSTER path information.

        Args:
            sample: Dictionary containing raw sample data

        Returns:
            Dictionary with 'obs', 'action', and path information
        """
        # Get base data from parent
        data = super()._sample_to_data(sample)

        # The sample should contain episode information
        # We'll add the path in __getitem__ after we know the episode index

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - obs: dict with point_cloud, agent_pos, hamster_path
                - action: action tensor
        """
        # Get the episode index for this sample
        episode_idx = self._get_episode_index(idx)

        # Get base sample from parent
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        # Add HAMSTER path for this episode
        # Shape: (max_path_length, path_dim)
        hamster_path = self.processed_paths[episode_idx].copy()

        # Repeat path for each timestep in the sequence if needed
        # Current shape: (max_path_length, path_dim)
        # Target shape: (To, max_path_length, path_dim) where To is n_obs_steps
        # However, since path is constant for the episode, we might just use (max_path_length, path_dim)
        # Let's keep it as (max_path_length, path_dim) for now, the encoder can handle it

        data['obs']['hamster_path'] = hamster_path

        # Convert to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)

        # Apply point cloud color augmentation from parent
        if self.aug_color:
            import random
            if random.random() <= self.aug_prob:
                T, N, C = torch_data['obs']['point_cloud'].shape
                pc_reshaped = torch_data['obs']['point_cloud'].reshape(-1, C)
                pc_reshaped = self.pc_jitter(pc_reshaped)
                torch_data['obs']['point_cloud'] = pc_reshaped.reshape(T, N, C)

        return torch_data

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Get normalizer for the dataset including HAMSTER path normalization.

        Args:
            mode: Normalization mode ('limits' or 'gaussian')
            **kwargs: Additional arguments

        Returns:
            LinearNormalizer with all fields
        """
        # Get base normalizer
        normalizer = super().get_normalizer(mode=mode, **kwargs)

        # Add normalizer for HAMSTER path
        # Path coordinates are already normalized to [0, 1] from HAMSTER
        # Gripper state is 0 or 1
        # So we use identity normalizer
        normalizer['hamster_path'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def get_path_statistics(self) -> Dict[str, float]:
        """
        Compute statistics about the HAMSTER paths.

        Returns:
            Dictionary with path statistics
        """
        path_lengths = []
        for path in self.hamster_paths:
            path_lengths.append(len(path))

        path_lengths = np.array(path_lengths)

        stats = {
            'min_length': int(path_lengths.min()),
            'max_length': int(path_lengths.max()),
            'mean_length': float(path_lengths.mean()),
            'std_length': float(path_lengths.std()),
            'n_episodes': len(self.hamster_paths),
            'n_empty': int(np.sum(path_lengths == 0)),
        }

        return stats


if __name__ == '__main__':
    # Test the dataset
    print("HAMSTERRoboTwinDataset Test")
    print("=" * 50)

    # Example usage (requires actual data files)
    zarr_path = '/path/to/robotwin_dataset.zarr'
    cache_path = '/path/to/hamster_paths.pkl'

    # Create dummy cache for testing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        dummy_cache_path = f.name
        # Create dummy paths for 10 episodes
        dummy_paths = {
            'paths': [
                [(0.1, 0.2, 0), (0.3, 0.4, 0), (0.5, 0.6, 1)]  # 3 points
                for _ in range(10)
            ],
            'task_description': 'Pick up the apple',
            'metadata': {'generated_at': '2025-01-01'}
        }
        pickle.dump(dummy_paths, f)

    print(f"Created dummy cache at: {dummy_cache_path}")

    # Test path preprocessing
    dataset_mock = type('MockDataset', (), {
        'hamster_paths': dummy_paths['paths'],
        'max_path_length': 50,
        'path_dim': 3
    })()

    # Test preprocessing function
    def test_preprocess(paths, max_len, dim):
        n_eps = len(paths)
        processed = np.zeros((n_eps, max_len, dim), dtype=np.float32)
        for ep_idx, path in enumerate(paths):
            path_array = np.array(path, dtype=np.float32)
            n_points = path_array.shape[0]
            if n_points >= max_len:
                processed[ep_idx] = path_array[:max_len]
            else:
                processed[ep_idx, :n_points] = path_array
                processed[ep_idx, n_points:] = path_array[-1]
        return processed

    processed = test_preprocess(dummy_paths['paths'], 50, 3)
    print(f"Processed paths shape: {processed.shape}")
    print(f"First path (first 5 points):\n{processed[0, :5]}")
    print(f"First path (last 5 points):\n{processed[0, -5:]}")

    # Clean up
    os.remove(dummy_cache_path)
    print(f"Removed dummy cache")

    print("\nTest completed successfully!")
    print("To use with real data, provide actual zarr_path and hamster_path_cache.")
