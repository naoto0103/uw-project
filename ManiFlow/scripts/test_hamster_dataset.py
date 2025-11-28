#!/usr/bin/env python3
"""
Test script for HAMSTERRoboTwinDataset.

This script verifies that the dataset can load and provide data correctly.
"""

import sys
import os

# Add ManiFlow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ManiFlow'))

import torch
from termcolor import cprint
from maniflow.dataset.hamster_robotwin_dataset import HAMSTERRoboTwinDataset


def test_dataset():
    """Test HAMSTERRoboTwinDataset functionality."""

    zarr_path = "data/synthetic_test_50/replay_buffer.zarr"
    cache_path = "data/synthetic_test_50/hamster_paths.pkl"

    cprint("=" * 60, "cyan")
    cprint("HAMSTERRoboTwinDataset Test", "cyan")
    cprint("=" * 60, "cyan")

    # Create dataset
    cprint("Creating dataset...", "yellow")
    dataset = HAMSTERRoboTwinDataset(
        zarr_path=zarr_path,
        hamster_path_cache=cache_path,
        max_path_length=50,
        path_dim=3,
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.2,
        task_name="synthetic_test"
    )

    cprint(f"Dataset created successfully!", "green")
    cprint(f"Dataset size: {len(dataset)}", "green")
    cprint(f"Train episodes: {dataset.train_episodes_num}", "green")
    cprint(f"Val episodes: {dataset.val_episodes_num}", "green")

    # Get path statistics
    stats = dataset.get_path_statistics()
    cprint("\nPath Statistics:", "cyan")
    for key, value in stats.items():
        cprint(f"  {key}: {value}", "cyan")

    # Test sampling
    cprint("\nTesting data sampling...", "yellow")
    sample = dataset[0]

    cprint("Sample structure:", "green")
    for key, value in sample.items():
        if isinstance(value, dict):
            cprint(f"  {key}:", "green")
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    cprint(f"    {subkey}: shape={subvalue.shape}, dtype={subvalue.dtype}", "green")
                else:
                    cprint(f"    {subkey}: {type(subvalue)}", "green")
        else:
            if hasattr(value, 'shape'):
                cprint(f"  {key}: shape={value.shape}, dtype={value.dtype}", "green")
            else:
                cprint(f"  {key}: {type(value)}", "green")

    # Verify HAMSTER path
    hamster_path = sample['obs']['hamster_path']
    cprint("\nHAMSTER Path details:", "cyan")
    cprint(f"  Shape: {hamster_path.shape}", "cyan")
    cprint(f"  Dtype: {hamster_path.dtype}", "cyan")
    cprint(f"  First 5 points:", "cyan")
    for i in range(min(5, hamster_path.shape[0])):
        cprint(f"    Point {i}: {hamster_path[i].tolist()}", "cyan")

    # Test normalizer
    cprint("\nTesting normalizer...", "yellow")
    normalizer = dataset.get_normalizer()
    # LinearNormalizer is a nn.Module with params_dict attribute
    if hasattr(normalizer, 'params_dict'):
        norm_keys = list(normalizer.params_dict.keys())
    else:
        norm_keys = dir(normalizer)
    cprint(f"Normalizer type: {type(normalizer).__name__}", "green")
    cprint(f"Normalizer has hamster_path: {'hamster_path' in str(normalizer)}", "green")

    # Verify normalizer works
    cprint("  Normalizer created successfully", "green")

    # Test validation dataset
    cprint("\nTesting validation dataset...", "yellow")
    val_dataset = dataset.get_validation_dataset()
    cprint(f"Validation dataset size: {len(val_dataset)}", "green")

    # Test batch loading (simulate DataLoader)
    cprint("\nTesting batch creation...", "yellow")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))

    cprint("Batch structure:", "green")
    for key, value in batch.items():
        if isinstance(value, dict):
            cprint(f"  {key}:", "green")
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    cprint(f"    {subkey}: shape={subvalue.shape}", "green")
        else:
            if hasattr(value, 'shape'):
                cprint(f"  {key}: shape={value.shape}", "green")

    cprint("\n" + "=" * 60, "green")
    cprint("All tests passed!", "green")
    cprint("=" * 60, "green")


if __name__ == "__main__":
    test_dataset()
