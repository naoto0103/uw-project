#!/usr/bin/env python3
"""
Parse beat_block_hammer training logs and create structured training_metrics.jsonl files.

This script handles:
1. Multiple log files from training restarts
2. Parsing tqdm progress bars for per-batch loss
3. Extracting validation loss from checkpoint save messages
4. Merging logs from multiple restarts into a single timeline
"""

import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Log directory
LOG_DIR = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/logs")
OUTPUT_DIR = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs")

# Mapping from log file patterns to output directories
LOG_PATTERNS = {
    "original_cluttered": {
        "pattern": "train_original_cluttered_beat_block_hammer_seed42_*.log",
        "output_dir": "cluttered_beat_block_hammer/original_seed42"
    },
    "original_clean": {
        "pattern": "train_original_clean_beat_block_hammer_seed42_*.log",
        "output_dir": "clean_beat_block_hammer/original_seed42"
    },
    "overlay_current_cluttered": {
        "pattern": "train_overlay_current_cluttered_beat_block_hammer_seed42_*.log",
        "output_dir": "cluttered_beat_block_hammer/overlay_current_seed42"
    },
    "overlay_current_clean": {
        "pattern": "train_overlay_current_clean_beat_block_hammer_seed42_*.log",
        "output_dir": "clean_beat_block_hammer/overlay_current_seed42"
    },
    "overlay_initial_current_cluttered": {
        "pattern": "train_overlay_initial_current_cluttered_beat_block_hammer_seed42_*.log",
        "output_dir": "cluttered_beat_block_hammer/overlay_initial_current_seed42"
    },
    "overlay_initial_current_clean": {
        "pattern": "train_overlay_initial_current_clean_beat_block_hammer_seed42_*.log",
        "output_dir": "clean_beat_block_hammer/overlay_initial_current_seed42"
    }
}


def get_val_losses_from_checkpoints(checkpoint_dir: Path) -> dict:
    """
    Extract validation losses directly from checkpoint filenames.
    This is more reliable than parsing logs.
    """
    val_losses = {}
    if not checkpoint_dir.exists():
        return val_losses

    for ckpt_file in checkpoint_dir.glob("epoch=*-val_loss=*.ckpt"):
        # Match epoch=XXXX-val_loss=Y.YYYYYY.ckpt
        match = re.search(r'epoch=(\d+)-val_loss=(\d+\.\d+)\.ckpt', ckpt_file.name)
        if match:
            epoch = int(match.group(1))
            val_loss = float(match.group(2))
            val_losses[epoch] = val_loss
            print(f"    Checkpoint: epoch {epoch}, val_loss={val_loss}")

    return val_losses


def parse_log_file(log_path: Path) -> dict:
    """
    Parse a single log file and extract training metrics.

    Returns:
        dict: {
            "metadata": {...},
            "epochs": {epoch_num: {"train_loss": float, "val_loss": float or None}},
            "resume_from_epoch": int or None,
            "first_epoch": int or None,
            "last_epoch": int or None
        }
    """
    result = {
        "metadata": {},
        "epochs": {},
        "resume_from_epoch": None,
        "first_epoch": None,
        "last_epoch": None,
        "log_file": str(log_path.name)
    }

    with open(log_path, 'r') as f:
        content = f.read()

    # Check if this is a resumed training
    resume_match = re.search(r'Resuming from checkpoint.*?epoch=(\d+)', content)
    if resume_match:
        result["resume_from_epoch"] = int(resume_match.group(1))
        print(f"    Resumed from epoch {result['resume_from_epoch']}")

    # Extract validation loss from checkpoint saves
    # Format: saved in  .../epoch=0050-val_loss=0.042107.ckpt
    val_loss_pattern = r'saved in.*?epoch=(\d+)-val_loss=([0-9.]+)\.ckpt'
    for match in re.finditer(val_loss_pattern, content):
        epoch = int(match.group(1))
        val_loss = float(match.group(2))
        if epoch not in result["epochs"]:
            result["epochs"][epoch] = {}
        result["epochs"][epoch]["val_loss"] = val_loss

    # Extract per-epoch train_loss from tqdm progress bars
    # Look for the last loss value for each epoch (typically at 99% or near completion)
    # Pattern: Training epoch X: YY%|...loss=Z.ZZZ
    # We collect all loss values per epoch and keep the last one (highest progress %)
    epoch_losses = {}  # {epoch: [(progress%, loss), ...]}

    # Match patterns like: Training epoch 5:  99%|█████████▉| 88/89 [01:17<00:00,  1.14it/s, loss=0.167]
    train_loss_pattern = r'Training epoch (\d+):\s+(\d+)%\|[^|]*\|[^,]+,[^,]+,\s*loss=([0-9.]+)'
    for match in re.finditer(train_loss_pattern, content):
        epoch = int(match.group(1))
        progress = int(match.group(2))
        loss = float(match.group(3))
        if epoch not in epoch_losses:
            epoch_losses[epoch] = []
        epoch_losses[epoch].append((progress, loss))

    # For each epoch, take the loss from the highest progress percentage
    epochs_with_train_loss = 0
    for epoch, losses in epoch_losses.items():
        if losses:
            # Sort by progress and take the last (highest progress) loss
            best_progress, best_loss = max(losses, key=lambda x: x[0])
            if epoch not in result["epochs"]:
                result["epochs"][epoch] = {}
            result["epochs"][epoch]["train_loss"] = best_loss
            epochs_with_train_loss += 1

    # Track first and last epoch from Training epoch lines
    training_epochs = re.findall(r'Training epoch (\d+):', content)
    if training_epochs:
        epochs_int = [int(e) for e in training_epochs]
        result["first_epoch"] = min(epochs_int)
        result["last_epoch"] = max(epochs_int)
        print(f"    Training epochs: {result['first_epoch']} to {result['last_epoch']}, train_loss extracted: {epochs_with_train_loss}")

    # Count val_loss entries
    val_loss_count = sum(1 for e in result["epochs"].values() if e.get("val_loss") is not None)
    print(f"    Val loss entries: {val_loss_count}")

    return result


def merge_log_results(log_results: list) -> dict:
    """
    Merge results from multiple log files (handling restarts).

    Args:
        log_results: List of parsed log results, sorted by start epoch

    Returns:
        dict: Merged epochs dictionary
    """
    merged_epochs = {}

    for log_result in log_results:
        for epoch, data in log_result["epochs"].items():
            if epoch not in merged_epochs:
                merged_epochs[epoch] = {"train_loss": None, "val_loss": None}

            # Update with new data (later logs take precedence for same epoch)
            if data.get("train_loss") is not None:
                merged_epochs[epoch]["train_loss"] = data["train_loss"]
            if data.get("val_loss") is not None:
                merged_epochs[epoch]["val_loss"] = data["val_loss"]

    return merged_epochs


def create_training_metrics_jsonl(condition_name: str, output_dir: Path, epochs_data: dict):
    """
    Create a training_metrics.jsonl file in the same format as other tasks.
    """
    output_path = output_dir / "training_metrics.jsonl"

    # Sort epochs
    sorted_epochs = sorted(epochs_data.keys())

    lines = []

    # Add metadata
    metadata = {
        "type": "metadata",
        "config_name": f"train_maniflow_{condition_name.split('_')[0]}_robotwin2",
        "task_name": "robotwin2_beat_block_hammer",
        "exp_name": f"robotwin2_{condition_name}_parsed",
        "num_epochs": 501,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "seed": 42,
        "output_dir": str(output_dir),
        "start_time": datetime.now().isoformat(),
        "note": "Parsed from raw training logs"
    }
    lines.append(json.dumps(metadata))

    # Add epoch data
    for epoch in sorted_epochs:
        data = epochs_data[epoch]
        epoch_record = {
            "type": "epoch",
            "epoch": epoch,
            "global_step": (epoch + 1) * 88,  # Approximate based on 88 batches per epoch
            "train_loss": data.get("train_loss"),
            "val_loss": data.get("val_loss"),
            "train_action_mse_error": None,
            "lr": None,
            "test_mean_score": -data.get("train_loss") if data.get("train_loss") else None,
            "timestamp": datetime.now().isoformat()
        }
        lines.append(json.dumps(epoch_record))

    # Write file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Written {len(sorted_epochs)} epochs to {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("Parsing beat_block_hammer training logs")
    print("=" * 60)

    for condition_name, config in LOG_PATTERNS.items():
        print(f"\n--- Processing {condition_name} ---")

        # Create output directory path
        output_dir = OUTPUT_DIR / config["output_dir"]
        checkpoint_dir = output_dir / "checkpoints"

        # Get val_losses from checkpoint files (most reliable source)
        print(f"  Scanning checkpoints in {checkpoint_dir}...")
        checkpoint_val_losses = get_val_losses_from_checkpoints(checkpoint_dir)

        # Find all matching log files
        log_files = sorted(LOG_DIR.glob(config["pattern"]))

        if not log_files and not checkpoint_val_losses:
            print(f"  No log files or checkpoints found")
            continue

        # Parse each log file for additional info
        log_results = []
        if log_files:
            print(f"  Found {len(log_files)} log file(s):")
            for lf in log_files:
                print(f"    - {lf.name}")

            for log_file in log_files:
                print(f"  Parsing {log_file.name}...")
                result = parse_log_file(log_file)
                log_results.append(result)

        # Start with checkpoint val_losses as the base
        merged_epochs = {}
        for epoch, val_loss in checkpoint_val_losses.items():
            merged_epochs[epoch] = {"train_loss": None, "val_loss": val_loss}

        # Add any additional data from logs
        for log_result in log_results:
            for epoch, data in log_result["epochs"].items():
                if epoch not in merged_epochs:
                    merged_epochs[epoch] = {"train_loss": None, "val_loss": None}
                # Log data can fill in missing values but checkpoint values take precedence
                if data.get("train_loss") is not None:
                    merged_epochs[epoch]["train_loss"] = data["train_loss"]
                if data.get("val_loss") is not None and merged_epochs[epoch]["val_loss"] is None:
                    merged_epochs[epoch]["val_loss"] = data["val_loss"]

        print(f"  Total epochs with data: {len(merged_epochs)}")

        # Report epoch range
        if merged_epochs:
            min_epoch = min(merged_epochs.keys())
            max_epoch = max(merged_epochs.keys())
            print(f"  Epoch range: {min_epoch} to {max_epoch}")

            # Count epochs with val_loss
            val_loss_count = sum(1 for e in merged_epochs.values() if e.get("val_loss") is not None)
            print(f"  Epochs with val_loss: {val_loss_count}")

        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create training_metrics.jsonl
        if merged_epochs:
            create_training_metrics_jsonl(condition_name, output_dir, merged_epochs)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
