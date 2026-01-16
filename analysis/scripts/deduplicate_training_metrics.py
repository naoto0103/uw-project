#!/usr/bin/env python3
"""
Deduplicate training_metrics.jsonl files.

When training is restarted from a checkpoint, some epochs get recorded multiple times.
This script keeps only the last occurrence of each epoch (the most recent training run).
"""

import json
from pathlib import Path

RAW_DATA_DIR = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/analysis/raw_data")

TASKS = ["click_bell", "move_can_pot", "beat_block_hammer"]


def deduplicate_jsonl(file_path: Path) -> tuple[int, int]:
    """
    Deduplicate a training_metrics.jsonl file.

    Returns:
        tuple: (original_count, deduplicated_count)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Separate metadata and epoch records
    metadata_lines = []
    epoch_records = {}  # epoch -> (line_index, json_str)

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if record.get("type") == "metadata":
                metadata_lines.append(line)
            elif record.get("type") == "epoch":
                epoch = record.get("epoch")
                if epoch is not None:
                    # Keep the last occurrence (overwrite if exists)
                    epoch_records[epoch] = (i, line)
        except json.JSONDecodeError:
            print(f"  Warning: Could not parse line {i}: {line[:50]}...")

    original_epoch_count = sum(1 for line in lines if '"type": "epoch"' in line)

    # Sort epochs and reconstruct file
    sorted_epochs = sorted(epoch_records.keys())

    output_lines = metadata_lines.copy()
    for epoch in sorted_epochs:
        _, line = epoch_records[epoch]
        output_lines.append(line)

    # Write back
    with open(file_path, 'w') as f:
        f.write('\n'.join(output_lines) + '\n')

    return original_epoch_count, len(sorted_epochs)


def main():
    print("=" * 60)
    print("Deduplicating training_metrics.jsonl files")
    print("=" * 60)

    total_fixed = 0

    for task in TASKS:
        task_dir = RAW_DATA_DIR / task
        if not task_dir.exists():
            print(f"\n{task}: directory not found")
            continue

        print(f"\n=== {task} ===")

        for condition_dir in sorted(task_dir.iterdir()):
            if not condition_dir.is_dir():
                continue

            jsonl_file = condition_dir / "training_metrics.jsonl"
            if not jsonl_file.exists():
                continue

            orig, dedup = deduplicate_jsonl(jsonl_file)

            if orig != dedup:
                print(f"  {condition_dir.name}: {orig} -> {dedup} epochs (removed {orig - dedup} duplicates)")
                total_fixed += 1
            else:
                print(f"  {condition_dir.name}: {dedup} epochs (no duplicates)")

    print(f"\n{'=' * 60}")
    print(f"Done! Fixed {total_fixed} files with duplicates.")
    print("=" * 60)


if __name__ == "__main__":
    main()
