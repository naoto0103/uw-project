#!/usr/bin/env python3
"""
Analyze training curves from training_metrics.jsonl files.
Creates visualizations comparing training dynamics across conditions and tasks.

Outputs figures to FIGURES_DIR and tables to TABLES_DIR as defined in config.py.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from config import (
    TASKS,
    TASK_DESCRIPTIONS,
    CONDITIONS,
    CONDITION_COLORS,
    RAW_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    FIGURE_DPI,
    FIGURE_FORMAT,
)

# Use the same style as other figures
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
})


def load_training_metrics(task: str, condition_id: int) -> dict:
    """Load training metrics for a task/condition pair."""
    task_dir = RAW_DATA_DIR / task
    condition_name = f"condition{condition_id}"

    # Find the condition directory
    for cond_dir in task_dir.iterdir():
        if cond_dir.is_dir() and cond_dir.name.startswith(condition_name):
            jsonl_file = cond_dir / "training_metrics.jsonl"
            if jsonl_file.exists():
                epochs = {}
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        record = json.loads(line.strip())
                        if record.get("type") == "epoch":
                            epoch = record.get("epoch")
                            epochs[epoch] = {
                                "train_loss": record.get("train_loss"),
                                "val_loss": record.get("val_loss")
                            }
                return epochs
    return {}


def smooth_curve(values, window=10):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')


def plot_training_curves_by_task():
    """Create training curve plots for each task (all 6 conditions)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, task in enumerate(TASKS):
        ax = axes[idx]

        for cond_id in range(1, 7):
            epochs_data = load_training_metrics(task, cond_id)
            if not epochs_data:
                continue

            # Get sorted epochs and values
            sorted_epochs = sorted(epochs_data.keys())
            train_losses = [epochs_data[e].get("train_loss") for e in sorted_epochs]

            # Filter out None values
            valid_data = [(e, l) for e, l in zip(sorted_epochs, train_losses) if l is not None]
            if not valid_data:
                continue

            epochs, losses = zip(*valid_data)

            # Apply smoothing for cleaner visualization
            if len(losses) > 20:
                smoothed = smooth_curve(np.array(losses), window=20)
                epochs_smoothed = epochs[10:-9] if len(epochs) > 20 else epochs
                ax.plot(epochs_smoothed, smoothed,
                       label=f"C{cond_id}: {CONDITIONS[cond_id]['short_name']}",
                       color=CONDITION_COLORS[cond_id],
                       linewidth=1.5,
                       alpha=0.9)
            else:
                ax.plot(epochs, losses,
                       label=f"C{cond_id}: {CONDITIONS[cond_id]['short_name']}",
                       color=CONDITION_COLORS[cond_id],
                       linewidth=1.5,
                       alpha=0.9)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title(TASK_DESCRIPTIONS[task])
        ax.set_xlim(0, 500)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        if idx == 2:  # Only show legend on last plot
            ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    save_path = FIGURES_DIR / f"training_curves_by_task.{FIGURE_FORMAT}"
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_val_loss_comparison():
    """Create validation loss comparison (every 50 epochs from checkpoints)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, task in enumerate(TASKS):
        ax = axes[idx]

        for cond_id in range(1, 7):
            epochs_data = load_training_metrics(task, cond_id)
            if not epochs_data:
                continue

            sorted_epochs = sorted(epochs_data.keys())
            val_data = [(e, epochs_data[e].get("val_loss")) for e in sorted_epochs
                       if epochs_data[e].get("val_loss") is not None]

            if not val_data:
                continue

            epochs, val_losses = zip(*val_data)

            ax.plot(epochs, val_losses,
                   label=f"C{cond_id}: {CONDITIONS[cond_id]['short_name']}",
                   color=CONDITION_COLORS[cond_id],
                   marker='o',
                   markersize=4,
                   linewidth=1.5,
                   alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.set_title(TASK_DESCRIPTIONS[task])
        ax.set_xlim(0, 500)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        if idx == 2:
            ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    save_path = FIGURES_DIR / f"validation_loss_comparison.{FIGURE_FORMAT}"
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_loss_comparison():
    """Create bar chart comparing final training/validation loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect final losses
    final_train_losses = defaultdict(dict)
    final_val_losses = defaultdict(dict)

    for task in TASKS:
        for cond_id in range(1, 7):
            epochs_data = load_training_metrics(task, cond_id)
            if not epochs_data:
                continue

            # Get final epoch data
            sorted_epochs = sorted(epochs_data.keys())
            if not sorted_epochs:
                continue

            # Final training loss (last available)
            for e in reversed(sorted_epochs):
                if epochs_data[e].get("train_loss") is not None:
                    final_train_losses[task][cond_id] = epochs_data[e]["train_loss"]
                    break

            # Final validation loss (last available)
            for e in reversed(sorted_epochs):
                if epochs_data[e].get("val_loss") is not None:
                    final_val_losses[task][cond_id] = epochs_data[e]["val_loss"]
                    break

    # Plot final train loss
    ax = axes[0]
    x = np.arange(len(TASKS))
    width = 0.12

    for i, cond_id in enumerate(range(1, 7)):
        values = [final_train_losses[task].get(cond_id, 0) for task in TASKS]
        ax.bar(x + i*width - 2.5*width, values, width,
               label=f"C{cond_id}: {CONDITIONS[cond_id]['short_name']}",
               color=CONDITION_COLORS[cond_id],
               edgecolor='white',
               linewidth=0.5)

    ax.set_ylabel("Final Training Loss (Epoch 500)")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DESCRIPTIONS[t] for t in TASKS], rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title("Final Training Loss by Condition")
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Plot final val loss
    ax = axes[1]

    for i, cond_id in enumerate(range(1, 7)):
        values = [final_val_losses[task].get(cond_id, 0) for task in TASKS]
        ax.bar(x + i*width - 2.5*width, values, width,
               label=f"C{cond_id}: {CONDITIONS[cond_id]['short_name']}",
               color=CONDITION_COLORS[cond_id],
               edgecolor='white',
               linewidth=0.5)

    ax.set_ylabel("Final Validation Loss (Epoch 500)")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DESCRIPTIONS[t] for t in TASKS], rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title("Final Validation Loss by Condition")
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_path = FIGURES_DIR / f"final_loss_comparison.{FIGURE_FORMAT}"
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_single_task_train_val_loss(task: str = "click_bell"):
    """
    Create a figure with Training Loss and Validation Loss for a single task.
    Two subplots stacked vertically, each showing all 6 conditions.

    Args:
        task: Task name (default: "click_bell")
    """
    # Larger font sizes for paper
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))

    # --- Top: Training Loss ---
    ax_train = axes[0]

    for cond_id in range(1, 7):
        epochs_data = load_training_metrics(task, cond_id)
        if not epochs_data:
            continue

        sorted_epochs = sorted(epochs_data.keys())
        train_losses = [epochs_data[e].get("train_loss") for e in sorted_epochs]

        valid_data = [(e, l) for e, l in zip(sorted_epochs, train_losses) if l is not None]
        if not valid_data:
            continue

        epochs, losses = zip(*valid_data)

        # Apply smoothing for cleaner visualization
        if len(losses) > 20:
            smoothed = smooth_curve(np.array(losses), window=20)
            epochs_smoothed = epochs[10:-9] if len(epochs) > 20 else epochs
            ax_train.plot(epochs_smoothed, smoothed,
                         label=f"C{cond_id}",
                         color=CONDITION_COLORS[cond_id],
                         linewidth=2.0,
                         alpha=0.9)
        else:
            ax_train.plot(epochs, losses,
                         label=f"C{cond_id}",
                         color=CONDITION_COLORS[cond_id],
                         linewidth=2.0,
                         alpha=0.9)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Training Loss")
    ax_train.set_title(f"Training Loss - {TASK_DESCRIPTIONS[task]}")
    ax_train.set_xlim(0, 500)
    ax_train.grid(True, alpha=0.3)
    ax_train.set_axisbelow(True)
    ax_train.legend(loc='upper right', ncol=2)

    # --- Bottom: Validation Loss ---
    ax_val = axes[1]

    for cond_id in range(1, 7):
        epochs_data = load_training_metrics(task, cond_id)
        if not epochs_data:
            continue

        sorted_epochs = sorted(epochs_data.keys())
        val_data = [(e, epochs_data[e].get("val_loss")) for e in sorted_epochs
                   if epochs_data[e].get("val_loss") is not None]

        if not val_data:
            continue

        epochs, val_losses = zip(*val_data)

        ax_val.plot(epochs, val_losses,
                   label=f"C{cond_id}",
                   color=CONDITION_COLORS[cond_id],
                   marker='o',
                   markersize=4,
                   linewidth=2.0,
                   alpha=0.8)

    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation Loss")
    ax_val.set_title(f"Validation Loss - {TASK_DESCRIPTIONS[task]}")
    ax_val.set_xlim(0, 500)
    ax_val.grid(True, alpha=0.3)
    ax_val.set_axisbelow(True)
    ax_val.legend(loc='upper right', ncol=2)

    plt.tight_layout()
    save_path = FIGURES_DIR / f"train_val_loss_{task}.{FIGURE_FORMAT}"
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Reset font sizes to default
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })


def plot_convergence_analysis():
    """Analyze convergence speed - epochs to reach certain loss thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate epochs to reach 50% of initial loss
    convergence_data = []

    for task in TASKS:
        for cond_id in range(1, 7):
            epochs_data = load_training_metrics(task, cond_id)
            if not epochs_data:
                continue

            sorted_epochs = sorted(epochs_data.keys())
            train_losses = [(e, epochs_data[e].get("train_loss")) for e in sorted_epochs
                           if epochs_data[e].get("train_loss") is not None]

            if len(train_losses) < 10:
                continue

            epochs, losses = zip(*train_losses)
            initial_loss = losses[0]
            target_loss = initial_loss * 0.5  # 50% of initial

            # Find first epoch where loss drops below target
            convergence_epoch = None
            for e, l in zip(epochs, losses):
                if l < target_loss:
                    convergence_epoch = e
                    break

            if convergence_epoch is not None:
                convergence_data.append({
                    "task": task,
                    "condition": cond_id,
                    "convergence_epoch": convergence_epoch,
                    "initial_loss": initial_loss,
                    "final_loss": losses[-1]
                })

    # Create grouped bar chart
    x = np.arange(len(TASKS))
    width = 0.12

    for i, cond_id in enumerate(range(1, 7)):
        values = []
        for task in TASKS:
            match = [d for d in convergence_data if d["task"] == task and d["condition"] == cond_id]
            values.append(match[0]["convergence_epoch"] if match else 0)

        ax.bar(x + i*width - 2.5*width, values, width,
               label=f"C{cond_id}: {CONDITIONS[cond_id]['short_name']}",
               color=CONDITION_COLORS[cond_id],
               edgecolor='white',
               linewidth=0.5)

    ax.set_ylabel("Epochs to 50% Loss Reduction")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DESCRIPTIONS[t] for t in TASKS], rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Convergence Speed Analysis")
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_path = FIGURES_DIR / f"convergence_analysis.{FIGURE_FORMAT}"
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_statistics_table():
    """Generate statistics for the markdown report."""
    stats = []

    for task in TASKS:
        for cond_id in range(1, 7):
            epochs_data = load_training_metrics(task, cond_id)
            if not epochs_data:
                continue

            sorted_epochs = sorted(epochs_data.keys())
            train_losses = [epochs_data[e].get("train_loss") for e in sorted_epochs
                           if epochs_data[e].get("train_loss") is not None]
            val_losses = [epochs_data[e].get("val_loss") for e in sorted_epochs
                         if epochs_data[e].get("val_loss") is not None]

            if train_losses:
                stats.append({
                    "task": TASK_DESCRIPTIONS[task],
                    "condition": cond_id,
                    "initial_train_loss": train_losses[0],
                    "final_train_loss": train_losses[-1],
                    "min_train_loss": min(train_losses),
                    "final_val_loss": val_losses[-1] if val_losses else None,
                    "min_val_loss": min(val_losses) if val_losses else None,
                    "loss_reduction": (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
                })

    return stats


def main():
    # Ensure output directories exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Curve Analysis")
    print("=" * 60)

    print("\nGenerating plots...")
    plot_training_curves_by_task()
    plot_val_loss_comparison()
    plot_final_loss_comparison()
    plot_convergence_analysis()
    plot_single_task_train_val_loss("click_bell")  # For paper Figure 5

    print("\nGenerating statistics...")
    stats = generate_statistics_table()

    # Print summary table
    print("\n--- Training Statistics Summary ---")
    print(f"{'Task':<25} {'C'} {'Init Loss':>10} {'Final Loss':>10} {'Min Loss':>10} {'Reduction':>10}")
    print("-" * 75)
    for s in stats:
        print(f"{s['task']:<25} {s['condition']} {s['initial_train_loss']:>10.4f} {s['final_train_loss']:>10.4f} {s['min_train_loss']:>10.4f} {s['loss_reduction']:>9.1f}%")

    # Save stats as JSON
    stats_path = TABLES_DIR / "training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {stats_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
