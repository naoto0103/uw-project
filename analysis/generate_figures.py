#!/usr/bin/env python3
"""
Generate publication-ready figures for HAMSTER + ManiFlow evaluation.

Outputs:
- Success rate bar charts
- Hypothesis comparison charts
- Path generation statistics charts

Usage:
    python generate_figures.py
    python generate_figures.py --no-show
    python generate_figures.py --format pdf
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import (
    TASKS,
    TASK_DESCRIPTIONS,
    CONDITIONS,
    CONDITION_COLORS,
    FIGURES_DIR,
    FIGURE_DPI,
    FIGURE_FORMAT,
)
from scripts.load_results import (
    load_all_results,
    TaskResults,
    get_success_rates_table,
    get_path_stats_table,
)
from scripts.utils import ensure_output_dirs
from analyze_all_tasks import compute_average_success_rates, aggregate_hypothesis_results


# =============================================================================
# Style Configuration
# =============================================================================

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Font settings
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
})

# Color schemes
IN_DOMAIN_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # C1, C2, C3
CROSS_DOMAIN_COLORS = ['#d62728', '#9467bd', '#8c564b']  # C4, C5, C6

MODE_COLORS = {
    'original': '#4a4a4a',       # gray - baseline
    'overlay_current': '#3498db', # blue - current path
    'overlay_initial_current': '#e74c3c',  # red - initial+current
}


# =============================================================================
# Main Results Figure: Success Rate Bar Chart
# =============================================================================

def plot_main_results(
    all_results: Dict[str, TaskResults],
    save_path: Optional[Path] = None,
    show: bool = True,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """
    Create a grouped bar chart of success rates across all tasks and conditions.

    Layout: Tasks on x-axis, grouped bars for each condition.
    """
    success_table = get_success_rates_table(all_results)

    # Larger font sizes for paper
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    # Clearer legend labels
    CONDITION_LABELS = {
        1: "C1: No Path (Cluttered)",
        2: "C2: Current (Cluttered)",
        3: "C3: Initial+Current (Cluttered)",
        4: "C4: No Path (Clean)",
        5: "C5: Current (Clean)",
        6: "C6: Initial+Current (Clean)",
    }

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar parameters
    n_tasks = len(TASKS)
    n_conditions = 6
    bar_width = 0.12
    x = np.arange(n_tasks)

    # Plot bars for each condition
    for i, cond_id in enumerate(range(1, 7)):
        rates = []
        for task in TASKS:
            rate = success_table.get(task, {}).get(cond_id)
            rates.append(rate if rate is not None else 0)

        offset = (i - (n_conditions - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            rates,
            bar_width,
            label=CONDITION_LABELS[cond_id],
            color=CONDITION_COLORS[cond_id],
            edgecolor='white',
            linewidth=0.5,
        )

        # Add value labels on top of bars
        for bar, rate in zip(bars, rates):
            if rate > 0:
                height = bar.get_height()
                ax.annotate(
                    f'{rate:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

    # Customize axes
    ax.set_xlabel('Task')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rates Across All Conditions')
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS])
    ax.set_ylim(0, 105)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
    )

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    # Reset font sizes to default
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    return fig


def plot_main_results_grouped(
    all_results: Dict[str, TaskResults],
    save_path: Optional[Path] = None,
    show: bool = True,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """
    Create a grouped bar chart with in-domain and cross-domain side by side.

    Layout: Two groups per task (in-domain: C1-C3, cross-domain: C4-C6)
    """
    success_table = get_success_rates_table(all_results)

    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    n_tasks = len(TASKS)
    bar_width = 0.25
    x = np.arange(n_tasks)

    # In-domain (C1-C3)
    ax1 = axes[0]
    for i, cond_id in enumerate([1, 2, 3]):
        rates = []
        for task in TASKS:
            rate = success_table.get(task, {}).get(cond_id)
            rates.append(rate if rate is not None else 0)

        offset = (i - 1) * bar_width
        bars = ax1.bar(
            x + offset,
            rates,
            bar_width,
            label=CONDITIONS[cond_id]['mode'].replace('overlay_', '').replace('_', '+'),
            color=IN_DOMAIN_COLORS[i],
            edgecolor='white',
            linewidth=0.5,
        )

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax1.annotate(
                    f'{rate:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

    ax1.set_xlabel('Task')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('In-Domain (Trained on Cluttered)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS], rotation=15, ha='right')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right', title='Mode')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Cross-domain (C4-C6)
    ax2 = axes[1]
    for i, cond_id in enumerate([4, 5, 6]):
        rates = []
        for task in TASKS:
            rate = success_table.get(task, {}).get(cond_id)
            rates.append(rate if rate is not None else 0)

        offset = (i - 1) * bar_width
        bars = ax2.bar(
            x + offset,
            rates,
            bar_width,
            label=CONDITIONS[cond_id]['mode'].replace('overlay_', '').replace('_', '+'),
            color=CROSS_DOMAIN_COLORS[i],
            edgecolor='white',
            linewidth=0.5,
        )

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax2.annotate(
                    f'{rate:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

    ax2.set_xlabel('Task')
    ax2.set_title('Cross-Domain (Trained on Clean, Eval on Cluttered)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS], rotation=15, ha='right')
    ax2.legend(loc='upper right', title='Mode')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# Hypothesis Comparison Figures
# =============================================================================

def plot_hypothesis_h1_h2(
    all_results: Dict[str, TaskResults],
    save_path: Optional[Path] = None,
    show: bool = True,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """
    Create a comparison chart for H1 (in-domain) and H2 (cross-domain) improvements.

    Shows the improvement from adding path guidance (C1→C3 vs C4→C6).
    """
    aggregated = aggregate_hypothesis_results(all_results)

    fig, ax = plt.subplots(figsize=(8, 5))

    n_tasks = len(TASKS)
    bar_width = 0.35
    x = np.arange(n_tasks)

    h1_improvements = []
    h2_improvements = []

    for task in TASKS:
        # H1: In-domain (C1 → C3)
        h1_data = aggregated["H1"]["by_task"].get(task, {})
        h1_imp = 0
        for comp in h1_data.get("comparisons", []):
            if "cond3" in comp["label"]:
                h1_imp = comp["improvement"]
                break
        h1_improvements.append(h1_imp)

        # H2: Cross-domain (C4 → C6)
        h2_data = aggregated["H2"]["by_task"].get(task, {})
        h2_imp = 0
        for comp in h2_data.get("comparisons", []):
            if "cond6" in comp["label"]:
                h2_imp = comp["improvement"]
                break
        h2_improvements.append(h2_imp)

    # Plot bars
    bars1 = ax.bar(
        x - bar_width/2,
        h1_improvements,
        bar_width,
        label='In-Domain (C1→C3)',
        color='#3498db',
        edgecolor='white',
    )
    bars2 = ax.bar(
        x + bar_width/2,
        h2_improvements,
        bar_width,
        label='Cross-Domain (C4→C6)',
        color='#e74c3c',
        edgecolor='white',
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            sign = '+' if height >= 0 else ''
            ax.annotate(
                f'{sign}{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=9,
            )

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title('H1 & H2: Effect of Path Guidance\n(Original → Initial+Current)')
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS])
    ax.legend(loc='best')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_hypothesis_h3(
    all_results: Dict[str, TaskResults],
    save_path: Optional[Path] = None,
    show: bool = True,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """
    Create a comparison chart for H3 (initial path benefit).

    Shows the improvement from current-only to initial+current (C2→C3 vs C5→C6).
    """
    aggregated = aggregate_hypothesis_results(all_results)

    fig, ax = plt.subplots(figsize=(8, 5))

    n_tasks = len(TASKS)
    bar_width = 0.35
    x = np.arange(n_tasks)

    in_domain_improvements = []
    cross_domain_improvements = []

    for task in TASKS:
        h3_data = aggregated["H3"]["by_task"].get(task, {})
        in_imp = 0
        cross_imp = 0
        for comp in h3_data.get("comparisons", []):
            if "in-domain" in comp["label"]:
                in_imp = comp["improvement"]
            elif "cross-domain" in comp["label"]:
                cross_imp = comp["improvement"]
        in_domain_improvements.append(in_imp)
        cross_domain_improvements.append(cross_imp)

    # Plot bars
    bars1 = ax.bar(
        x - bar_width/2,
        in_domain_improvements,
        bar_width,
        label='In-Domain (C2→C3)',
        color='#3498db',
        edgecolor='white',
    )
    bars2 = ax.bar(
        x + bar_width/2,
        cross_domain_improvements,
        bar_width,
        label='Cross-Domain (C5→C6)',
        color='#e74c3c',
        edgecolor='white',
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            sign = '+' if height >= 0 else ''
            ax.annotate(
                f'{sign}{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=9,
            )

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title('H3: Benefit of Initial Path (Memory Function)\n(Current-only → Initial+Current)')
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS])
    ax.legend(loc='best')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_hypothesis_h4(
    all_results: Dict[str, TaskResults],
    save_path: Optional[Path] = None,
    show: bool = True,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """
    Create a comparison chart for H4 (larger effect in cross-domain).

    Shows side-by-side comparison of in-domain vs cross-domain improvements.
    """
    aggregated = aggregate_hypothesis_results(all_results)

    fig, ax = plt.subplots(figsize=(8, 5))

    n_tasks = len(TASKS)
    bar_width = 0.35
    x = np.arange(n_tasks)

    in_domain_effects = []
    cross_domain_effects = []

    for task in TASKS:
        h4_data = aggregated["H4"]["by_task"].get(task, {})
        comps = h4_data.get("comparisons", [])
        if comps:
            # Use the initial+current comparison (second one)
            comp = comps[1] if len(comps) > 1 else comps[0]
            in_domain_effects.append(comp["in_domain_improvement"])
            cross_domain_effects.append(comp["cross_domain_improvement"])
        else:
            in_domain_effects.append(0)
            cross_domain_effects.append(0)

    # Plot bars
    bars1 = ax.bar(
        x - bar_width/2,
        in_domain_effects,
        bar_width,
        label='In-Domain Effect',
        color='#3498db',
        edgecolor='white',
    )
    bars2 = ax.bar(
        x + bar_width/2,
        cross_domain_effects,
        bar_width,
        label='Cross-Domain Effect',
        color='#e74c3c',
        edgecolor='white',
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            sign = '+' if height >= 0 else ''
            ax.annotate(
                f'{sign}{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=9,
            )

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title('H4: Path Guidance Effect Size Comparison\n(Δ In-Domain vs Δ Cross-Domain)')
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS])
    ax.legend(loc='best')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# Path Generation Statistics Figure
# =============================================================================

def plot_path_stats(
    all_results: Dict[str, TaskResults],
    save_path: Optional[Path] = None,
    show: bool = True,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """
    Create a figure showing path generation statistics for VILA conditions.
    """
    path_table = get_path_stats_table(all_results)
    vila_conditions = [cid for cid, c in CONDITIONS.items() if c["uses_vila"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    n_tasks = len(TASKS)
    bar_width = 0.2
    x = np.arange(n_tasks)

    colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']  # C2, C3, C5, C6

    # Plot 1: Path Success Rate
    ax1 = axes[0]
    for i, cond_id in enumerate(vila_conditions):
        rates = []
        for task in TASKS:
            stats = path_table.get(task, {}).get(cond_id)
            rates.append(stats['path_success_rate'] if stats else 0)

        offset = (i - (len(vila_conditions) - 1) / 2) * bar_width
        ax1.bar(
            x + offset,
            rates,
            bar_width,
            label=f"C{cond_id}",
            color=colors[i],
            edgecolor='white',
        )

    ax1.set_xlabel('Task')
    ax1.set_ylabel('Path Success Rate (%)')
    ax1.set_title('VILA Path Generation Success')
    ax1.set_xticks(x)
    ax1.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS], rotation=15, ha='right')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='best', ncol=2)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Plot 2: Frame 0 Success Rate
    ax2 = axes[1]
    for i, cond_id in enumerate(vila_conditions):
        rates = []
        for task in TASKS:
            stats = path_table.get(task, {}).get(cond_id)
            rates.append(stats['frame0_success_rate'] if stats else 0)

        offset = (i - (len(vila_conditions) - 1) / 2) * bar_width
        ax2.bar(
            x + offset,
            rates,
            bar_width,
            label=f"C{cond_id}",
            color=colors[i],
            edgecolor='white',
        )

    ax2.set_xlabel('Task')
    ax2.set_ylabel('Frame 0 Success Rate (%)')
    ax2.set_title('Initial Frame Path Success')
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS], rotation=15, ha='right')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='best', ncol=2)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    # Plot 3: Average Fallbacks
    ax3 = axes[2]
    for i, cond_id in enumerate(vila_conditions):
        fallbacks = []
        for task in TASKS:
            stats = path_table.get(task, {}).get(cond_id)
            fallbacks.append(stats['avg_fallbacks'] if stats else 0)

        offset = (i - (len(vila_conditions) - 1) / 2) * bar_width
        ax3.bar(
            x + offset,
            fallbacks,
            bar_width,
            label=f"C{cond_id}",
            color=colors[i],
            edgecolor='white',
        )

    ax3.set_xlabel('Task')
    ax3.set_ylabel('Avg Fallbacks per Episode')
    ax3.set_title('Fallback Usage')
    ax3.set_xticks(x)
    ax3.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS], rotation=15, ha='right')
    ax3.legend(loc='best', ncol=2)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax3.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# Summary Figure (for paper)
# =============================================================================

def plot_summary_figure(
    all_results: Dict[str, TaskResults],
    save_path: Optional[Path] = None,
    show: bool = True,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """
    Create a comprehensive summary figure with multiple panels.

    This is the main figure for the paper, showing:
    - (a) In-domain results
    - (b) Cross-domain results
    - (c) Path guidance effect comparison
    """
    success_table = get_success_rates_table(all_results)
    aggregated = aggregate_hypothesis_results(all_results)

    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    n_tasks = len(TASKS)
    bar_width = 0.25
    x = np.arange(n_tasks)

    # Panel A: In-domain results (C1-C3)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, cond_id in enumerate([1, 2, 3]):
        rates = []
        for task in TASKS:
            rate = success_table.get(task, {}).get(cond_id)
            rates.append(rate if rate is not None else 0)

        offset = (i - 1) * bar_width
        ax1.bar(
            x + offset,
            rates,
            bar_width,
            label=CONDITIONS[cond_id]['mode'].replace('overlay_', '').replace('_', '+'),
            color=IN_DOMAIN_COLORS[i],
            edgecolor='white',
        )

    ax1.set_xlabel('Task')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('(a) In-Domain: Trained & Evaluated on Cluttered')
    ax1.set_xticks(x)
    ax1.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS])
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right', title='Mode')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Panel B: Cross-domain results (C4-C6)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, cond_id in enumerate([4, 5, 6]):
        rates = []
        for task in TASKS:
            rate = success_table.get(task, {}).get(cond_id)
            rates.append(rate if rate is not None else 0)

        offset = (i - 1) * bar_width
        ax2.bar(
            x + offset,
            rates,
            bar_width,
            label=CONDITIONS[cond_id]['mode'].replace('overlay_', '').replace('_', '+'),
            color=CROSS_DOMAIN_COLORS[i],
            edgecolor='white',
        )

    ax2.set_xlabel('Task')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('(b) Cross-Domain: Trained on Clean, Eval on Cluttered')
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS])
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right', title='Mode')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    # Panel C: H1 & H2 comparison (path guidance effect)
    ax3 = fig.add_subplot(gs[1, 0])

    h1_improvements = []
    h2_improvements = []

    for task in TASKS:
        h1_data = aggregated["H1"]["by_task"].get(task, {})
        h1_imp = 0
        for comp in h1_data.get("comparisons", []):
            if "cond3" in comp["label"]:
                h1_imp = comp["improvement"]
                break
        h1_improvements.append(h1_imp)

        h2_data = aggregated["H2"]["by_task"].get(task, {})
        h2_imp = 0
        for comp in h2_data.get("comparisons", []):
            if "cond6" in comp["label"]:
                h2_imp = comp["improvement"]
                break
        h2_improvements.append(h2_imp)

    bar_width_c = 0.35
    bars1 = ax3.bar(
        x - bar_width_c/2,
        h1_improvements,
        bar_width_c,
        label='In-Domain (C1→C3)',
        color='#3498db',
        edgecolor='white',
    )
    bars2 = ax3.bar(
        x + bar_width_c/2,
        h2_improvements,
        bar_width_c,
        label='Cross-Domain (C4→C6)',
        color='#e74c3c',
        edgecolor='white',
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            sign = '+' if height >= 0 else ''
            ax3.annotate(
                f'{sign}{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=9,
            )

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Improvement (pp)')
    ax3.set_title('(c) Path Guidance Effect (Original → Initial+Current)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS])
    ax3.legend(loc='best')
    ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax3.set_axisbelow(True)

    # Panel D: H3 comparison (initial path benefit)
    ax4 = fig.add_subplot(gs[1, 1])

    h3_in_domain = []
    h3_cross_domain = []

    for task in TASKS:
        h3_data = aggregated["H3"]["by_task"].get(task, {})
        in_imp = 0
        cross_imp = 0
        for comp in h3_data.get("comparisons", []):
            if "in-domain" in comp["label"]:
                in_imp = comp["improvement"]
            elif "cross-domain" in comp["label"]:
                cross_imp = comp["improvement"]
        h3_in_domain.append(in_imp)
        h3_cross_domain.append(cross_imp)

    bars1 = ax4.bar(
        x - bar_width_c/2,
        h3_in_domain,
        bar_width_c,
        label='In-Domain (C2→C3)',
        color='#3498db',
        edgecolor='white',
    )
    bars2 = ax4.bar(
        x + bar_width_c/2,
        h3_cross_domain,
        bar_width_c,
        label='Cross-Domain (C5→C6)',
        color='#e74c3c',
        edgecolor='white',
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            sign = '+' if height >= 0 else ''
            ax4.annotate(
                f'{sign}{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=9,
            )

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('Task')
    ax4.set_ylabel('Improvement (pp)')
    ax4.set_title('(d) Initial Path Benefit (Current → Initial+Current)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([TASK_DESCRIPTIONS.get(t, t) for t in TASKS])
    ax4.legend(loc='best')
    ax4.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax4.set_axisbelow(True)

    plt.suptitle('HAMSTER + ManiFlow Evaluation Results', fontsize=14, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# Average Results Bar Chart
# =============================================================================

def plot_average_results(
    all_results: Dict[str, TaskResults],
    save_path: Optional[Path] = None,
    show: bool = True,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """
    Create a bar chart showing average success rates across all tasks.
    """
    avg_rates = compute_average_success_rates(all_results)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(6)
    colors = [CONDITION_COLORS[i+1] for i in range(6)]

    rates = [avg_rates.get(i+1, 0) for i in range(6)]

    bars = ax.bar(x, rates, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.annotate(
            f'{rate:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    # Add condition labels
    labels = [f"C{i+1}\n{CONDITIONS[i+1]['short_name']}" for i in range(6)]

    ax.set_xlabel('Condition')
    ax.set_ylabel('Average Success Rate (%)')
    ax.set_title('Average Success Rate Across All Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(rates) * 1.15 if rates else 100)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Add divider between in-domain and cross-domain
    ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1, ax.get_ylim()[1] * 0.95, 'In-Domain', ha='center', fontsize=10, style='italic')
    ax.text(4, ax.get_ylim()[1] * 0.95, 'Cross-Domain', ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for HAMSTER + ManiFlow evaluation."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display figures (just save)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default=FIGURE_FORMAT,
        help="Output format for figures",
    )

    args = parser.parse_args()
    show = not args.no_show
    fig_format = args.format

    print("Loading evaluation results...")
    all_results = load_all_results()

    ensure_output_dirs()

    print(f"\nGenerating figures (format: {fig_format})...")

    # Main results
    plot_main_results(
        all_results,
        save_path=FIGURES_DIR / f"main_results_all.{fig_format}",
        show=show,
        fig_format=fig_format,
    )

    plot_main_results_grouped(
        all_results,
        save_path=FIGURES_DIR / f"main_results_grouped.{fig_format}",
        show=show,
        fig_format=fig_format,
    )

    # Hypothesis figures
    plot_hypothesis_h1_h2(
        all_results,
        save_path=FIGURES_DIR / f"hypothesis_h1_h2.{fig_format}",
        show=show,
        fig_format=fig_format,
    )

    plot_hypothesis_h3(
        all_results,
        save_path=FIGURES_DIR / f"hypothesis_h3.{fig_format}",
        show=show,
        fig_format=fig_format,
    )

    plot_hypothesis_h4(
        all_results,
        save_path=FIGURES_DIR / f"hypothesis_h4.{fig_format}",
        show=show,
        fig_format=fig_format,
    )

    # Path statistics
    plot_path_stats(
        all_results,
        save_path=FIGURES_DIR / f"path_stats.{fig_format}",
        show=show,
        fig_format=fig_format,
    )

    # Average results
    plot_average_results(
        all_results,
        save_path=FIGURES_DIR / f"average_results.{fig_format}",
        show=show,
        fig_format=fig_format,
    )

    # Summary figure (main paper figure)
    plot_summary_figure(
        all_results,
        save_path=FIGURES_DIR / f"summary_figure.{fig_format}",
        show=show,
        fig_format=fig_format,
    )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
