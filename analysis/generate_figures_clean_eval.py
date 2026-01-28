#!/usr/bin/env python3
"""
Generate publication-ready figures for Beat Block Hammer clean environment evaluation.

Produces VILA (C1-C6) and GT (GT-C2,C3,C5,C6) figures matching the exact style
of existing cluttered evaluation figures.

Usage:
    python generate_figures_clean_eval.py
    python generate_figures_clean_eval.py --no-show
    python generate_figures_clean_eval.py --format pdf
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use('Agg')
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
    HYPOTHESES,
)
from scripts.load_results import (
    load_all_results as load_all_results_vila,
    TaskResults,
    get_success_rates_table,
    get_path_stats_table,
)
from scripts.load_results_gt import (
    load_all_results as load_all_results_gt,
    get_success_rates_table as get_success_rates_table_gt,
)
from config_gt import CONDITIONS as GT_CONDITIONS, CONDITION_COLORS as GT_CONDITION_COLORS
from scripts.utils import ensure_output_dirs


# =============================================================================
# Style Configuration (identical to generate_figures.py)
# =============================================================================

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

IN_DOMAIN_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # C1, C2, C3
CROSS_DOMAIN_COLORS = ['#d62728', '#9467bd', '#8c564b']  # C4, C5, C6
GT_IN_DOMAIN_COLORS = ['#ff7f0e', '#2ca02c']  # GT-C2, GT-C3
GT_CROSS_DOMAIN_COLORS = ['#9467bd', '#8c564b']  # GT-C5, GT-C6

PAPER_FONT = {
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
}

DEFAULT_FONT = {
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
}

TASK = "beat_block_hammer"
TASK_LABEL = TASK_DESCRIPTIONS[TASK]
EVAL_ENV = "clean"


# =============================================================================
# Helper: add value label on bar
# =============================================================================

def add_bar_labels(ax, bars, rates, fontsize=9, include_zero=False, fmt='{:.0f}'):
    for bar, rate in zip(bars, rates):
        if rate > 0 or include_zero:
            height = bar.get_height()
            ax.annotate(
                fmt.format(rate),
                xy=(bar.get_x() + bar.get_width() / 2, max(height, 1)),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=fontsize,
            )


# =============================================================================
# VILA Figures
# =============================================================================

def plot_vila_main_results_all(
    success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """All 6 VILA conditions bar chart (single task)."""
    plt.rcParams.update(PAPER_FONT)

    CONDITION_LABELS = {
        1: "C1: No Path (Cluttered)",
        2: "C2: Current (Cluttered)",
        3: "C3: Initial+Current (Cluttered)",
        4: "C4: No Path (Clean)",
        5: "C5: Current (Clean)",
        6: "C6: Initial+Current (Clean)",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(6)
    rates = [success_rates.get(i+1, 0) or 0 for i in range(6)]
    colors = [CONDITION_COLORS[i+1] for i in range(6)]

    for i in range(6):
        bar = ax.bar(
            x[i], rates[i],
            color=colors[i], edgecolor='white', linewidth=0.5,
            label=CONDITION_LABELS[i+1],
        )

    # Add value labels
    for i, rate in enumerate(rates):
        ax.annotate(
            f'{rate:.0f}',
            xy=(x[i], max(rate, 1)),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
        )

    ax.set_xlabel('Condition')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'{TASK_LABEL}: Success Rates (Eval on Clean)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i+1}" for i in range(6)])
    ax.set_ylim(0, 105)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Divider
    ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1, ax.get_ylim()[1] * 0.95, 'In-Domain', ha='center', fontsize=12, style='italic')
    ax.text(4, ax.get_ylim()[1] * 0.95, 'Cross-Domain', ha='center', fontsize=12, style='italic')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_vila_grouped(
    success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """In-domain (C1-C3) vs Cross-domain (C4-C6) side by side."""
    plt.rcParams.update(PAPER_FONT)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bar_width = 0.5

    # In-domain
    ax1 = axes[0]
    for i, cond_id in enumerate([1, 2, 3]):
        rate = success_rates.get(cond_id, 0) or 0
        bars = ax1.bar(
            i, rate, bar_width,
            label=CONDITIONS[cond_id]['mode'].replace('overlay_', '').replace('_', '+'),
            color=IN_DOMAIN_COLORS[i],
            edgecolor='white', linewidth=0.5,
        )
        if rate > 0:
            ax1.annotate(
                f'{rate:.0f}',
                xy=(i, rate),
                xytext=(0, 2),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8,
            )

    ax1.set_xlabel('Mode')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('In-Domain (Trained on Cluttered)')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(['original', 'current', 'initial\n+current'])
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right', title='Mode')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Cross-domain
    ax2 = axes[1]
    for i, cond_id in enumerate([4, 5, 6]):
        rate = success_rates.get(cond_id, 0) or 0
        bars = ax2.bar(
            i, rate, bar_width,
            label=CONDITIONS[cond_id]['mode'].replace('overlay_', '').replace('_', '+'),
            color=CROSS_DOMAIN_COLORS[i],
            edgecolor='white', linewidth=0.5,
        )
        if rate > 0:
            ax2.annotate(
                f'{rate:.0f}',
                xy=(i, rate),
                xytext=(0, 2),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8,
            )

    ax2.set_xlabel('Mode')
    ax2.set_title('Cross-Domain (Trained on Clean)')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['original', 'current', 'initial\n+current'])
    ax2.legend(loc='upper right', title='Mode')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    plt.suptitle(f'{TASK_LABEL} - Eval on Clean', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_vila_in_domain(
    success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """In-domain results (C1 vs C2)."""
    plt.rcParams.update(PAPER_FONT)

    CONDITION_LABELS = {1: "C1: No Path", 2: "C2: Current"}

    fig, ax = plt.subplots(figsize=(6, 6))

    bar_width = 0.35
    x = np.arange(2)
    rates = [success_rates.get(1, 0) or 0, success_rates.get(2, 0) or 0]
    colors = [IN_DOMAIN_COLORS[0], IN_DOMAIN_COLORS[1]]
    labels = [CONDITION_LABELS[1], CONDITION_LABELS[2]]

    for i in range(2):
        bars = ax.bar(
            x[i], rates[i], bar_width,
            label=labels[i], color=colors[i],
            edgecolor='white', linewidth=0.5,
        )
        ax.annotate(
            f'{rates[i]:.0f}',
            xy=(x[i], max(rates[i], 1)),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=10,
        )

    ax.set_xlabel('Condition')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'In-Domain: {TASK_LABEL}\n(Eval on Clean)')
    ax.set_xticks(x)
    ax.set_xticklabels(['C1', 'C2'])
    ax.set_ylim(0, 105)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_vila_cross_domain(
    success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """Cross-domain results (C4 vs C5)."""
    plt.rcParams.update(PAPER_FONT)

    CONDITION_LABELS = {4: "C4: No Path", 5: "C5: Current"}

    fig, ax = plt.subplots(figsize=(6, 6))

    bar_width = 0.35
    x = np.arange(2)
    rates = [success_rates.get(4, 0) or 0, success_rates.get(5, 0) or 0]
    colors = [CROSS_DOMAIN_COLORS[0], CROSS_DOMAIN_COLORS[1]]
    labels = [CONDITION_LABELS[4], CONDITION_LABELS[5]]

    for i in range(2):
        bars = ax.bar(
            x[i], rates[i], bar_width,
            label=labels[i], color=colors[i],
            edgecolor='white', linewidth=0.5,
        )
        ax.annotate(
            f'{rates[i]:.0f}',
            xy=(x[i], max(rates[i], 1)),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=10,
        )

    ax.set_xlabel('Condition')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Cross-Domain: {TASK_LABEL}\n(Trained on Clean, Eval on Clean)')
    ax.set_xticks(x)
    ax.set_xticklabels(['C4', 'C5'])
    ax.set_ylim(0, 105)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_vila_memory_function(
    success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """Memory function comparison: C2 vs C3 and C5 vs C6."""
    plt.rcParams.update(PAPER_FONT)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    bar_width = 0.35
    x = np.arange(1)  # single task

    # Left: In-domain C2 vs C3
    ax1 = axes[0]
    rate_c2 = success_rates.get(2, 0) or 0
    rate_c3 = success_rates.get(3, 0) or 0

    bars1 = ax1.bar(x - bar_width/2, [rate_c2], bar_width,
                     label='C2: Current', color='#ff7f0e', edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x + bar_width/2, [rate_c3], bar_width,
                     label='C3: Initial+Current', color='#2ca02c', edgecolor='white', linewidth=0.5)

    for bar, rate in [(bars1[0], rate_c2), (bars2[0], rate_c3)]:
        ax1.annotate(f'{rate:.0f}', xy=(bar.get_x() + bar.get_width()/2, max(rate, 1)),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title(f'In-Domain: Memory Function Effect\n({TASK_LABEL}, Eval on Clean)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([TASK_LABEL])
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Right: Cross-domain C5 vs C6
    ax2 = axes[1]
    rate_c5 = success_rates.get(5, 0) or 0
    rate_c6 = success_rates.get(6, 0) or 0

    bars3 = ax2.bar(x - bar_width/2, [rate_c5], bar_width,
                     label='C5: Current', color='#9467bd', edgecolor='white', linewidth=0.5)
    bars4 = ax2.bar(x + bar_width/2, [rate_c6], bar_width,
                     label='C6: Initial+Current', color='#8c564b', edgecolor='white', linewidth=0.5)

    for bar, rate in [(bars3[0], rate_c5), (bars4[0], rate_c6)]:
        ax2.annotate(f'{rate:.0f}', xy=(bar.get_x() + bar.get_width()/2, max(rate, 1)),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    ax2.set_title(f'Cross-Domain: Memory Function Effect\n({TASK_LABEL}, Eval on Clean)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASK_LABEL])
    ax2.legend(loc='upper right')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_vila_training_env_comparison(
    success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """Compare in-domain (C1-C3 avg) vs cross-domain (C4-C6 avg)."""
    plt.rcParams.update(PAPER_FONT)

    fig, ax = plt.subplots(figsize=(6, 6))

    in_domain_rates = [(success_rates.get(i, 0) or 0) for i in [1, 2, 3]]
    cross_domain_rates = [(success_rates.get(i, 0) or 0) for i in [4, 5, 6]]
    in_avg = sum(in_domain_rates) / 3
    cross_avg = sum(cross_domain_rates) / 3

    bar_width = 0.35
    x = np.arange(2)

    bars1 = ax.bar(x[0], in_avg, bar_width,
                   label='Trained on Cluttered (C1-C3 avg)', color='#3498db',
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x[1], cross_avg, bar_width,
                   label='Trained on Clean (C4-C6 avg)', color='#e74c3c',
                   edgecolor='white', linewidth=0.5)

    for xi, rate in [(x[0], in_avg), (x[1], cross_avg)]:
        ax.annotate(f'{rate:.1f}', xy=(xi, max(rate, 1)),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Training Environment')
    ax.set_ylabel('Average Success Rate (%)')
    ax.set_title(f'Training Environment Comparison\n({TASK_LABEL}, Eval on Clean)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Cluttered', 'Clean'])
    ax.set_ylim(0, max(in_avg, cross_avg) * 1.2 if max(in_avg, cross_avg) > 0 else 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_vila_hypothesis_h1_h2(
    success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """H1 & H2: Path guidance effect (C1->C3, C4->C6)."""
    plt.rcParams.update(PAPER_FONT)

    fig, ax = plt.subplots(figsize=(6, 5))

    bar_width = 0.35
    x = np.arange(1)

    h1_imp = (success_rates.get(3, 0) or 0) - (success_rates.get(1, 0) or 0)
    h2_imp = (success_rates.get(6, 0) or 0) - (success_rates.get(4, 0) or 0)

    bars1 = ax.bar(x - bar_width/2, [h1_imp], bar_width,
                   label='In-Domain (C1->C3)', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + bar_width/2, [h2_imp], bar_width,
                   label='Cross-Domain (C4->C6)', color='#e74c3c', edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            sign = '+' if h >= 0 else ''
            ax.annotate(f'{sign}{h:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3 if h >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title(f'H1 & H2: Effect of Path Guidance\n({TASK_LABEL}, Eval on Clean)')
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL])
    ax.legend(loc='best')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_vila_hypothesis_h3(
    success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """H3: Initial path benefit (C2->C3, C5->C6)."""
    plt.rcParams.update(PAPER_FONT)

    fig, ax = plt.subplots(figsize=(6, 5))

    bar_width = 0.35
    x = np.arange(1)

    in_imp = (success_rates.get(3, 0) or 0) - (success_rates.get(2, 0) or 0)
    cross_imp = (success_rates.get(6, 0) or 0) - (success_rates.get(5, 0) or 0)

    bars1 = ax.bar(x - bar_width/2, [in_imp], bar_width,
                   label='In-Domain (C2->C3)', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + bar_width/2, [cross_imp], bar_width,
                   label='Cross-Domain (C5->C6)', color='#e74c3c', edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            sign = '+' if h >= 0 else ''
            ax.annotate(f'{sign}{h:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3 if h >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title(f'H3: Benefit of Initial Path (Memory Function)\n({TASK_LABEL}, Eval on Clean)')
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL])
    ax.legend(loc='best')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_vila_hypothesis_h4(
    success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """H4: Larger effect in cross-domain."""
    plt.rcParams.update(PAPER_FONT)

    fig, ax = plt.subplots(figsize=(6, 5))

    bar_width = 0.35
    x = np.arange(1)

    in_effect = (success_rates.get(3, 0) or 0) - (success_rates.get(1, 0) or 0)
    cross_effect = (success_rates.get(6, 0) or 0) - (success_rates.get(4, 0) or 0)

    bars1 = ax.bar(x - bar_width/2, [in_effect], bar_width,
                   label='In-Domain Effect', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + bar_width/2, [cross_effect], bar_width,
                   label='Cross-Domain Effect', color='#e74c3c', edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            sign = '+' if h >= 0 else ''
            ax.annotate(f'{sign}{h:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3 if h >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title(f'H4: Path Guidance Effect Size Comparison\n({TASK_LABEL}, Eval on Clean)')
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL])
    ax.legend(loc='best')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_vila_path_stats(
    task_results: TaskResults,
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """Path generation statistics for VILA conditions."""
    plt.rcParams.update(PAPER_FONT)

    vila_cond_ids = [cid for cid, c in CONDITIONS.items() if c["uses_vila"]]
    colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    bar_width = 0.18
    x = np.arange(1)

    # Path success rate
    ax1 = axes[0]
    for i, cid in enumerate(vila_cond_ids):
        cond = task_results.get_condition(cid)
        rate = cond.path_success_rate * 100 if cond else 0
        offset = (i - (len(vila_cond_ids) - 1) / 2) * bar_width
        ax1.bar(x + offset, [rate], bar_width, label=f"C{cid}", color=colors[i], edgecolor='white')

    ax1.set_ylabel('Path Success Rate (%)')
    ax1.set_title('VILA Path Generation Success')
    ax1.set_xticks(x)
    ax1.set_xticklabels([TASK_LABEL])
    ax1.set_ylim(0, 105)
    ax1.legend(loc='best', ncol=2)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Frame0 success rate
    ax2 = axes[1]
    for i, cid in enumerate(vila_cond_ids):
        cond = task_results.get_condition(cid)
        rate = cond.frame0_success_rate * 100 if cond else 0
        offset = (i - (len(vila_cond_ids) - 1) / 2) * bar_width
        ax2.bar(x + offset, [rate], bar_width, label=f"C{cid}", color=colors[i], edgecolor='white')

    ax2.set_ylabel('Frame 0 Success Rate (%)')
    ax2.set_title('Initial Frame Path Success')
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASK_LABEL])
    ax2.set_ylim(0, 105)
    ax2.legend(loc='best', ncol=2)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    # Fallbacks
    ax3 = axes[2]
    for i, cid in enumerate(vila_cond_ids):
        cond = task_results.get_condition(cid)
        fb = cond.avg_fallbacks_per_episode if cond else 0
        offset = (i - (len(vila_cond_ids) - 1) / 2) * bar_width
        ax3.bar(x + offset, [fb], bar_width, label=f"C{cid}", color=colors[i], edgecolor='white')

    ax3.set_ylabel('Avg Fallbacks per Episode')
    ax3.set_title('Fallback Usage')
    ax3.set_xticks(x)
    ax3.set_xticklabels([TASK_LABEL])
    ax3.legend(loc='best', ncol=2)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax3.set_axisbelow(True)

    plt.suptitle(f'{TASK_LABEL} - Path Stats (Eval on Clean)', fontsize=16)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


# =============================================================================
# GT Figures
# =============================================================================

def plot_gt_main_results_all(
    gt_success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """All 4 GT conditions bar chart."""
    plt.rcParams.update(PAPER_FONT)

    CONDITION_LABELS = {
        2: "GT-C2: Current (Cluttered)",
        3: "GT-C3: Initial+Current (Cluttered)",
        5: "GT-C5: Current (Clean)",
        6: "GT-C6: Initial+Current (Clean)",
    }

    gt_cond_ids = [2, 3, 5, 6]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(4)
    rates = [gt_success_rates.get(c, 0) or 0 for c in gt_cond_ids]
    colors = [GT_CONDITION_COLORS[c] for c in gt_cond_ids]

    for i, cond_id in enumerate(gt_cond_ids):
        ax.bar(
            x[i], rates[i],
            color=colors[i], edgecolor='white', linewidth=0.5,
            label=CONDITION_LABELS[cond_id],
        )

    # Add value labels
    for i, rate in enumerate(rates):
        ax.annotate(
            f'{rate:.0f}',
            xy=(x[i], max(rate, 1)),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
        )

    ax.set_xlabel('Condition')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'GT Path Trained Models: {TASK_LABEL}\n(Eval on Clean)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"GT-C{c}" for c in gt_cond_ids])
    ax.set_ylim(0, 105)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Divider between in-domain and cross-domain
    ax.axvline(x=1.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, 'In-Domain', ha='center', fontsize=12, style='italic')
    ax.text(2.5, ax.get_ylim()[1] * 0.95, 'Cross-Domain', ha='center', fontsize=12, style='italic')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


def plot_gt_grouped(
    gt_success_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """GT In-domain (GT-C2,C3) vs Cross-domain (GT-C5,C6) side by side."""
    plt.rcParams.update(PAPER_FONT)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bar_width = 0.35
    x = np.arange(1)

    # In-domain
    ax1 = axes[0]
    for i, cond_id in enumerate([2, 3]):
        rate = gt_success_rates.get(cond_id, 0) or 0
        bars = ax1.bar(
            x + (i - 0.5) * bar_width, [rate], bar_width,
            label=GT_CONDITIONS[cond_id]['mode'].replace('overlay_', '').replace('_', '+'),
            color=GT_IN_DOMAIN_COLORS[i],
            edgecolor='white', linewidth=0.5,
        )
        if rate > 0:
            ax1.annotate(f'{rate:.0f}',
                         xy=(bars[0].get_x() + bars[0].get_width()/2, rate),
                         xytext=(0, 2), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('GT Path In-Domain (Trained on Cluttered)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([TASK_LABEL])
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right', title='Mode')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # Cross-domain
    ax2 = axes[1]
    for i, cond_id in enumerate([5, 6]):
        rate = gt_success_rates.get(cond_id, 0) or 0
        bars = ax2.bar(
            x + (i - 0.5) * bar_width, [rate], bar_width,
            label=GT_CONDITIONS[cond_id]['mode'].replace('overlay_', '').replace('_', '+'),
            color=GT_CROSS_DOMAIN_COLORS[i],
            edgecolor='white', linewidth=0.5,
        )
        if rate > 0:
            ax2.annotate(f'{rate:.0f}',
                         xy=(bars[0].get_x() + bars[0].get_width()/2, rate),
                         xytext=(0, 2), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

    ax2.set_title('GT Path Cross-Domain (Trained on Clean)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([TASK_LABEL])
    ax2.legend(loc='upper right', title='Mode')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    plt.suptitle(f'{TASK_LABEL} - GT Path (Eval on Clean)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


# =============================================================================
# Summary Figure
# =============================================================================

def plot_summary_figure(
    vila_rates: Dict[int, float],
    gt_rates: Dict[int, float],
    save_path: Optional[Path] = None,
    fig_format: str = FIGURE_FORMAT,
) -> plt.Figure:
    """4-panel summary: (a) VILA all, (b) GT all, (c) VILA vs GT current, (d) VILA vs GT init+curr."""
    plt.rcParams.update(PAPER_FONT)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # (a) VILA all conditions
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(6)
    rates_vila = [vila_rates.get(i+1, 0) or 0 for i in range(6)]
    colors = [CONDITION_COLORS[i+1] for i in range(6)]
    for i in range(6):
        ax1.bar(x[i], rates_vila[i], color=colors[i], edgecolor='white', linewidth=0.5)
        ax1.annotate(f'{rates_vila[i]:.0f}', xy=(x[i], max(rates_vila[i], 1)),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    ax1.set_title('(a) VILA Path: All Conditions')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{i+1}' for i in range(6)])
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim(0, 105)
    ax1.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    # (b) GT all conditions
    ax2 = fig.add_subplot(gs[0, 1])
    gt_conds = [2, 3, 5, 6]
    x_gt = np.arange(4)
    rates_gt = [gt_rates.get(c, 0) or 0 for c in gt_conds]
    gt_colors = [GT_CONDITION_COLORS[c] for c in gt_conds]
    for i in range(4):
        ax2.bar(x_gt[i], rates_gt[i], color=gt_colors[i], edgecolor='white', linewidth=0.5)
        ax2.annotate(f'{rates_gt[i]:.0f}', xy=(x_gt[i], max(rates_gt[i], 1)),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    ax2.set_title('(b) GT Path: All Conditions')
    ax2.set_xticks(x_gt)
    ax2.set_xticklabels([f'GT-C{c}' for c in gt_conds])
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(0, 105)
    ax2.axvline(x=1.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    # (c) VILA vs GT: current path conditions
    ax3 = fig.add_subplot(gs[1, 0])
    bar_width = 0.35
    x_c = np.arange(2)  # in-domain, cross-domain
    vila_cur = [vila_rates.get(2, 0) or 0, vila_rates.get(5, 0) or 0]
    gt_cur = [gt_rates.get(2, 0) or 0, gt_rates.get(5, 0) or 0]

    bars1 = ax3.bar(x_c - bar_width/2, vila_cur, bar_width, label='VILA Path', color='#3498db', edgecolor='white')
    bars2 = ax3.bar(x_c + bar_width/2, gt_cur, bar_width, label='GT Path', color='#e74c3c', edgecolor='white')

    for bars, vals in [(bars1, vila_cur), (bars2, gt_cur)]:
        for bar, v in zip(bars, vals):
            ax3.annotate(f'{v:.0f}', xy=(bar.get_x() + bar.get_width()/2, max(v, 1)),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)

    ax3.set_title('(c) VILA vs GT: Current Path')
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(['In-Domain\n(C2 vs GT-C2)', 'Cross-Domain\n(C5 vs GT-C5)'])
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_ylim(0, 105)
    ax3.legend(loc='upper right')
    ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax3.set_axisbelow(True)

    # (d) VILA vs GT: initial+current path conditions
    ax4 = fig.add_subplot(gs[1, 1])
    vila_ic = [vila_rates.get(3, 0) or 0, vila_rates.get(6, 0) or 0]
    gt_ic = [gt_rates.get(3, 0) or 0, gt_rates.get(6, 0) or 0]

    bars1 = ax4.bar(x_c - bar_width/2, vila_ic, bar_width, label='VILA Path', color='#3498db', edgecolor='white')
    bars2 = ax4.bar(x_c + bar_width/2, gt_ic, bar_width, label='GT Path', color='#e74c3c', edgecolor='white')

    for bars, vals in [(bars1, vila_ic), (bars2, gt_ic)]:
        for bar, v in zip(bars, vals):
            ax4.annotate(f'{v:.0f}', xy=(bar.get_x() + bar.get_width()/2, max(v, 1)),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)

    ax4.set_title('(d) VILA vs GT: Initial+Current Path')
    ax4.set_xticks(x_c)
    ax4.set_xticklabels(['In-Domain\n(C3 vs GT-C3)', 'Cross-Domain\n(C6 vs GT-C6)'])
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_ylim(0, 105)
    ax4.legend(loc='upper right')
    ax4.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax4.set_axisbelow(True)

    plt.suptitle(f'{TASK_LABEL}: Clean Env Evaluation Summary', fontsize=16, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', format=fig_format)
        print(f"  Saved: {save_path}")

    plt.rcParams.update(DEFAULT_FONT)
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate figures for beat_block_hammer clean environment evaluation."
    )
    parser.add_argument("--no-show", action="store_true", help="Don't display figures")
    parser.add_argument("--format", type=str, choices=["png", "pdf", "svg"],
                        default=FIGURE_FORMAT, help="Output format")

    args = parser.parse_args()
    fig_format = args.format

    ensure_output_dirs()

    # Output subdirectory for clean eval figures
    clean_fig_dir = FIGURES_DIR / "clean_eval"
    clean_fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Load VILA results ---
    print("Loading VILA clean eval results...")
    vila_results = load_all_results_vila(tasks=[TASK], eval_env=EVAL_ENV)
    vila_task = vila_results[TASK]

    vila_rates = {}
    for cid in range(1, 7):
        cond = vila_task.get_condition(cid)
        if cond:
            vila_rates[cid] = cond.success_rate_percent
            print(f"  C{cid}: {cond.success_rate_percent:.1f}% ({cond.num_successes}/{cond.num_episodes})")
        else:
            vila_rates[cid] = 0
            print(f"  C{cid}: NOT FOUND")

    # --- Load GT results ---
    print("\nLoading GT clean eval results...")
    gt_results = load_all_results_gt(tasks=[TASK], eval_env=EVAL_ENV)
    gt_task = gt_results[TASK]

    gt_rates = {}
    for cid in [2, 3, 5, 6]:
        cond = gt_task.get_condition(cid)
        if cond:
            gt_rates[cid] = cond.success_rate_percent
            print(f"  GT-C{cid}: {cond.success_rate_percent:.1f}% ({cond.num_successes}/{cond.num_episodes})")
        else:
            gt_rates[cid] = 0
            print(f"  GT-C{cid}: NOT FOUND")

    # --- Generate VILA figures ---
    print(f"\nGenerating VILA figures (format: {fig_format})...")

    plot_vila_main_results_all(
        vila_rates, save_path=clean_fig_dir / f"vila_main_results_all.{fig_format}",
        fig_format=fig_format)

    plot_vila_grouped(
        vila_rates, save_path=clean_fig_dir / f"vila_main_results_grouped.{fig_format}",
        fig_format=fig_format)

    plot_vila_in_domain(
        vila_rates, save_path=clean_fig_dir / f"vila_in_domain_results.{fig_format}",
        fig_format=fig_format)

    plot_vila_cross_domain(
        vila_rates, save_path=clean_fig_dir / f"vila_cross_domain_results.{fig_format}",
        fig_format=fig_format)

    plot_vila_memory_function(
        vila_rates, save_path=clean_fig_dir / f"vila_memory_function_comparison.{fig_format}",
        fig_format=fig_format)

    plot_vila_training_env_comparison(
        vila_rates, save_path=clean_fig_dir / f"vila_training_env_comparison.{fig_format}",
        fig_format=fig_format)

    plot_vila_hypothesis_h1_h2(
        vila_rates, save_path=clean_fig_dir / f"vila_hypothesis_h1_h2.{fig_format}",
        fig_format=fig_format)

    plot_vila_hypothesis_h3(
        vila_rates, save_path=clean_fig_dir / f"vila_hypothesis_h3.{fig_format}",
        fig_format=fig_format)

    plot_vila_hypothesis_h4(
        vila_rates, save_path=clean_fig_dir / f"vila_hypothesis_h4.{fig_format}",
        fig_format=fig_format)

    plot_vila_path_stats(
        vila_task, save_path=clean_fig_dir / f"vila_path_stats.{fig_format}",
        fig_format=fig_format)

    # --- Generate GT figures ---
    print(f"\nGenerating GT figures (format: {fig_format})...")

    plot_gt_main_results_all(
        gt_rates, save_path=clean_fig_dir / f"gt_main_results_all.{fig_format}",
        fig_format=fig_format)

    plot_gt_grouped(
        gt_rates, save_path=clean_fig_dir / f"gt_main_results_grouped.{fig_format}",
        fig_format=fig_format)

    # --- Generate combined summary ---
    print(f"\nGenerating summary figure (format: {fig_format})...")

    plot_summary_figure(
        vila_rates, gt_rates,
        save_path=clean_fig_dir / f"summary_figure.{fig_format}",
        fig_format=fig_format)

    print(f"\nAll figures saved to: {clean_fig_dir}")
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
