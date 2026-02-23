#!/usr/bin/env python3
"""
Test different smoothing options for trajectory plots.

This script creates side-by-side comparisons of different smoothing approaches
to help choose the best one for reducing noise in b9d2 pair analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.data_loading import _load_df03_format

# Configuration
EXPERIMENT_ID = '20251119'
METRIC_NAME = 'baseline_deviation_normalized'
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
PAIR_COL = 'pair'
GENOTYPE_COL = 'genotype'

# Test parameters
TEST_PAIR = 'b9d2_pair_7'  # Has more data
TEST_GENOTYPE = 'b9d2_homozygous'

# Genotype color
COLOR = '#D32F2F'  # Red for homozygous

# Output
OUTPUT_DIR = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251204_initial_b9d2_pair_analysis')
OUTPUT_PATH = OUTPUT_DIR / 'smoothing_comparison.png'


def apply_savgol_smooth(y_values, window_length=7, polyorder=2):
    """Apply Savitzky-Golay filter."""
    if len(y_values) < window_length:
        return y_values
    return savgol_filter(y_values, window_length=window_length, polyorder=polyorder)


def apply_gaussian_smooth(y_values, sigma=1.0):
    """Apply Gaussian smoothing."""
    return gaussian_filter1d(y_values, sigma=sigma)


def apply_rolling_mean(times, values, bin_width):
    """Apply rolling mean via binning."""
    time_bins = np.arange(np.floor(times.min()), np.ceil(times.max()) + 1, bin_width)
    bin_means = []
    bin_times = []
    for i in range(len(time_bins) - 1):
        mask = (times >= time_bins[i]) & (times < time_bins[i+1])
        if mask.sum() > 0:
            bin_means.append(values[mask].mean())
            bin_times.append((time_bins[i] + time_bins[i+1]) / 2)
    return np.array(bin_times), np.array(bin_means)


def load_test_data():
    """Load and filter data for testing."""
    print(f"Loading data for {EXPERIMENT_ID}...")
    df = _load_df03_format(EXPERIMENT_ID)

    # Filter
    df = df[df['use_embryo_flag'] == 1].copy()
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, METRIC_NAME, PAIR_COL, GENOTYPE_COL])

    # Get test subset
    filtered = df[(df[PAIR_COL] == TEST_PAIR) & (df[GENOTYPE_COL] == TEST_GENOTYPE)].copy()

    print(f"Test data: {TEST_PAIR} - {TEST_GENOTYPE}")
    print(f"  {len(filtered)} timepoints, {filtered[EMBRYO_ID_COL].nunique()} embryos")

    # Extract trajectories
    embryo_ids = filtered[EMBRYO_ID_COL].unique()
    trajectories = []

    for embryo_id in embryo_ids:
        embryo_data = filtered[filtered[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)
        if len(embryo_data) > 1:
            trajectories.append({
                'embryo_id': embryo_id,
                'times': embryo_data[TIME_COL].values,
                'metrics': embryo_data[METRIC_NAME].values,
            })

    return trajectories


def plot_smoothing_comparison(trajectories):
    """Create comparison plot of different smoothing methods."""
    print("\nCreating smoothing comparison plot...")

    # Set up 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Smoothing Comparison - {TEST_PAIR} {TEST_GENOTYPE}\n'
                 f'{METRIC_NAME}', fontsize=14, fontweight='bold')

    # Prepare data
    all_times = np.concatenate([t['times'] for t in trajectories])
    all_metrics = np.concatenate([t['metrics'] for t in trajectories])

    # Get axis limits
    time_min, time_max = all_times.min(), all_times.max()
    metric_min, metric_max = all_metrics.min(), all_metrics.max()
    metric_padding = (metric_max - metric_min) * 0.1

    # --- Row 1: Individual trajectories + smoothed individuals ---

    # 1. No smoothing (original)
    ax = axes[0, 0]
    for traj in trajectories:
        ax.plot(traj['times'], traj['metrics'], alpha=0.3, linewidth=1, color=COLOR)
    # Binned mean
    bin_times, bin_means = apply_rolling_mean(all_times, all_metrics, 0.5)
    ax.plot(bin_times, bin_means, color=COLOR, linewidth=2.5, label='Mean (0.5 hpf bins)', zorder=5)
    ax.set_title('Original (No Smoothing)', fontweight='bold')
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('Normalized Baseline Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(metric_min - metric_padding, metric_max + metric_padding)

    # 2. Savitzky-Golay on individuals
    ax = axes[0, 1]
    for traj in trajectories:
        smoothed = apply_savgol_smooth(traj['metrics'], window_length=7, polyorder=2)
        ax.plot(traj['times'], smoothed, alpha=0.3, linewidth=1, color=COLOR)
    # Mean of smoothed
    all_smoothed_metrics = np.concatenate([
        apply_savgol_smooth(t['metrics'], window_length=7, polyorder=2)
        for t in trajectories
    ])
    bin_times, bin_means = apply_rolling_mean(all_times, all_smoothed_metrics, 0.5)
    ax.plot(bin_times, bin_means, color=COLOR, linewidth=2.5, label='Mean of smoothed', zorder=5)
    ax.set_title('Savitzky-Golay (window=7, poly=2)', fontweight='bold')
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('Normalized Baseline Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(metric_min - metric_padding, metric_max + metric_padding)

    # 3. Gaussian on individuals
    ax = axes[0, 2]
    for traj in trajectories:
        smoothed = apply_gaussian_smooth(traj['metrics'], sigma=1.5)
        ax.plot(traj['times'], smoothed, alpha=0.3, linewidth=1, color=COLOR)
    # Mean of smoothed
    all_smoothed_metrics = np.concatenate([
        apply_gaussian_smooth(t['metrics'], sigma=1.5)
        for t in trajectories
    ])
    bin_times, bin_means = apply_rolling_mean(all_times, all_smoothed_metrics, 0.5)
    ax.plot(bin_times, bin_means, color=COLOR, linewidth=2.5, label='Mean of smoothed', zorder=5)
    ax.set_title('Gaussian (sigma=1.5)', fontweight='bold')
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('Normalized Baseline Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(metric_min - metric_padding, metric_max + metric_padding)

    # --- Row 2: Mean smoothing only ---

    # 4. Larger bins (1.0 hpf)
    ax = axes[1, 0]
    for traj in trajectories:
        ax.plot(traj['times'], traj['metrics'], alpha=0.3, linewidth=1, color=COLOR)
    bin_times, bin_means = apply_rolling_mean(all_times, all_metrics, 1.0)
    ax.plot(bin_times, bin_means, color=COLOR, linewidth=2.5, label='Mean (1.0 hpf bins)', zorder=5)
    ax.set_title('Larger Bins (1.0 hpf)', fontweight='bold')
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('Normalized Baseline Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(metric_min - metric_padding, metric_max + metric_padding)

    # 5. Savitzky-Golay on mean only
    ax = axes[1, 1]
    for traj in trajectories:
        ax.plot(traj['times'], traj['metrics'], alpha=0.3, linewidth=1, color=COLOR)
    bin_times, bin_means = apply_rolling_mean(all_times, all_metrics, 0.5)
    smoothed_mean = apply_savgol_smooth(bin_means, window_length=7, polyorder=2)
    ax.plot(bin_times, smoothed_mean, color=COLOR, linewidth=2.5, label='Smoothed mean', zorder=5)
    ax.set_title('SavGol on Mean Only (window=7)', fontweight='bold')
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('Normalized Baseline Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(metric_min - metric_padding, metric_max + metric_padding)

    # 6. Gaussian on mean only
    ax = axes[1, 2]
    for traj in trajectories:
        ax.plot(traj['times'], traj['metrics'], alpha=0.3, linewidth=1, color=COLOR)
    bin_times, bin_means = apply_rolling_mean(all_times, all_metrics, 0.5)
    smoothed_mean = apply_gaussian_smooth(bin_means, sigma=2.0)
    ax.plot(bin_times, smoothed_mean, color=COLOR, linewidth=2.5, label='Smoothed mean', zorder=5)
    ax.set_title('Gaussian on Mean Only (sigma=2.0)', fontweight='bold')
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('Normalized Baseline Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_min, time_max)
    ax.set_ylim(metric_min - metric_padding, metric_max + metric_padding)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


def main():
    """Main function."""
    print("=" * 60)
    print("SMOOTHING OPTIONS TEST")
    print("=" * 60)

    trajectories = load_test_data()
    plot_smoothing_comparison(trajectories)

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
