"""
Quick diagnostic: Plot individual WT embryo trajectories to assess noise vs threshold artifacts.

Uses IQR 2.0 calibration to show:
- WT envelope bounds (acceptable range)
- Average WT trajectory (reference)
- Top 3 embryos with most outlier instances (spiky/high-variance)
- Top 3 embryos with least outlier instances (smooth/low-variance)
- Highlights points where embryos exceed threshold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import utilities
from load_data import get_analysis_dataframe

# Configuration
METRIC_NAME = 'normalized_baseline_deviation'
TIME_BIN_WIDTH = 2.0
WT_GENOTYPE = 'cep290_wildtype'

# IQR 2.0 parameters
k = 2.0

# Output
OUTPUT_DIR = Path(__file__).parent / 'outputs' / '06c_wt_noise_diagnosis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def bin_data_by_time(df, bin_width=2.0, time_col='predicted_stage_hpf'):
    """Bin data by developmental time."""
    df = df.copy()
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    bin_edges = np.arange(
        np.floor(min_time / bin_width) * bin_width,
        np.ceil(max_time / bin_width) * bin_width + bin_width,
        bin_width
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    df['time_bin'] = pd.cut(df[time_col], bins=bin_edges, labels=bin_centers)
    df['time_bin'] = df['time_bin'].astype(float)

    return df, bin_centers


def compute_iqr_bounds(wt_df, time_bins, metric=METRIC_NAME, k=2.0):
    """Compute IQR bounds per time bin."""
    envelope = {}

    for time_bin in time_bins:
        bin_df = wt_df[wt_df['time_bin'] == time_bin]
        if len(bin_df) == 0:
            continue

        values = bin_df[metric].values
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        envelope[time_bin] = {
            'low': q1 - k * iqr,
            'high': q3 + k * iqr,
            'median': np.median(values),
            'mean': np.mean(values),
        }

    return envelope


def count_outlier_instances(df, envelope, metric=METRIC_NAME):
    """Count how many times each embryo is outside the bounds."""
    outlier_counts = {}

    for embryo_id in df['embryo_id'].unique():
        embryo_df = df[df['embryo_id'] == embryo_id]
        n_outliers = 0

        for _, row in embryo_df.iterrows():
            time_bin = row['time_bin']
            if pd.isna(time_bin) or time_bin not in envelope:
                continue

            metric_val = row[metric]
            bounds = envelope[time_bin]

            if metric_val < bounds['low'] or metric_val > bounds['high']:
                n_outliers += 1

        outlier_counts[embryo_id] = n_outliers

    return outlier_counts


def main():
    print("=" * 80)
    print("WT NOISE DIAGNOSIS: Individual Embryo Trajectories")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df, _ = get_analysis_dataframe()

    # Bin by time
    print("Binning data...")
    df, time_bins = bin_data_by_time(df, bin_width=TIME_BIN_WIDTH)

    # Extract WT
    wt_df = df[df['genotype'] == WT_GENOTYPE].copy()
    print(f"WT: {wt_df['embryo_id'].nunique()} embryos, {len(wt_df)} timepoints")

    # Compute IQR 2.0 bounds
    print("Computing IQR 2.0 bounds...")
    envelope = compute_iqr_bounds(wt_df, time_bins, metric=METRIC_NAME, k=2.0)

    # Count outliers per embryo
    print("Counting outliers per embryo...")
    outlier_counts = count_outlier_instances(wt_df, envelope, metric=METRIC_NAME)

    # Sort embryos by outlier count
    sorted_embryos = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\nOutlier counts:")
    print(f"  Max: {sorted_embryos[0][1]}")
    print(f"  Min: {sorted_embryos[-1][1]}")
    print(f"  Mean: {np.mean([c for _, c in sorted_embryos]):.1f}")

    # Get top 3 highest and lowest
    top_3_high = [e for e, c in sorted_embryos[:3]]
    top_3_low = [e for e, c in sorted_embryos[-3:]]

    print(f"\nTop 3 spiky embryos (most outliers):")
    for embryo_id, count in sorted_embryos[:3]:
        print(f"  {embryo_id}: {count} outliers")

    print(f"\nTop 3 smooth embryos (least outliers):")
    for embryo_id, count in sorted_embryos[-3:]:
        print(f"  {embryo_id}: {count} outliers")

    # Create plots
    print("\nGenerating plots...")

    # PLOT 1: Top 3 spiky embryos
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot envelope
    times = sorted(envelope.keys())
    lows = [envelope[t]['low'] for t in times]
    highs = [envelope[t]['high'] for t in times]
    medians = [envelope[t]['mean'] for t in times]

    ax.fill_between(times, lows, highs, alpha=0.2, color='blue', label='IQR 2.0 bounds')
    ax.plot(times, medians, 'b-', linewidth=2.5, label='WT mean (reference)')

    # Plot top 3 spiky embryos
    colors = ['red', 'orange', 'darkorange']
    for i, embryo_id in enumerate(top_3_high):
        embryo_df = wt_df[wt_df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')

        # Plot trajectory
        ax.plot(embryo_df['time_bin'], embryo_df[METRIC_NAME],
                'o-', color=colors[i], linewidth=1.5, markersize=4, alpha=0.7,
                label=f'Embryo {embryo_id} ({outlier_counts[embryo_id]} outliers)')

        # Highlight outlier points
        for _, row in embryo_df.iterrows():
            time_bin = row['time_bin']
            metric_val = row[METRIC_NAME]

            if pd.isna(time_bin) or time_bin not in envelope:
                continue

            bounds = envelope[time_bin]
            if metric_val < bounds['low'] or metric_val > bounds['high']:
                ax.scatter(time_bin, metric_val, s=150, edgecolor=colors[i],
                          facecolor='none', linewidth=2.5, zorder=10)

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel(METRIC_NAME, fontsize=12)
    ax.set_title('Top 3 Spiky WT Embryos (Most Outliers)\nCircles mark points outside IQR 2.0 bounds',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'wt_spiky_embryos.png', dpi=300, bbox_inches='tight')
    print("  Saved: wt_spiky_embryos.png")
    plt.close()

    # PLOT 2: Top 3 smooth embryos
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.fill_between(times, lows, highs, alpha=0.2, color='blue', label='IQR 2.0 bounds')
    ax.plot(times, medians, 'b-', linewidth=2.5, label='WT mean (reference)')

    colors = ['green', 'lightgreen', 'yellowgreen']
    for i, embryo_id in enumerate(top_3_low):
        embryo_df = wt_df[wt_df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')

        ax.plot(embryo_df['time_bin'], embryo_df[METRIC_NAME],
                'o-', color=colors[i], linewidth=1.5, markersize=4, alpha=0.7,
                label=f'Embryo {embryo_id} ({outlier_counts[embryo_id]} outliers)')

        # Highlight outlier points (should be rare)
        for _, row in embryo_df.iterrows():
            time_bin = row['time_bin']
            metric_val = row[METRIC_NAME]

            if pd.isna(time_bin) or time_bin not in envelope:
                continue

            bounds = envelope[time_bin]
            if metric_val < bounds['low'] or metric_val > bounds['high']:
                ax.scatter(time_bin, metric_val, s=150, edgecolor=colors[i],
                          facecolor='none', linewidth=2.5, zorder=10)

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel(METRIC_NAME, fontsize=12)
    ax.set_title('Top 3 Smooth WT Embryos (Least Outliers)\nCircles mark points outside IQR 2.0 bounds',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'wt_smooth_embryos.png', dpi=300, bbox_inches='tight')
    print("  Saved: wt_smooth_embryos.png")
    plt.close()

    # PLOT 3: All WT embryos overlaid with reference
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.fill_between(times, lows, highs, alpha=0.15, color='blue', label='IQR 2.0 bounds')
    ax.plot(times, medians, 'b-', linewidth=3, label='WT mean (reference)', zorder=100)

    # Plot all WT embryos in light gray
    for embryo_id in wt_df['embryo_id'].unique():
        embryo_df = wt_df[wt_df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')
        ax.plot(embryo_df['time_bin'], embryo_df[METRIC_NAME],
                '-', color='gray', linewidth=0.8, alpha=0.3)

    # Highlight top 3 spiky in red
    for i, embryo_id in enumerate(top_3_high):
        embryo_df = wt_df[wt_df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')
        ax.plot(embryo_df['time_bin'], embryo_df[METRIC_NAME],
                'o-', color='red', linewidth=2, markersize=4, alpha=0.8,
                label=f'Spiky: {embryo_id}' if i == 0 else '')

    # Highlight top 3 smooth in green
    for i, embryo_id in enumerate(top_3_low):
        embryo_df = wt_df[wt_df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')
        ax.plot(embryo_df['time_bin'], embryo_df[METRIC_NAME],
                's-', color='green', linewidth=2, markersize=4, alpha=0.8,
                label=f'Smooth: {embryo_id}' if i == 0 else '')

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel(METRIC_NAME, fontsize=12)
    ax.set_title('All WT Embryos with Spiky/Smooth Highlighted\nGray: All embryos, Red: Spiky (top 3), Green: Smooth (top 3)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'wt_all_embryos_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: wt_all_embryos_comparison.png")
    plt.close()

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")

    # Summary
    print(f"\nInterpretation guide:")
    print(f"- Spiky embryos: Check if actual high variance (noisy data) or artifacts")
    print(f"- Smooth embryos: Should show clean, monotonic-like trajectories")
    print(f"- Outlier circles: Points where embryo exceeds IQR 2.0 bounds")
    print(f"- If many red circles = genuine outliers OR threshold too tight")
    print(f"- If few green circles = WT envelope is appropriate")


if __name__ == '__main__':
    main()
