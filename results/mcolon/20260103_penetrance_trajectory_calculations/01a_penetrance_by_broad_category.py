"""
Phase A: Penetrance Analysis by Broad Trajectory Categories

Analyzes penetrance rates over time for 4 broad trajectory categories:
- Low_to_High
- High_to_Low
- Intermediate
- Not Penetrant

Uses hybrid threshold approach:
- Time-binned IQR for <30 hpf (dynamic curvature)
- Global IQR for >=30 hpf (stable curvature)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

from config import (
    METRIC_NAME,
    TIME_COL,
    EMBRYO_COL,
    GENOTYPE_COL,
    CATEGORY_COL,
    IQR_K,
    EARLY_CUTOFF_HPF,
    TIME_BIN_WIDTH,
    BROAD_CATEGORIES,
    CATEGORY_COLORS,
    FIGURE_DIR,
    TABLE_DIR,
    DPI,
)

from data_loading import (
    load_trajectory_data,
    extract_wt_data,
    bin_data_by_time,
)

warnings.filterwarnings('ignore')


# ============================================================================
# Hybrid Threshold Computation
# ============================================================================

def compute_hybrid_wt_envelope(
    wt_df,
    time_bins,
    metric=METRIC_NAME,
    k=IQR_K,
    early_cutoff_hpf=EARLY_CUTOFF_HPF,
    bin_col='time_bin',
):
    """
    Compute hybrid WT envelope: time-binned for <30 hpf, global for >=30 hpf.

    Parameters
    ----------
    wt_df : pd.DataFrame
        WT embryo data with time_bin column
    time_bins : np.ndarray
        Array of time bin centers
    metric : str
        Curvature metric column name
    k : float
        IQR multiplier (default 2.0 for ~95% coverage)
    early_cutoff_hpf : float
        Threshold for switching from time-binned to global (default 30.0)
    bin_col : str
        Name of time bin column

    Returns
    -------
    envelope : dict
        Mapping of time_bin -> {'low': float, 'high': float, 'median': float, 'method': str}
    """
    envelope = {}

    # Separate bins into early (<30 hpf) and late (>=30 hpf)
    early_bins = time_bins[time_bins < early_cutoff_hpf]
    late_bins = time_bins[time_bins >= early_cutoff_hpf]

    print(f"\nComputing hybrid WT envelope (k={k}, cutoff={early_cutoff_hpf} hpf):")
    print(f"  Early bins (<{early_cutoff_hpf} hpf): {len(early_bins)} bins - time-binned IQR")
    print(f"  Late bins (>={early_cutoff_hpf} hpf): {len(late_bins)} bins - global IQR")

    # EARLY BINS: Time-binned IQR
    for time_bin in early_bins:
        bin_df = wt_df[wt_df[bin_col] == time_bin]

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
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'n_samples': len(values),
            'method': 'time-binned',
        }

    # LATE BINS: Global IQR (pooled from all WT data >= cutoff)
    late_wt_df = wt_df[wt_df[bin_col] >= early_cutoff_hpf]
    late_values = late_wt_df[metric].values

    if len(late_values) > 0:
        q1_global = np.percentile(late_values, 25)
        q3_global = np.percentile(late_values, 75)
        iqr_global = q3_global - q1_global

        global_bounds = {
            'low': q1_global - k * iqr_global,
            'high': q3_global + k * iqr_global,
            'median': np.median(late_values),
            'q1': q1_global,
            'q3': q3_global,
            'iqr': iqr_global,
            'n_samples': len(late_values),
            'method': 'global',
        }

        print(f"  Global IQR (>={early_cutoff_hpf} hpf):")
        print(f"    Pooled {len(late_values)} WT samples")
        print(f"    Bounds: [{global_bounds['low']:.6f}, {global_bounds['high']:.6f}]")

        # Apply global bounds to all late bins
        for time_bin in late_bins:
            envelope[time_bin] = global_bounds.copy()

    return envelope


def mark_penetrant_hybrid(
    df,
    wt_envelope,
    metric=METRIC_NAME,
    bin_col='time_bin',
    penetrant_col='penetrant',
):
    """
    Mark frames as penetrant if outside WT envelope.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time_bin column
    wt_envelope : dict
        Envelope from compute_hybrid_wt_envelope
    metric : str
        Metric column name
    bin_col : str
        Time bin column name
    penetrant_col : str
        Output column name for penetrance flag

    Returns
    -------
    df : pd.DataFrame
        Dataframe with penetrant column added (0, 1, or NaN)
    """
    df = df.copy()
    df[penetrant_col] = 0

    for idx, row in df.iterrows():
        time_bin = row[bin_col]

        if pd.isna(time_bin) or time_bin not in wt_envelope:
            df.loc[idx, penetrant_col] = np.nan
            continue

        metric_value = row[metric]
        envelope = wt_envelope[time_bin]

        if metric_value < envelope['low'] or metric_value > envelope['high']:
            df.loc[idx, penetrant_col] = 1

    return df


# ============================================================================
# Penetrance Computation
# ============================================================================

def compute_penetrance_by_group_and_time(
    df,
    wt_envelope,
    time_bins,
    group_col=CATEGORY_COL,
    metric=METRIC_NAME,
    embryo_col=EMBRYO_COL,
    bin_col='time_bin',
):
    """
    Compute penetrance per group per time bin.

    Parameters
    ----------
    df : pd.DataFrame
        Data with group assignments and time bins
    wt_envelope : dict
        WT threshold envelope
    time_bins : np.ndarray
        Time bin centers
    group_col : str
        Column containing group labels
    metric : str
        Metric column name
    embryo_col : str
        Embryo ID column name
    bin_col : str
        Time bin column name

    Returns
    -------
    penetrance_df : pd.DataFrame
        Columns: [group, time_bin, penetrance, n_embryos, n_penetrant, se]
    """
    # Mark penetrant frames
    df = mark_penetrant_hybrid(df, wt_envelope, metric=metric, bin_col=bin_col)

    results = []

    # Get unique groups
    groups = df[group_col].dropna().unique()

    for group in groups:
        group_df = df[df[group_col] == group].copy()

        for time_bin in time_bins:
            bin_df = group_df[group_df[bin_col] == time_bin].dropna(subset=['penetrant'])

            if len(bin_df) == 0:
                continue

            # Embryo-level penetrance: if ANY frame is penetrant, embryo is penetrant
            embryos_in_bin = bin_df[embryo_col].unique()
            embryos_penetrant = bin_df[bin_df['penetrant'] == 1][embryo_col].unique()

            n_embryos = len(embryos_in_bin)
            n_penetrant = len(embryos_penetrant)
            penetrance = n_penetrant / n_embryos if n_embryos > 0 else 0.0

            # Standard error
            se = np.sqrt(penetrance * (1 - penetrance) / n_embryos) if n_embryos > 0 else 0.0

            results.append({
                'group': group,
                'time_bin': time_bin,
                'penetrance': penetrance,
                'n_embryos': n_embryos,
                'n_penetrant': n_penetrant,
                'se': se,
            })

    penetrance_df = pd.DataFrame(results)
    return penetrance_df


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_penetrance_curves(penetrance_df, output_path, title="Penetrance by Trajectory Category"):
    """
    Plot penetrance curves with SE error bands.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for group in BROAD_CATEGORIES:
        group_data = penetrance_df[penetrance_df['group'] == group]

        if len(group_data) == 0:
            continue

        times = group_data['time_bin'].values
        pens = group_data['penetrance'].values * 100
        ses = group_data['se'].values * 100

        color = CATEGORY_COLORS.get(group, '#888888')

        # Main line
        ax.plot(times, pens, marker='o', color=color, linewidth=2.5,
                markersize=6, label=group, alpha=0.9)

        # Error band
        ax.fill_between(times, pens - ses, pens + ses, color=color, alpha=0.2)

    # Add vertical line at 30 hpf cutoff
    ax.axvline(EARLY_CUTOFF_HPF, color='gray', linestyle='--', linewidth=2,
               alpha=0.7, label=f'{EARLY_CUTOFF_HPF} hpf cutoff')

    ax.set_xlabel('Developmental Time (hpf)', fontsize=13)
    ax.set_ylabel('Penetrance (%)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def plot_penetrance_heatmap(penetrance_df, output_path, title="Penetrance Heatmap"):
    """
    Heatmap: rows = groups, columns = time bins, color = penetrance %.
    """
    # Pivot to matrix form
    heatmap_data = penetrance_df.pivot(
        index='group',
        columns='time_bin',
        values='penetrance'
    ) * 100  # Convert to percentage

    # Reorder rows
    heatmap_data = heatmap_data.reindex([g for g in BROAD_CATEGORIES if g in heatmap_data.index])

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(
        heatmap_data,
        cmap='YlOrRd',
        annot=False,
        fmt='.0f',
        cbar_kws={'label': 'Penetrance (%)'},
        vmin=0,
        vmax=100,
        ax=ax,
    )

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('Trajectory Category', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main analysis pipeline for Phase A."""

    print("=" * 80)
    print("PHASE A: PENETRANCE ANALYSIS BY BROAD TRAJECTORY CATEGORIES")
    print("=" * 80)

    # Load data
    df = load_trajectory_data()

    # Bin by time
    print(f"\nBinning data by {TIME_BIN_WIDTH} hpf windows...")
    df, time_bins = bin_data_by_time(df, bin_width=TIME_BIN_WIDTH)
    print(f"  Created {len(time_bins)} time bins ({time_bins[0]:.1f} - {time_bins[-1]:.1f} hpf)")

    # Extract WT data
    wt_df = extract_wt_data(df)

    # Compute hybrid WT envelope
    wt_envelope = compute_hybrid_wt_envelope(
        wt_df,
        time_bins,
        metric=METRIC_NAME,
        k=IQR_K,
        early_cutoff_hpf=EARLY_CUTOFF_HPF,
    )

    # Compute penetrance by group and time
    print(f"\nComputing penetrance by broad category...")
    penetrance_df = compute_penetrance_by_group_and_time(
        df,
        wt_envelope,
        time_bins,
        group_col=CATEGORY_COL,
    )

    # Print summary
    print(f"\nPenetrance summary:")
    for group in BROAD_CATEGORIES:
        group_data = penetrance_df[penetrance_df['group'] == group]
        if len(group_data) > 0:
            mean_pen = group_data['penetrance'].mean() * 100
            max_pen = group_data['penetrance'].max() * 100
            print(f"  {group:20s}: mean={mean_pen:5.1f}%, max={max_pen:5.1f}%")

    # Save tables
    print(f"\nSaving outputs...")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    penetrance_df.to_csv(TABLE_DIR / 'category_penetrance_by_time.csv', index=False)
    print(f"  Saved: category_penetrance_by_time.csv")

    # Save WT threshold summary
    threshold_summary = []
    for time_bin, bounds in wt_envelope.items():
        threshold_summary.append({
            'time_bin': time_bin,
            'low': bounds['low'],
            'high': bounds['high'],
            'median': bounds['median'],
            'iqr': bounds['iqr'],
            'n_samples': bounds['n_samples'],
            'method': bounds['method'],
        })
    threshold_df = pd.DataFrame(threshold_summary).sort_values('time_bin')
    threshold_df.to_csv(TABLE_DIR / 'wt_threshold_summary.csv', index=False)
    print(f"  Saved: wt_threshold_summary.csv")

    # Generate plots
    print(f"\nGenerating plots...")
    plot_penetrance_curves(
        penetrance_df,
        FIGURE_DIR / 'penetrance_curves_by_category.png',
        title="Penetrance by Broad Trajectory Category"
    )
    plot_penetrance_heatmap(
        penetrance_df,
        FIGURE_DIR / 'penetrance_heatmap_category.png',
        title="Penetrance Heatmap: Broad Categories"
    )

    print(f"\n{'='*80}")
    print("PHASE A ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Outputs saved to:")
    print(f"  Tables: {TABLE_DIR}")
    print(f"  Figures: {FIGURE_DIR}")


if __name__ == '__main__':
    main()
