"""
Phase B: Penetrance Analysis by Trajectory Subcategories

Analyzes penetrance rates over time for 6 trajectory subcategories:
- Low_to_High_A
- Low_to_High_B
- High_to_Low_A
- High_to_Low_B
- Intermediate
- Not Penetrant

Reuses the same hybrid threshold approach from Phase A.
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
    SUBCATEGORY_COL,
    IQR_K,
    EARLY_CUTOFF_HPF,
    TIME_BIN_WIDTH,
    SUBCATEGORIES,
    SUBCATEGORY_COLORS,
    FIGURE_DIR,
    TABLE_DIR,
    DPI,
)

from data_loading import (
    load_trajectory_data,
    extract_wt_data,
    bin_data_by_time,
)

# Import functions from Phase A
import sys
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "phase_a",
    Path(__file__).parent / "01a_penetrance_by_broad_category.py"
)
phase_a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase_a)

compute_hybrid_wt_envelope = phase_a.compute_hybrid_wt_envelope
mark_penetrant_hybrid = phase_a.mark_penetrant_hybrid
compute_penetrance_by_group_and_time = phase_a.compute_penetrance_by_group_and_time

warnings.filterwarnings('ignore')


# ============================================================================
# Plotting Functions (Subcategory-specific)
# ============================================================================

def plot_penetrance_curves_subcategory(penetrance_df, output_path, title="Penetrance by Trajectory Subcategory"):
    """
    Plot penetrance curves for subcategories with SE error bands.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    for group in SUBCATEGORIES:
        group_data = penetrance_df[penetrance_df['group'] == group]

        if len(group_data) == 0:
            continue

        times = group_data['time_bin'].values
        pens = group_data['penetrance'].values * 100
        ses = group_data['se'].values * 100

        color = SUBCATEGORY_COLORS.get(group, '#888888')

        # Differentiate line styles for A vs B variants
        if 'A' in group:
            linestyle = '-'
            marker = 'o'
        elif 'B' in group:
            linestyle = '--'
            marker = 's'
        else:
            linestyle = '-'
            marker = '^'

        # Main line
        ax.plot(times, pens, marker=marker, color=color, linewidth=2.5,
                markersize=6, label=group, alpha=0.9, linestyle=linestyle)

        # Error band
        ax.fill_between(times, pens - ses, pens + ses, color=color, alpha=0.15)

    # Add vertical line at 30 hpf cutoff
    ax.axvline(EARLY_CUTOFF_HPF, color='gray', linestyle=':', linewidth=2.5,
               alpha=0.7, label=f'{EARLY_CUTOFF_HPF} hpf cutoff')

    ax.set_xlabel('Developmental Time (hpf)', fontsize=13)
    ax.set_ylabel('Penetrance (%)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def plot_penetrance_heatmap_subcategory(penetrance_df, output_path, title="Penetrance Heatmap"):
    """
    Heatmap: rows = subcategories, columns = time bins, color = penetrance %.
    """
    # Pivot to matrix form
    heatmap_data = penetrance_df.pivot(
        index='group',
        columns='time_bin',
        values='penetrance'
    ) * 100  # Convert to percentage

    # Reorder rows
    heatmap_data = heatmap_data.reindex([g for g in SUBCATEGORIES if g in heatmap_data.index])

    fig, ax = plt.subplots(figsize=(16, 7))

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
    ax.set_ylabel('Trajectory Subcategory', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main analysis pipeline for Phase B."""

    print("=" * 80)
    print("PHASE B: PENETRANCE ANALYSIS BY TRAJECTORY SUBCATEGORIES")
    print("=" * 80)

    # Load data
    df = load_trajectory_data()

    # Bin by time
    print(f"\nBinning data by {TIME_BIN_WIDTH} hpf windows...")
    df, time_bins = bin_data_by_time(df, bin_width=TIME_BIN_WIDTH)
    print(f"  Created {len(time_bins)} time bins ({time_bins[0]:.1f} - {time_bins[-1]:.1f} hpf)")

    # Extract WT data
    wt_df = extract_wt_data(df)

    # Compute hybrid WT envelope (same as Phase A)
    wt_envelope = compute_hybrid_wt_envelope(
        wt_df,
        time_bins,
        metric=METRIC_NAME,
        k=IQR_K,
        early_cutoff_hpf=EARLY_CUTOFF_HPF,
    )

    # Compute penetrance by subcategory and time
    print(f"\nComputing penetrance by subcategory...")
    penetrance_df = compute_penetrance_by_group_and_time(
        df,
        wt_envelope,
        time_bins,
        group_col=SUBCATEGORY_COL,
    )

    # Print summary
    print(f"\nPenetrance summary:")
    for group in SUBCATEGORIES:
        group_data = penetrance_df[penetrance_df['group'] == group]
        if len(group_data) > 0:
            mean_pen = group_data['penetrance'].mean() * 100
            max_pen = group_data['penetrance'].max() * 100
            n_embryos = group_data['n_embryos'].iloc[0]  # Approximate
            print(f"  {group:20s}: mean={mean_pen:5.1f}%, max={max_pen:5.1f}%")

    # Save tables
    print(f"\nSaving outputs...")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    penetrance_df.to_csv(TABLE_DIR / 'subcategory_penetrance_by_time.csv', index=False)
    print(f"  Saved: subcategory_penetrance_by_time.csv")

    # Generate plots
    print(f"\nGenerating plots...")
    plot_penetrance_curves_subcategory(
        penetrance_df,
        FIGURE_DIR / 'penetrance_curves_by_subcategory.png',
        title="Penetrance by Trajectory Subcategory"
    )
    plot_penetrance_heatmap_subcategory(
        penetrance_df,
        FIGURE_DIR / 'penetrance_heatmap_subcategory.png',
        title="Penetrance Heatmap: Subcategories"
    )

    print(f"\n{'='*80}")
    print("PHASE B ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Outputs saved to:")
    print(f"  Tables: {TABLE_DIR}")
    print(f"  Figures: {FIGURE_DIR}")


if __name__ == '__main__':
    main()
