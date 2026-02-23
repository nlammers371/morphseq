"""
Global IQR Calibration Analysis

Instead of binning by time, apply a single IQR threshold across ALL WT timepoints
to get a global WT reference band. This simpler approach avoids sparse bins and
shows what penetrance looks like with a single, global calibration.

Tests: IQR ±1.5σ, ±2.0σ, ±3.0σ on all data pooled together.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings

from load_data import (
    get_analysis_dataframe,
    get_genotype_short_name,
    get_genotype_color,
)

warnings.filterwarnings('ignore')

# Configuration
METRIC_NAME = 'normalized_baseline_deviation'
RANDOM_SEED = 42

# Output directories
OUTPUT_DIR = Path(__file__).parent / 'outputs' / '06e_global_iqr_calibration'
FIGURE_DIR = OUTPUT_DIR / 'figures'
TABLE_DIR = OUTPUT_DIR / 'tables'

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Genotypes
WT_GENOTYPE = 'cep290_wildtype'
ANALYSIS_GENOTYPES = [
    'cep290_heterozygous',
    'cep290_homozygous'
]

# Calibrations: global IQR on pooled WT data
CALIBRATIONS = {
    'global_iqr_1.5': {'k': 1.5, 'label': 'Global IQR ±1.5σ'},
    'global_iqr_2.0': {'k': 2.0, 'label': 'Global IQR ±2.0σ'},
    'global_iqr_3.0': {'k': 3.0, 'label': 'Global IQR ±3.0σ'},
}

np.random.seed(RANDOM_SEED)


# ============================================================================
# Global IQR Computation
# ============================================================================

def compute_global_iqr_bounds(wt_df, metric=METRIC_NAME, k=2.0):
    """
    Compute a single IQR band from ALL WT data pooled together.

    Parameters
    ----------
    wt_df : pd.DataFrame
        WT data
    metric : str
        Metric column to use
    k : float
        IQR multiplier (±k*IQR)

    Returns
    -------
    dict
        'low', 'high', 'median' bounds
    """
    values = wt_df[metric].values

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    bounds = {
        'low': q1 - k * iqr,
        'high': q3 + k * iqr,
        'median': np.median(values),
        'mean': np.mean(values),
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'n_samples': len(values),
    }

    return bounds


def mark_penetrant_global(df, wt_bounds, metric=METRIC_NAME):
    """Mark timepoints as penetrant if outside global WT bounds."""
    df = df.copy()
    df['penetrant'] = 0

    for idx, row in df.iterrows():
        metric_value = row[metric]

        if metric_value < wt_bounds['low'] or metric_value > wt_bounds['high']:
            df.loc[idx, 'penetrant'] = 1

    return df


def compute_penetrance_global(df, metric_col='penetrant'):
    """
    Compute embryo-level penetrance (% embryos with ≥1 penetrant frame).
    """
    df_clean = df.dropna(subset=[metric_col])

    if len(df_clean) == 0:
        return 0.0, []

    # Embryo-level: any timepoint outside bounds = penetrant embryo
    embryos = df_clean['embryo_id'].unique()
    embryos_penetrant = df_clean[df_clean[metric_col] == 1]['embryo_id'].unique()

    penetrance = len(embryos_penetrant) / len(embryos) if len(embryos) > 0 else 0

    return penetrance, {
        'n_embryos': len(embryos),
        'n_penetrant': len(embryos_penetrant),
        'penetrance': penetrance,
        'n_samples': len(df_clean)
    }


# ============================================================================
# Time-binned Penetrance (for temporal analysis)
# ============================================================================

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


def compute_penetrance_by_time(df, time_bins, metric_col='penetrant'):
    """Compute embryo-level penetrance per time bin."""
    penetrance_results = []

    for time_bin in time_bins:
        bin_df = df[df['time_bin'] == time_bin].dropna(subset=[metric_col])

        if len(bin_df) == 0:
            continue

        # Embryo-level penetrance
        embryos_in_bin = bin_df['embryo_id'].unique()
        embryos_penetrant = bin_df[bin_df[metric_col] == 1]['embryo_id'].unique()
        n_embryos = len(embryos_in_bin)
        n_embryos_penetrant = len(embryos_penetrant)
        embryo_penetrance = n_embryos_penetrant / n_embryos if n_embryos > 0 else 0

        penetrance_results.append({
            'time_bin': time_bin,
            'embryo_penetrance': embryo_penetrance,
            'n_embryos': n_embryos,
            'n_penetrant': n_embryos_penetrant
        })

    return penetrance_results


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_histogram_with_bounds(wt_df, wt_bounds, calib_name, figure_dir):
    """Plot histogram of WT data with bounds overlay."""
    fig, ax = plt.subplots(figsize=(12, 6))

    values = wt_df[METRIC_NAME].values

    # Histogram
    ax.hist(values, bins=50, alpha=0.6, color='blue', edgecolor='black', label='WT data')

    # Bounds
    ax.axvline(wt_bounds['low'], color='red', linestyle='--', linewidth=2.5,
               label=f'Low bound: {wt_bounds["low"]:.4f}')
    ax.axvline(wt_bounds['high'], color='red', linestyle='--', linewidth=2.5,
               label=f'High bound: {wt_bounds["high"]:.4f}')
    ax.axvline(wt_bounds['median'], color='green', linestyle='-', linewidth=2,
               label=f'Median: {wt_bounds["median"]:.4f}')

    ax.set_xlabel(METRIC_NAME, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'WT Distribution with Global IQR Bounds\n{CALIBRATIONS[calib_name]["label"]}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figure_dir / f'histogram_{calib_name}.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: histogram_{calib_name}.png")
    plt.close()


def plot_scatter_with_bounds(df, wt_bounds, calib_name, figure_dir):
    """Scatter plot of all data with WT bounds."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot each genotype
    for genotype in [WT_GENOTYPE, 'cep290_heterozygous', 'cep290_homozygous']:
        genotype_df = df[df['genotype'] == genotype]
        color = get_genotype_color(genotype)
        short = get_genotype_short_name(genotype)

        # Sort by time for line plot
        genotype_df = genotype_df.sort_values('predicted_stage_hpf')

        # Check penetrance status
        penetrant_mask = (genotype_df[METRIC_NAME] < wt_bounds['low']) | \
                         (genotype_df[METRIC_NAME] > wt_bounds['high'])

        # Plot non-penetrant
        non_pens = genotype_df[~penetrant_mask]
        ax.scatter(non_pens['predicted_stage_hpf'], non_pens[METRIC_NAME],
                  alpha=0.4, s=30, color=color, label=f'{short} (within bounds)')

        # Plot penetrant
        pens = genotype_df[penetrant_mask]
        ax.scatter(pens['predicted_stage_hpf'], pens[METRIC_NAME],
                  alpha=0.8, s=60, color=color, marker='X', edgecolor='black', linewidth=1,
                  label=f'{short} (outside bounds)')

    # WT bounds
    time_range = df['predicted_stage_hpf'].values
    ax.axhline(wt_bounds['low'], color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(wt_bounds['high'], color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f"WT bounds: [{wt_bounds['low']:.4f}, {wt_bounds['high']:.4f}]")
    ax.axhline(wt_bounds['median'], color='green', linestyle='-', linewidth=2,
               alpha=0.5, label=f"WT median: {wt_bounds['median']:.4f}")

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel(METRIC_NAME, fontsize=12)
    ax.set_title(f'All Genotypes with Global WT Bounds\n{CALIBRATIONS[calib_name]["label"]}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figure_dir / f'scatter_{calib_name}.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: scatter_{calib_name}.png")
    plt.close()


def plot_calibration_comparison(calibration_results, figure_dir):
    """Compare penetrance across the 3 global IQR calibrations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    genotypes = [WT_GENOTYPE, 'cep290_heterozygous', 'cep290_homozygous']
    genotype_short = ['WT', 'Het', 'Homo']
    colors = [get_genotype_color(g) for g in genotypes]

    calib_names = ['global_iqr_1.5', 'global_iqr_2.0', 'global_iqr_3.0']

    for ax_idx, calib_name in enumerate(calib_names):
        results = calibration_results[calib_name]
        ax = axes[ax_idx]

        penetrance_vals = [
            results['wt_penetrance'] * 100,
            results['het_penetrance'] * 100,
            results['homo_penetrance'] * 100,
        ]

        bars = ax.bar(genotype_short, penetrance_vals, color=colors, edgecolor='black', linewidth=1.5)

        # Color code by stringency
        for bar, val in zip(bars, penetrance_vals):
            if val < 1:
                bar.set_color('darkgreen')
            elif val < 5:
                bar.set_edgecolor('green')
                bar.set_linewidth(3)

        ax.axhline(1, color='darkgreen', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(5, color='green', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_ylim(0, max(100, max(penetrance_vals) * 1.1))
        ax.set_ylabel('Penetrance (%)', fontsize=11)
        ax.set_title(CALIBRATIONS[calib_name]['label'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, penetrance_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Penetrance Comparison: Global IQR Calibrations',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figure_dir / 'penetrance_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: penetrance_comparison.png")
    plt.close()


def plot_temporal_penetrance_by_calibration(calibration_results, time_bins, figure_dir):
    """
    Plot penetrance over time (in time bins) for WT/Het/Homo across all calibrations.

    Creates 3 subplots (one per calibration) showing how penetrance changes over
    developmental time for each genotype as line plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    genotypes = [WT_GENOTYPE, 'cep290_heterozygous', 'cep290_homozygous']
    genotype_short = ['WT', 'Het', 'Homo']
    colors = [get_genotype_color(g) for g in genotypes]
    markers = ['^', 'o', 's']

    calib_names = ['global_iqr_1.5', 'global_iqr_2.0', 'global_iqr_3.0']

    for ax_idx, calib_name in enumerate(calib_names):
        results = calibration_results[calib_name]
        ax = axes[ax_idx]

        # Plot WT
        if 'wt_temporal_penetrance' in results:
            wt_times = [r['time_bin'] for r in results['wt_temporal_penetrance']]
            wt_pens = [r['embryo_penetrance'] * 100 for r in results['wt_temporal_penetrance']]
            ax.plot(wt_times, wt_pens, marker=markers[0], color=colors[0], linewidth=2.5,
                   markersize=7, label=genotype_short[0], alpha=0.8)

        # Plot Het
        if 'het_temporal_penetrance' in results:
            het_times = [r['time_bin'] for r in results['het_temporal_penetrance']]
            het_pens = [r['embryo_penetrance'] * 100 for r in results['het_temporal_penetrance']]
            ax.plot(het_times, het_pens, marker=markers[1], color=colors[1], linewidth=2.5,
                   markersize=7, label=genotype_short[1], alpha=0.8)

        # Plot Homo
        if 'homo_temporal_penetrance' in results:
            homo_times = [r['time_bin'] for r in results['homo_temporal_penetrance']]
            homo_pens = [r['embryo_penetrance'] * 100 for r in results['homo_temporal_penetrance']]
            ax.plot(homo_times, homo_pens, marker=markers[2], color=colors[2], linewidth=2.5,
                   markersize=7, label=genotype_short[2], alpha=0.8)

        ax.set_xlabel('Time (hpf)', fontsize=11)
        ax.set_ylabel('Penetrance (%)', fontsize=11)
        ax.set_ylim(-5, 105)
        ax.set_title(CALIBRATIONS[calib_name]['label'], fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Temporal Penetrance: WT/Het/Homo Over Development',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figure_dir / 'temporal_penetrance_by_calibration.png', dpi=300, bbox_inches='tight')
    print("  Saved: temporal_penetrance_by_calibration.png")
    plt.close()


def plot_wt_bounds_summary(calibration_results, figure_dir):
    """Visualize WT bounds for each calibration."""
    fig, ax = plt.subplots(figsize=(10, 6))

    calib_names = ['global_iqr_1.5', 'global_iqr_2.0', 'global_iqr_3.0']

    for idx, calib_name in enumerate(calib_names):
        bounds = calibration_results[calib_name]['wt_bounds']

        # Draw bounds as vertical lines
        ax.axvline(bounds['low'], color='red', linestyle='--', alpha=0.3 + idx*0.2, linewidth=1.5)
        ax.axvline(bounds['high'], color='red', linestyle='--', alpha=0.3 + idx*0.2, linewidth=1.5)

        # Fill between bounds
        ax.axvspan(bounds['low'], bounds['high'], alpha=0.1 + idx*0.1, color='blue',
                   label=f"{CALIBRATIONS[calib_name]['label']}: [{bounds['low']:.4f}, {bounds['high']:.4f}]")

    ax.axvline(calibration_results[calib_names[0]]['wt_bounds']['median'],
               color='green', linestyle='-', linewidth=2, label='WT median')

    ax.set_xlabel(METRIC_NAME, fontsize=12)
    ax.set_ylabel('Calibration', fontsize=12)
    ax.set_title('WT Bounds Comparison: Global IQR Calibrations', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(figure_dir / 'bounds_summary.png', dpi=300, bbox_inches='tight')
    print("  Saved: bounds_summary.png")
    plt.close()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main analysis pipeline."""

    print("=" * 80)
    print("GLOBAL IQR CALIBRATION ANALYSIS (No Time Binning)")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df, metadata = get_analysis_dataframe()
    print(f"  Loaded {len(df)} timepoints from {df['embryo_id'].nunique()} embryos")

    # Bin data by time (for temporal analysis)
    print("\nBinning data by 2 hpf windows for temporal analysis...")
    df, time_bins = bin_data_by_time(df, bin_width=2.0)
    print(f"  Created {len(time_bins)} time bins")

    # Extract WT data
    wt_df = df[df['genotype'] == WT_GENOTYPE].copy()
    print(f"  WT: {wt_df['embryo_id'].nunique()} embryos, {len(wt_df)} timepoints")

    # Test all calibrations
    print(f"\n{'='*80}")
    print("TESTING GLOBAL IQR CALIBRATIONS")
    print(f"{'='*80}\n")

    calibration_results = {}

    for calib_name, calib_info in CALIBRATIONS.items():
        k = calib_info['k']
        print(f"Testing: {calib_info['label']} (k={k})")

        # Compute global WT bounds
        wt_bounds = compute_global_iqr_bounds(wt_df, metric=METRIC_NAME, k=k)

        print(f"  WT bounds: [{wt_bounds['low']:.6f}, {wt_bounds['high']:.6f}]")
        print(f"  WT median: {wt_bounds['median']:.6f}, IQR: {wt_bounds['iqr']:.6f}")

        # Compute WT baseline penetrance
        wt_df_marked = mark_penetrant_global(wt_df, wt_bounds, metric=METRIC_NAME)
        wt_penetrance, wt_stats = compute_penetrance_global(wt_df_marked)
        print(f"  WT penetrance: {wt_penetrance*100:.2f}% ({wt_stats['n_penetrant']}/{wt_stats['n_embryos']} embryos)")

        # Compute Het and Homo penetrance
        het_df = df[df['genotype'] == 'cep290_heterozygous'].copy()
        het_df_marked = mark_penetrant_global(het_df, wt_bounds, metric=METRIC_NAME)
        het_penetrance, het_stats = compute_penetrance_global(het_df_marked)
        print(f"  Het penetrance: {het_penetrance*100:.2f}% ({het_stats['n_penetrant']}/{het_stats['n_embryos']} embryos)")

        homo_df = df[df['genotype'] == 'cep290_homozygous'].copy()
        homo_df_marked = mark_penetrant_global(homo_df, wt_bounds, metric=METRIC_NAME)
        homo_penetrance, homo_stats = compute_penetrance_global(homo_df_marked)
        print(f"  Homo penetrance: {homo_penetrance*100:.2f}% ({homo_stats['n_penetrant']}/{homo_stats['n_embryos']} embryos)")

        # Compute temporal penetrance (per time bin)
        print(f"  Computing temporal penetrance...")
        wt_temporal = compute_penetrance_by_time(wt_df_marked, time_bins, metric_col='penetrant')
        het_temporal = compute_penetrance_by_time(het_df_marked, time_bins, metric_col='penetrant')
        homo_temporal = compute_penetrance_by_time(homo_df_marked, time_bins, metric_col='penetrant')
        print(f"    WT temporal: {len(wt_temporal)} time bins")
        print(f"    Het temporal: {len(het_temporal)} time bins")
        print(f"    Homo temporal: {len(homo_temporal)} time bins\n")

        calibration_results[calib_name] = {
            'label': calib_info['label'],
            'wt_bounds': wt_bounds,
            'wt_penetrance': wt_penetrance,
            'wt_stats': wt_stats,
            'het_penetrance': het_penetrance,
            'het_stats': het_stats,
            'homo_penetrance': homo_penetrance,
            'homo_stats': homo_stats,
            'wt_temporal_penetrance': wt_temporal,
            'het_temporal_penetrance': het_temporal,
            'homo_temporal_penetrance': homo_temporal,
        }

        # Generate diagnostic plots
        print(f"  Generating plots for {calib_name}...")
        plot_histogram_with_bounds(wt_df, wt_bounds, calib_name, FIGURE_DIR)
        plot_scatter_with_bounds(df, wt_bounds, calib_name, FIGURE_DIR)

    # Generate comparison plots
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*80}\n")

    plot_calibration_comparison(calibration_results, FIGURE_DIR)
    plot_temporal_penetrance_by_calibration(calibration_results, time_bins, FIGURE_DIR)
    plot_wt_bounds_summary(calibration_results, FIGURE_DIR)

    # Save summary table
    print("Saving summary statistics...")
    summary_df = pd.DataFrame([
        {
            'calibration': calib_name,
            'label': results['label'],
            'wt_bounds_low': results['wt_bounds']['low'],
            'wt_bounds_high': results['wt_bounds']['high'],
            'wt_median': results['wt_bounds']['median'],
            'wt_penetrance_%': results['wt_penetrance'] * 100,
            'het_penetrance_%': results['het_penetrance'] * 100,
            'homo_penetrance_%': results['homo_penetrance'] * 100,
        }
        for calib_name, results in calibration_results.items()
    ])

    summary_df.to_csv(TABLE_DIR / 'global_iqr_summary.csv', index=False)
    print(f"  Saved: global_iqr_summary.csv\n")
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables: {TABLE_DIR}")

    # Recommendation
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    print("\nGlobal IQR calibrations with WT baseline <5%:")
    for calib_name, results in calibration_results.items():
        if results['wt_penetrance'] < 0.05:
            print(f"  ✓ {results['label']}: {results['wt_penetrance']*100:.2f}% WT penetrance")

    print(f"\nComparing to time-binned analysis (06b):")
    print(f"  - Global approach: Single threshold across all timepoints")
    print(f"  - Time-binned approach: Separate threshold per 2hpf bin")
    print(f"  - Global is simpler but may miss temporal variation")
    print(f"  - Time-binned avoids sparse bin issues but creates more parameters")


if __name__ == '__main__':
    main()
