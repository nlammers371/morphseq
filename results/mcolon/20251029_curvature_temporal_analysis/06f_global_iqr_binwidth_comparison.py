"""
Binwidth Sensitivity Analysis: Global IQR ±1.5σ

Uses a single global IQR ±1.5σ threshold (WT reference) but bins temporal data
at different resolutions: 2hpf, 5hpf, 10hpf.

Goal: Determine if penetrance curve "spikiness" is due to:
- Normal fluctuation (smooths with larger bins)
- Real biological variation (persists across bin sizes)

Approach:
1. Compute global IQR ±1.5σ bounds from all WT data
2. Mark penetrant frames using this threshold
3. For each bin width, compute temporal penetrance curves
4. Compare how curves change with binning resolution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

from load_data import (
    get_analysis_dataframe,
    get_genotype_short_name,
    get_genotype_color,
)

warnings.filterwarnings('ignore')

# Configuration
METRIC_NAME = 'baseline_deviation_normalized'
ALT_METRIC_NAME = 'normalized_baseline_deviation'
RANDOM_SEED = 42
BINWIDTHS = [2.0, 5.0, 10.0]  # hpf

# Output directories
OUTPUT_DIR = Path(__file__).parent / 'outputs' / '06f_binwidth_comparison'
FIGURE_DIR = OUTPUT_DIR / 'figures'
TABLE_DIR = OUTPUT_DIR / 'tables'

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Genotypes
WT_GENOTYPE = 'cep290_wildtype'

np.random.seed(RANDOM_SEED)


# ============================================================================
# Metric Column Handling
# ============================================================================

def ensure_metric_column(df, metric=METRIC_NAME, alt_metric=ALT_METRIC_NAME):
    """
    Ensure the preferred metric column exists, falling back to alternative names.

    If the preferred column is missing but the alternative exists, duplicate the
    alternative into the preferred column name so the rest of the analysis can
    consistently reference `metric`.
    """
    if metric in df.columns:
        return df

    if alt_metric in df.columns:
        df = df.copy()
        df[metric] = df[alt_metric]
        print(
            f"  NOTE: '{metric}' not found. Using '{alt_metric}' values instead.",
        )
        return df

    raise KeyError(
        f"Required metric column not found. Expected '{metric}' or '{alt_metric}'."
    )


# ============================================================================
# Global IQR Computation (fixed at ±1.5σ)
# ============================================================================

def compute_global_iqr_bounds(wt_df, metric=METRIC_NAME, k=1.5):
    """Compute single IQR band from ALL WT data pooled."""
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
        'k': k,
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


# ============================================================================
# Time Binning Functions
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
    """Compute embryo-level penetrance per time bin with error estimates."""
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

        # Compute standard error (binomial)
        if n_embryos > 0:
            se = np.sqrt(embryo_penetrance * (1 - embryo_penetrance) / n_embryos)
        else:
            se = 0.0

        penetrance_results.append({
            'time_bin': time_bin,
            'embryo_penetrance': embryo_penetrance,
            'n_embryos': n_embryos,
            'n_penetrant': n_embryos_penetrant,
            'se': se,
        })

    return penetrance_results


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_temporal_by_binwidth(temporal_results, binwidths, figure_dir):
    """
    Plot temporal penetrance for each binwidth in separate subplots.

    Creates a figure with 3 subplots (one per binwidth) showing WT/Het/Homo
    penetrance curves.
    """
    fig, axes = plt.subplots(1, len(binwidths), figsize=(5 * len(binwidths), 5))

    if len(binwidths) == 1:
        axes = [axes]

    genotypes = [WT_GENOTYPE, 'cep290_heterozygous', 'cep290_homozygous']
    genotype_short = ['WT', 'Het', 'Homo']
    colors = [get_genotype_color(g) for g in genotypes]
    markers = ['^', 'o', 's']

    for ax_idx, binwidth in enumerate(binwidths):
        results = temporal_results[binwidth]
        ax = axes[ax_idx]

        # Plot WT
        wt_times = [r['time_bin'] for r in results['wt']]
        wt_pens = [r['embryo_penetrance'] * 100 for r in results['wt']]
        wt_ses = [r['se'] * 100 for r in results['wt']]
        ax.errorbar(wt_times, wt_pens, yerr=wt_ses, marker=markers[0], color=colors[0],
                   linewidth=2.5, markersize=7, label=genotype_short[0], alpha=0.8, capsize=4)

        # Plot Het
        het_times = [r['time_bin'] for r in results['het']]
        het_pens = [r['embryo_penetrance'] * 100 for r in results['het']]
        het_ses = [r['se'] * 100 for r in results['het']]
        ax.errorbar(het_times, het_pens, yerr=het_ses, marker=markers[1], color=colors[1],
                   linewidth=2.5, markersize=7, label=genotype_short[1], alpha=0.8, capsize=4)

        # Plot Homo
        homo_times = [r['time_bin'] for r in results['homo']]
        homo_pens = [r['embryo_penetrance'] * 100 for r in results['homo']]
        homo_ses = [r['se'] * 100 for r in results['homo']]
        ax.errorbar(homo_times, homo_pens, yerr=homo_ses, marker=markers[2], color=colors[2],
                   linewidth=2.5, markersize=7, label=genotype_short[2], alpha=0.8, capsize=4)

        ax.set_xlabel('Time (hpf)', fontsize=11)
        ax.set_ylabel('Penetrance (%)', fontsize=11)
        ax.set_ylim(-5, 105)
        ax.set_title(f'{binwidth:.1f} hpf bins (n={len(results["wt"])} bins)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Temporal Penetrance: Effect of Bin Width (IQR ±1.5σ)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figure_dir / 'temporal_penetrance_by_binwidth.png', dpi=300, bbox_inches='tight')
    print("  Saved: temporal_penetrance_by_binwidth.png")
    plt.close()


def plot_genotype_smoothing(temporal_results, binwidths, figure_dir):
    """
    Plot how a single genotype's curve smooths with different binwidths.
    Creates 3 subplots (one per genotype) with overlaid curves for each binwidth.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    genotypes = [WT_GENOTYPE, 'cep290_heterozygous', 'cep290_homozygous']
    genotype_short = ['WT', 'Het', 'Homo']
    genotype_keys = ['wt', 'het', 'homo']
    colors_binwidth = ['blue', 'orange', 'red']
    linestyles = ['-', '--', ':']

    for ax_idx, (genotype, genotype_short_name, geno_key) in enumerate(zip(genotypes, genotype_short, genotype_keys)):
        ax = axes[ax_idx]

        for binwidth_idx, binwidth in enumerate(binwidths):
            results = temporal_results[binwidth]
            data = results[geno_key]

            times = [r['time_bin'] for r in data]
            pens = [r['embryo_penetrance'] * 100 for r in data]
            ses = [r['se'] * 100 for r in data]

            ax.errorbar(times, pens, yerr=ses, marker='o', color=colors_binwidth[binwidth_idx],
                       linewidth=2.5, markersize=6, label=f'{binwidth:.1f} hpf',
                       alpha=0.7, linestyle=linestyles[binwidth_idx], capsize=3)

        ax.set_xlabel('Time (hpf)', fontsize=11)
        ax.set_ylabel('Penetrance (%)', fontsize=11)
        ax.set_ylim(-5, 105)
        ax.set_title(f'{genotype_short_name}\n(shows smoothing effect)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Smoothing Effect of Bin Width: Each Genotype Across 2/5/10 hpf Bins',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figure_dir / 'genotype_smoothing_by_binwidth.png', dpi=300, bbox_inches='tight')
    print("  Saved: genotype_smoothing_by_binwidth.png")
    plt.close()


def plot_wt_focus(temporal_results, binwidths, figure_dir):
    """Focus on WT only to clearly show noise vs signal."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors_binwidth = ['blue', 'orange', 'red']
    markers = ['^', 'o', 's']
    sizes = [8, 7, 6]

    for binwidth_idx, binwidth in enumerate(binwidths):
        results = temporal_results[binwidth]
        data = results['wt']

        times = [r['time_bin'] for r in data]
        pens = [r['embryo_penetrance'] * 100 for r in data]
        ses = [r['se'] * 100 for r in data]

        ax.errorbar(times, pens, yerr=ses, marker=markers[binwidth_idx],
                   color=colors_binwidth[binwidth_idx], linewidth=2.5,
                   markersize=sizes[binwidth_idx],
                   label=f'{binwidth:.1f} hpf bins (n={len(data)} bins)',
                   alpha=0.8, capsize=4)

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('WT Penetrance (%)', fontsize=12)
    ax.set_ylim(-5, 60)
    ax.set_title('WT Penetrance: Noise vs Signal Assessment\nLarger bins reveal underlying trend if noise dominates',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figure_dir / 'wt_penetrance_smoothing_focus.png', dpi=300, bbox_inches='tight')
    print("  Saved: wt_penetrance_smoothing_focus.png")
    plt.close()


# ============================================================================
# Summary Statistics
# ============================================================================

def compute_summary_stats(temporal_results, binwidths):
    """Compute summary statistics for each genotype × binwidth combination."""
    summary_data = []

    genotype_keys = ['wt', 'het', 'homo']
    genotype_names = ['WT', 'Het', 'Homo']

    for binwidth in binwidths:
        results = temporal_results[binwidth]

        for geno_key, geno_name in zip(genotype_keys, genotype_names):
            data = results[geno_key]

            if len(data) == 0:
                continue

            pens = [r['embryo_penetrance'] for r in data]
            ses = [r['se'] for r in data]

            summary_data.append({
                'binwidth_hpf': binwidth,
                'genotype': geno_name,
                'n_time_bins': len(data),
                'mean_penetrance_%': np.mean(pens) * 100,
                'std_penetrance_%': np.std(pens) * 100,
                'min_penetrance_%': np.min(pens) * 100,
                'max_penetrance_%': np.max(pens) * 100,
                'range_penetrance_%': (np.max(pens) - np.min(pens)) * 100,
                'mean_se_%': np.mean(ses) * 100,
            })

    return pd.DataFrame(summary_data)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main analysis pipeline."""

    print("=" * 80)
    print("BINWIDTH SENSITIVITY ANALYSIS: Global IQR ±1.5σ")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df, metadata = get_analysis_dataframe()
    df = ensure_metric_column(df, metric=METRIC_NAME, alt_metric=ALT_METRIC_NAME)
    print(f"  Loaded {len(df)} timepoints from {df['embryo_id'].nunique()} embryos")

    # Extract WT data for global threshold
    wt_df = df[df['genotype'] == WT_GENOTYPE].copy()
    print(f"  WT: {wt_df['embryo_id'].nunique()} embryos, {len(wt_df)} timepoints")

    # Compute global IQR ±1.5σ bounds
    print(f"\nComputing global IQR ±1.5σ bounds...")
    wt_bounds = compute_global_iqr_bounds(wt_df, metric=METRIC_NAME, k=1.5)
    print(f"  WT bounds: [{wt_bounds['low']:.6f}, {wt_bounds['high']:.6f}]")
    print(f"  WT median: {wt_bounds['median']:.6f}, IQR: {wt_bounds['iqr']:.6f}")

    # Mark penetrant frames using global threshold
    print(f"\nMarking penetrant frames using global threshold...")
    df_marked = mark_penetrant_global(df, wt_bounds, metric=METRIC_NAME)

    # Compute temporal penetrance for each binwidth
    print(f"\n{'='*80}")
    print("COMPUTING TEMPORAL PENETRANCE FOR EACH BINWIDTH")
    print(f"{'='*80}\n")

    temporal_results = {}

    for binwidth in BINWIDTHS:
        print(f"Binwidth: {binwidth} hpf")

        # Bin data
        df_binned, time_bins = bin_data_by_time(df_marked, bin_width=binwidth)
        print(f"  Created {len(time_bins)} time bins")

        # Extract genotypes
        wt_df_binned = df_binned[df_binned['genotype'] == WT_GENOTYPE].copy()
        het_df_binned = df_binned[df_binned['genotype'] == 'cep290_heterozygous'].copy()
        homo_df_binned = df_binned[df_binned['genotype'] == 'cep290_homozygous'].copy()

        # Compute penetrance per time bin
        wt_temporal = compute_penetrance_by_time(wt_df_binned, time_bins, metric_col='penetrant')
        het_temporal = compute_penetrance_by_time(het_df_binned, time_bins, metric_col='penetrant')
        homo_temporal = compute_penetrance_by_time(homo_df_binned, time_bins, metric_col='penetrant')

        print(f"    WT:   {len(wt_temporal)} bins, mean penetrance: {np.mean([r['embryo_penetrance'] for r in wt_temporal])*100:.2f}%")
        print(f"    Het:  {len(het_temporal)} bins, mean penetrance: {np.mean([r['embryo_penetrance'] for r in het_temporal])*100:.2f}%")
        print(f"    Homo: {len(homo_temporal)} bins, mean penetrance: {np.mean([r['embryo_penetrance'] for r in homo_temporal])*100:.2f}%\n")

        temporal_results[binwidth] = {
            'wt': wt_temporal,
            'het': het_temporal,
            'homo': homo_temporal,
        }

    # Generate plots
    print(f"{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}\n")

    plot_temporal_by_binwidth(temporal_results, BINWIDTHS, FIGURE_DIR)
    plot_genotype_smoothing(temporal_results, BINWIDTHS, FIGURE_DIR)
    plot_wt_focus(temporal_results, BINWIDTHS, FIGURE_DIR)

    # Save summary statistics
    print("Saving summary statistics...")
    summary_df = compute_summary_stats(temporal_results, BINWIDTHS)
    summary_df.to_csv(TABLE_DIR / 'binwidth_summary.csv', index=False)
    print(f"  Saved: binwidth_summary.csv\n")
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables: {TABLE_DIR}")

    # Interpretation guide
    print(f"\n{'='*80}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*80}")
    print("\nComparing binwidth curves:")
    print("  - If curves align across binwidths → true biological signal")
    print("  - If 2hpf is spiky but 10hpf is smooth → noise in data")
    print("  - Larger bins reduce variance but may hide real biology")
    print("\nWT penetrance interpretation:")
    print(f"  - Current (2hpf): {summary_df[(summary_df['binwidth_hpf']==2) & (summary_df['genotype']=='WT')]['mean_penetrance_%'].values[0]:.1f}%")
    print(f"  - With 5hpf:     {summary_df[(summary_df['binwidth_hpf']==5) & (summary_df['genotype']=='WT')]['mean_penetrance_%'].values[0]:.1f}%")
    print(f"  - With 10hpf:    {summary_df[(summary_df['binwidth_hpf']==10) & (summary_df['genotype']=='WT')]['mean_penetrance_%'].values[0]:.1f}%")


if __name__ == '__main__':
    main()
