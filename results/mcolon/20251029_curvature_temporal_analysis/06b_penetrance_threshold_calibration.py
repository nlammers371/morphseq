"""
WT Penetrance Threshold Calibration Analysis

This script explores how different WT reference thresholds affect penetrance estimates.
The goal is to find a calibration where WT has minimal baseline penetrance (~1-5%)
while maintaining biological interpretability.

Strategies tested:
1. Percentile-based bands (90%, 95%, 99%, 99.9%)
2. IQR-based outlier detection (k=1.5, 2.0, 3.0)
3. Custom stringency levels

Per-genotype analysis: WT, Het, and Homo analyzed independently.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

# Import utilities from existing analysis
from load_data import (
    get_analysis_dataframe,
    get_genotype_short_name,
    get_genotype_color,
)

warnings.filterwarnings('ignore')

# Configuration
METRIC_NAME = 'normalized_baseline_deviation'
TIME_BIN_WIDTH = 2.0  # hpf
N_BOOTSTRAP = 50
BOOTSTRAP_HOLDOUT_FRACTION = 0.2
RANDOM_SEED = 42

# Output directories
OUTPUT_DIR = Path(__file__).parent / 'outputs' / '06b_penetrance_calibration'
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

# Calibration strategies to test
CALIBRATIONS = {
    'perc_90': {'type': 'percentile', 'params': {'low': 5, 'high': 95}, 'label': '90% band (5-95%)'},
    'perc_95': {'type': 'percentile', 'params': {'low': 2.5, 'high': 97.5}, 'label': '95% band (2.5-97.5%)'},
    'perc_99': {'type': 'percentile', 'params': {'low': 0.5, 'high': 99.5}, 'label': '99% band (0.5-99.5%)'},
    'perc_999': {'type': 'percentile', 'params': {'low': 0.05, 'high': 99.95}, 'label': '99.9% band (0.05-99.95%)'},
    'iqr_1.5': {'type': 'iqr', 'params': {'k': 1.5}, 'label': 'IQR ±1.5σ'},
    'iqr_2.0': {'type': 'iqr', 'params': {'k': 2.0}, 'label': 'IQR ±2.0σ'},
    'iqr_3.0': {'type': 'iqr', 'params': {'k': 3.0}, 'label': 'IQR ±3.0σ'},
}

np.random.seed(RANDOM_SEED)


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


def compute_wt_reference_envelope_percentile(
    wt_df,
    time_bins,
    metric=METRIC_NAME,
    percentile_low=2.5,
    percentile_high=97.5
):
    """Compute WT reference using percentile method."""
    envelope = {}

    for time_bin in time_bins:
        bin_df = wt_df[wt_df['time_bin'] == time_bin]

        if len(bin_df) == 0:
            continue

        values = bin_df[metric].values

        envelope[time_bin] = {
            'low': np.percentile(values, percentile_low),
            'high': np.percentile(values, percentile_high),
            'median': np.percentile(values, 50),
        }

    return envelope


def compute_wt_reference_envelope_iqr(
    wt_df,
    time_bins,
    metric=METRIC_NAME,
    k=1.5
):
    """Compute WT reference using IQR method: Q1 - k*IQR to Q3 + k*IQR."""
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
        }

    return envelope


def compute_wt_reference_envelope(wt_df, time_bins, metric, calibration_name):
    """Compute WT reference envelope for a specific calibration."""
    if calibration_name not in CALIBRATIONS:
        raise ValueError(f"Unknown calibration: {calibration_name}")

    calib = CALIBRATIONS[calibration_name]

    if calib['type'] == 'percentile':
        return compute_wt_reference_envelope_percentile(
            wt_df, time_bins, metric,
            percentile_low=calib['params']['low'],
            percentile_high=calib['params']['high']
        )
    elif calib['type'] == 'iqr':
        return compute_wt_reference_envelope_iqr(
            wt_df, time_bins, metric,
            k=calib['params']['k']
        )
    else:
        raise ValueError(f"Unknown calibration type: {calib['type']}")


def mark_penetrant_timepoints(df, wt_envelope, metric=METRIC_NAME):
    """Mark timepoints as penetrant if outside WT envelope."""
    df = df.copy()
    df['penetrant'] = 0

    for idx, row in df.iterrows():
        time_bin = row['time_bin']

        if pd.isna(time_bin) or time_bin not in wt_envelope:
            df.loc[idx, 'penetrant'] = np.nan
            continue

        metric_value = row[metric]
        envelope = wt_envelope[time_bin]

        if metric_value < envelope['low'] or metric_value > envelope['high']:
            df.loc[idx, 'penetrant'] = 1

    return df


def compute_penetrance_by_time(df, time_bins, metric_col='penetrant'):
    """Compute embryo-level penetrance over time."""
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


def compute_wt_baseline_penetrance(wt_df, wt_envelope, time_bins, metric=METRIC_NAME):
    """Compute WT baseline penetrance (should be minimal)."""
    wt_df = mark_penetrant_timepoints(wt_df, wt_envelope, metric=metric)
    penetrance_results = compute_penetrance_by_time(wt_df, time_bins)

    # Average across all time bins
    if len(penetrance_results) == 0:
        return 0.0, []

    embryo_pens = [r['embryo_penetrance'] for r in penetrance_results]
    mean_wt_penetrance = np.mean(embryo_pens)

    return mean_wt_penetrance, penetrance_results


def main():
    """Main analysis pipeline."""

    print("=" * 80)
    print("WT PENETRANCE THRESHOLD CALIBRATION ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df, metadata = get_analysis_dataframe()
    print(f"  Loaded {len(df)} timepoints from {df['embryo_id'].nunique()} embryos")

    # Bin data by time
    print(f"\nBinning data by {TIME_BIN_WIDTH} hpf windows...")
    df, time_bins = bin_data_by_time(df, bin_width=TIME_BIN_WIDTH)
    print(f"  Created {len(time_bins)} time bins")

    # Extract WT data
    wt_df = df[df['genotype'] == WT_GENOTYPE].copy()
    print(f"  WT: {wt_df['embryo_id'].nunique()} embryos")

    # Test all calibrations
    print(f"\n{'='*80}")
    print("TESTING CALIBRATIONS")
    print(f"{'='*80}\n")

    calibration_results = {}

    for calib_name, calib_info in CALIBRATIONS.items():
        print(f"Testing: {calib_info['label']}")

        # Compute WT envelope
        wt_envelope = compute_wt_reference_envelope(wt_df, time_bins, METRIC_NAME, calib_name)

        # Compute WT baseline penetrance
        wt_baseline, wt_penetrance = compute_wt_baseline_penetrance(
            wt_df, wt_envelope, time_bins, metric=METRIC_NAME
        )

        print(f"  WT baseline penetrance: {wt_baseline*100:.2f}%")

        # Compute Het and Homo penetrance
        het_results = {}
        homo_results = {}

        for genotype in ANALYSIS_GENOTYPES:
            genotype_df = df[df['genotype'] == genotype].copy()
            genotype_df = mark_penetrant_timepoints(genotype_df, wt_envelope, metric=METRIC_NAME)
            penetrance_results = compute_penetrance_by_time(genotype_df, time_bins)

            embryo_pens = [r['embryo_penetrance'] for r in penetrance_results]
            mean_penetrance = np.mean(embryo_pens)

            if genotype == 'cep290_heterozygous':
                het_results = {
                    'mean': mean_penetrance,
                    'penetrance': penetrance_results
                }
                print(f"  Het penetrance: {mean_penetrance*100:.2f}%")
            else:
                homo_results = {
                    'mean': mean_penetrance,
                    'penetrance': penetrance_results
                }
                print(f"  Homo penetrance: {mean_penetrance*100:.2f}%")

        calibration_results[calib_name] = {
            'label': calib_info['label'],
            'wt_baseline': wt_baseline,
            'wt_penetrance': wt_penetrance,
            'het': het_results,
            'homo': homo_results,
            'wt_envelope': wt_envelope
        }

    # Generate comparison plots
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*80}\n")

    plot_calibration_comparison(calibration_results, time_bins, FIGURE_DIR)
    plot_wt_baseline_by_calibration(calibration_results, TABLE_DIR, FIGURE_DIR)
    plot_dose_response_by_calibration(calibration_results, FIGURE_DIR)

    # Generate diagnostic plots for IQR methods
    print(f"\n{'='*80}")
    print("GENERATING DIAGNOSTIC PLOTS")
    print(f"{'='*80}\n")

    iqr_calibrations = ['iqr_1.5', 'iqr_2.0', 'iqr_3.0']
    for calib_name in iqr_calibrations:
        if calib_name in calibration_results:
            results = calibration_results[calib_name]
            print(f"Diagnostic plot for: {results['label']}")
            plot_threshold_bounds_diagnostic(
                wt_df,
                results['wt_envelope'],
                time_bins,
                results['label'],
                FIGURE_DIR / f'threshold_diagnostic_{calib_name}.png'
            )

    # Save summary table
    print("\nSaving summary statistics...")
    summary_df = pd.DataFrame([
        {
            'calibration': calib_name,
            'label': results['label'],
            'wt_baseline_%': results['wt_baseline'] * 100,
            'het_mean_%': results['het']['mean'] * 100,
            'homo_mean_%': results['homo']['mean'] * 100,
        }
        for calib_name, results in calibration_results.items()
    ])

    summary_df.to_csv(TABLE_DIR / 'calibration_summary.csv', index=False)
    print(f"  Saved: calibration_summary.csv\n")
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables: {TABLE_DIR}")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    print("\nCalibrations with WT baseline <5%:")
    for calib_name, results in calibration_results.items():
        if results['wt_baseline'] < 0.05:
            print(f"  ✓ {results['label']}: {results['wt_baseline']*100:.2f}% WT penetrance")


def plot_calibration_comparison(calibration_results, time_bins, output_dir):
    """Plot penetrance curves for all calibrations."""
    n_calib = len(calibration_results)
    calib_names = list(calibration_results.keys())

    # Create grid: 3 rows (WT, Het, Homo) x N cols (calibrations)
    fig, axes = plt.subplots(3, n_calib, figsize=(4 * n_calib, 12))

    if n_calib == 1:
        axes = axes.reshape(3, 1)

    genotypes = [WT_GENOTYPE, 'cep290_heterozygous', 'cep290_homozygous']
    genotype_short = ['WT', 'Het', 'Homo']
    colors = [get_genotype_color(g) for g in genotypes]

    for col_idx, calib_name in enumerate(calib_names):
        results = calibration_results[calib_name]

        # WT row
        ax = axes[0, col_idx]
        wt_times = [r['time_bin'] for r in results['wt_penetrance']]
        wt_pens = [r['embryo_penetrance'] * 100 for r in results['wt_penetrance']]
        ax.plot(wt_times, wt_pens, 'o-', color=colors[0], linewidth=2, markersize=5)
        ax.set_ylabel('Penetrance (%)', fontsize=10)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel('WT\nPenetrance (%)', fontsize=10, fontweight='bold')
        ax.set_title(results['label'], fontsize=10, fontweight='bold')

        # Het row
        ax = axes[1, col_idx]
        het_times = [r['time_bin'] for r in results['het']['penetrance']]
        het_pens = [r['embryo_penetrance'] * 100 for r in results['het']['penetrance']]
        ax.plot(het_times, het_pens, 'o-', color=colors[1], linewidth=2, markersize=5)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel('Het\nPenetrance (%)', fontsize=10, fontweight='bold')

        # Homo row
        ax = axes[2, col_idx]
        homo_times = [r['time_bin'] for r in results['homo']['penetrance']]
        homo_pens = [r['embryo_penetrance'] * 100 for r in results['homo']['penetrance']]
        ax.plot(homo_times, homo_pens, 'o-', color=colors[2], linewidth=2, markersize=5)
        ax.set_xlabel('Time (hpf)', fontsize=10)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel('Homo\nPenetrance (%)', fontsize=10, fontweight='bold')

    plt.suptitle('Penetrance Across Calibrations', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'penetrance_calibration_sweep.png', dpi=300, bbox_inches='tight')
    print("  Saved: penetrance_calibration_sweep.png")
    plt.close()


def plot_wt_baseline_by_calibration(calibration_results, table_dir, figure_dir):
    """Plot WT baseline penetrance vs calibration stringency."""
    calib_names = list(calibration_results.keys())
    wt_baselines = [calibration_results[c]['wt_baseline'] * 100 for c in calib_names]
    labels = [calibration_results[c]['label'] for c in calib_names]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(calib_names)), wt_baselines, color='steelblue', edgecolor='black', linewidth=1.5)

    # Color bars by stringency level
    for i, bar in enumerate(bars):
        if wt_baselines[i] < 1:
            bar.set_color('darkgreen')
        elif wt_baselines[i] < 5:
            bar.set_color('green')
        elif wt_baselines[i] < 10:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # Add reference lines
    ax.axhline(1, color='darkgreen', linestyle='--', linewidth=2, alpha=0.5, label='Target: 1%')
    ax.axhline(5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target: 5%')

    ax.set_xticks(range(len(calib_names)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('WT Baseline Penetrance (%)', fontsize=12)
    ax.set_title('WT Baseline Penetrance by Calibration Strategy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, wt_baselines)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figure_dir / 'wt_baseline_penetrance_by_calibration.png', dpi=300, bbox_inches='tight')
    print("  Saved: wt_baseline_penetrance_by_calibration.png")
    plt.close()


def plot_dose_response_by_calibration(calibration_results, figure_dir):
    """Plot dose response (WT/Het/Homo) for selected calibrations."""
    # Select 4 representative calibrations
    selected = ['perc_95', 'perc_99', 'iqr_1.5', 'iqr_3.0']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for plot_idx, calib_name in enumerate(selected):
        if calib_name not in calibration_results:
            continue

        results = calibration_results[calib_name]
        ax = axes[plot_idx]

        genotypes = [WT_GENOTYPE, 'cep290_heterozygous', 'cep290_homozygous']
        genotype_short = ['WT', 'Het', 'Homo']
        colors = [get_genotype_color(g) for g in genotypes]
        markers = ['^', 'o', 's']

        # Plot WT
        wt_times = [r['time_bin'] for r in results['wt_penetrance']]
        wt_pens = [r['embryo_penetrance'] * 100 for r in results['wt_penetrance']]
        ax.plot(wt_times, wt_pens, marker=markers[0], color=colors[0], linewidth=2,
                markersize=6, label=genotype_short[0], alpha=0.7)

        # Plot Het
        het_times = [r['time_bin'] for r in results['het']['penetrance']]
        het_pens = [r['embryo_penetrance'] * 100 for r in results['het']['penetrance']]
        ax.plot(het_times, het_pens, marker=markers[1], color=colors[1], linewidth=2,
                markersize=6, label=genotype_short[1], alpha=0.7)

        # Plot Homo
        homo_times = [r['time_bin'] for r in results['homo']['penetrance']]
        homo_pens = [r['embryo_penetrance'] * 100 for r in results['homo']['penetrance']]
        ax.plot(homo_times, homo_pens, marker=markers[2], color=colors[2], linewidth=2,
                markersize=6, label=genotype_short[2], alpha=0.7)

        ax.set_xlabel('Developmental Time (hpf)', fontsize=11)
        ax.set_ylabel('Penetrance (%)', fontsize=11)
        ax.set_title(results['label'], fontsize=12, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Dose Response: WT, Het, Homo Penetrance by Calibration', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figure_dir / 'dose_response_by_calibration.png', dpi=300, bbox_inches='tight')
    print("  Saved: dose_response_by_calibration.png")
    plt.close()


def plot_threshold_bounds_diagnostic(wt_df, wt_envelope, time_bins, calibration_label, output_path):
    """
    Plot threshold bounds with sample counts.

    Top panel: IQR threshold bounds (low, median, high) over time
    Bottom panel: Sample count per time bin
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # Extract threshold values
    times = []
    lows = []
    medians = []
    highs = []
    sample_counts = []

    for time_bin in time_bins:
        if time_bin not in wt_envelope:
            continue

        times.append(time_bin)
        envelope = wt_envelope[time_bin]
        lows.append(envelope['low'])
        medians.append(envelope['median'])
        highs.append(envelope['high'])

        # Count samples at this time bin
        bin_df = wt_df[wt_df['time_bin'] == time_bin]
        sample_counts.append(len(bin_df))

    times = np.array(times)
    lows = np.array(lows)
    medians = np.array(medians)
    highs = np.array(highs)
    sample_counts = np.array(sample_counts)

    # TOP PANEL: Threshold bounds
    ax_top.fill_between(times, lows, highs, alpha=0.3, color='blue', label='Acceptable WT range')
    ax_top.plot(times, medians, 'b-', linewidth=2.5, label='WT median')
    ax_top.plot(times, lows, 'b--', linewidth=1.5, alpha=0.7, label='Lower threshold (Q1 - k×IQR)')
    ax_top.plot(times, highs, 'b--', linewidth=1.5, alpha=0.7, label='Upper threshold (Q3 + k×IQR)')

    ax_top.set_xlabel('Developmental Time (hpf)', fontsize=12)
    ax_top.set_ylabel(f'{METRIC_NAME}', fontsize=12)
    ax_top.set_title(f'WT Threshold Bounds: {calibration_label}', fontsize=13, fontweight='bold')
    ax_top.legend(fontsize=10, loc='best')
    ax_top.grid(True, alpha=0.3)

    # BOTTOM PANEL: Sample counts
    colors = ['red' if n < 10 else 'steelblue' for n in sample_counts]
    ax_bottom.bar(times, sample_counts, width=1.8, color=colors, edgecolor='black', linewidth=0.5, alpha=0.7)

    # Add reference line for sparse data
    ax_bottom.axhline(10, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Sparse threshold (N<10)')

    ax_bottom.set_xlabel('Developmental Time (hpf)', fontsize=12)
    ax_bottom.set_ylabel('# WT Samples per Bin', fontsize=12)
    ax_bottom.set_title('Sample Count by Time Bin', fontsize=12, fontweight='bold')
    ax_bottom.legend(fontsize=10, loc='best')
    ax_bottom.grid(True, alpha=0.3, axis='y')

    # Align x-axes
    ax_top.set_xlim(times[0] - 1, times[-1] + 1)
    ax_bottom.set_xlim(times[0] - 1, times[-1] + 1)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


if __name__ == '__main__':
    main()
