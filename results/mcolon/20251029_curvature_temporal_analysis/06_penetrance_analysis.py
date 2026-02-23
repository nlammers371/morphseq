"""
WT Reference-Based Penetrance Analysis

This script computes penetrance as deviation beyond wild-type reference bands.
Uses WT percentile envelope (2.5-97.5%) as the reference range for normal morphology.
Analyzes Het and Homo separately against the same WT reference.

Bootstrap validation: 50 iterations with 20% embryo holdout per iteration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
import random
from scipy.stats import binom
from multiprocessing import Pool
import os

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
WT_PERCENTILE_LOW = 2.5
WT_PERCENTILE_HIGH = 97.5
N_BOOTSTRAP = 50
BOOTSTRAP_HOLDOUT_FRACTION = 0.2
RANDOM_SEED = 42

# Output directories
OUTPUT_DIR = Path(__file__).parent / 'outputs' / '06_penetrance_analysis'
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

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def bin_data_by_time(df, bin_width=2.0, time_col='predicted_stage_hpf'):
    """
    Bin data by developmental time.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time column
    bin_width : float
        Width of each bin in hpf
    time_col : str
        Column name for developmental time

    Returns
    -------
    df : pd.DataFrame
        Data with added 'time_bin' column
    time_bins : np.ndarray
        Array of bin centers
    """
    df = df.copy()
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    # Create bins
    bin_edges = np.arange(
        np.floor(min_time / bin_width) * bin_width,
        np.ceil(max_time / bin_width) * bin_width + bin_width,
        bin_width
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign bins
    df['time_bin'] = pd.cut(df[time_col], bins=bin_edges, labels=bin_centers)
    df['time_bin'] = df['time_bin'].astype(float)

    return df, bin_centers


def compute_wt_reference_envelope(
    wt_df,
    time_bins,
    metric=METRIC_NAME,
    percentile_low=WT_PERCENTILE_LOW,
    percentile_high=WT_PERCENTILE_HIGH
):
    """
    Compute WT reference envelope (percentile bands).

    For each time bin, computes the percentile band from WT embryos.
    Returns both the envelope and the full distribution statistics.

    Parameters
    ----------
    wt_df : pd.DataFrame
        WT embryo data with time_bin and metric columns
    time_bins : np.ndarray
        Array of time bin centers
    metric : str
        Metric column name
    percentile_low : float
        Lower percentile (e.g., 2.5)
    percentile_high : float
        Upper percentile (e.g., 97.5)

    Returns
    -------
    envelope : dict
        Dictionary mapping time_bin -> {'low': value, 'median': value, 'high': value}
    """
    envelope = {}

    for time_bin in time_bins:
        bin_df = wt_df[wt_df['time_bin'] == time_bin]

        if len(bin_df) == 0:
            continue

        values = bin_df[metric].values

        envelope[time_bin] = {
            'low': np.percentile(values, percentile_low),
            'median': np.percentile(values, 50),
            'high': np.percentile(values, percentile_high),
            'mean': np.mean(values),
            'std': np.std(values),
            'n': len(values)
        }

    return envelope


def mark_penetrant_timepoints(
    df,
    wt_envelope,
    metric=METRIC_NAME
):
    """
    Mark timepoints as penetrant if outside WT envelope.

    A timepoint is marked as penetrant (1) if the metric value falls outside
    the WT reference band (below low or above high percentile).

    Parameters
    ----------
    df : pd.DataFrame
        Data with time_bin and metric columns
    wt_envelope : dict
        WT reference envelope from compute_wt_reference_envelope()
    metric : str
        Metric column name

    Returns
    -------
    df : pd.DataFrame
        Data with added 'penetrant' column (0 or 1)
    """
    df = df.copy()
    df['penetrant'] = 0

    for idx, row in df.iterrows():
        time_bin = row['time_bin']

        if pd.isna(time_bin) or time_bin not in wt_envelope:
            df.loc[idx, 'penetrant'] = np.nan
            continue

        metric_value = row[metric]
        envelope = wt_envelope[time_bin]

        # Penetrant if outside WT band
        if metric_value < envelope['low'] or metric_value > envelope['high']:
            df.loc[idx, 'penetrant'] = 1

    return df


def compute_penetrance_by_time(
    df,
    time_bins,
    metric_col='penetrant'
):
    """
    Compute sample-level and embryo-level penetrance over time.

    Sample-level: % of measurements penetrant in each bin
    Embryo-level: % of embryos with ≥1 penetrant measurement in each bin

    Parameters
    ----------
    df : pd.DataFrame
        Data with time_bin and metric_col columns
    time_bins : np.ndarray
        Array of time bin centers
    metric_col : str
        Name of penetrance column

    Returns
    -------
    penetrance_by_time : list of dict
        List with one dict per time bin containing:
        - time_bin
        - sample_penetrance, sample_ci_low, sample_ci_high
        - embryo_penetrance, embryo_ci_low, embryo_ci_high
        - n_samples, n_embryos
    """
    penetrance_results = []

    for time_bin in time_bins:
        bin_df = df[df['time_bin'] == time_bin].dropna(subset=[metric_col])

        if len(bin_df) == 0:
            continue

        # Sample-level penetrance
        n_penetrant_samples = (bin_df[metric_col] == 1).sum()
        n_total_samples = len(bin_df)
        sample_penetrance = n_penetrant_samples / n_total_samples if n_total_samples > 0 else 0

        # Embryo-level penetrance
        embryos_in_bin = bin_df['embryo_id'].unique()
        embryos_penetrant = bin_df[bin_df[metric_col] == 1]['embryo_id'].unique()
        n_embryos = len(embryos_in_bin)
        n_embryos_penetrant = len(embryos_penetrant)
        embryo_penetrance = n_embryos_penetrant / n_embryos if n_embryos > 0 else 0

        # Wilson confidence intervals
        sample_ci_low, sample_ci_high = wilson_ci(n_penetrant_samples, n_total_samples)
        embryo_ci_low, embryo_ci_high = wilson_ci(n_embryos_penetrant, n_embryos)

        penetrance_results.append({
            'time_bin': time_bin,
            'sample_penetrance': sample_penetrance,
            'sample_ci_low': sample_ci_low,
            'sample_ci_high': sample_ci_high,
            'embryo_penetrance': embryo_penetrance,
            'embryo_ci_low': embryo_ci_low,
            'embryo_ci_high': embryo_ci_high,
            'n_samples': n_total_samples,
            'n_embryos': n_embryos
        })

    return penetrance_results


def compute_onset_times(df, metric_col='penetrant'):
    """
    Compute first penetrant timepoint per embryo.

    For each embryo, finds the first time bin where metric_col == 1.

    Parameters
    ----------
    df : pd.DataFrame
        Data with embryo_id, time_bin, and metric_col columns (sorted by time)
    metric_col : str
        Name of penetrance column

    Returns
    -------
    onset_times : dict
        Dictionary mapping embryo_id -> first_time_bin_with_penetrant_flag
        (returns np.nan if embryo never shows penetrant phenotype)
    """
    onset_times = {}

    for embryo_id in df['embryo_id'].unique():
        embryo_df = df[df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')

        # Find first penetrant timepoint
        penetrant_rows = embryo_df[embryo_df[metric_col] == 1]

        if len(penetrant_rows) > 0:
            onset_times[embryo_id] = penetrant_rows.iloc[0]['predicted_stage_hpf']
        else:
            onset_times[embryo_id] = np.nan

    return onset_times


def wilson_ci(successes, n, confidence=0.95):
    """
    Compute Wilson confidence interval for binomial proportion.

    More robust than normal approximation, especially for small n.

    Parameters
    ----------
    successes : int
        Number of successes
    n : int
        Total number of trials
    confidence : float
        Confidence level (default 0.95 for 95% CI)

    Returns
    -------
    ci_low, ci_high : float
        Lower and upper bounds of CI
    """
    if n == 0:
        return np.nan, np.nan

    p_hat = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # ~1.96 for 95% CI

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator

    ci_low = max(0, center - margin)
    ci_high = min(1, center + margin)

    return ci_low, ci_high


def _bootstrap_penetrance_single_iteration(args):
    """Helper function for parallel bootstrap penetrance computation."""
    df, embryo_ids, wt_envelope, time_bins, holdout_fraction, metric, iter_idx = args
    n_holdout = max(1, int(len(embryo_ids) * holdout_fraction))
    holdout_embryos = set(random.sample(list(embryo_ids), n_holdout))
    train_df = df[~df['embryo_id'].isin(holdout_embryos)].copy()

    # Mark penetrant and compute penetrance
    train_df = mark_penetrant_timepoints(train_df, wt_envelope, metric=metric)
    penetrance_iter = compute_penetrance_by_time(train_df, time_bins)
    return (iter_idx, penetrance_iter)


def bootstrap_penetrance_stability(
    df,
    wt_envelope,
    time_bins,
    genotype,
    n_iterations=N_BOOTSTRAP,
    holdout_fraction=BOOTSTRAP_HOLDOUT_FRACTION,
    metric=METRIC_NAME,
    n_jobs=-1
):
    """
    Bootstrap penetrance computation to assess stability.
    Uses parallel processing for speedup.

    For each iteration:
    1. Randomly hold out holdout_fraction of embryos
    2. Re-compute penetrance metrics on remaining embryos
    3. Track variation in penetrance across iterations

    Parameters
    ----------
    df : pd.DataFrame
        Data for this genotype
    wt_envelope : dict
        WT reference envelope
    time_bins : np.ndarray
        Array of time bin centers
    genotype : str
        Genotype name
    n_iterations : int
        Number of bootstrap iterations
    holdout_fraction : float
        Fraction of embryos to hold out per iteration
    metric : str
        Metric column name
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)

    Returns
    -------
    penetrance_mean : list
        Mean penetrance across bootstraps (one dict per time bin)
    penetrance_ci : list
        95% CI of penetrance (one dict per time bin)
    bootstrap_results : list
        Raw results from all iterations
    """
    embryo_ids = df['embryo_id'].unique()
    n_embryos = len(embryo_ids)
    n_holdout = max(1, int(n_embryos * holdout_fraction))

    genotype_short = get_genotype_short_name(genotype)
    print(f"  Bootstrapping {n_iterations} iterations (holding out {n_holdout}/{n_embryos} embryos)...")

    # Create task list for parallel processing
    tasks = [
        (df, embryo_ids, wt_envelope, time_bins, holdout_fraction, metric, iter_idx)
        for iter_idx in range(n_iterations)
    ]

    # Determine number of workers
    if n_jobs == -1:
        n_workers = max(1, os.cpu_count() - 1)
    else:
        n_workers = n_jobs

    print(f"  Using {n_workers} parallel workers for {len(tasks)} bootstrap iterations...")

    # Parallel execution
    with Pool(n_workers) as pool:
        results = pool.map(_bootstrap_penetrance_single_iteration, tasks)

    # Sort results by iteration index and extract penetrance results
    results.sort(key=lambda x: x[0])
    bootstrap_results = [pen_iter for _, pen_iter in results]

    print(f"  Completed all {n_iterations} bootstrap iterations")

    # Aggregate results
    time_bins_present = [r['time_bin'] for r in bootstrap_results[0]]

    penetrance_mean = []
    penetrance_ci = []

    for time_bin in time_bins_present:
        sample_pens = [
            next((r['sample_penetrance'] for r in iter_results if r['time_bin'] == time_bin), np.nan)
            for iter_results in bootstrap_results
        ]
        embryo_pens = [
            next((r['embryo_penetrance'] for r in iter_results if r['time_bin'] == time_bin), np.nan)
            for iter_results in bootstrap_results
        ]

        sample_pens = np.array([p for p in sample_pens if not np.isnan(p)])
        embryo_pens = np.array([p for p in embryo_pens if not np.isnan(p)])

        penetrance_mean.append({
            'time_bin': time_bin,
            'sample_penetrance_mean': np.mean(sample_pens),
            'embryo_penetrance_mean': np.mean(embryo_pens)
        })

        penetrance_ci.append({
            'time_bin': time_bin,
            'sample_penetrance_ci': (np.percentile(sample_pens, 2.5), np.percentile(sample_pens, 97.5)),
            'embryo_penetrance_ci': (np.percentile(embryo_pens, 2.5), np.percentile(embryo_pens, 97.5))
        })

    return penetrance_mean, penetrance_ci, bootstrap_results


def plot_wt_reference_envelope(
    wt_envelope,
    time_bins,
    metric=METRIC_NAME,
    output_path=None,
    figsize=(12, 6)
):
    """
    Plot WT reference envelope.

    Parameters
    ----------
    wt_envelope : dict
        WT reference envelope
    time_bins : np.ndarray
        Time bin centers
    metric : str
        Metric name
    output_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    times = []
    medians = []
    lows = []
    highs = []

    for time_bin in time_bins:
        if time_bin in wt_envelope:
            times.append(time_bin)
            medians.append(wt_envelope[time_bin]['median'])
            lows.append(wt_envelope[time_bin]['low'])
            highs.append(wt_envelope[time_bin]['high'])

    times = np.array(times)
    medians = np.array(medians)
    lows = np.array(lows)
    highs = np.array(highs)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot envelope
    ax.fill_between(times, lows, highs, alpha=0.3, color='blue', label='95% WT envelope (2.5-97.5%)')
    ax.plot(times, medians, 'b-', linewidth=2, label='WT median')
    ax.plot(times, lows, 'b--', alpha=0.5, linewidth=1)
    ax.plot(times, highs, 'b--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Developmental Time (hpf)', fontsize=12)
    ax.set_ylabel(f'{metric}', fontsize=12)
    ax.set_title('Wild-Type Reference Envelope', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")

    plt.close()


def plot_penetrance_curves(
    penetrance_results,
    bootstrap_mean,
    bootstrap_ci,
    genotype,
    output_path=None,
    figsize=(12, 6)
):
    """
    Plot penetrance vs time with confidence intervals.

    Parameters
    ----------
    penetrance_results : list
        Results from compute_penetrance_by_time()
    bootstrap_mean : list
        Mean penetrance from bootstrap
    bootstrap_ci : list
        CI of penetrance from bootstrap
    genotype : str
        Genotype name
    output_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    genotype_short = get_genotype_short_name(genotype)
    color = get_genotype_color(genotype)

    times = [r['time_bin'] for r in penetrance_results]
    sample_pens = [r['sample_penetrance'] * 100 for r in penetrance_results]  # Convert to %
    sample_ci_low = [r['sample_ci_low'] * 100 for r in penetrance_results]
    sample_ci_high = [r['sample_ci_high'] * 100 for r in penetrance_results]

    embryo_pens = [r['embryo_penetrance'] * 100 for r in penetrance_results]  # Convert to %
    embryo_ci_low = [r['embryo_ci_low'] * 100 for r in penetrance_results]
    embryo_ci_high = [r['embryo_ci_high'] * 100 for r in penetrance_results]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sample-level penetrance
    ax = axes[0]
    ax.fill_between(times, sample_ci_low, sample_ci_high, alpha=0.3, color=color, label='95% Wilson CI')
    ax.plot(times, sample_pens, 'o-', color=color, linewidth=2, markersize=6, label='Sample penetrance')
    ax.set_xlabel('Developmental Time (hpf)', fontsize=12)
    ax.set_ylabel('Penetrance (%)', fontsize=12)
    ax.set_title(f'{genotype_short}: Sample-Level Penetrance', fontsize=12, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Embryo-level penetrance
    ax = axes[1]
    ax.fill_between(times, embryo_ci_low, embryo_ci_high, alpha=0.3, color=color, label='95% Wilson CI')
    ax.plot(times, embryo_pens, 'o-', color=color, linewidth=2, markersize=6, label='Embryo penetrance')
    ax.set_xlabel('Developmental Time (hpf)', fontsize=12)
    ax.set_ylabel('Penetrance (%)', fontsize=12)
    ax.set_title(f'{genotype_short}: Embryo-Level Penetrance', fontsize=12, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{genotype_short} Penetrance Over Time', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")

    plt.close()


def plot_onset_distribution(
    onset_times_dict,
    genotype,
    output_path=None,
    figsize=(10, 6)
):
    """
    Plot histogram of onset times.

    Parameters
    ----------
    onset_times_dict : dict
        Dictionary mapping embryo_id -> onset_time
    genotype : str
        Genotype name
    output_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    genotype_short = get_genotype_short_name(genotype)
    color = get_genotype_color(genotype)

    # Extract valid onset times
    onset_times = np.array([t for t in onset_times_dict.values() if not np.isnan(t)])

    if len(onset_times) == 0:
        return

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(onset_times, bins=15, color=color, alpha=0.7, edgecolor='black')
    ax.axvline(np.median(onset_times), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(onset_times):.1f} hpf')

    ax.set_xlabel('First Penetrant Timepoint (hpf)', fontsize=12)
    ax.set_ylabel('Number of Embryos', fontsize=12)
    ax.set_title(f'{genotype_short}: Penetrance Onset Time Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    stats_text = f'n = {len(onset_times)}\nMean: {np.mean(onset_times):.1f} hpf\nMedian: {np.median(onset_times):.1f} hpf\nSD: {np.std(onset_times):.1f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")

    plt.close()


def plot_penetrance_comparison(
    penetrance_het,
    penetrance_homo,
    penetrance_wt=None,
    output_path=None,
    figsize=(12, 6)
):
    """
    Compare penetrance curves for Het vs Homo (and optionally WT).

    Parameters
    ----------
    penetrance_het : list
        Penetrance results for Het
    penetrance_homo : list
        Penetrance results for Homo
    penetrance_wt : list, optional
        Penetrance results for WT (for baseline comparison)
    output_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    times_het = [r['time_bin'] for r in penetrance_het]
    embryo_pens_het = [r['embryo_penetrance'] * 100 for r in penetrance_het]  # Convert to %
    embryo_ci_low_het = [r['embryo_ci_low'] * 100 for r in penetrance_het]
    embryo_ci_high_het = [r['embryo_ci_high'] * 100 for r in penetrance_het]

    times_homo = [r['time_bin'] for r in penetrance_homo]
    embryo_pens_homo = [r['embryo_penetrance'] * 100 for r in penetrance_homo]  # Convert to %
    embryo_ci_low_homo = [r['embryo_ci_low'] * 100 for r in penetrance_homo]
    embryo_ci_high_homo = [r['embryo_ci_high'] * 100 for r in penetrance_homo]

    fig, ax = plt.subplots(figsize=figsize)

    # WT baseline (if provided)
    if penetrance_wt is not None:
        times_wt = [r['time_bin'] for r in penetrance_wt]
        embryo_pens_wt = [r['embryo_penetrance'] * 100 for r in penetrance_wt]
        embryo_ci_low_wt = [r['embryo_ci_low'] * 100 for r in penetrance_wt]
        embryo_ci_high_wt = [r['embryo_ci_high'] * 100 for r in penetrance_wt]

        color_wt = get_genotype_color('cep290_wildtype')
        ax.fill_between(times_wt, embryo_ci_low_wt, embryo_ci_high_wt, alpha=0.2, color=color_wt)
        ax.plot(times_wt, embryo_pens_wt, '^-', color=color_wt, linewidth=2, markersize=6, label='WT (baseline)')

    # Het
    color_het = get_genotype_color('cep290_heterozygous')
    ax.fill_between(times_het, embryo_ci_low_het, embryo_ci_high_het, alpha=0.2, color=color_het)
    ax.plot(times_het, embryo_pens_het, 'o-', color=color_het, linewidth=2, markersize=6, label='Het')

    # Homo
    color_homo = get_genotype_color('cep290_homozygous')
    ax.fill_between(times_homo, embryo_ci_low_homo, embryo_ci_high_homo, alpha=0.2, color=color_homo)
    ax.plot(times_homo, embryo_pens_homo, 's-', color=color_homo, linewidth=2, markersize=6, label='Homo')

    ax.set_xlabel('Developmental Time (hpf)', fontsize=12)
    ax.set_ylabel('Embryo-Level Penetrance (%)', fontsize=12)
    title = 'Penetrance Comparison: Het vs Homo' if penetrance_wt is None else 'Penetrance Comparison: WT, Het, and Homo'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")

    plt.close()


def save_penetrance_table(penetrance_results, output_path):
    """Save penetrance results to CSV."""
    df = pd.DataFrame(penetrance_results)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path.name}")


def main():
    """Main analysis pipeline."""

    print("=" * 80)
    print("WT REFERENCE-BASED PENETRANCE ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df, metadata = get_analysis_dataframe()
    print(f"  Loaded {len(df)} timepoints from {df['embryo_id'].nunique()} embryos")
    print(f"  Genotypes: {df['genotype'].unique()}")

    # Bin data by time
    print(f"\nBinning data by {TIME_BIN_WIDTH} hpf windows...")
    df, time_bins = bin_data_by_time(df, bin_width=TIME_BIN_WIDTH)
    print(f"  Created {len(time_bins)} time bins")
    print(f"  Time range: {time_bins[0]:.1f} - {time_bins[-1]:.1f} hpf")

    # Compute WT reference envelope
    print(f"\n{'='*80}")
    print("COMPUTING WT REFERENCE ENVELOPE")
    print(f"{'='*80}\n")

    wt_df = df[df['genotype'] == WT_GENOTYPE].copy()
    n_wt_embryos = wt_df['embryo_id'].nunique()
    n_wt_timepoints = len(wt_df)

    print(f"  {n_wt_embryos} WT embryos, {n_wt_timepoints} timepoints")
    print(f"  Computing {WT_PERCENTILE_LOW}-{WT_PERCENTILE_HIGH}% envelope...")

    wt_envelope = compute_wt_reference_envelope(
        wt_df,
        time_bins,
        metric=METRIC_NAME,
        percentile_low=WT_PERCENTILE_LOW,
        percentile_high=WT_PERCENTILE_HIGH
    )

    print(f"  Envelope computed for {len(wt_envelope)} time bins")

    # Save WT reference
    wt_ref_df = pd.DataFrame([
        {
            'time_bin': t,
            'percentile_2.5': wt_envelope[t]['low'],
            'percentile_50_median': wt_envelope[t]['median'],
            'percentile_97.5': wt_envelope[t]['high'],
            'mean': wt_envelope[t]['mean'],
            'std': wt_envelope[t]['std'],
            'n_samples': wt_envelope[t]['n']
        }
        for t in time_bins if t in wt_envelope
    ])
    wt_ref_df.to_csv(TABLE_DIR / 'wt_reference_percentiles.csv', index=False)
    print(f"  Saved: wt_reference_percentiles.csv")

    # Plot WT envelope
    plot_wt_reference_envelope(
        wt_envelope,
        time_bins,
        metric=METRIC_NAME,
        output_path=FIGURE_DIR / 'wt_reference_envelope.png'
    )

    # Per-genotype analysis
    results_all = {}

    # Include WT in analysis for baseline comparison
    all_genotypes_to_analyze = [WT_GENOTYPE] + ANALYSIS_GENOTYPES

    for genotype in all_genotypes_to_analyze:
        print(f"\n{'='*80}")
        print(f"ANALYZING GENOTYPE: {genotype}")
        print(f"{'='*80}\n")

        genotype_df = df[df['genotype'] == genotype].copy()
        n_embryos = genotype_df['embryo_id'].nunique()
        n_timepoints = len(genotype_df)

        print(f"  {n_embryos} embryos, {n_timepoints} timepoints")

        # Mark penetrant timepoints
        print(f"\nMarking penetrant timepoints (outside WT envelope)...")
        genotype_df = mark_penetrant_timepoints(genotype_df, wt_envelope, metric=METRIC_NAME)

        # Compute penetrance
        print(f"Computing penetrance over time...")
        penetrance_results = compute_penetrance_by_time(genotype_df, time_bins)

        # Compute onset times
        print(f"Computing onset times...")
        onset_times = compute_onset_times(genotype_df)
        n_with_onset = sum(1 for t in onset_times.values() if not np.isnan(t))
        print(f"  {n_with_onset}/{n_embryos} embryos show penetrant phenotype")

        # Bootstrap stability
        print(f"\nAssessing penetrance stability via bootstrap...")
        pens_mean, pens_ci, pens_bootstrap = bootstrap_penetrance_stability(
            genotype_df, wt_envelope, time_bins, genotype
        )

        # Store results
        results_all[genotype] = {
            'penetrance': penetrance_results,
            'onset_times': onset_times,
            'bootstrap_mean': pens_mean,
            'bootstrap_ci': pens_ci
        }

        # Save results
        genotype_short = get_genotype_short_name(genotype)

        print(f"\nSaving results for {genotype_short}...")
        save_penetrance_table(
            penetrance_results,
            TABLE_DIR / f'penetrance_by_time_{genotype_short}.csv'
        )

        # Save onset times
        onset_df = pd.DataFrame([
            {'embryo_id': e, 'onset_time_hpf': t}
            for e, t in onset_times.items()
        ])
        onset_df.to_csv(TABLE_DIR / f'embryo_onset_times_{genotype_short}.csv', index=False)
        print(f"  Saved: embryo_onset_times_{genotype_short}.csv")

        # Generate plots
        print(f"\nGenerating figures for {genotype_short}...")

        plot_penetrance_curves(
            penetrance_results,
            pens_mean,
            pens_ci,
            genotype,
            output_path=FIGURE_DIR / f'penetrance_vs_time_{genotype_short}.png'
        )

        plot_onset_distribution(
            onset_times,
            genotype,
            output_path=FIGURE_DIR / f'penetrance_onset_distribution_{genotype_short}.png'
        )

    # Cross-genotype comparison
    print(f"\n{'='*80}")
    print("CROSS-GENOTYPE COMPARISON")
    print(f"{'='*80}\n")

    # Add WT penetrance to comparison if available
    penetrance_wt = None
    if 'cep290_wildtype' in results_all:
        penetrance_wt = results_all['cep290_wildtype']['penetrance']

    plot_penetrance_comparison(
        results_all['cep290_heterozygous']['penetrance'],
        results_all['cep290_homozygous']['penetrance'],
        penetrance_wt=penetrance_wt,
        output_path=FIGURE_DIR / 'penetrance_comparison_WT_Het_Homo.png'
    )

    # Onset comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    onset_het = np.array([t for t in results_all['cep290_heterozygous']['onset_times'].values() if not np.isnan(t)])
    onset_homo = np.array([t for t in results_all['cep290_homozygous']['onset_times'].values() if not np.isnan(t)])

    color_het = get_genotype_color('cep290_heterozygous')
    color_homo = get_genotype_color('cep290_homozygous')

    ax.hist(onset_het, bins=15, alpha=0.6, label=f'Het (n={len(onset_het)})', color=color_het, edgecolor='black')
    ax.hist(onset_homo, bins=15, alpha=0.6, label=f'Homo (n={len(onset_homo)})', color=color_homo, edgecolor='black')

    ax.set_xlabel('First Penetrant Timepoint (hpf)', fontsize=12)
    ax.set_ylabel('Number of Embryos', fontsize=12)
    ax.set_title('Penetrance Onset Time Distribution: Het vs Homo', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'penetrance_onset_comparison_Het_vs_Homo.png', dpi=300, bbox_inches='tight')
    print("  Saved: penetrance_onset_comparison_Het_vs_Homo.png")
    plt.close()

    # Summary statistics
    print(f"\nGenerating summary statistics...")

    summary_stats = []

    for genotype in all_genotypes_to_analyze:
        penetrance_results = results_all[genotype]['penetrance']
        onset_times = results_all[genotype]['onset_times']

        embryo_pens = [r['embryo_penetrance'] for r in penetrance_results]
        onset_times_valid = [t for t in onset_times.values() if not np.isnan(t)]

        summary_stats.append({
            'genotype': get_genotype_short_name(genotype),
            'n_embryos_with_onset': len(onset_times_valid),
            'n_total_embryos': len(onset_times),
            'mean_penetrance': np.mean(embryo_pens),
            'max_penetrance': np.max(embryo_pens),
            'mean_onset_hpf': np.mean(onset_times_valid) if len(onset_times_valid) > 0 else np.nan,
            'median_onset_hpf': np.median(onset_times_valid) if len(onset_times_valid) > 0 else np.nan,
            'earliest_onset_hpf': np.min(onset_times_valid) if len(onset_times_valid) > 0 else np.nan,
            'latest_onset_hpf': np.max(onset_times_valid) if len(onset_times_valid) > 0 else np.nan
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(TABLE_DIR / 'summary_statistics.csv', index=False)
    print(f"  Saved: summary_statistics.csv\n")
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables: {TABLE_DIR}")


if __name__ == '__main__':
    main()
