"""
Reusable utilities for comprehensive group comparison figures.

This module provides functions for creating 3-panel comparison figures that combine:
- Panel A: Classification performance (AUROC) over time with null distribution context
- Panel B: Morphological divergence showing when groups actually differ
- Panel C: Individual trajectories with group means

The module emphasizes:
- No extrapolation: Interpolation only within each embryo's observed time range
- Null distribution context: Shows expected variation under the null hypothesis
- Gaussian smoothing: Reduces noise in divergence trends (default sigma=1.5)

Typical usage:
    >>> from comparison_plotting_utils import create_full_comparison
    >>> fig, divergence_df = create_full_comparison(
    ...     df=raw_data,
    ...     df_results=classification_results,
    ...     group1_ids=['emb1', 'emb2'],
    ...     group2_ids=['emb3', 'emb4'],
    ...     group1_label='Mutant',
    ...     group2_label='Wildtype',
    ...     metric_col='baseline_deviation_normalized',
    ...     metric_label='Baseline Deviation (normalized)',
    ...     save_path=Path('output/mutant_vs_wt_comprehensive.png'),
    ...     time_col='predicted_stage_hpf',
    ...     embryo_id_col='embryo_id'
    ... )
    >>> plt.close(fig)  # Prevent memory buildup
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def compute_morphological_divergence(
    df: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    metric_col: str,
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id'
) -> pd.DataFrame:
    """
    Compute mean morphological metric difference between groups over time.

    This function interpolates individual embryo trajectories to a common temporal grid
    (0.5 hpf resolution) to enable point-wise comparison. Critically, interpolation is
    performed ONLY within each embryo's observed time range (no extrapolation) to
    preserve data integrity.

    The interpolation strategy:
    1. Identify global time range across all embryos
    2. Create uniform grid at 0.5 hpf steps
    3. For each embryo, interpolate to grid points within its observed range
    4. Compute per-timepoint statistics (mean, SEM) for each group
    5. Calculate absolute difference between group means

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe containing trajectory data for all embryos
    group1_ids : List[str]
        Embryo IDs for group 1
    group2_ids : List[str]
        Embryo IDs for group 2
    metric_col : str
        Column name for morphological metric to analyze
    time_col : str, optional
        Column name for developmental time (default: 'predicted_stage_hpf')
    embryo_id_col : str, optional
        Column name for embryo identifiers (default: 'embryo_id')

    Returns
    -------
    divergence_df : pd.DataFrame
        DataFrame with columns:
        - hpf : float - timepoint on common grid
        - group1_mean : float - mean metric value for group 1
        - group1_sem : float - standard error of mean for group 1
        - group2_mean : float - mean metric value for group 2
        - group2_sem : float - standard error of mean for group 2
        - abs_difference : float - |group2_mean - group1_mean|
        - n_group1 : int - number of embryos contributing to group 1 at this timepoint
        - n_group2 : int - number of embryos contributing to group 2 at this timepoint

    Raises
    ------
    ValueError
        If fewer than 2 embryos per group, or if groups have no temporal overlap

    Notes
    -----
    The 0.5 hpf grid step provides sufficient temporal resolution for zebrafish
    developmental dynamics while maintaining statistical power. Finer grids may
    introduce excessive interpolation artifacts.

    The no-extrapolation policy ensures we only compare groups during developmental
    windows where both have actual observations, avoiding spurious differences from
    extrapolated values.
    """
    # Filter to relevant embryos
    df_filtered = df[df[embryo_id_col].isin(group1_ids + group2_ids)].copy()

    if len(df_filtered) == 0:
        raise ValueError("No data found for the specified embryo IDs")

    # Add group labels
    df_filtered['group'] = df_filtered[embryo_id_col].apply(
        lambda x: 'group1' if x in group1_ids else 'group2'
    )

    # Check group sizes
    n_group1_embryos = len([eid for eid in group1_ids if eid in df_filtered[embryo_id_col].values])
    n_group2_embryos = len([eid for eid in group2_ids if eid in df_filtered[embryo_id_col].values])

    if n_group1_embryos < 2:
        raise ValueError(f"Group 1 has only {n_group1_embryos} embryo(s) with data. Need at least 2.")
    if n_group2_embryos < 2:
        raise ValueError(f"Group 2 has only {n_group2_embryos} embryo(s) with data. Need at least 2.")

    # Drop missing values
    df_filtered = df_filtered.dropna(subset=[time_col, metric_col])

    if len(df_filtered) == 0:
        raise ValueError(f"No valid data after dropping NaN values in {time_col} or {metric_col}")

    # Interpolate trajectories to common grid (NO extrapolation)
    grid_step = 0.5
    time_min = np.floor(df_filtered[time_col].min() / grid_step) * grid_step
    time_max = np.ceil(df_filtered[time_col].max() / grid_step) * grid_step
    common_grid = np.arange(time_min, time_max + grid_step, grid_step)

    # Interpolate each embryo
    interpolated_records = []
    for embryo_id in df_filtered[embryo_id_col].unique():
        embryo_data = df_filtered[df_filtered[embryo_id_col] == embryo_id].sort_values(time_col)

        if len(embryo_data) < 2:
            continue  # Skip embryos with insufficient data

        group = embryo_data['group'].iloc[0]

        # Get this embryo's actual time range (no extrapolation!)
        embryo_time_min = embryo_data[time_col].min()
        embryo_time_max = embryo_data[time_col].max()

        # Interpolate only within embryo's time range
        interp_values = np.interp(
            common_grid,
            embryo_data[time_col].values,
            embryo_data[metric_col].values
        )

        for t, v in zip(common_grid, interp_values):
            # Only keep values within embryo's actual time range
            if embryo_time_min <= t <= embryo_time_max:
                interpolated_records.append({
                    'embryo_id': embryo_id,
                    'hpf': t,
                    'metric_value': v,
                    'group': group
                })

    if len(interpolated_records) == 0:
        raise ValueError("No interpolated data generated. Check input data quality.")

    df_interp = pd.DataFrame(interpolated_records)

    # Compute stats per timepoint
    divergence_records = []
    for hpf in sorted(df_interp['hpf'].unique()):
        df_t = df_interp[df_interp['hpf'] == hpf]

        group1_values = df_t[df_t['group'] == 'group1']['metric_value'].values
        group2_values = df_t[df_t['group'] == 'group2']['metric_value'].values

        if len(group1_values) > 0 and len(group2_values) > 0:
            group1_mean = np.mean(group1_values)
            group1_sem = stats.sem(group1_values) if len(group1_values) > 1 else 0

            group2_mean = np.mean(group2_values)
            group2_sem = stats.sem(group2_values) if len(group2_values) > 1 else 0

            abs_diff = abs(group2_mean - group1_mean)

            divergence_records.append({
                'hpf': hpf,
                'group1_mean': group1_mean,
                'group1_sem': group1_sem,
                'group2_mean': group2_mean,
                'group2_sem': group2_sem,
                'abs_difference': abs_diff,
                'n_group1': len(group1_values),
                'n_group2': len(group2_values)
            })

    if len(divergence_records) == 0:
        raise ValueError("No temporal overlap between groups. Cannot compute divergence.")

    return pd.DataFrame(divergence_records)


def create_comprehensive_figure(
    df_results: pd.DataFrame,
    divergence_df: pd.DataFrame,
    df_raw: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    group1_label: str,
    group2_label: str,
    metric_col: str,
    metric_label: str,
    save_path: Path,
    colors: Optional[Dict[str, str]] = None,
    smooth_sigma: float = 1.5,
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id'
) -> plt.Figure:
    """
    Create 3-panel comprehensive comparison figure.

    This function generates a publication-quality figure showing:
    - Panel A: Classification performance (AUROC) over time
      * Shows when groups can be distinguished based on VAE embeddings
      * Null distribution band provides context for expected random variation
      * Green circles mark significant timepoints (p < 0.05)
      * Error bars show uncertainty from null distribution standard deviation
    
    - Panel B: Morphological divergence over time
      * Shows when groups actually differ in morphological space
      * Gaussian smoothing (sigma=1.5) reduces noise while preserving trends
      * Gray bands show combined standard error from both groups
      * Reveals the timing of phenotypic divergence
    
    - Panel C: Individual trajectories with group means
      * Shows biological data underlying the predictions
      * Faint lines = individual embryos (smoothed with sigma=1.5)
      * Bold lines = group means for interpretability
      * Enables assessment of within-group heterogeneity

    The smoothing rationale: Gaussian filtering with sigma=1.5 removes high-frequency
    noise from measurement error and short-term fluctuations while preserving
    biologically meaningful developmental trends. This value balances noise reduction
    with temporal resolution preservation.

    Parameters
    ----------
    df_results : pd.DataFrame
        Classification results with columns: time_bin, auroc_observed, auroc_null_mean,
        auroc_null_std, pval, n_samples
    divergence_df : pd.DataFrame
        Morphological divergence over time (from compute_morphological_divergence)
    df_raw : pd.DataFrame
        Raw trajectory data for plotting individual embryos
    group1_ids : List[str]
        Embryo IDs for group 1
    group2_ids : List[str]
        Embryo IDs for group 2
    group1_label : str
        Display label for group 1 (used in titles and legends)
    group2_label : str
        Display label for group 2 (used in titles and legends)
    metric_col : str
        Column name for morphological metric
    metric_label : str
        Display label for metric axis
    save_path : Path
        Path to save PNG file (300 DPI)
    colors : Dict[str, str], optional
        Color mapping {group_label: hex_color}. If None, uses default colors.
    smooth_sigma : float, optional
        Gaussian smoothing parameter (default: 1.5)
    time_col : str, optional
        Column name for developmental time (default: 'predicted_stage_hpf')
    embryo_id_col : str, optional
        Column name for embryo identifiers (default: 'embryo_id')

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure (NOT closed, allowing caller to display or modify)

    Notes
    -----
    The figure is saved at 300 DPI for publication quality. The returned figure
    should be explicitly closed by the caller to prevent memory buildup when
    processing multiple comparisons.

    The null distribution interpretation: The gray band in Panel A shows the mean ± 1 SD
    of AUROC values from permutation tests where group labels are shuffled. This
    represents expected performance under the null hypothesis of no group difference.
    Observed AUROC values outside this band suggest genuine distinguishability.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Filter for valid results
    df_res = df_results[df_results['auroc_observed'].notna()].copy()

    if len(df_res) == 0:
        print(f"  Warning: No valid AUROC results for {group1_label} vs {group2_label}")
        plt.close(fig)
        return fig

    # Get time range
    time_min = min(df_res['time_bin'].min(), divergence_df['hpf'].min())
    time_max = max(df_res['time_bin'].max(), divergence_df['hpf'].max())

    # Colors
    default_colors = {
        group1_label: '#d62728',  # Red
        group2_label: '#1f77b4',  # Blue
    }
    if colors is None:
        colors = default_colors
    
    color1 = colors.get(group1_label, default_colors[group1_label])
    color2 = colors.get(group2_label, default_colors[group2_label])

    # =========================================================================
    # Panel A: AUROC vs Time (with null distribution band)
    # =========================================================================
    ax1 = axes[0]

    ALPHA = 0.05

    # Split data by significance
    df_sig = df_res[df_res['pval'] < ALPHA]
    df_nonsig = df_res[df_res['pval'] >= ALPHA]

    # Plot null distribution band (mean ± 1 SD)
    if 'auroc_null_mean' in df_res.columns and 'auroc_null_std' in df_res.columns:
        ax1.fill_between(
            df_res['time_bin'],
            df_res['auroc_null_mean'] - df_res['auroc_null_std'],
            df_res['auroc_null_mean'] + df_res['auroc_null_std'],
            color='gray',
            alpha=0.2,
            label='Null mean ± 1 SD',
            linewidth=0
        )

    # Plot connecting line (all points)
    ax1.plot(
        df_res['time_bin'],
        df_res['auroc_observed'],
        linewidth=2,
        color='black',
        alpha=0.7,
        zorder=2
    )

    # Plot significant points (filled green circles)
    if len(df_sig) > 0:
        ax1.scatter(
            df_sig['time_bin'],
            df_sig['auroc_observed'],
            s=100,
            c='#2ca02c',  # Green
            marker='o',
            edgecolors='darkgreen',
            linewidths=1.5,
            label=f'Significant (p < {ALPHA})',
            zorder=3
        )

    # Plot non-significant points (open gray circles)
    if len(df_nonsig) > 0:
        ax1.scatter(
            df_nonsig['time_bin'],
            df_nonsig['auroc_observed'],
            s=100,
            c='white',
            marker='o',
            edgecolors='gray',
            linewidths=1.5,
            label=f'Not significant (p ≥ {ALPHA})',
            zorder=2
        )

    # Mark highly significant points with stars
    very_sig_mask = df_res['pval'] < 0.01
    if very_sig_mask.any():
        for _, row in df_res[very_sig_mask].iterrows():
            ax1.annotate(
                '*',
                (row['time_bin'], row['auroc_observed'] + 0.03),
                ha='center',
                fontsize=14,
                fontweight='bold',
                color='darkgreen'
            )

    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=2, label='Chance (AUROC=0.5)')

    ax1.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('AUROC', fontsize=14, fontweight='bold')
    ax1.set_title(
        f'(A) Group Distinguishability: {group1_label} vs {group2_label}\n'
        f'Can we predict group membership from VAE embeddings?',
        fontsize=14, fontweight='bold', loc='left'
    )
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(time_min, time_max)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # =========================================================================
    # Panel B: Morphological Divergence
    # =========================================================================
    ax2 = axes[1]

    # Apply Gaussian smoothing to divergence line
    abs_diff_smoothed = gaussian_filter1d(
        divergence_df['abs_difference'].values,
        sigma=smooth_sigma
    )

    ax2.plot(
        divergence_df['hpf'],
        abs_diff_smoothed,
        linewidth=3,
        color='black',
        label=f'|Mean difference| (smoothed σ={smooth_sigma})',
        zorder=100
    )

    # Error bands (using smoothed values as center)
    combined_sem = np.sqrt(divergence_df['group1_sem']**2 + divergence_df['group2_sem']**2)
    ax2.fill_between(
        divergence_df['hpf'],
        abs_diff_smoothed - combined_sem,
        abs_diff_smoothed + combined_sem,
        alpha=0.3,
        color='gray',
        label='± SEM'
    )

    ax2.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f'Absolute Difference ({metric_label})', fontsize=14, fontweight='bold')
    ax2.set_title(
        f'(B) Morphological Divergence Over Time\n'
        f'When do {group1_label} and {group2_label} groups actually diverge?',
        fontsize=14, fontweight='bold', loc='left'
    )
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(time_min, time_max)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # =========================================================================
    # Panel C: Individual Trajectories
    # =========================================================================
    ax3 = axes[2]

    # Filter raw data
    df_plot = df_raw[df_raw[embryo_id_col].isin(group1_ids + group2_ids)].copy()
    df_plot = df_plot.dropna(subset=[time_col, metric_col])

    # Plot individual trajectories with smoothing
    for embryo_id in group1_ids:
        embryo_data = df_plot[df_plot[embryo_id_col] == embryo_id].sort_values(time_col)
        if len(embryo_data) > 2:  # Need at least 3 points for smoothing
            # Apply Gaussian smoothing to individual embryo trajectory
            metric_values_smoothed = gaussian_filter1d(
                embryo_data[metric_col].values,
                sigma=smooth_sigma
            )
            ax3.plot(
                embryo_data[time_col],
                metric_values_smoothed,
                alpha=0.25,
                linewidth=0.8,
                color=color1
            )

    for embryo_id in group2_ids:
        embryo_data = df_plot[df_plot[embryo_id_col] == embryo_id].sort_values(time_col)
        if len(embryo_data) > 2:  # Need at least 3 points for smoothing
            # Apply Gaussian smoothing to individual embryo trajectory
            metric_values_smoothed = gaussian_filter1d(
                embryo_data[metric_col].values,
                sigma=smooth_sigma
            )
            ax3.plot(
                embryo_data[time_col],
                metric_values_smoothed,
                alpha=0.25,
                linewidth=0.8,
                color=color2
            )

    # Smooth group means
    group1_mean_smoothed = gaussian_filter1d(
        divergence_df['group1_mean'].values,
        sigma=smooth_sigma
    )
    group2_mean_smoothed = gaussian_filter1d(
        divergence_df['group2_mean'].values,
        sigma=smooth_sigma
    )

    # Plot group means
    ax3.plot(
        divergence_df['hpf'],
        group1_mean_smoothed,
        linewidth=4,
        color=color1,
        label=f'{group1_label} mean (n={len(group1_ids)})',
        zorder=100
    )
    ax3.plot(
        divergence_df['hpf'],
        group2_mean_smoothed,
        linewidth=4,
        color=color2,
        label=f'{group2_label} mean (n={len(group2_ids)})',
        zorder=100
    )

    ax3.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax3.set_ylabel(metric_label, fontsize=14, fontweight='bold')
    ax3.set_title(
        f'(C) Individual Trajectories and Group Means\n'
        f'Morphological data underlying the classification and divergence',
        fontsize=14, fontweight='bold', loc='left'
    )
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(time_min, time_max)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Set x-axis ticks to every 5 hpf for all panels
    tick_start = np.ceil(time_min / 5) * 5  # Round up to nearest 5
    tick_end = np.floor(time_max / 5) * 5 + 5  # Include endpoint
    tick_positions = np.arange(tick_start, tick_end, 5)
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(tick_positions)

    # Overall title
    fig.suptitle(
        f'Comprehensive Comparison: {group1_label} vs {group2_label}',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comprehensive figure: {save_path}")

    return fig


def create_full_comparison(
    df: pd.DataFrame,
    df_results: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    group1_label: str,
    group2_label: str,
    metric_col: str,
    metric_label: str,
    save_path: Path,
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    colors: Optional[Dict[str, str]] = None,
    smooth_sigma: float = 1.5
) -> Tuple[Optional[plt.Figure], Optional[pd.DataFrame]]:
    """
    Convenience function for full comparison workflow.

    This wrapper function performs the complete analysis pipeline:
    1. Compute morphological divergence between groups
    2. Create 3-panel comprehensive figure
    3. Auto-save divergence data to CSV
    4. Auto-save figure to PNG (300 DPI)
    5. Return both for optional further processing

    CSV output columns (in order):
    - hpf : Developmental time on common grid
    - group1_mean : Mean metric for group 1
    - group1_sem : Standard error for group 1
    - group2_mean : Mean metric for group 2
    - group2_sem : Standard error for group 2
    - abs_difference : Absolute difference between groups
    - n_group1 : Sample size for group 1
    - n_group2 : Sample size for group 2

    Parameters
    ----------
    df : pd.DataFrame
        Raw trajectory data
    df_results : pd.DataFrame
        Classification results from permutation testing
    group1_ids : List[str]
        Embryo IDs for group 1
    group2_ids : List[str]
        Embryo IDs for group 2
    group1_label : str
        Display label for group 1
    group2_label : str
        Display label for group 2
    metric_col : str
        Column name for morphological metric
    metric_label : str
        Display label for metric axis
    save_path : Path
        Path to save PNG file (divergence CSV will be saved alongside)
    time_col : str, optional
        Column name for developmental time (default: 'predicted_stage_hpf')
    embryo_id_col : str, optional
        Column name for embryo identifiers (default: 'embryo_id')
    colors : Dict[str, str], optional
        Color mapping {group_label: hex_color}
    smooth_sigma : float, optional
        Gaussian smoothing parameter (default: 1.5)

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The created figure (caller should close it), or None if analysis failed
    divergence_df : pd.DataFrame or None
        Divergence data, or None if analysis failed

    Notes
    -----
    This function handles errors gracefully by catching exceptions from divergence
    computation (e.g., insufficient data, no temporal overlap) and returning
    (None, None) rather than crashing. This allows batch processing to continue
    even if some comparisons fail.

    Memory management: The caller is responsible for closing the returned figure
    to prevent memory buildup:
        >>> fig, div_df = create_full_comparison(...)
        >>> plt.close(fig)
    """
    try:
        # Compute morphological divergence
        divergence_df = compute_morphological_divergence(
            df=df,
            group1_ids=group1_ids,
            group2_ids=group2_ids,
            metric_col=metric_col,
            time_col=time_col,
            embryo_id_col=embryo_id_col
        )

        # Save divergence CSV
        csv_path = save_path.parent / f"{save_path.stem}_divergence.csv"
        divergence_df.to_csv(csv_path, index=False)
        print(f"  Saved divergence data: {csv_path}")

        # Create comprehensive figure
        fig = create_comprehensive_figure(
            df_results=df_results,
            divergence_df=divergence_df,
            df_raw=df,
            group1_ids=group1_ids,
            group2_ids=group2_ids,
            group1_label=group1_label,
            group2_label=group2_label,
            metric_col=metric_col,
            metric_label=metric_label,
            save_path=save_path,
            colors=colors,
            smooth_sigma=smooth_sigma,
            time_col=time_col,
            embryo_id_col=embryo_id_col
        )

        return fig, divergence_df

    except ValueError as e:
        print(f"  Skipping {group1_label} vs {group2_label}: {e}")
        return None, None
    except Exception as e:
        print(f"  Error processing {group1_label} vs {group2_label}: {e}")
        return None, None


# =============================================================================
# AUROC Overlay Plotting Functions
# =============================================================================

def plot_auroc_overlay(
    results_dict: Dict[str, Dict],
    colors: Dict[str, str],
    title: str = 'Classification Performance Over Time',
    baseline_key: Optional[str] = None,
    time_bin_width: float = 2.0,
    figsize: Tuple[float, float] = (14, 7)
) -> plt.Figure:
    """
    Plot AUROC overlay for multiple comparisons with significance annotations.
    
    Creates an overlay plot showing AUROC trajectories for multiple group comparisons,
    with null distribution bands and significance markers.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping comparison labels to results from compare_groups()
        Each value should have 'classification' key with DataFrame
    colors : Dict[str, str]
        Mapping of comparison labels to hex colors
    title : str, optional
        Plot title
    baseline_key : str, optional
        Key for baseline comparison to plot as dashed line (e.g., 'Het_vs_WT')
    time_bin_width : float, optional
        Width of time bins in hours (for display only)
    figsize : Tuple[float, float], optional
        Figure size (width, height)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    
    Notes
    -----
    - Null distribution bands show mean ± 1 SD (10% alpha)
    - Large circles mark p < 0.05
    - Stars mark p < 0.01
    - Baseline (if specified) shown as dashed line
    
    Example
    -------
    >>> colors = {'Homo_vs_WT': '#D32F2F', 'Het_vs_WT': '#888888'}
    >>> fig = plot_auroc_overlay(
    ...     results_dict={'Homo_vs_WT': results1, 'Het_vs_WT': results2},
    ...     colors=colors,
    ...     title='Pooled Classification',
    ...     baseline_key='Het_vs_WT'
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot baseline first if specified
    if baseline_key and baseline_key in results_dict:
        df_baseline = results_dict[baseline_key]['classification']
        color = colors.get(baseline_key, '#888888')
        
        ax.plot(df_baseline['time_bin'], df_baseline['auroc_observed'],
                '--', color=color, linewidth=2, alpha=0.7, 
                label=f'{baseline_key} (baseline)')
        
        if 'auroc_null_mean' in df_baseline.columns and 'auroc_null_std' in df_baseline.columns:
            ax.fill_between(
                df_baseline['time_bin'],
                df_baseline['auroc_null_mean'] - df_baseline['auroc_null_std'],
                df_baseline['auroc_null_mean'] + df_baseline['auroc_null_std'],
                color=color,
                alpha=0.10,
                linewidth=0
            )
    
    # Plot each comparison
    for label, results in results_dict.items():
        if label == baseline_key:
            continue  # Already plotted
        
        df_class = results['classification']
        if df_class.empty:
            continue
        
        color = colors.get(label, '#666666')
        
        # Plot AUROC line
        ax.plot(df_class['time_bin'], df_class['auroc_observed'],
                'o-', color=color, linewidth=2, markersize=5, label=label)
        
        # Null distribution band
        if 'auroc_null_mean' in df_class.columns and 'auroc_null_std' in df_class.columns:
            ax.fill_between(
                df_class['time_bin'],
                df_class['auroc_null_mean'] - df_class['auroc_null_std'],
                df_class['auroc_null_mean'] + df_class['auroc_null_std'],
                color=color,
                alpha=0.10,
                linewidth=0
            )
        
        # Significance markers (p < 0.05)
        sig_mask = df_class['pval'] < 0.05
        if sig_mask.any():
            ax.scatter(df_class.loc[sig_mask, 'time_bin'],
                      df_class.loc[sig_mask, 'auroc_observed'],
                      s=200, facecolors='none', edgecolors=color, 
                      linewidths=2.5, zorder=5)
        
        # Stars for highly significant (p < 0.01)
        very_sig_mask = df_class['pval'] < 0.01
        if very_sig_mask.any():
            for _, row in df_class[very_sig_mask].iterrows():
                ax.annotate('*', (row['time_bin'], row['auroc_observed'] + 0.03),
                           ha='center', fontsize=14, fontweight='bold', color=color)
    
    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')
    
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(f'{title}\n(shaded = null mean ± 1 SD, circles = p<0.05, * = p<0.01)',
                fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_temporal_emergence(
    results_dict: Dict[str, Dict],
    colors: Dict[str, str],
    time_bin_width: float = 2.0,
    title_prefix: str = "",
    figsize_per_panel: float = 5.0
) -> plt.Figure:
    """
    Plot temporal emergence of phenotypic differences with significance highlighting.
    
    Creates bar plots showing when differences emerge, with significance markers
    and vertical lines indicating first significant timepoint.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping comparison labels to results from compare_groups()
    colors : Dict[str, str]
        Mapping of comparison labels to hex colors
    time_bin_width : float, optional
        Width of time bins in hours
    title_prefix : str, optional
        Prefix for figure title (e.g., "Per-Cluster: ")
    figsize_per_panel : float, optional
        Width of each subplot panel
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure with multiple subplots
    
    Notes
    -----
    - Bars show AUROC at each time bin
    - Significant bins (p < 0.05) have dark edges and full opacity
    - Highly significant bins (p < 0.01) marked with stars
    - Green vertical line shows earliest significant timepoint
    
    Example
    -------
    >>> colors = {'bumpy_vs_WT': '#9467BD', 'high_to_low_vs_WT': '#E377C2'}
    >>> fig = plot_temporal_emergence(
    ...     results_dict={'bumpy_vs_WT': results1, 'high_to_low_vs_WT': results2},
    ...     colors=colors,
    ...     title_prefix='Per-Cluster: '
    ... )
    """
    n_comparisons = len(results_dict)
    fig, axes = plt.subplots(1, n_comparisons, 
                            figsize=(figsize_per_panel * n_comparisons, 5), 
                            sharey=True)
    
    if n_comparisons == 1:
        axes = [axes]
    
    for ax, (label, results) in zip(axes, results_dict.items()):
        df_class = results['classification']
        summary = results['summary']
        color = colors.get(label, '#666666')
        
        if df_class.empty:
            ax.set_title(f"{label}\n(No data)")
            continue
        
        # Bar plot of AUROC per time bin
        bars = ax.bar(df_class['time_bin'], df_class['auroc_observed'],
                     width=time_bin_width * 0.8, color=color, alpha=0.6)
        
        # Highlight significant bins
        for i, (idx, row) in enumerate(df_class.iterrows()):
            if row['pval'] < 0.05:
                bars[i].set_alpha(1.0)
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
            if row['pval'] < 0.01:
                ax.annotate('*', (row['time_bin'], row['auroc_observed'] + 0.02),
                           ha='center', fontsize=12, fontweight='bold')
        
        # Reference line
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Mark earliest significant
        if summary['earliest_significant_hpf'] is not None:
            ax.axvline(x=summary['earliest_significant_hpf'],
                      color='green', linestyle='-', alpha=0.7, linewidth=2)
            ax.annotate(f"First sig:\n{summary['earliest_significant_hpf']}h",
                       xy=(summary['earliest_significant_hpf'], 0.95),
                       ha='center', fontsize=9, color='green')
        
        ax.set_xlabel('Hours Post Fertilization')
        max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
        ax.set_title(f"{label}\nMax AUROC: {max_auroc:.2f}")
        ax.set_ylim(0.3, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].set_ylabel('AUROC')
    fig.suptitle(f'{title_prefix}Temporal Emergence of Phenotypic Differences', 
                fontsize=14, y=1.02)
    
    plt.tight_layout()
    return fig
