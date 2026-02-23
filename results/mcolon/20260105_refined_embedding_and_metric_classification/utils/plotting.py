"""Plotting utilities for cryptic phenotype comparison figures."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from scipy.ndimage import gaussian_filter1d


def create_comparison_figure(
    auroc_df: pd.DataFrame,
    divergence_df: pd.DataFrame,
    df_trajectories: pd.DataFrame,
    group1_label: str,
    group2_label: str,
    metric_cols: List[str],
    embedding_auroc_df: Optional[pd.DataFrame] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    time_landmarks: Optional[Dict[float, str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
    time_col_auroc: str = 'time_bin',
    time_col_traj: str = 'predicted_stage_hpf',
    auroc_color: str = '#2ca02c',
    baseline_auroc_df: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    Create 3-panel comparison figure for cryptic phenotype analysis.

    Panel A: AUROC over time (metric-based, optionally with embedding overlay)
    Panel B: Metric divergence (Z-scored for multi-metric comparison)
    Panel C: Raw trajectories with group means

    IMPORTANT: group1_label is the POSITIVE class (phenotype), group2_label is NEGATIVE (reference).
    This matches compare_groups() convention where group1='phenotype', group2='reference'.

    Parameters
    ----------
    auroc_df : pd.DataFrame
        Metric-based AUROC from compare_groups()['classification']
        Must have: time_bin, auroc_observed, auroc_null_std, pval
    divergence_df : pd.DataFrame
        From compute_multi_metric_divergence() with zscore
        Must have: hpf, metric, abs_difference_zscore
    df_trajectories : pd.DataFrame
        Raw trajectory data for Panel C (with 'group' column)
    group1_label : str
        Label for positive (phenotype) group
    group2_label : str
        Label for negative (reference) group
    metric_cols : List[str]
        Metrics to plot in Panel B and C
    embedding_auroc_df : Optional[pd.DataFrame]
        Embedding-based AUROC to overlay on Panel A
    metric_labels : Optional[Dict[str, str]]
        Display labels for metrics (e.g., {'baseline_deviation_normalized': 'Curvature'})
    time_landmarks : Optional[Dict[float, str]]
        Vertical lines to mark (e.g., {24.0: '24 hpf'})
    title : Optional[str]
        Overall figure title
    save_path : Optional[Path]
        Path to save figure (creates parent dirs if needed)
    figsize : tuple
        Figure size (width, height) in inches
    time_col_auroc : str
        Time column in AUROC DataFrames (default: 'time_bin')
    time_col_traj : str
        Time column in trajectory DataFrame (default: 'predicted_stage_hpf')
    auroc_color : str
        Color for main AUROC line and null bands (default: '#2ca02c')
    baseline_auroc_df : Optional[pd.DataFrame]
        Optional baseline comparison AUROC (e.g., Het vs WT) shown as dashed gray

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Example
    -------
    >>> fig = create_comparison_figure(
    ...     auroc_df=metric_auroc,
    ...     divergence_df=divergence,
    ...     df_trajectories=df_prep,
    ...     group1_label='CE',
    ...     group2_label='WT',
    ...     metric_cols=['baseline_deviation_normalized', 'total_length_um'],
    ...     embedding_auroc_df=embedding_auroc,
    ...     save_path=Path('output/ce_vs_wt.png'),
    ... )
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[1, 1, 1.2])

    metric_labels = metric_labels or {m: m for m in metric_cols}

    # ==========================================================================
    # Panel A: AUROC over time (Enhanced with preferred style)
    # ==========================================================================
    ax_auroc = axes[0]

    # Prefer plotting at bin centers when available to avoid interpreting `time_bin`
    # as an exact timepoint (e.g., with 4h bins, `time_bin=12` represents 12–16 hpf).
    x_col_main = 'time_bin_center' if 'time_bin_center' in auroc_df.columns else time_col_auroc

    bin_width_label = None
    if 'bin_width' in auroc_df.columns:
        uniq = auroc_df['bin_width'].dropna().unique()
        if len(uniq) == 1:
            bin_width_label = float(uniq[0])

    # Plot baseline comparison first if provided (dashed gray)
    if baseline_auroc_df is not None:
        x_col_baseline = (
            'time_bin_center'
            if 'time_bin_center' in baseline_auroc_df.columns
            else time_col_auroc
        )
        ax_auroc.plot(
            baseline_auroc_df[x_col_baseline],
            baseline_auroc_df['auroc_observed'],
            '--',
            color='#888888',
            linewidth=2,
            alpha=0.7,
            label='Baseline'
        )
        # Baseline null distribution band
        if 'auroc_null_mean' in baseline_auroc_df.columns and 'auroc_null_std' in baseline_auroc_df.columns:
            ax_auroc.fill_between(
                baseline_auroc_df[x_col_baseline],
                baseline_auroc_df['auroc_null_mean'] - baseline_auroc_df['auroc_null_std'],
                baseline_auroc_df['auroc_null_mean'] + baseline_auroc_df['auroc_null_std'],
                color='#888888',
                alpha=0.10,
                linewidth=0
            )

    # Plot main metric AUROC
    ax_auroc.plot(
        auroc_df[x_col_main],
        auroc_df['auroc_observed'],
        'o-',
        label='Metric AUROC',
        color=auroc_color,
        linewidth=2,
        markersize=5,
    )

    # Color-matched null distribution band (more subtle than before)
    if 'auroc_null_mean' in auroc_df.columns and 'auroc_null_std' in auroc_df.columns:
        ax_auroc.fill_between(
            auroc_df[x_col_main],
            auroc_df['auroc_null_mean'] - auroc_df['auroc_null_std'],
            auroc_df['auroc_null_mean'] + auroc_df['auroc_null_std'],
            color=auroc_color,
            alpha=0.10,  # More subtle
            linewidth=0
        )

    # Overlay embedding AUROC if provided
    if embedding_auroc_df is not None:
        embedding_color = '#1f77b4'
        x_col_embedding = (
            'time_bin_center'
            if 'time_bin_center' in embedding_auroc_df.columns
            else time_col_auroc
        )
        ax_auroc.plot(
            embedding_auroc_df[x_col_embedding],
            embedding_auroc_df['auroc_observed'],
            's--',
            label='Embedding AUROC',
            color=embedding_color,
            linewidth=2,
            markersize=6,
        )
        # Embedding null band
        if 'auroc_null_mean' in embedding_auroc_df.columns and 'auroc_null_std' in embedding_auroc_df.columns:
            ax_auroc.fill_between(
                embedding_auroc_df[x_col_embedding],
                embedding_auroc_df['auroc_null_mean'] - embedding_auroc_df['auroc_null_std'],
                embedding_auroc_df['auroc_null_mean'] + embedding_auroc_df['auroc_null_std'],
                color=embedding_color,
                alpha=0.10,
                linewidth=0
            )

    # OPEN CIRCLE significance markers (p < 0.05) - PREFERRED STYLE
    if 'pval' in auroc_df.columns:
        sig_mask = auroc_df['pval'] < 0.05
        if sig_mask.any():
            ax_auroc.scatter(
                auroc_df.loc[sig_mask, x_col_main],
                auroc_df.loc[sig_mask, 'auroc_observed'],
                s=200,  # Large circles
                facecolors='none',  # OPEN (not filled)
                edgecolors=auroc_color,  # Color-matched edge
                linewidths=2.5,
                zorder=5
            )

        # Stars for highly significant (p < 0.01)
        very_sig_mask = auroc_df['pval'] < 0.01
        if very_sig_mask.any():
            for _, row in auroc_df[very_sig_mask].iterrows():
                ax_auroc.annotate(
                    '*',
                    (row[x_col_main], row['auroc_observed'] + 0.03),
                    ha='center',
                    fontsize=14,
                    fontweight='bold',
                    color=auroc_color  # Color-matched star
                )

    # Reference line at chance
    ax_auroc.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

    # Labels and styling
    ax_auroc.set_ylabel('AUROC', fontsize=11)
    ax_auroc.set_xlabel('')
    bin_note = f", bin={bin_width_label:g}h" if bin_width_label is not None else ""
    ax_auroc.set_title(
        f'A. Classification (positive={group1_label}, negative={group2_label}{bin_note})',
        fontsize=12
    )
    ax_auroc.legend(loc='upper left', fontsize=9)
    ax_auroc.set_ylim(0.3, 1.05)
    ax_auroc.grid(True, alpha=0.3)
    ax_auroc.spines['top'].set_visible(False)
    ax_auroc.spines['right'].set_visible(False)

    # ==========================================================================
    # Panel B: Metric Divergence (with smoothing) - RAW absolute difference, NOT z-scored
    # ==========================================================================
    ax_div = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_cols)))
    smooth_sigma = 1.5  # Smoothing parameter

    for i, metric in enumerate(metric_cols):
        metric_data = divergence_df[divergence_df['metric'] == metric].sort_values('hpf')
        if len(metric_data) == 0:
            continue
        label = metric_labels.get(metric, metric)

        # Apply Gaussian smoothing to RAW divergence (not z-scored!)
        abs_diff_smoothed = gaussian_filter1d(
            metric_data['abs_difference'].values,
            sigma=smooth_sigma
        )

        ax_div.plot(
            metric_data['hpf'],
            abs_diff_smoothed,
            '-',
            label=f'{label} (smoothed)',
            color=colors[i],
            linewidth=2,
        )

        # Add SEM band for primary metric
        if i == 0 and 'group1_sem' in metric_data.columns and 'group2_sem' in metric_data.columns:
            combined_sem = np.sqrt(metric_data['group1_sem']**2 + metric_data['group2_sem']**2)
            ax_div.fill_between(
                metric_data['hpf'],
                abs_diff_smoothed - combined_sem.values,
                abs_diff_smoothed + combined_sem.values,
                alpha=0.2,
                color=colors[i],
            )

    ax_div.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax_div.set_ylabel('Absolute Difference', fontsize=11)
    ax_div.set_xlabel('')
    ax_div.set_title('B. Morphological Divergence Over Time', fontsize=12)
    ax_div.legend(loc='upper left', fontsize=9)
    ax_div.grid(True, alpha=0.3)

    # ==========================================================================
    # Panel C: Individual Trajectories (with smoothing)
    # ==========================================================================
    ax_traj = axes[2]
    primary_metric = metric_cols[0]
    smooth_sigma_traj = 1.5  # Smoothing parameter for trajectories

    # Colors for groups
    group_colors = {group1_label: '#d62728', group2_label: '#1f77b4'}

    for group_label in [group2_label, group1_label]:  # Plot reference first, phenotype on top
        color = group_colors[group_label]
        group_data = df_trajectories[df_trajectories['group'] == group_label]

        if len(group_data) == 0:
            continue

        # Individual trajectories (faint, with smoothing)
        embryo_ids = group_data['embryo_id'].unique()
        n_to_plot = min(20, len(embryo_ids))  # Limit for clarity
        for embryo_id in embryo_ids[:n_to_plot]:
            emb_data = group_data[group_data['embryo_id'] == embryo_id].sort_values(time_col_traj)
            if len(emb_data) < 3:  # Need at least 3 points for smoothing
                continue

            # Apply Gaussian smoothing to individual trajectory
            metric_smoothed = gaussian_filter1d(
                emb_data[primary_metric].values,
                sigma=smooth_sigma_traj
            )

            ax_traj.plot(
                emb_data[time_col_traj],
                metric_smoothed,
                alpha=0.15,
                color=color,
                linewidth=0.5,
            )

        # Group mean trajectory (with smoothing)
        mean_traj = group_data.groupby(time_col_traj)[primary_metric].agg(['mean', 'sem']).reset_index()

        if len(mean_traj) >= 3:
            # Apply smoothing to group mean
            mean_smoothed = gaussian_filter1d(
                mean_traj['mean'].values,
                sigma=smooth_sigma_traj
            )

            ax_traj.plot(
                mean_traj[time_col_traj],
                mean_smoothed,
                linewidth=3,
                color=color,
                label=f'{group_label} (n={len(embryo_ids)})',
            )
            # Add SEM bands (centered on smoothed mean)
            ax_traj.fill_between(
                mean_traj[time_col_traj],
                mean_smoothed - mean_traj['sem'].values,
                mean_smoothed + mean_traj['sem'].values,
                alpha=0.2,
                color=color,
            )
        else:
            # Fall back to non-smoothed if too few points
            ax_traj.plot(
                mean_traj[time_col_traj],
                mean_traj['mean'],
                linewidth=3,
                color=color,
                label=f'{group_label} (n={len(embryo_ids)})',
            )

    ax_traj.set_xlabel('Time (hpf)', fontsize=11)
    ax_traj.set_ylabel(metric_labels.get(primary_metric, primary_metric), fontsize=11)
    ax_traj.set_title(f'C. Individual Trajectories: {metric_labels.get(primary_metric, primary_metric)}', fontsize=12)
    ax_traj.legend(loc='upper left', fontsize=9)
    ax_traj.grid(True, alpha=0.3)

    # ==========================================================================
    # Add time landmarks to all panels
    # ==========================================================================
    if time_landmarks:
        for t, label in time_landmarks.items():
            for ax in axes:
                ax.axvline(t, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            # Add label at top of first panel
            ax_auroc.text(
                t,
                ax_auroc.get_ylim()[1],
                f' {label}',
                ha='left',
                va='bottom',
                fontsize=9,
                color='red',
            )

    # ==========================================================================
    # Overall title and layout
    # ==========================================================================
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def create_auroc_comparison_figure(
    auroc_dfs: Dict[str, pd.DataFrame],
    title: str,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    time_col: str = 'time_bin',
) -> plt.Figure:
    """
    Create a single-panel figure comparing AUROC curves from multiple comparisons.

    Useful for overlay plots comparing different reference groups or methods.

    Parameters
    ----------
    auroc_dfs : Dict[str, pd.DataFrame]
        Mapping of label -> AUROC DataFrame
    title : str
        Figure title
    save_path : Optional[Path]
        Path to save figure
    figsize : tuple
        Figure size
    time_col : str
        Time column name

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(auroc_dfs)))

    for (label, df), color in zip(auroc_dfs.items(), colors):
        ax.plot(
            df[time_col],
            df['auroc_observed'],
            'o-',
            label=label,
            color=color,
            linewidth=2,
            markersize=5,
        )
        if 'auroc_null_std' in df.columns:
            ax.fill_between(
                df[time_col],
                df['auroc_observed'] - df['auroc_null_std'],
                df['auroc_observed'] + df['auroc_null_std'],
                alpha=0.15,
                color=color,
            )

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='Chance')
    ax.set_xlabel('Time (hpf)', fontsize=11)
    ax.set_ylabel('AUROC', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig
"""Temporal emergence plotting function to add to plotting.py"""

def plot_temporal_emergence(
    results_dict: Dict[str, Dict],
    colors: Dict[str, str],
    time_bin_width: float = 4.0,
    title_prefix: str = "",
    figsize_per_panel: float = 5.0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot temporal emergence of phenotypic differences with significance highlighting.

    Creates bar plots showing when differences emerge, with:
    - Bars showing AUROC at each time bin
    - Significant bins (p < 0.05) have dark edges and full opacity
    - Highly significant bins (p < 0.01) marked with stars
    - Green vertical line shows earliest significant timepoint

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping comparison labels to results from compare_groups()
        Each value should have 'classification' and 'summary' keys
        Example: {'Penetrant_vs_Control': {'classification': df, 'summary': dict}}
    colors : Dict[str, str]
        Mapping of comparison labels to hex colors
        Example: {'Penetrant_vs_Control': '#D32F2F'}
    time_bin_width : float, optional
        Width of time bins in hours (from compare_groups bin_width), default 4.0
    title_prefix : str, optional
        Prefix for figure title (e.g., "CEP290: "), default ""
    figsize_per_panel : float, optional
        Width of each subplot panel, default 5.0
    save_path : Optional[Path], optional
        Path to save figure, default None

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Example
    -------
    >>> results_dict = {'CE_vs_WT': {'classification': df, 'summary': summary_dict}}
    >>> colors = {'CE_vs_WT': '#D32F2F'}
    >>> fig = plot_temporal_emergence(
    ...     results_dict,
    ...     colors=colors,
    ...     time_bin_width=4.0,
    ...     title_prefix='B9D2: ',
    ... )
    """
    n_comparisons = len(results_dict)
    fig, axes = plt.subplots(
        1, n_comparisons,
        figsize=(figsize_per_panel * n_comparisons, 5),
        sharey=True
    )

    # Handle single comparison case
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
        bars = ax.bar(
            df_class['time_bin'],
            df_class['auroc_observed'],
            width=time_bin_width * 0.8,
            color=color,
            alpha=0.6
        )

        # Highlight significant bins
        for i, (idx, row) in enumerate(df_class.iterrows()):
            if row['pval'] < 0.05:
                bars[i].set_alpha(1.0)  # Full opacity
                bars[i].set_edgecolor('black')  # Black border
                bars[i].set_linewidth(2)
            if row['pval'] < 0.01:
                ax.annotate(
                    '*',
                    (row['time_bin'], row['auroc_observed'] + 0.02),
                    ha='center',
                    fontsize=12,
                    fontweight='bold'
                )

        # Reference line at chance
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Mark earliest significant timepoint
        if summary['earliest_significant_hpf'] is not None:
            ax.axvline(
                x=summary['earliest_significant_hpf'],
                color='green',
                linestyle='-',
                alpha=0.7,
                linewidth=2
            )
            ax.annotate(
                f"First sig:\n{summary['earliest_significant_hpf']}h",
                xy=(summary['earliest_significant_hpf'], 0.95),
                ha='center',
                fontsize=9,
                color='green'
            )

        # Labels and styling
        ax.set_xlabel('Hours Post Fertilization')
        max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
        ax.set_title(f"{label}\nMax AUROC: {max_auroc:.2f}")
        ax.set_ylim(0.3, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Y-label on first panel only
    axes[0].set_ylabel('AUROC')

    # Overall title
    fig.suptitle(
        f'{title_prefix}Temporal Emergence of Phenotypic Differences',
        fontsize=14,
        y=1.02
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig
