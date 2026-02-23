#!/usr/bin/env python3
"""
Plot individual embryo divergence trajectories.

Creates plots where line color/darkness indicates maximum divergence.
Darker lines = embryos with higher maximum divergence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from scipy import stats
import config_new as config


def plot_genotype_comparison(
    df_combined: pd.DataFrame,
    metric: str = "mahalanobis_distance",
    remove_outliers: bool = True,
    outlier_percentile: float = 99.0,
    figsize: tuple = (18, 8),
    plot_type: str = "side_by_side"
) -> plt.Figure:
    """
    Compare homozygous vs heterozygous embryo divergence from wildtype reference.

    Both genotypes are measured against the SAME wildtype reference.
    This shows whether hets have lower divergence values than homs.

    Parameters
    ----------
    df_combined : pd.DataFrame
        Combined dataframe with both hom and het embryos (from WT reference)
        Must have columns: embryo_id, time_bin, genotype, [metric]
    metric : str
        Distance metric to compare
    remove_outliers : bool
        Whether to remove extreme outliers
    outlier_percentile : float
        Percentile threshold for outlier removal
    figsize : tuple
        Figure size
    plot_type : str
        'side_by_side' or 'overlay'

    Returns
    -------
    plt.Figure
        Comparison figure
    """
    # Separate genotypes - auto-detect homozygous and heterozygous
    genotypes = df_combined['genotype'].unique()
    hom_genotypes = [g for g in genotypes if 'homozygous' in g.lower()]
    het_genotypes = [g for g in genotypes if 'hetero' in g.lower()]

    df_hom = df_combined[df_combined['genotype'].isin(hom_genotypes)].copy()
    df_het = df_combined[df_combined['genotype'].isin(het_genotypes)].copy()

    print(f"\n  Homozygous: {len(df_hom)} timepoints, {df_hom['embryo_id'].nunique()} embryos")
    print(f"  Heterozygous: {len(df_het)} timepoints, {df_het['embryo_id'].nunique()} embryos")

    # Remove outliers if requested
    if remove_outliers:
        threshold = np.percentile(df_combined[metric], outlier_percentile)
        df_hom = df_hom[df_hom[metric] <= threshold].copy()
        df_het = df_het[df_het[metric] <= threshold].copy()
        print(f"  Outlier threshold (>{outlier_percentile}%): {threshold:.2f}")

    # Compute statistics
    hom_mean = df_hom[metric].mean()
    hom_std = df_hom[metric].std()
    het_mean = df_het[metric].mean()
    het_std = df_het[metric].std()

    # Statistical test
    t_stat, p_value = stats.ttest_ind(df_hom[metric], df_het[metric])

    print(f"\n  Homozygous: mean={hom_mean:.2f} ± {hom_std:.2f}")
    print(f"  Heterozygous: mean={het_mean:.2f} ± {het_std:.2f}")
    print(f"  T-test: t={t_stat:.2f}, p={p_value:.2e}")

    if plot_type == "side_by_side":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        axes = [(ax1, df_hom, 'cep290_homozygous', '#d62728', 'Homozygous'),  # red
                (ax2, df_het, 'cep290_heterozygous', '#1f77b4', 'Heterozygous')]  # blue

        for ax, df, genotype, color, label in axes:
            # Plot individual trajectories
            for embryo_id in df['embryo_id'].unique():
                embryo_data = df[df['embryo_id'] == embryo_id].sort_values('time_bin')
                if len(embryo_data) < 2:
                    continue
                ax.plot(
                    embryo_data['time_bin'],
                    embryo_data[metric],
                    color=color,
                    alpha=0.3,
                    linewidth=1.0
                )

            # Compute and plot mean trajectory
            mean_trajectory = df.groupby('time_bin')[metric].agg(['mean', 'sem']).reset_index()
            ax.plot(
                mean_trajectory['time_bin'],
                mean_trajectory['mean'],
                color=color,
                linewidth=3,
                label=f'Mean (n={df["embryo_id"].nunique()})',
                zorder=10
            )

            # Add confidence interval
            ax.fill_between(
                mean_trajectory['time_bin'],
                mean_trajectory['mean'] - 1.96 * mean_trajectory['sem'],
                mean_trajectory['mean'] + 1.96 * mean_trajectory['sem'],
                color=color,
                alpha=0.2,
                label='95% CI'
            )

            # Formatting
            ax.set_xlabel('Developmental Time (hpf)', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(label, fontsize=14, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=10)

            # Add stats
            stats_text = (
                f'Mean: {df[metric].mean():.1f} ± {df[metric].std():.1f}\n'
                f'Median: {df[metric].median():.1f}\n'
                f'Range: {df[metric].min():.1f} - {df[metric].max():.1f}'
            )
            ax.text(
                0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color, linewidth=2),
                fontsize=9,
                family='monospace'
            )

        # Overall title
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        fig.suptitle(
            f'Genotype Comparison: Divergence from Wildtype Reference\n'
            f'{metric.replace("_", " ").title()} | '
            f'Hom mean: {hom_mean:.1f}, Het mean: {het_mean:.1f} | '
            f'Difference: {sig_marker} (p={p_value:.2e})',
            fontsize=14,
            fontweight='bold',
            y=1.00
        )

    else:  # overlay
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot homozygous
        for embryo_id in df_hom['embryo_id'].unique():
            embryo_data = df_hom[df_hom['embryo_id'] == embryo_id].sort_values('time_bin')
            if len(embryo_data) < 2:
                continue
            ax.plot(
                embryo_data['time_bin'],
                embryo_data[metric],
                color='#d62728',
                alpha=0.2,
                linewidth=0.8
            )

        # Plot heterozygous
        for embryo_id in df_het['embryo_id'].unique():
            embryo_data = df_het[df_het['embryo_id'] == embryo_id].sort_values('time_bin')
            if len(embryo_data) < 2:
                continue
            ax.plot(
                embryo_data['time_bin'],
                embryo_data[metric],
                color='#1f77b4',
                alpha=0.2,
                linewidth=0.8
            )

        # Plot means
        hom_mean_traj = df_hom.groupby('time_bin')[metric].agg(['mean', 'sem']).reset_index()
        het_mean_traj = df_het.groupby('time_bin')[metric].agg(['mean', 'sem']).reset_index()

        ax.plot(
            hom_mean_traj['time_bin'],
            hom_mean_traj['mean'],
            color='#d62728',
            linewidth=3,
            label=f'Homozygous mean (n={df_hom["embryo_id"].nunique()})',
            zorder=10
        )
        ax.plot(
            het_mean_traj['time_bin'],
            het_mean_traj['mean'],
            color='#1f77b4',
            linewidth=3,
            label=f'Heterozygous mean (n={df_het["embryo_id"].nunique()})',
            zorder=10
        )

        # Formatting
        ax.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        ax.set_title(
            f'Genotype Comparison: Divergence from Wildtype Reference\n'
            f'{metric.replace("_", " ").title()} | '
            f'Hom: {hom_mean:.1f}±{hom_std:.1f}, Het: {het_mean:.1f}±{het_std:.1f} | '
            f'{sig_marker} (p={p_value:.2e})',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    return fig


def plot_embryo_trajectories_by_divergence(
    df_divergence: pd.DataFrame,
    metric: str = "mahalanobis_distance",
    genotype: Optional[str] = None,
    reference_genotype: Optional[str] = None,
    top_n: Optional[int] = None,
    remove_outliers: bool = True,
    outlier_percentile: float = 99.0,
    figsize: tuple = (14, 8),
    cmap: str = 'RdYlBu_r'
) -> plt.Figure:
    """
    Plot individual embryo trajectories with color based on divergence intensity.

    Parameters
    ----------
    df_divergence : pd.DataFrame
        Divergence scores with columns: embryo_id, time_bin, genotype, [metric]
    metric : str
        Distance metric column to plot
    genotype : Optional[str]
        If provided, filter to only this genotype
    reference_genotype : Optional[str]
        Reference genotype label for plot title
    top_n : Optional[int]
        If provided, plot only the top N embryos by max divergence
        If None, plots ALL embryos
    remove_outliers : bool
        Whether to remove extreme outliers before plotting
    outlier_percentile : float
        Percentile threshold for outlier removal (e.g., 99.0 removes top 1%)
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Matplotlib colormap name for line colors

    Returns
    -------
    plt.Figure
        The generated matplotlib figure
    """
    # Filter by genotype if requested
    if genotype is not None:
        df_divergence = df_divergence[df_divergence['genotype'] == genotype].copy()
        genotype_label = genotype
    else:
        genotype_label = "All genotypes"
    
    # Remove extreme outliers if requested
    n_before = len(df_divergence)
    if remove_outliers:
        threshold = np.percentile(df_divergence[metric], outlier_percentile)
        df_divergence = df_divergence[df_divergence[metric] <= threshold].copy()
        n_removed = n_before - len(df_divergence)
        print(f"\nRemoving extreme outliers (>{outlier_percentile}th percentile: {threshold:.2f})")
        print(f"  Removed {n_removed} / {n_before} points ({100*n_removed/n_before:.1f}%)")
    
    print(f"\nPlotting embryo trajectories...")
    print(f"  Genotype: {genotype_label}")
    print(f"  Metric: {metric}")
    
    # Compute per-embryo max divergence for coloring (darker = higher max divergence)
    embryo_scores = df_divergence.groupby('embryo_id')[metric].max()
    
    print(f"  Total embryos: {len(embryo_scores)}")
    
    # Select top embryos if requested
    if top_n is not None and len(embryo_scores) > top_n:
        top_embryos = embryo_scores.nlargest(top_n).index
        df_plot = df_divergence[df_divergence['embryo_id'].isin(top_embryos)].copy()
        embryo_scores = embryo_scores[top_embryos]
        print(f"  Showing top {top_n} embryos (highest divergence)")
    else:
        df_plot = df_divergence.copy()
        print(f"  Showing ALL {len(embryo_scores)} embryos")
    
    # Normalize scores for colormap (0-1)
    score_min = embryo_scores.min()
    score_max = embryo_scores.max()
    if score_max == score_min:
        score_max = score_min + 1  # Avoid division by zero
    embryo_colors = (embryo_scores - score_min) / (score_max - score_min)
    
    print(f"  Divergence range: {score_min:.2f} - {score_max:.2f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colormap (viridis: light=low, dark=high)
    cmap_obj = plt.get_cmap(cmap)
    
    # Plot each embryo
    n_plotted = 0
    for embryo_id in embryo_scores.index:
        embryo_data = df_plot[df_plot['embryo_id'] == embryo_id].sort_values('time_bin')
        
        if len(embryo_data) < 2:
            continue
        
        # Get color intensity (darker = higher divergence)
        color_value = embryo_colors[embryo_id]
        color = cmap_obj(color_value)
        
        ax.plot(
            embryo_data['time_bin'],
            embryo_data[metric],
            color=color,
            alpha=0.7,
            linewidth=1.5
        )
        n_plotted += 1
    
    print(f"  Plotted {n_plotted} embryo trajectories")
    
    # Create title with reference info
    if reference_genotype:
        ref_label = f"Reference: {reference_genotype}"
    else:
        ref_label = "Reference: wildtype"
    
    # Formatting
    ax.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_title(
        f'Individual Embryo Divergence Trajectories - {genotype_label} (n={n_plotted})\n'
        f'{ref_label} | {metric.replace("_", " ").title()}\n'
        f'Line darkness = divergence intensity (darker = more divergent)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=score_min, vmax=score_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(
        f'Max {metric.replace("_", " ").title()}',
        fontsize=12,
        fontweight='bold'
    )
    cbar.ax.tick_params(labelsize=10)
    
    # Add statistics box
    stats_text = (
        f'Divergence Statistics:\n'
        f'  Range: {df_plot[metric].min():.2f} - {df_plot[metric].max():.2f}\n'
        f'  Mean: {df_plot[metric].mean():.2f} ± {df_plot[metric].std():.2f}\n'
        f'  Median: {df_plot[metric].median():.2f}'
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2),
        fontsize=10,
        family='monospace'
    )
    
    plt.tight_layout()

    return fig


def main():
    print("="*80)
    print("EMBRYO TRAJECTORY VISUALIZATION")
    print("="*80)
    
    # Load divergence results
    data_dir = Path(config.DATA_DIR) / "cep290" / "divergence"
    plot_dir = Path(config.PLOT_DIR) / "cep290" / "divergence"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading divergence data from: {data_dir}")
    
    # Load divergence data (both genotypes measured from WT reference)
    df_hom_wt = pd.read_csv(data_dir / "hom_vs_wt_divergence.csv")
    print(f"Loaded hom_vs_wt: {len(df_hom_wt)} embryo-timepoints")
    print(f"  Unique embryos: {df_hom_wt['embryo_id'].nunique()}")

    df_het_wt = pd.read_csv(data_dir / "het_vs_wt_divergence.csv")
    print(f"Loaded het_vs_wt: {len(df_het_wt)} embryo-timepoints")
    print(f"  Unique embryos: {df_het_wt['embryo_id'].nunique()}")

    # Combine both genotypes (both from WT reference)
    df_combined_wt = pd.concat([df_hom_wt, df_het_wt], ignore_index=True)
    print(f"\nCombined dataset: {len(df_combined_wt)} embryo-timepoints")
    print(f"  Genotypes: {df_combined_wt['genotype'].unique().tolist()}")
    
    # Plot 1: All embryos colored by max Mahalanobis distance
    print("\n" + "="*80)
    print("PLOT 1: All embryos combined - Mahalanobis distance")
    print("="*80)
    
    fig1 = plot_embryo_trajectories_by_divergence(
        df_hom_wt,
        metric="mahalanobis_distance",
        genotype=None,  # All genotypes
        reference_genotype="cep290_wildtype",
        top_n=None,  # Plot all
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8)
    )
    fig1.savefig(plot_dir / "trajectories_all_mahalanobis.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Get unique genotypes
    genotypes = df_hom_wt['genotype'].unique()
    print(f"\nGenerating plots for genotypes: {genotypes.tolist()}")
    
    # Plot for each genotype separately - ALL embryos, colored by max divergence
    for genotype in genotypes:
        geno_short = genotype.replace('cep290_', '')
        
        print("\n" + "="*80)
        print(f"PLOTTING: {genotype} - ALL embryos (Mahalanobis, colored by max)")
        print("="*80)
        
        fig = plot_embryo_trajectories_by_divergence(
            df_hom_wt,
            metric="mahalanobis_distance",
            genotype=genotype,
            reference_genotype="cep290_wildtype",
            top_n=None,  # Plot ALL embryos
            remove_outliers=True,  # Remove extreme outliers
            outlier_percentile=99.0,  # Remove top 1%
            figsize=(14, 8)
        )
        fig.savefig(plot_dir / f"trajectories_{geno_short}_all_mahalanobis.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("\n" + "="*80)
        print(f"PLOTTING: {genotype} - ALL embryos (Euclidean, colored by max)")
        print("="*80)
        
        fig = plot_embryo_trajectories_by_divergence(
            df_hom_wt,
            metric="euclidean_distance",
            genotype=genotype,
            reference_genotype="cep290_wildtype",
            top_n=None,  # Plot ALL embryos
            remove_outliers=True,  # Remove extreme outliers
            outlier_percentile=99.0,  # Remove top 1%
            figsize=(14, 8)
        )
        fig.savefig(plot_dir / f"trajectories_{geno_short}_all_euclidean.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Summary plot with Euclidean for all genotypes
    print("\n" + "="*80)
    print("PLOTTING: All genotypes combined - Euclidean distance")
    print("="*80)
    
    fig = plot_embryo_trajectories_by_divergence(
        df_hom_wt,
        metric="euclidean_distance",
        genotype=None,  # All genotypes
        reference_genotype="cep290_wildtype",
        top_n=None,  # Plot ALL embryos
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8)
    )
    fig.savefig(plot_dir / "trajectories_all_euclidean.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # GENOTYPE COMPARISON: Homozygous vs Heterozygous (both from WT reference)
    # ========================================================================

    print("\n" + "="*80)
    print("GENOTYPE COMPARISON: Homozygous vs Heterozygous (WT reference)")
    print("="*80)
    print("Hypothesis: Heterozygous embryos have lower divergence from WT than homozygous")

    # Side-by-side comparison - Mahalanobis
    print("\n" + "="*80)
    print("COMPARISON PLOT: Mahalanobis distance (side-by-side)")
    print("="*80)

    fig = plot_genotype_comparison(
        df_combined=df_combined_wt,
        metric="mahalanobis_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(18, 8),
        plot_type="side_by_side"
    )
    fig.savefig(plot_dir / "genotype_comparison_mahalanobis_sidebyside.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Overlay comparison - Mahalanobis
    print("\n" + "="*80)
    print("COMPARISON PLOT: Mahalanobis distance (overlay)")
    print("="*80)

    fig = plot_genotype_comparison(
        df_combined=df_combined_wt,
        metric="mahalanobis_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8),
        plot_type="overlay"
    )
    fig.savefig(plot_dir / "genotype_comparison_mahalanobis_overlay.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Side-by-side comparison - Euclidean
    print("\n" + "="*80)
    print("COMPARISON PLOT: Euclidean distance (side-by-side)")
    print("="*80)

    fig = plot_genotype_comparison(
        df_combined=df_combined_wt,
        metric="euclidean_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(18, 8),
        plot_type="side_by_side"
    )
    fig.savefig(plot_dir / "genotype_comparison_euclidean_sidebyside.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Overlay comparison - Euclidean
    print("\n" + "="*80)
    print("COMPARISON PLOT: Euclidean distance (overlay)")
    print("="*80)

    fig = plot_genotype_comparison(
        df_combined=df_combined_wt,
        metric="euclidean_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8),
        plot_type="overlay"
    )
    fig.savefig(plot_dir / "genotype_comparison_euclidean_overlay.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\n" + "="*80)
    print("ALL PLOTS GENERATED!")
    print("="*80)
    print(f"\nPlots saved to: {plot_dir}")
    print("\nGenerated plots:")
    print("\n1. INDIVIDUAL GENOTYPE TRAJECTORIES (from WT reference):")
    print("   - trajectories_all_mahalanobis.png (all genotypes combined)")
    print("   - trajectories_all_euclidean.png (all genotypes combined)")
    print("   - trajectories_homozygous_all_mahalanobis.png")
    print("   - trajectories_homozygous_all_euclidean.png")
    print("   - trajectories_wildtype_all_mahalanobis.png")
    print("   - trajectories_wildtype_all_euclidean.png")
    print("\n2. GENOTYPE COMPARISONS (Hom vs Het, both from WT reference):")
    print("   MAHALANOBIS DISTANCE:")
    print("   - genotype_comparison_mahalanobis_sidebyside.png")
    print("   - genotype_comparison_mahalanobis_overlay.png")
    print("   EUCLIDEAN DISTANCE:")
    print("   - genotype_comparison_euclidean_sidebyside.png")
    print("   - genotype_comparison_euclidean_overlay.png")
    print("\nKey features:")
    print("  - Individual embryo trajectories with mean ± 95% CI")
    print("  - Statistical comparison (t-test) between genotypes")
    print("  - Tests hypothesis: Do hets have lower divergence than homs?")
    print("  - Both genotypes measured against SAME wildtype reference")

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
