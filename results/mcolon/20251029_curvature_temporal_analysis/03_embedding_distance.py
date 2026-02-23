#!/usr/bin/env python3
"""
Analyze relationship between embedding distance and curvature differences.

This script investigates:
1. How much is distance in morphology space (embedding space) reflected by curvature?
2. Do embryos with similar curvature profiles occupy similar regions of embedding space?
3. Are there genotype-specific patterns in the embedding-curvature relationship?

For each embryo:
- Compute pairwise Euclidean distances between all timepoint pairs in embedding space
- Compute pairwise curvature differences (absolute differences in metric values)
- Correlate these distances to see if morphology changes align with curvature changes

Outputs:
- Scatter plots: embedding distance vs curvature difference
- Correlation matrices and statistics by genotype
- Temporal trajectories in embedding space colored by curvature
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

# Import data loading from this directory
from load_data import get_analysis_dataframe, get_genotype_short_name, get_genotype_color


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent
FIGURE_DIR = RESULTS_DIR / 'outputs' / 'figures' / '03_embedding_distance'
TABLE_DIR = RESULTS_DIR / 'outputs' / 'tables' / '03_embedding_distance'

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Metrics to analyze
METRICS = ['arc_length_ratio', 'normalized_baseline_deviation']
METRIC_LABELS = {
    'arc_length_ratio': 'Arc Length Ratio',
    'normalized_baseline_deviation': 'Normalized Baseline Deviation'
}


# ============================================================================
# Embedding Distance Computation
# ============================================================================

def compute_pairwise_embedding_distances(embeddings):
    """
    Compute pairwise Euclidean distances in embedding space.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_timepoints, n_embedding_dims)

    Returns
    -------
    np.ndarray
        Symmetric distance matrix (n_timepoints, n_timepoints)
    """
    if len(embeddings) < 2:
        return None

    distances = squareform(pdist(embeddings, metric='euclidean'))
    return distances


def compute_pairwise_curvature_differences(curvature_values):
    """
    Compute pairwise absolute differences in curvature metric.

    Parameters
    ----------
    curvature_values : np.ndarray
        Shape (n_timepoints,)

    Returns
    -------
    np.ndarray
        Symmetric difference matrix (n_timepoints, n_timepoints)
    """
    if len(curvature_values) < 2:
        return None

    differences = np.abs(curvature_values[:, None] - curvature_values[None, :])
    return differences


def compute_correlation_for_embryo(
    embryo_data,
    embedding_cols,
    metric='arc_length_ratio'
):
    """
    Compute correlation between embedding distance and curvature difference for one embryo.

    Parameters
    ----------
    embryo_data : pd.DataFrame
        One embryo's data
    embedding_cols : list of str
        Column names containing embeddings
    metric : str
        Which curvature metric to use

    Returns
    -------
    dict
        {
            'embryo_id': str,
            'genotype': str,
            'n_timepoints': int,
            'pearson_r': float,
            'pearson_p': float,
            'spearman_rho': float,
            'spearman_p': float,
            'metric': str
        }
    """
    if len(embryo_data) < 3:
        return None

    # Extract data
    embeddings = embryo_data[embedding_cols].values
    curvature = embryo_data[metric].values

    # Compute distances
    embed_distances = compute_pairwise_embedding_distances(embeddings)
    curv_differences = compute_pairwise_curvature_differences(curvature)

    if embed_distances is None or curv_differences is None:
        return None

    # Flatten for correlation
    upper_tri = np.triu_indices(len(embeddings), k=1)
    embed_dist_flat = embed_distances[upper_tri]
    curv_diff_flat = curv_differences[upper_tri]

    # Compute correlations
    if len(embed_dist_flat) > 2:
        pearson_r, pearson_p = pearsonr(embed_dist_flat, curv_diff_flat)
        spearman_rho, spearman_p = spearmanr(embed_dist_flat, curv_diff_flat)
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = np.nan

    return {
        'embryo_id': embryo_data['embryo_id'].iloc[0],
        'genotype': embryo_data['genotype'].iloc[0],
        'n_timepoints': len(embryo_data),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'metric': metric
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_embedding_vs_curvature_scatter(
    df,
    embedding_cols,
    metric='arc_length_ratio',
    save_dir=FIGURE_DIR
):
    """
    Create scatter plot: embedding distance vs curvature difference.

    For a sample of embryos, overlay their data points colored by genotype.

    Parameters
    ----------
    df : pd.DataFrame
        Full analysis dataframe
    embedding_cols : list of str
    metric : str
    save_dir : Path

    Returns
    -------
    Path
        Path to saved figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    genotypes = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']

    for ax, genotype in zip(axes, genotypes):
        genotype_df = df[df['genotype'] == genotype]

        all_embed_dists = []
        all_curv_diffs = []

        # Collect data from all embryos
        for embryo_id in genotype_df['embryo_id'].unique()[:20]:  # Sample first 20
            embryo_data = genotype_df[genotype_df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')

            if len(embryo_data) < 3:
                continue

            embeddings = embryo_data[embedding_cols].values
            curvature = embryo_data[metric].values

            embed_distances = compute_pairwise_embedding_distances(embeddings)
            curv_differences = compute_pairwise_curvature_differences(curvature)

            upper_tri = np.triu_indices(len(embeddings), k=1)
            all_embed_dists.extend(embed_distances[upper_tri])
            all_curv_diffs.extend(curv_differences[upper_tri])

        if len(all_embed_dists) > 0:
            ax.scatter(
                all_embed_dists,
                all_curv_diffs,
                alpha=0.3,
                s=20,
                color=get_genotype_color(genotype)
            )

            # Fit trend line
            z = np.polyfit(all_embed_dists, all_curv_diffs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(all_embed_dists), max(all_embed_dists), 100)
            ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.7)

            # Correlation
            corr, p_val = spearmanr(all_embed_dists, all_curv_diffs)
            ax.text(
                0.05, 0.95,
                f'ρ = {corr:.3f}\np = {p_val:.3e}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10
            )

        genotype_short = get_genotype_short_name(genotype)
        ax.set_title(f'{genotype_short}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Embedding Distance (Euclidean)', fontsize=11)

        if ax == axes[0]:
            ax.set_ylabel(f'{METRIC_LABELS[metric]} Difference', fontsize=11)

        ax.grid(alpha=0.3)

    metric_label = METRIC_LABELS.get(metric, metric)
    plt.suptitle(f'Embedding Distance vs Curvature - {metric_label}',
                 fontweight='bold', fontsize=13, y=1.00)
    plt.tight_layout()

    save_path = save_dir / f'embedding_vs_curvature_{metric}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path


def plot_trajectory_in_embedding_space(
    df,
    embedding_cols,
    metric='arc_length_ratio',
    n_embryos_per_genotype=3,
    save_dir=FIGURE_DIR
):
    """
    Plot temporal trajectories in embedding space, colored by curvature.

    Parameters
    ----------
    df : pd.DataFrame
    embedding_cols : list of str
    metric : str
    n_embryos_per_genotype : int
    save_dir : Path

    Returns
    -------
    Path
    """
    # Use first 3 embedding dimensions
    emb_cols = embedding_cols[:3]

    if len(emb_cols) < 2:
        print("  Warning: Not enough embedding dimensions for visualization")
        return None

    is_3d = len(emb_cols) == 3

    fig = plt.figure(figsize=(15, 5))

    genotypes = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']

    for idx, genotype in enumerate(genotypes, 1):
        if is_3d:
            ax = fig.add_subplot(1, 3, idx, projection='3d')
        else:
            ax = fig.add_subplot(1, 3, idx)

        genotype_df = df[df['genotype'] == genotype]
        embryo_ids = genotype_df['embryo_id'].unique()[:n_embryos_per_genotype]

        for embryo_id in embryo_ids:
            embryo_data = genotype_df[genotype_df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')

            if len(embryo_data) < 2:
                continue

            # Get embedding values and curvature for coloring
            positions = embryo_data[emb_cols].values
            curvature = embryo_data[metric].values

            if is_3d:
                scatter = ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    c=curvature,
                    cmap='viridis',
                    s=50,
                    alpha=0.7,
                    label=embryo_id
                )

                # Plot trajectory
                ax.plot(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    'k-',
                    alpha=0.3,
                    linewidth=1
                )

                ax.set_xlabel(emb_cols[0], fontsize=10)
                ax.set_ylabel(emb_cols[1], fontsize=10)
                ax.set_zlabel(emb_cols[2], fontsize=10)
            else:
                scatter = ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    c=curvature,
                    cmap='viridis',
                    s=50,
                    alpha=0.7,
                    label=embryo_id
                )

                ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1)

                ax.set_xlabel(emb_cols[0], fontsize=10)
                ax.set_ylabel(emb_cols[1], fontsize=10)

        genotype_short = get_genotype_short_name(genotype)
        ax.set_title(f'{genotype_short}', fontweight='bold', fontsize=11)

        # Colorbar for the last subplot
        if idx == len(genotypes):
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label(METRIC_LABELS.get(metric, metric), fontsize=10)

    metric_label = METRIC_LABELS.get(metric, metric)
    fig_type = "3D Embedding Space" if is_3d else "2D Embedding Space"
    plt.suptitle(f'{fig_type} Trajectories (colored by {metric_label})',
                 fontweight='bold', fontsize=13, y=0.98)
    plt.tight_layout()

    save_path = save_dir / f'embedding_trajectories_{"3d" if is_3d else "2d"}_{metric}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_embedding_curvature_correlations(df, embedding_cols, metrics=None):
    """
    Compute embedding-curvature correlations for all embryos.

    Returns
    -------
    pd.DataFrame
        One row per embryo per metric with correlation statistics
    """
    if metrics is None:
        metrics = METRICS

    results = []

    for embryo_id in df['embryo_id'].unique():
        embryo_data = df[df['embryo_id'] == embryo_id].sort_values('frame_index')

        for metric in metrics:
            result = compute_correlation_for_embryo(embryo_data, embedding_cols, metric)

            if result is not None:
                results.append(result)

    return pd.DataFrame(results)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("EMBEDDING DISTANCE vs CURVATURE ANALYSIS")
    print("="*80)

    # Load data
    print("\nSTEP 1: LOADING DATA")
    df, metadata = get_analysis_dataframe()
    embedding_cols = metadata['embedding_cols']

    print(f"\n  Found {len(embedding_cols)} embedding dimensions")
    print(f"  Sample dimensions: {embedding_cols[:5]}")

    # Compute correlations
    print("\nSTEP 2: COMPUTING EMBEDDING-CURVATURE CORRELATIONS")

    corr_df = compute_embedding_curvature_correlations(df, embedding_cols)

    corr_file = TABLE_DIR / 'embedding_curvature_correlations.csv'
    corr_df.to_csv(corr_file, index=False)
    print(f"  Saved correlations to {corr_file}")

    # Summary by genotype
    print("\nSummary of Correlations by Genotype:")
    for metric in METRICS:
        print(f"\n  {METRIC_LABELS[metric]}:")
        metric_data = corr_df[corr_df['metric'] == metric]

        for genotype in metadata['genotypes']:
            genotype_short = get_genotype_short_name(genotype)
            genotype_data = metric_data[metric_data['genotype'] == genotype]

            if len(genotype_data) > 0:
                mean_rho = genotype_data['spearman_rho'].mean()
                mean_p = genotype_data['spearman_p'].mean()
                print(f"    {genotype_short}: ρ = {mean_rho:7.3f}, p = {mean_p:.3e}")

    # Create scatter plots
    print("\nSTEP 3: CREATING SCATTER PLOTS")

    for metric in METRICS:
        print(f"\n  {METRIC_LABELS[metric]}...")
        fig_path = plot_embedding_vs_curvature_scatter(df, embedding_cols, metric)
        print(f"    Saved: {fig_path}")

    # Create trajectory plots
    print("\nSTEP 4: CREATING EMBEDDING SPACE TRAJECTORIES")

    for metric in METRICS:
        print(f"\n  {METRIC_LABELS[metric]}...")
        fig_path = plot_trajectory_in_embedding_space(df, embedding_cols, metric)
        if fig_path:
            print(f"    Saved: {fig_path}")

    # Done
    print("\n" + "="*80)
    print("EMBEDDING DISTANCE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Data: {TABLE_DIR / 'embedding_curvature_correlations.csv'}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
