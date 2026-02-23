#!/usr/bin/env python3
"""
Plot curvature over time for individual embryos and aggregate by genotype.

This analysis shows how curvature changes across development, revealing whether
specific genotypes have distinct temporal signatures. We plot both individual
embryo trajectories and aggregated mean ± SEM trends for comparison.

Metrics analyzed:
- arc_length_ratio (normalized, size-independent)
- normalized_baseline_deviation (baseline deviation / embryo length)

Outputs:
- Individual embryo trajectory PDFs
- Aggregate genotype comparison plots
- Statistical summary tables
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import data loading from this directory
from load_data import get_analysis_dataframe, get_genotype_short_name, get_genotype_color

# Setup
RESULTS_DIR = Path(__file__).parent
FIGURE_DIR = RESULTS_DIR / 'outputs' / 'figures' / '01_individual_trajectories'
TABLE_DIR = RESULTS_DIR / 'outputs' / 'tables' / '01_individual_trajectories'

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Plotting parameters
METRICS = ['arc_length_ratio', 'normalized_baseline_deviation']
METRIC_LABELS = {
    'arc_length_ratio': 'Arc Length Ratio',
    'normalized_baseline_deviation': 'Normalized Baseline Deviation'
}


# ============================================================================
# Individual Embryo Trajectories
# ============================================================================

def plot_individual_embryo_trajectories(df, embryo_id, metrics=None, save_dir=FIGURE_DIR):
    """
    Create trajectory plot for a single embryo across all metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Full analysis dataframe
    embryo_id : str
        Embryo ID to plot
    metrics : list of str, optional
        Which metrics to plot. Default: METRICS
    save_dir : Path
        Directory to save figure

    Returns
    -------
    Path
        Path to saved figure
    """
    if metrics is None:
        metrics = METRICS

    # Sort by predicted_stage_hpf (developmental stage) for proper temporal ordering
    embryo_data = df[df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')

    if len(embryo_data) < 2:
        return None

    genotype = embryo_data['genotype'].iloc[0]
    genotype_short = get_genotype_short_name(genotype)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(
            embryo_data['predicted_stage_hpf'],
            embryo_data[metric],
            'o-',
            linewidth=2,
            markersize=6,
            color=get_genotype_color(genotype),
            label=genotype_short
        )

        ax.set_xlabel('Developmental Stage (hpf)', fontsize=11)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_title(f'{embryo_id} - {METRIC_LABELS.get(metric, metric)}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=10)

    plt.suptitle(f'Curvature Trajectory - {embryo_id} ({genotype_short})', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = save_dir / f'trajectory_{embryo_id}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path


# ============================================================================
# Aggregate Genotype Comparisons
# ============================================================================

def plot_aggregate_trajectories(df, metrics=None, save_dir=FIGURE_DIR):
    """
    Create aggregate comparison plots showing all three genotypes.

    For each metric, plots:
    - Individual embryo trajectories (transparent lines)
    - Mean trajectory ± SEM (bold line with error band)

    Parameters
    ----------
    df : pd.DataFrame
        Full analysis dataframe
    metrics : list of str, optional
    save_dir : Path

    Returns
    -------
    list of Path
        Paths to saved figures
    """
    if metrics is None:
        metrics = METRICS

    saved_paths = []

    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        genotypes = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']
        genotype_labels = ['Wildtype', 'Heterozygous', 'Homozygous']

        for ax, genotype, label in zip(axes, genotypes, genotype_labels):
            genotype_df = df[df['genotype'] == genotype]

            # Plot individual embryo trajectories
            for embryo_id in genotype_df['embryo_id'].unique():
                embryo_data = genotype_df[genotype_df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')

                ax.plot(
                    embryo_data['predicted_stage_hpf'],
                    embryo_data[metric],
                    'o-',
                    linewidth=1,
                    markersize=3,
                    alpha=0.15,
                    color=get_genotype_color(genotype)
                )

            # Plot mean trajectory with SEM
            grouped = genotype_df.groupby('predicted_stage_hpf')[metric].agg(['mean', 'sem'])
            grouped = grouped.reset_index()

            ax.plot(
                grouped['predicted_stage_hpf'],
                grouped['mean'],
                'o-',
                linewidth=3,
                markersize=8,
                color=get_genotype_color(genotype),
                label='Mean',
                zorder=10
            )

            ax.fill_between(
                grouped['predicted_stage_hpf'],
                grouped['mean'] - grouped['sem'],
                grouped['mean'] + grouped['sem'],
                alpha=0.2,
                color=get_genotype_color(genotype)
            )

            ax.set_xlabel('Developmental Stage (hpf)', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)

            if ax == axes[0]:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)

        plt.suptitle(f'Aggregate Trajectories - {METRIC_LABELS.get(metric, metric)}',
                     fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()

        save_path = save_dir / f'aggregate_trajectories_{metric}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        saved_paths.append(save_path)

    return saved_paths


# ============================================================================
# Statistical Comparisons
# ============================================================================

def create_summary_statistics(df, metrics=None):
    """
    Compute summary statistics for each genotype.

    Parameters
    ----------
    df : pd.DataFrame
    metrics : list of str, optional

    Returns
    -------
    pd.DataFrame
        Summary statistics (one row per genotype per metric)
    """
    if metrics is None:
        metrics = METRICS

    summary_data = []

    for genotype in ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']:
        genotype_df = df[df['genotype'] == genotype]
        genotype_short = get_genotype_short_name(genotype)

        for metric in metrics:
            values = genotype_df[metric].dropna()

            summary_data.append({
                'genotype': genotype_short,
                'metric': metric,
                'n_embryos': genotype_df['embryo_id'].nunique(),
                'n_samples': len(genotype_df),
                'mean': values.mean(),
                'std': values.std(),
                'sem': values.sem(),
                'median': values.median(),
                'min': values.min(),
                'max': values.max(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75)
            })

    return pd.DataFrame(summary_data)


def perform_statistical_tests(df, metrics=None):
    """
    Perform genotype comparisons using ANOVA and post-hoc tests.

    Parameters
    ----------
    df : pd.DataFrame
    metrics : list of str, optional

    Returns
    -------
    pd.DataFrame
        Test results: genotype_pair, metric, statistic, p_value
    """
    if metrics is None:
        metrics = METRICS

    test_results = []
    genotypes = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']

    for metric in metrics:
        # Get data per genotype
        groups = [df[df['genotype'] == g][metric].dropna().values for g in genotypes]

        # ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        test_results.append({
            'comparison': 'All Genotypes (ANOVA)',
            'metric': metric,
            'statistic': f_stat,
            'p_value': p_value,
            'test': 'ANOVA'
        })

        # Pairwise comparisons
        for i in range(len(genotypes)):
            for j in range(i + 1, len(genotypes)):
                g1 = genotypes[i]
                g2 = genotypes[j]
                short1 = get_genotype_short_name(g1)
                short2 = get_genotype_short_name(g2)

                values1 = df[df['genotype'] == g1][metric].dropna().values
                values2 = df[df['genotype'] == g2][metric].dropna().values

                if len(values1) > 0 and len(values2) > 0:
                    t_stat, p_val = stats.ttest_ind(values1, values2)

                    test_results.append({
                        'comparison': f'{short1} vs {short2}',
                        'metric': metric,
                        'statistic': t_stat,
                        'p_value': p_val,
                        'test': "t-test",
                        'n1': len(values1),
                        'n2': len(values2)
                    })

    return pd.DataFrame(test_results)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("INDIVIDUAL TRAJECTORIES: CURVATURE OVER DEVELOPMENT")
    print("="*80)

    # Load data
    print("\nStep 1: Loading data...")
    df, metadata = get_analysis_dataframe()

    # Summary statistics
    print("\nStep 2: Computing summary statistics...")
    summary_df = create_summary_statistics(df)

    summary_file = TABLE_DIR / 'summary_statistics.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"  Saved to {summary_file}")

    print("\nSummary Statistics by Genotype:")
    print(summary_df.to_string(index=False))

    # Statistical tests
    print("\nStep 3: Performing statistical tests...")
    test_df = perform_statistical_tests(df)

    test_file = TABLE_DIR / 'statistical_tests.csv'
    test_df.to_csv(test_file, index=False)
    print(f"  Saved to {test_file}")

    print("\nKey Statistical Tests:")
    for metric in METRICS:
        print(f"\n{METRIC_LABELS[metric]}:")
        metric_tests = test_df[test_df['metric'] == metric]
        for _, row in metric_tests.iterrows():
            print(f"  {row['comparison']:25s}: t={row['statistic']:7.3f}, p={row['p_value']:.4f}")

    # Individual embryo plots
    print("\nStep 4: Creating individual embryo trajectory plots...")
    n_embryos = df['embryo_id'].nunique()
    saved_count = 0

    for i, embryo_id in enumerate(df['embryo_id'].unique()):
        if i % 20 == 0:
            print(f"  Processing embryo {i+1}/{n_embryos}...")

        path = plot_individual_embryo_trajectories(df, embryo_id)
        if path:
            saved_count += 1

    print(f"  Saved {saved_count} individual embryo plots")

    # Aggregate plots
    print("\nStep 5: Creating aggregate genotype comparison plots...")
    aggregate_paths = plot_aggregate_trajectories(df)

    for path in aggregate_paths:
        print(f"  Saved to {path}")

    # Summary
    print("\n" + "="*80)
    print("TRAJECTORY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Individual trajectories: {FIGURE_DIR / 'trajectory_*.png'}")
    print(f"  Aggregate plots: {FIGURE_DIR / 'aggregate_trajectories_*.png'}")
    print(f"  Statistics: {TABLE_DIR / 'summary_statistics.csv'}")
    print(f"  Tests: {TABLE_DIR / 'statistical_tests.csv'}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
