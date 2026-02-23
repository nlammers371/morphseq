#!/usr/bin/env python3
"""
Compare WT vs Homo Full Time Matrix Results

Loads saved results from run_wt_full_time_matrix.py and run_homo_full_time_matrix.py
and creates side-by-side heatmap comparisons with shared color scales.

This allows direct visual comparison of:
- Which model predicts better at which timepoints
- How prediction accuracy differs between WT-trained vs Homo-trained models
- Genotype-specific differences in predictability
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'penetrance'
PLOT_DIR = BASE_DIR / 'plots' / 'penetrance'

WT_DATA_DIR = DATA_DIR / 'wt_full_time_matrix'
HOMO_DATA_DIR = DATA_DIR / 'homo_full_time_matrix'

OUTPUT_DATA_DIR = DATA_DIR / 'wt_vs_homo_comparison'
OUTPUT_PLOT_DIR = PLOT_DIR / 'wt_vs_homo_comparison'

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Test genotypes to compare
GENOTYPES = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']

# ============================================================================
# Load Results
# ============================================================================

def load_model_results(data_dir: Path, model_name: str) -> Dict[str, pd.DataFrame]:
    """Load saved full time matrix results for one model."""
    # The full time matrix scripts save all genotypes in one file
    file_path = data_dir / 'full_time_matrix_metrics.csv'

    if not file_path.exists():
        print(f"  ERROR: {file_path} not found")
        return {}

    print(f"  Loading {file_path}...")
    df = pd.read_csv(file_path)

    # Split by genotype
    results = {}
    for genotype in GENOTYPES:
        genotype_df = df[df['genotype'] == genotype].copy()
        if len(genotype_df) > 0:
            results[genotype] = genotype_df
            print(f"    {genotype}: {len(genotype_df)} timepoint pairs")

    return results


# ============================================================================
# Create Comparison Matrices
# ============================================================================

def create_comparison_matrices(
    wt_results: Dict[str, pd.DataFrame],
    homo_results: Dict[str, pd.DataFrame],
    genotype: str
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create paired matrices for WT vs Homo models for one genotype.

    Returns
    -------
    dict
        {metric: (wt_matrix, homo_matrix)} for mae, r2, error_std
    """
    if genotype not in wt_results or genotype not in homo_results:
        print(f"  WARNING: Missing results for {genotype}")
        return {}

    wt_df = wt_results[genotype]
    homo_df = homo_results[genotype]

    # Get all unique start and target times from BOTH models
    all_start_times = sorted(set(wt_df['start_time'].unique()) | set(homo_df['start_time'].unique()))
    all_target_times = sorted(set(wt_df['target_time'].unique()) | set(homo_df['target_time'].unique()))

    matrices = {}

    for metric in ['mae', 'r2', 'error_std']:
        # Create matrices for both models
        wt_matrix = pd.DataFrame(
            index=all_start_times,
            columns=all_target_times,
            dtype=float
        )

        homo_matrix = pd.DataFrame(
            index=all_start_times,
            columns=all_target_times,
            dtype=float
        )

        # Fill WT matrix
        for _, row in wt_df.iterrows():
            wt_matrix.loc[row['start_time'], row['target_time']] = row[metric]

        # Fill Homo matrix
        for _, row in homo_df.iterrows():
            homo_matrix.loc[row['start_time'], row['target_time']] = row[metric]

        matrices[metric] = (wt_matrix, homo_matrix)

    return matrices


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_side_by_side_comparison(
    wt_results: Dict[str, pd.DataFrame],
    homo_results: Dict[str, pd.DataFrame],
    metric: str = 'mae',
    save_path: Optional[Path] = None
):
    """
    Create side-by-side heatmaps comparing WT vs Homo models for all genotypes.

    Parameters
    ----------
    wt_results : dict
        {genotype: df} from WT model
    homo_results : dict
        {genotype: df} from Homo model
    metric : str
        'mae', 'r2', or 'error_std'
    save_path : Path, optional
        Where to save the plot
    """
    n_genotypes = len(GENOTYPES)
    fig, axes = plt.subplots(n_genotypes, 2, figsize=(16, 5 * n_genotypes))

    if n_genotypes == 1:
        axes = axes.reshape(1, -1)

    # Determine shared color scale across ALL matrices
    all_values = []

    for genotype in GENOTYPES:
        if genotype in wt_results:
            all_values.extend(wt_results[genotype][metric].dropna().values)
        if genotype in homo_results:
            all_values.extend(homo_results[genotype][metric].dropna().values)

    if len(all_values) == 0:
        print(f"  WARNING: No data for metric {metric}")
        return None

    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)

    # Handle R² which can be negative
    if metric == 'r2':
        vmin = max(vmin, -1.0)
        vmax = min(vmax, 1.0)

    print(f"\n  {metric.upper()} color scale: {vmin:.3f} - {vmax:.3f}")

    # Create heatmaps for each genotype
    for idx, genotype in enumerate(GENOTYPES):
        genotype_label = genotype.replace('cep290_', '').replace('_', ' ').title()

        # Get matrices
        matrices = create_comparison_matrices(wt_results, homo_results, genotype)

        if metric not in matrices:
            continue

        wt_matrix, homo_matrix = matrices[metric]

        # WT Model (left column)
        ax_wt = axes[idx, 0]

        sns.heatmap(
            wt_matrix,
            ax=ax_wt,
            cmap='viridis' if metric != 'r2' else 'RdYlGn',
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_kws={'label': metric.upper()},
            linewidths=0.5,
            linecolor='white'
        )

        ax_wt.set_title(f'WT Model → {genotype_label}', fontsize=12, fontweight='bold')
        ax_wt.set_xlabel('Target Time (hpf)', fontsize=10)
        ax_wt.set_ylabel('Start Time (hpf)', fontsize=10)
        ax_wt.set_xticklabels(ax_wt.get_xticklabels(), rotation=45, ha='right')
        ax_wt.set_yticklabels(ax_wt.get_yticklabels(), rotation=0)

        # Homo Model (right column)
        ax_homo = axes[idx, 1]

        sns.heatmap(
            homo_matrix,
            ax=ax_homo,
            cmap='viridis' if metric != 'r2' else 'RdYlGn',
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_kws={'label': metric.upper()},
            linewidths=0.5,
            linecolor='white'
        )

        ax_homo.set_title(f'Homo Model → {genotype_label}', fontsize=12, fontweight='bold')
        ax_homo.set_xlabel('Target Time (hpf)', fontsize=10)
        ax_homo.set_ylabel('Start Time (hpf)', fontsize=10)
        ax_homo.set_xticklabels(ax_homo.get_xticklabels(), rotation=45, ha='right')
        ax_homo.set_yticklabels(ax_homo.get_yticklabels(), rotation=0)

    metric_labels = {
        'mae': 'Mean Absolute Error',
        'r2': 'R² Score',
        'error_std': 'Error Std Deviation'
    }

    plt.suptitle(
        f'WT vs Homo Model Comparison: {metric_labels[metric]}',
        fontsize=16,
        fontweight='bold',
        y=0.998
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    return fig


def plot_difference_heatmaps(
    wt_results: Dict[str, pd.DataFrame],
    homo_results: Dict[str, pd.DataFrame],
    metric: str = 'mae',
    save_path: Optional[Path] = None
):
    """
    Create heatmaps showing the DIFFERENCE between WT and Homo models.

    Positive values = Homo model has higher metric value
    Negative values = WT model has higher metric value

    For MAE/error_std: Negative is better (WT predicts better)
    For R²: Positive is better (Homo predicts better)
    """
    n_genotypes = len(GENOTYPES)
    fig, axes = plt.subplots(1, n_genotypes, figsize=(6 * n_genotypes, 5))

    if n_genotypes == 1:
        axes = [axes]

    # Determine shared color scale for differences
    all_diffs = []

    for genotype in GENOTYPES:
        matrices = create_comparison_matrices(wt_results, homo_results, genotype)

        if metric not in matrices:
            continue

        wt_matrix, homo_matrix = matrices[metric]
        diff_matrix = homo_matrix - wt_matrix
        all_diffs.extend(diff_matrix.values.flatten())

    all_diffs = [x for x in all_diffs if not np.isnan(x)]

    if len(all_diffs) == 0:
        print(f"  WARNING: No difference data for metric {metric}")
        return None

    # Use symmetric scale around zero
    abs_max = max(abs(np.nanmin(all_diffs)), abs(np.nanmax(all_diffs)))
    vmin = -abs_max
    vmax = abs_max

    print(f"\n  {metric.upper()} difference scale: {vmin:.3f} - {vmax:.3f}")

    # Create difference heatmaps
    for idx, genotype in enumerate(GENOTYPES):
        genotype_label = genotype.replace('cep290_', '').replace('_', ' ').title()

        matrices = create_comparison_matrices(wt_results, homo_results, genotype)

        if metric not in matrices:
            continue

        wt_matrix, homo_matrix = matrices[metric]
        diff_matrix = homo_matrix - wt_matrix

        ax = axes[idx]

        sns.heatmap(
            diff_matrix,
            ax=ax,
            cmap='RdBu_r',  # Red = Homo higher, Blue = WT higher
            vmin=vmin,
            vmax=vmax,
            center=0,
            cbar=True,
            cbar_kws={'label': f'Δ {metric.upper()} (Homo - WT)'},
            linewidths=0.5,
            linecolor='white'
        )

        ax.set_title(f'{genotype_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Time (hpf)', fontsize=10)
        ax.set_ylabel('Start Time (hpf)', fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    metric_labels = {
        'mae': 'Mean Absolute Error',
        'r2': 'R² Score',
        'error_std': 'Error Std Deviation'
    }

    interpretation = {
        'mae': '(Red = Homo worse, Blue = WT worse)',
        'r2': '(Red = Homo better, Blue = WT better)',
        'error_std': '(Red = Homo more variable, Blue = WT more variable)'
    }

    plt.suptitle(
        f'Model Difference: {metric_labels[metric]} {interpretation[metric]}',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    return fig


def create_summary_statistics(
    wt_results: Dict[str, pd.DataFrame],
    homo_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create summary statistics comparing WT vs Homo models.

    Returns
    -------
    pd.DataFrame
        Summary statistics for each genotype
    """
    summary_data = []

    for genotype in GENOTYPES:
        if genotype not in wt_results or genotype not in homo_results:
            continue

        wt_df = wt_results[genotype]
        homo_df = homo_results[genotype]

        for metric in ['mae', 'r2', 'error_std']:
            wt_mean = wt_df[metric].mean()
            wt_std = wt_df[metric].std()

            homo_mean = homo_df[metric].mean()
            homo_std = homo_df[metric].std()

            summary_data.append({
                'genotype': genotype,
                'metric': metric,
                'wt_mean': wt_mean,
                'wt_std': wt_std,
                'homo_mean': homo_mean,
                'homo_std': homo_std,
                'diff_mean': homo_mean - wt_mean,
                'n_timepoints_wt': len(wt_df),
                'n_timepoints_homo': len(homo_df)
            })

    return pd.DataFrame(summary_data)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("COMPARING WT vs HOMO FULL TIME MATRIX RESULTS")
    print("="*80)

    # ------------------------------------------------------------------------
    # Step 1: Load results
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: LOADING SAVED RESULTS")
    print("="*80)

    print("\nLoading WT model results...")
    wt_results = load_model_results(WT_DATA_DIR, 'WT')

    print("\nLoading Homo model results...")
    homo_results = load_model_results(HOMO_DATA_DIR, 'Homo')

    if len(wt_results) == 0 or len(homo_results) == 0:
        print("\nERROR: Missing results files. Make sure you've run both:")
        print("  - run_wt_full_time_matrix.py")
        print("  - run_homo_full_time_matrix.py")
        return

    # ------------------------------------------------------------------------
    # Step 2: Create summary statistics
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: COMPUTING SUMMARY STATISTICS")
    print("="*80)

    summary_df = create_summary_statistics(wt_results, homo_results)

    summary_file = OUTPUT_DATA_DIR / 'wt_vs_homo_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary to {summary_file}")

    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------------
    # Step 3: Create side-by-side comparisons
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: CREATING SIDE-BY-SIDE COMPARISONS")
    print("="*80)

    for metric in ['mae', 'r2', 'error_std']:
        print(f"\nCreating {metric.upper()} comparison...")

        save_path = OUTPUT_PLOT_DIR / f'side_by_side_{metric}.png'
        plot_side_by_side_comparison(
            wt_results,
            homo_results,
            metric=metric,
            save_path=save_path
        )

    # ------------------------------------------------------------------------
    # Step 4: Create difference heatmaps
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: CREATING DIFFERENCE HEATMAPS")
    print("="*80)

    for metric in ['mae', 'r2', 'error_std']:
        print(f"\nCreating {metric.upper()} difference heatmap...")

        save_path = OUTPUT_PLOT_DIR / f'difference_{metric}.png'
        plot_difference_heatmaps(
            wt_results,
            homo_results,
            metric=metric,
            save_path=save_path
        )

    # ------------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nKey Insights:")
    print(f"  - Side-by-side comparisons show absolute performance")
    print(f"  - Difference heatmaps show which model predicts better where")
    print(f"  - All plots use SHARED color scales for direct comparison")
    print(f"\nOutputs saved to:")
    print(f"  Data: {OUTPUT_DATA_DIR}")
    print(f"  Plots: {OUTPUT_PLOT_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
