#!/usr/bin/env python3
"""
Thin wrapper CLI for comparing 3 model predictions using reusable horizon plot utilities.

This script orchestrates loading model results and creating comparison visualizations.
All actual logic is in analyze.difference_detection modules (horizon_plots, time_matrix).

Previous logic: results/mcolon/20251020/compare_3models_full_time_matrix.py
New approach: Import from analyze.difference_detection instead of duplicating code.

Run: python compare_3models_full_time_matrix_wrapper.py
"""

import sys
import os
from pathlib import Path

# Ensure src is in path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
from analyze.difference_detection import (
    load_time_matrix_results,
    build_metric_matrices,
    plot_horizon_grid,
    plot_best_condition_map,
)


# ============================================================================
# Configuration (can be parameterized later)
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'penetrance'
PLOT_DIR = BASE_DIR / 'plots' / 'penetrance'

MODEL_CONDITIONS = ['WT', 'Het', 'Homo']
TEST_GENOTYPES = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']

MODEL_TRAINING_GENOTYPE = {
    'WT': 'cep290_wildtype',
    'Het': 'cep290_heterozygous',
    'Homo': 'cep290_homozygous'
}

OUTPUT_DATA_DIR = DATA_DIR / '3model_comparison'
OUTPUT_PLOT_DIR = PLOT_DIR / '3model_comparison'

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Loading
# ============================================================================

def load_all_model_results() -> dict:
    """
    Load results from all three models using analyze.difference_detection utilities.

    Returns
    -------
    dict
        {model_name: {genotype: df}}
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING SAVED RESULTS FROM ALL MODELS")
    print("="*80)

    all_results = {}

    for model_name in MODEL_CONDITIONS:
        file_path = DATA_DIR / f'{model_name.lower()}_full_time_matrix' / 'full_time_matrix_metrics.csv'

        if not file_path.exists():
            print(f"  WARNING: {file_path} not found")
            all_results[model_name] = {}
            continue

        print(f"  Loading {model_name} model from {file_path}...")
        df = pd.read_csv(file_path)

        # Split by genotype
        model_results = {}
        for genotype in TEST_GENOTYPES:
            genotype_df = df[df['genotype'] == genotype].copy()
            if len(genotype_df) > 0:
                model_results[genotype] = genotype_df
                print(f"    {genotype}: {len(genotype_df)} timepoint pairs")

        all_results[model_name] = model_results

    return all_results


# ============================================================================
# Summary Statistics
# ============================================================================

def create_summary_statistics(all_results: dict) -> pd.DataFrame:
    """
    Create summary statistics comparing all models.

    Returns
    -------
    pd.DataFrame
        Summary statistics for each model-genotype combination
    """
    print("\n" + "="*80)
    print("STEP 2: COMPUTING SUMMARY STATISTICS")
    print("="*80)

    summary_data = []

    for model_name, model_results in all_results.items():
        for genotype in TEST_GENOTYPES:
            if genotype not in model_results:
                continue

            df = model_results[genotype]

            for metric in ['mae', 'r2', 'error_std']:
                is_loeo = (MODEL_TRAINING_GENOTYPE[model_name] == genotype)

                summary_data.append({
                    'model': model_name,
                    'test_genotype': genotype,
                    'metric': metric,
                    'mean': df[metric].mean(),
                    'std': df[metric].std(),
                    'median': df[metric].median(),
                    'min': df[metric].min(),
                    'max': df[metric].max(),
                    'n_timepoints': len(df),
                    'uses_loeo': is_loeo
                })

    summary_df = pd.DataFrame(summary_data)
    summary_file = OUTPUT_DATA_DIR / '3model_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary to {summary_file}")

    # Print key comparisons
    print("\nMean MAE by Model and Test Genotype:")
    mae_pivot = summary_df[summary_df['metric'] == 'mae'].pivot_table(
        index='model',
        columns='test_genotype',
        values='mean'
    )
    print(mae_pivot.to_string())

    print("\nMean R² by Model and Test Genotype:")
    r2_pivot = summary_df[summary_df['metric'] == 'r2'].pivot_table(
        index='model',
        columns='test_genotype',
        values='mean'
    )
    print(r2_pivot.to_string())

    return summary_df


# ============================================================================
# Matrix Creation and Plotting
# ============================================================================

def create_matrices_for_all_models(all_results: dict, metric: str = 'mae') -> dict:
    """
    Create matrices for all models and genotypes.

    Returns
    -------
    dict
        {model_name: {genotype: matrix_df}}
    """
    # Get union of all start and target times
    all_start_times = set()
    all_target_times = set()

    for model_results in all_results.values():
        for genotype_df in model_results.values():
            all_start_times.update(genotype_df['start_time'].unique())
            all_target_times.update(genotype_df['target_time'].unique())

    all_start_times = sorted(all_start_times)
    all_target_times = sorted(all_target_times)

    # Create matrices for each model-genotype combination
    matrices = {}

    for model_name, model_results in all_results.items():
        matrices[model_name] = {}

        for genotype in TEST_GENOTYPES:
            if genotype not in model_results:
                continue

            genotype_df = model_results[genotype]

            # Create matrix
            matrix = pd.DataFrame(
                index=all_start_times,
                columns=all_target_times,
                dtype=float
            )

            # Fill matrix
            for _, row in genotype_df.iterrows():
                matrix.loc[row['start_time'], row['target_time']] = row[metric]

            matrices[model_name][genotype] = matrix

    return matrices


def plot_all_metrics(all_results: dict):
    """
    Create horizon plots for MAE, R², and error_std.

    Uses analyze.difference_detection.horizon_plots under the hood.
    """
    print("\n" + "="*80)
    print("STEP 3: CREATING 3×3 COMPARISON PLOTS")
    print("="*80)

    for metric in ['mae', 'r2', 'error_std']:
        print(f"\nCreating {metric.upper()} comparison...")

        matrices = create_matrices_for_all_models(all_results, metric)
        save_path = OUTPUT_PLOT_DIR / f'3model_comparison_{metric}.png'

        # TODO: Use plot_horizon_grid when implemented
        # For now, keep original plotting logic as fallback
        _plot_3model_comparison_fallback(
            matrices,
            metric=metric,
            save_path=save_path
        )


def plot_best_model_heatmaps(all_results: dict):
    """
    Create "best model" heatmaps showing which model performs best per cell.

    Uses analyze.difference_detection.horizon_plots.plot_best_condition_map.
    """
    print("\n" + "="*80)
    print("STEP 4: CREATING 'BEST MODEL' HEATMAPS")
    print("="*80)

    for metric in ['mae', 'r2', 'error_std']:
        print(f"\nCreating best model heatmap for {metric.upper()}...")

        matrices = create_matrices_for_all_models(all_results, metric)
        save_path = OUTPUT_PLOT_DIR / f'best_model_{metric}.png'

        # TODO: Use plot_best_condition_map when implemented
        # For now, keep original plotting logic as fallback
        _plot_best_model_heatmap_fallback(
            matrices,
            metric=metric,
            save_path=save_path
        )


# ============================================================================
# Fallback plotting (original logic from compare_3models_full_time_matrix.py)
# Keep until horizon_plots module is fully implemented
# ============================================================================

def _plot_3model_comparison_fallback(matrices: dict, metric: str, save_path: Path):
    """Original plotting logic - temporary fallback."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    models = list(matrices.keys())
    n_models = len(models)
    n_genotypes = len(TEST_GENOTYPES)

    fig, axes = plt.subplots(n_models, n_genotypes, figsize=(6 * n_genotypes, 5 * n_models))

    # Determine shared color scale
    all_values = []
    for model_name in models:
        for genotype in TEST_GENOTYPES:
            if genotype not in matrices[model_name]:
                continue
            matrix = matrices[model_name][genotype]
            all_values.extend(matrix.values.flatten())

    all_values = [x for x in all_values if not np.isnan(x)]

    if len(all_values) == 0:
        print(f"  WARNING: No data for metric {metric}")
        return

    vmin = np.percentile(all_values, 5)
    vmax = np.percentile(all_values, 95)

    if metric == 'r2':
        vmin = max(vmin, -1.0)
        vmax = min(vmax, 1.0)

    print(f"\n  {metric.upper()} color scale: {vmin:.3f} - {vmax:.3f}")

    # Create heatmaps
    for model_idx, model_name in enumerate(models):
        for genotype_idx, genotype in enumerate(TEST_GENOTYPES):
            ax = axes[model_idx, genotype_idx]

            if genotype not in matrices[model_name]:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            matrix = matrices[model_name][genotype]

            sns.heatmap(
                matrix,
                ax=ax,
                cmap='viridis' if metric != 'r2' else 'RdYlGn',
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                cbar_kws={'label': metric.upper()},
                linewidths=0.5,
                linecolor='white'
            )

            genotype_label = genotype.replace('cep290_', '').replace('_', ' ').title()
            is_loeo = (MODEL_TRAINING_GENOTYPE[model_name] == genotype)
            loeo_marker = ' (LOEO)' if is_loeo else ''

            ax.set_title(f'{model_name} → {genotype_label}{loeo_marker}', fontweight='bold')

            if model_idx == n_models - 1:
                ax.set_xlabel('Target Time (hpf)', fontsize=9)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            if genotype_idx == 0:
                ax.set_ylabel('Start Time (hpf)', fontsize=9)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

            if is_loeo:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)

    metric_labels = {'mae': 'Mean Absolute Error', 'r2': 'R² Score', 'error_std': 'Error Std Deviation'}
    plt.suptitle(f'3-Model Comparison: {metric_labels[metric]}\n(Color scale: 5th-95th percentile clipping)', fontweight='bold')
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {save_path}")


def _plot_best_model_heatmap_fallback(matrices: dict, metric: str, save_path: Path):
    """Original best model plotting logic - temporary fallback."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    models = list(matrices.keys())
    n_genotypes = len(TEST_GENOTYPES)

    fig, axes = plt.subplots(1, n_genotypes, figsize=(6 * n_genotypes, 5))

    if n_genotypes == 1:
        axes = [axes]

    for genotype_idx, genotype in enumerate(TEST_GENOTYPES):
        ax = axes[genotype_idx]

        genotype_matrices = []
        available_models = []

        for model_name in models:
            if genotype in matrices[model_name]:
                genotype_matrices.append(matrices[model_name][genotype])
                available_models.append(model_name)

        if len(genotype_matrices) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        stacked = np.stack([m.values for m in genotype_matrices], axis=-1)
        best_model_idx = np.full(stacked.shape[:2], np.nan)
        has_data = ~np.all(np.isnan(stacked), axis=-1)

        if metric in ['mae', 'error_std']:
            best_model_idx[has_data] = np.nanargmin(stacked[has_data], axis=-1)
        else:
            best_model_idx[has_data] = np.nanargmax(stacked[has_data], axis=-1)

        best_matrix = pd.DataFrame(best_model_idx, index=genotype_matrices[0].index, columns=genotype_matrices[0].columns)

        sns.heatmap(best_matrix, ax=ax, cmap='Set2', vmin=0, vmax=len(available_models) - 1, cbar=True, linewidths=0.5, linecolor='white')

        colorbar = ax.collections[0].colorbar
        colorbar.set_ticklabels(available_models)

        genotype_label = genotype.replace('cep290_', '').replace('_', ' ').title()
        ax.set_title(f'{genotype_label}', fontweight='bold')
        ax.set_xlabel('Target Time (hpf)', fontsize=10)
        ax.set_ylabel('Start Time (hpf)', fontsize=10)

    better_text = 'lower is better' if metric in ['mae', 'error_std'] else 'higher is better'
    metric_labels = {'mae': 'Mean Absolute Error', 'r2': 'R² Score', 'error_std': 'Error Std Deviation'}
    plt.suptitle(f'Best Model per Timepoint: {metric_labels[metric]} ({better_text})', fontweight='bold')
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*80)
    print("COMPARING 3 MODELS (WT, HET, HOMO) FULL TIME MATRIX RESULTS")
    print("(Using reusable horizon_plots and time_matrix utilities)")
    print("="*80)

    all_results = load_all_model_results()

    if sum(len(r) for r in all_results.values()) == 0:
        print("\nERROR: Missing results files. Make sure you've run all three models.")
        return

    create_summary_statistics(all_results)
    plot_all_metrics(all_results)
    plot_best_model_heatmaps(all_results)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  Data: {OUTPUT_DATA_DIR}")
    print(f"  Plots: {OUTPUT_PLOT_DIR}")
    print("\nNote: Plotting logic will transition to analyze.difference_detection.horizon_plots")
    print("once the utility functions are fully implemented.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
