#!/usr/bin/env python3
"""
CEP290 Cross-Genotype Trajectory Prediction with LOEO Cross-Validation

Tests if embeddings at time i can predict morphological distance at time i+k,
separately for each CEP290 genotype (WT, Het, Homo).

Each model is tested on all three genotypes:
- Diagonal (WT→WT, Het→Het, Homo→Homo): LOEO cross-validation
- Off-diagonal (e.g., WT→Homo): Cross-genotype testing with full model

Penetrance classification: Compare WT model vs Homo model performance on Homo embryos.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add 20251016 utils to path for data loading
sys.path.insert(0, str(Path(__file__).parent.parent / '20251016'))

from utils.data_loading import load_experiments
from utils.binning import bin_embryos_by_time
import config

from penetrance_analysis.trajectory_loeo import (
    create_trajectory_pairs,
    train_loeo_and_full_model,
    test_model_on_genotype,
    compute_overall_metrics,
    compute_per_embryo_metrics,
    compute_error_vs_horizon,
    classify_penetrance_dual_model
)

from penetrance_analysis.trajectory_viz_loeo import (
    create_aggregated_heatmap,
    create_per_embryo_heatmaps,
    compute_r2_per_cell,
    plot_aggregated_heatmap,
    plot_per_embryo_grid,
    plot_error_vs_horizon,
    plot_temporal_breakdown,
    plot_per_embryo_error_distribution,
    plot_model_comparison_3x3,
    plot_penetrance_classification
)

# ============================================================================
# Configuration
# ============================================================================

GENE = 'cep290'

GENOTYPES = {
    'wt': 'cep290_wildtype',
    'het': 'cep290_heterozygous',
    'homo': 'cep290_homozygous'
}

MIN_TIME = None  # No minimum time filter - use all available data
MIN_DELTA_T = 2  # Minimum prediction horizon (hpf)
MODEL_TYPE = 'random_forest'
N_ESTIMATORS = 100
MAX_DEPTH = None
ERROR_RATIO_THRESHOLD = 1.5

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'penetrance'
PLOT_DIR = BASE_DIR / 'plots' / 'penetrance'

OUTPUT_DATA_DIR = DATA_DIR / 'trajectory_loeo' / GENE
OUTPUT_PLOT_DIR = PLOT_DIR / 'trajectory_loeo' / GENE

# Create output directories
for model_geno in ['wt', 'het', 'homo']:
    (OUTPUT_DATA_DIR / f'{model_geno}_model').mkdir(parents=True, exist_ok=True)
    for test_geno in ['wt', 'het', 'homo']:
        (OUTPUT_PLOT_DIR / f'{model_geno}_model' / f'tested_on_{test_geno}').mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def load_binned_data(genotype_name: str) -> pd.DataFrame:
    """Load binned embedding data for a genotype."""
    genotype_family = genotype_name.split('_')[0]  # 'cep290' or 'tmem67'

    # Load raw experiments
    if genotype_family == 'cep290':
        experiments = config.CEP290_EXPERIMENTS
    elif genotype_family == 'tmem67':
        experiments = config.TMEM67_EXPERIMENTS
    else:
        raise ValueError(f"Unknown gene family: {genotype_family}")

    print(f"\n  Loading experiments for {genotype_family}: {experiments}")

    df_raw = load_experiments(
        experiment_ids=experiments,
        build_dir=config.BUILD06_DIR,
        verbose=False
    )

    # Filter to specific genotype
    df_raw = df_raw[df_raw['genotype'] == genotype_name].copy()

    print(f"  Loaded {len(df_raw)} raw timepoints for {genotype_name}")
    print(f"    Unique embryos: {df_raw['embryo_id'].nunique()}")

    # Bin by time
    df_binned = bin_embryos_by_time(df_raw, bin_width=2.0)

    # Filter to time range (if specified)
    if MIN_TIME is not None:
        df_binned = df_binned[df_binned['time_bin'] >= MIN_TIME].copy()
        print(f"  Binned to {len(df_binned)} timepoints (>= {MIN_TIME} hpf)")
    else:
        print(f"  Binned to {len(df_binned)} timepoints (all times)")

    print(f"    Unique embryos: {df_binned['embryo_id'].nunique()}")
    print(f"    Time range: {df_binned['time_bin'].min():.1f} - {df_binned['time_bin'].max():.1f} hpf")

    # Rename z_mu_b columns to embedding_dim
    z_cols = [col for col in df_binned.columns if col.startswith('z_mu_b_') and col.endswith('_binned')]

    if len(z_cols) == 0:
        raise ValueError(f"No embedding columns found in binned data")

    # Rename columns
    rename_dict = {}
    for i, z_col in enumerate(sorted(z_cols)):
        rename_dict[z_col] = f'embedding_dim_{i}'

    df_binned = df_binned.rename(columns=rename_dict)

    print(f"    Embedding dimensions: {len(rename_dict)}")

    # Load distances and merge
    distances_file = DATA_DIR / f'{genotype_family}_distances.csv'
    if distances_file.exists():
        df_distances = pd.read_csv(distances_file)
        df_distances = df_distances[df_distances['genotype'] == genotype_name].copy()

        # Merge on embryo_id and time_bin
        df_binned = df_binned.merge(
            df_distances[['embryo_id', 'time_bin', 'euclidean_distance']],
            on=['embryo_id', 'time_bin'],
            how='left'
        )

        # Rename to distance_from_wt for consistency
        df_binned = df_binned.rename(columns={'euclidean_distance': 'distance_from_wt'})

        # Drop rows with NaN distances
        n_before = len(df_binned)
        df_binned = df_binned.dropna(subset=['distance_from_wt']).copy()
        n_after = len(df_binned)

        if n_after < n_before:
            print(f"    Dropped {n_before - n_after} timepoints with missing distances")
            print(f"    Remaining: {n_after} timepoints from {df_binned['embryo_id'].nunique()} embryos")
    else:
        print(f"    Warning: No distances file found at {distances_file}")
        print(f"    Computing distances on the fly...")
        # Would need WT reference here - skip for now
        raise FileNotFoundError(f"Distances file required: {distances_file}")

    return df_binned


def generate_all_plots(
    predictions: pd.DataFrame,
    model_geno: str,
    test_geno: str,
    output_dir: Path
):
    """Generate all visualization plots for one model-test combination."""

    print(f"\n  Generating visualizations...")

    model_name = f'{GENE.upper()} {model_geno.upper()} model'
    test_genotype = GENOTYPES[test_geno]

    # 1. Aggregated heatmaps (3 versions: abs error, rel error, R²)
    print(f"    Creating aggregated heatmaps...")

    for metric in ['absolute_error', 'relative_error']:
        heatmap = create_aggregated_heatmap(predictions, metric)
        fig = plot_aggregated_heatmap(
            heatmap, GENE, model_name, test_genotype, metric,
            save_path=output_dir / f'aggregated_heatmap_{metric}.png'
        )
        plt.close(fig)

    # R² heatmap
    heatmap_r2 = compute_r2_per_cell(predictions)
    fig = plot_aggregated_heatmap(
        heatmap_r2, GENE, model_name, test_genotype, 'r2',
        cmap='RdYlGn',
        save_path=output_dir / 'aggregated_heatmap_r2.png'
    )
    plt.close(fig)

    # 2. Per-embryo grid
    print(f"    Creating per-embryo grid...")
    embryo_heatmaps = create_per_embryo_heatmaps(predictions, 'absolute_error')
    fig = plot_per_embryo_grid(
        embryo_heatmaps, GENE, model_name, test_genotype, 'absolute_error',
        save_path=output_dir / 'per_embryo_grid.png'
    )
    plt.close(fig)

    # 3. Error vs horizon
    print(f"    Creating error vs horizon plot...")
    error_vs_dt = compute_error_vs_horizon(predictions)
    fig = plot_error_vs_horizon(
        error_vs_dt, GENE, model_name, test_genotype,
        save_path=output_dir / 'error_vs_horizon.png'
    )
    plt.close(fig)

    # 4. Temporal breakdown
    print(f"    Creating temporal breakdown...")
    fig = plot_temporal_breakdown(
        predictions, GENE, model_name, test_genotype,
        save_path=output_dir / 'temporal_breakdown.png'
    )
    plt.close(fig)

    # 5. Per-embryo error distribution
    print(f"    Creating per-embryo error distribution...")
    per_embryo_metrics = compute_per_embryo_metrics(predictions)
    fig = plot_per_embryo_error_distribution(
        per_embryo_metrics, GENE, model_name, test_genotype,
        save_path=output_dir / 'per_embryo_error_distribution.png'
    )
    plt.close(fig)

    print(f"    Saved 7 plots to {output_dir}")


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CEP290 CROSS-GENOTYPE TRAJECTORY PREDICTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Gene: {GENE.upper()}")
    print(f"  Genotypes: {list(GENOTYPES.keys())}")
    print(f"  Minimum time: {'All data (no filter)' if MIN_TIME is None else f'>= {MIN_TIME} hpf'}")
    print(f"  Minimum delta_t: {MIN_DELTA_T} hpf")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Error ratio threshold: {ERROR_RATIO_THRESHOLD}")
    print(f"\nOutput:")
    print(f"  Data: {OUTPUT_DATA_DIR}")
    print(f"  Plots: {OUTPUT_PLOT_DIR}")

    # ------------------------------------------------------------------------
    # Step 1: Load data for all genotypes
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    genotype_data = {}
    for geno_key, genotype_name in GENOTYPES.items():
        try:
            df_binned = load_binned_data(genotype_name)
            genotype_data[geno_key] = {
                'name': genotype_name,
                'df_binned': df_binned
            }
        except Exception as e:
            print(f"\n✗ ERROR loading {genotype_name}: {e}")
            import traceback
            traceback.print_exc()
            return

    # ------------------------------------------------------------------------
    # Step 2: Create trajectory pairs
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: CREATING TRAJECTORY PAIRS")
    print("="*80)

    for geno_key in GENOTYPES.keys():
        print(f"\nCreating pairs for {GENOTYPES[geno_key]}...")
        df_pairs = create_trajectory_pairs(
            genotype_data[geno_key]['df_binned'],
            GENOTYPES[geno_key],
            min_delta_t=MIN_DELTA_T
        )
        genotype_data[geno_key]['pairs'] = df_pairs

        # Mark if genotype has no data
        if len(df_pairs) == 0:
            genotype_data[geno_key]['has_data'] = False
        else:
            genotype_data[geno_key]['has_data'] = True

    # ------------------------------------------------------------------------
    # Step 3: Train models (LOEO within each genotype)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: TRAINING MODELS")
    print("="*80)

    models = {}
    for geno_key in GENOTYPES.keys():
        if not genotype_data[geno_key]['has_data']:
            print(f"\n{'─'*80}")
            print(f"SKIPPING {GENE.upper()}_{geno_key.upper()} model - no data")
            print(f"{'─'*80}")
            continue

        print(f"\n{'─'*80}")
        print(f"Training {GENE.upper()}_{geno_key.upper()} model")
        print(f"{'─'*80}")

        result = train_loeo_and_full_model(
            df_pairs=genotype_data[geno_key]['pairs'],
            model_name=f'{GENE}_{geno_key}_model',
            model_type=MODEL_TYPE,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH
        )

        models[geno_key] = result

    # ------------------------------------------------------------------------
    # Step 4: Test each model on all 3 genotypes (9 combinations)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: TESTING MODELS (9 COMBINATIONS)")
    print("="*80)

    all_predictions = {}

    for model_geno in GENOTYPES.keys():
        for test_geno in GENOTYPES.keys():

            combo_key = f'{model_geno}_model_on_{test_geno}'
            print(f"\n{'─'*80}")
            print(f"{combo_key.upper()}")
            print(f"{'─'*80}")

            # Skip if either genotype has no data
            if not genotype_data[model_geno]['has_data']:
                print(f"  SKIPPING - {model_geno} has no model")
                continue

            if not genotype_data[test_geno]['has_data']:
                print(f"  SKIPPING - {test_geno} has no data to test on")
                continue

            if model_geno == test_geno:
                # Use LOEO predictions (diagonal)
                print(f"  Using LOEO predictions (within-genotype)")
                preds = models[model_geno]['loeo_predictions'].copy()

            else:
                # Cross-genotype test (off-diagonal)
                print(f"  Cross-genotype test (full model)")
                preds = test_model_on_genotype(
                    model=models[model_geno]['full_model'],
                    feature_cols=models[model_geno]['feature_cols'],
                    df_test_pairs=genotype_data[test_geno]['pairs'],
                    model_name=f'{GENE}_{model_geno}_model',
                    test_genotype=GENOTYPES[test_geno]
                )

            all_predictions[combo_key] = preds

            # Save predictions
            save_path = OUTPUT_DATA_DIR / f'{model_geno}_model' / f'predictions_on_{test_geno}.csv'
            preds.to_csv(save_path, index=False)
            print(f"  Saved predictions to {save_path}")

            # Generate visualizations
            plot_output_dir = OUTPUT_PLOT_DIR / f'{model_geno}_model' / f'tested_on_{test_geno}'
            generate_all_plots(preds, model_geno, test_geno, plot_output_dir)

    # ------------------------------------------------------------------------
    # Step 5: Penetrance classification (WT model vs Homo model on Homo)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: PENETRANCE CLASSIFICATION")
    print("="*80)

    penetrance = classify_penetrance_dual_model(
        wt_model_predictions=all_predictions['wt_model_on_homo'],
        homo_model_predictions=all_predictions['homo_model_on_homo'],
        error_ratio_threshold=ERROR_RATIO_THRESHOLD
    )

    # Save classification
    penetrance_path = OUTPUT_DATA_DIR / 'penetrance_classification.csv'
    penetrance.to_csv(penetrance_path, index=False)
    print(f"\n  Saved penetrance classification to {penetrance_path}")

    # Plot penetrance classification
    print(f"\n  Generating penetrance classification plots...")
    fig = plot_penetrance_classification(
        penetrance,
        GENE,
        save_path=OUTPUT_PLOT_DIR / 'penetrance_classification.png'
    )
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Step 6: Model comparison (3×3 grid)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 6: MODEL COMPARISON")
    print("="*80)

    print(f"\n  Generating 3×3 model comparison plot...")
    fig = plot_model_comparison_3x3(
        all_predictions,
        GENE,
        save_path=OUTPUT_PLOT_DIR / 'model_comparison_3x3.png'
    )
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Step 7: Save summary metrics
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 7: SAVING SUMMARY METRICS")
    print("="*80)

    summary_data = []
    for combo_key, preds in all_predictions.items():
        metrics = compute_overall_metrics(preds, combo_key)
        summary_data.append(metrics)

    summary_df = pd.DataFrame(summary_data)
    summary_path = OUTPUT_DATA_DIR / 'model_performance_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved summary metrics to {summary_path}")

    # Print summary table
    print(f"\n  Model Performance Summary:")
    print(f"  {'-'*80}")
    print(summary_df[['model_name', 'mean_abs_error', 'r2_overall', 'n_predictions']].to_string(index=False))

    # ------------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Data: {OUTPUT_DATA_DIR}")
    print(f"  Plots: {OUTPUT_PLOT_DIR}")
    print(f"\nKey outputs:")
    print(f"  - 9 model-test combinations (3 models × 3 test sets)")
    print(f"  - ~63 plots (7 per combination)")
    print(f"  - Penetrance classification: {penetrance_path}")
    print(f"  - Model comparison: {OUTPUT_PLOT_DIR / 'model_comparison_3x3.png'}")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
