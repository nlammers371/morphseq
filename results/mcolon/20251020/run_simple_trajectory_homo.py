#!/usr/bin/env python3
"""
Simple Trajectory Prediction: Model Comparison

Predicts future developmental trajectories from a single starting timepoint (30 hpf)
for CEP290 homozygous embryos.

NEW APPROACH: Trains SEPARATE model for EACH future timepoint to test how far ahead
we can predict from a 30 hpf snapshot. Uses parallel CPU processing for speed.

Goal: Identify which machine learning model best predicts at different horizons.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).parent.parent / '20251016'))

from utils.data_loading import load_experiments
from utils.binning import bin_embryos_by_time
import config

from penetrance_analysis.simple_trajectory import (
    create_start_time_pairs,
    train_all_horizons_parallel,
    compare_models_across_horizons,
    get_best_model_per_horizon
)

from penetrance_analysis.simple_viz import (
    plot_model_comparison_curves,
    plot_trajectory_examples,
    plot_model_performance_heatmap,
    plot_error_distributions,
    plot_model_ranking_table,
    plot_best_model_per_horizon
)

# ============================================================================
# Configuration
# ============================================================================

GENE = 'cep290'
GENOTYPE = 'cep290_homozygous'
START_TIME = 30.0  # hpf

# Models to test
MODELS_TO_TEST = [
    'linear',
    'ridge',
    'lasso',
    'random_forest',
    'gradient_boosting',
    # 'xgboost',  # Uncomment if XGBoost installed
    'svr',
    'mlp'
]

# Model hyperparameters (optional tuning)
MODEL_PARAMS = {
    'ridge': {'alpha': 1.0},
    'lasso': {'alpha': 0.1},
    'random_forest': {'n_estimators': 100, 'max_depth': None},
    'gradient_boosting': {'n_estimators': 100, 'max_depth': 3},
    'xgboost': {'n_estimators': 100, 'max_depth': 3},
    'svr': {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1},
    'mlp': {'hidden_layer_sizes': (100, 50), 'max_iter': 1000}
}

# Parallelization
N_JOBS = -1  # -1 = use all available CPUs

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'penetrance'
PLOT_DIR = BASE_DIR / 'plots' / 'penetrance'

OUTPUT_DATA_DIR = DATA_DIR / 'simple_trajectory'
OUTPUT_PLOT_DIR = PLOT_DIR / 'simple_trajectory'

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DATA_DIR / 'predictions').mkdir(exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def load_binned_data(genotype_name: str) -> pd.DataFrame:
    """Load and bin data for a genotype."""
    genotype_family = genotype_name.split('_')[0]

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

    print(f"  Binned to {len(df_binned)} timepoints")
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
        raise FileNotFoundError(f"Distances file required: {distances_file}")

    return df_binned


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("SIMPLE TRAJECTORY PREDICTION: PER-HORIZON MODEL COMPARISON")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Gene: {GENE.upper()}")
    print(f"  Genotype: {GENOTYPE}")
    print(f"  Starting time: {START_TIME} hpf")
    print(f"  Models to test: {len(MODELS_TO_TEST)}")
    print(f"    {', '.join(MODELS_TO_TEST)}")
    print(f"  Parallelization: n_jobs={N_JOBS} ({'all CPUs' if N_JOBS == -1 else f'{N_JOBS} CPUs'})")
    print(f"\nApproach:")
    print(f"  - Train SEPARATE model for EACH future timepoint")
    print(f"  - Test how far ahead we can predict from {START_TIME} hpf")
    print(f"  - NO TIME LIMITS - use all available data")
    print(f"\nOutput:")
    print(f"  Data: {OUTPUT_DATA_DIR}")
    print(f"  Plots: {OUTPUT_PLOT_DIR}")

    # ------------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    df_binned = load_binned_data(GENOTYPE)

    # ------------------------------------------------------------------------
    # Step 2: Create training pairs from start time
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: CREATING TRAINING PAIRS")
    print("="*80)

    df_pairs = create_start_time_pairs(df_binned, start_time=START_TIME)

    if len(df_pairs) == 0:
        print(f"\n✗ ERROR: No training pairs created!")
        print(f"  No embryos have data at start_time={START_TIME} hpf")
        return

    # Save pairs
    pairs_file = OUTPUT_DATA_DIR / 'homo_training_pairs.csv'
    df_pairs.to_csv(pairs_file, index=False)
    print(f"\nSaved training pairs to {pairs_file}")

    # Show horizon distribution
    print(f"\nPrediction horizons (hours ahead):")
    horizon_counts = df_pairs.groupby('delta_t').size().sort_index()
    for horizon, count in horizon_counts.items():
        print(f"  {horizon:5.1f} hours → {count:4d} pairs")

    # ------------------------------------------------------------------------
    # Step 3: Train models (parallel per horizon)
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: TRAINING MODELS (PARALLEL PER HORIZON)")
    print("="*80)
    print(f"\nTraining strategy:")
    print(f"  - For each model type: train {len(df_pairs['target_time'].unique())} separate models")
    print(f"  - Each model predicts ONE future timepoint")
    print(f"  - Parallelized across horizons (n_jobs={N_JOBS})")

    results_dict = {}
    successful_models = []

    for model_type in MODELS_TO_TEST:
        print(f"\n{'─'*80}")
        print(f"Model: {model_type.upper()}")
        print(f"{'─'*80}")

        try:
            # Get model params
            params = MODEL_PARAMS.get(model_type, {})

            # Train across all horizons in parallel
            results_by_horizon = train_all_horizons_parallel(
                df_pairs,
                model_type,
                n_jobs=N_JOBS,
                verbose=True,
                **params
            )

            results_dict[model_type] = results_by_horizon
            successful_models.append(model_type)

            # Save predictions for each horizon
            for target_time, result in results_by_horizon.items():
                pred_file = OUTPUT_DATA_DIR / 'predictions' / f'{model_type}_t{int(target_time)}_predictions.csv'
                result['predictions'].to_csv(pred_file, index=False)

            print(f"    ✓ Saved predictions for all horizons")

        except Exception as e:
            print(f"\n  ✗ ERROR training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(successful_models) == 0:
        print(f"\n✗ ERROR: No models trained successfully!")
        return

    print(f"\n✓ Successfully trained {len(successful_models)}/{len(MODELS_TO_TEST)} models")

    # ------------------------------------------------------------------------
    # Step 4: Compare models
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: COMPARING MODELS")
    print("="*80)

    # Compare across all horizons
    comparison_df = compare_models_across_horizons(results_dict)

    print(f"\nModel Performance Summary:")
    print(f"  Total model×horizon combinations: {len(comparison_df)}")

    # Show average per model
    avg_by_model = comparison_df.groupby('model_type').agg({
        'mae': 'mean',
        'r2': 'mean'
    }).sort_values('mae')

    print(f"\nAverage Performance (across all horizons):")
    for idx, (model_type, row) in enumerate(avg_by_model.iterrows(), 1):
        print(f"  {idx}. {model_type:20s} MAE={row['mae']:.4f}  R²={row['r2']:.4f}")

    # Save comparison
    comparison_file = OUTPUT_DATA_DIR / 'model_comparison_all_horizons.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nSaved full comparison to {comparison_file}")

    # Get best model per horizon
    best_per_horizon = get_best_model_per_horizon(comparison_df)

    print(f"\nBest model at each horizon:")
    for _, row in best_per_horizon.head(10).iterrows():
        print(f"  {row['horizon']:5.1f}h ahead: {row['best_model']:20s} (MAE={row['mae']:.4f})")

    # Save best per horizon
    best_file = OUTPUT_DATA_DIR / 'best_model_per_horizon.csv'
    best_per_horizon.to_csv(best_file, index=False)
    print(f"\nSaved best-per-horizon to {best_file}")

    # Get top 3 models overall
    top_models = avg_by_model.head(3).index.tolist()
    print(f"\nTop 3 models overall: {', '.join(top_models)}")

    # ------------------------------------------------------------------------
    # Step 5: Generate visualizations
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*80)

    # Plot 1: Model comparison curves (ERROR VS HORIZON)
    print(f"\n  Creating model comparison curves (error vs horizon)...")
    fig1 = plot_model_comparison_curves(
        results_dict,
        save_path=OUTPUT_PLOT_DIR / 'model_comparison_vs_horizon.png'
    )
    plt.close(fig1)

    # Plot 2: Trajectory examples (top 3 models)
    print(f"  Creating trajectory examples...")
    fig2 = plot_trajectory_examples(
        results_dict,
        top_models=top_models,
        n_examples=6,
        save_path=OUTPUT_PLOT_DIR / 'trajectory_examples_top3.png'
    )
    plt.close(fig2)

    # Plot 3: Performance heatmap (Model × Horizon)
    print(f"  Creating performance heatmap...")
    fig3 = plot_model_performance_heatmap(
        results_dict,
        save_path=OUTPUT_PLOT_DIR / 'model_performance_heatmap.png'
    )
    plt.close(fig3)

    # Plot 4: Error distributions
    print(f"  Creating error distributions...")
    fig4 = plot_error_distributions(
        results_dict,
        save_path=OUTPUT_PLOT_DIR / 'error_distributions.png'
    )
    plt.close(fig4)

    # Plot 5: Model ranking table
    print(f"  Creating model ranking table...")
    fig5 = plot_model_ranking_table(
        comparison_df,
        save_path=OUTPUT_PLOT_DIR / 'model_ranking_table.png'
    )
    plt.close(fig5)

    # Plot 6: Best model per horizon
    print(f"  Creating best-model-per-horizon plot...")
    fig6 = plot_best_model_per_horizon(
        best_per_horizon,
        save_path=OUTPUT_PLOT_DIR / 'best_model_per_horizon.png'
    )
    plt.close(fig6)

    # ------------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  Best model overall: {top_models[0]}")
    print(f"  Average MAE: {avg_by_model.iloc[0]['mae']:.4f}")
    print(f"  Average R²: {avg_by_model.iloc[0]['r2']:.4f}")
    print(f"\nAll outputs saved to:")
    print(f"  Data: {OUTPUT_DATA_DIR}")
    print(f"  Plots: {OUTPUT_PLOT_DIR}")
    print(f"\nKey files:")
    print(f"  - model_comparison_all_horizons.csv")
    print(f"  - best_model_per_horizon.csv")
    print(f"  - model_comparison_vs_horizon.png")
    print(f"  - trajectory_examples_top3.png")
    print(f"  - model_performance_heatmap.png")
    print(f"  - best_model_per_horizon.png")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
