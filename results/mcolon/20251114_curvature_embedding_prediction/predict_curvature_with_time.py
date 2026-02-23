#!/usr/bin/env python3
"""
Predict curvature from embeddings + time: Goal 1B Analysis

Same as predict_curvature_from_embeddings.py but includes predicted_stage_hpf
as a feature to see if time improves prediction.

Key Question:
    Does adding time information improve curvature prediction from embeddings?
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Setup paths
RESULTS_DIR = Path(__file__).parent
sys.path.insert(0, str(RESULTS_DIR / 'utils'))
sys.path.insert(0, str(RESULTS_DIR.parent.parent.parent / 'src'))

from utils import (
    prepare_features_and_target,
    validate_data_completeness,
    filter_by_genotype,
    train_regression_model_holdout,
    compute_regression_metrics,
    get_feature_importance,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_feature_importance,
    plot_metrics_table,
)

# Create output directories
FIGURE_DIR = RESULTS_DIR / 'outputs' / 'figures_with_time'
TABLE_DIR = RESULTS_DIR / 'outputs' / 'tables_with_time'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Configuration
# ============================================================================

DATA_FILE = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251106.csv')

# Target curvature metrics to predict
CURVATURE_METRICS = ['baseline_deviation_normalized']

# Embeddings to use
EMBEDDING_PATTERN = 'z_mu_b_'

# Time column to include as feature
TIME_COL = 'predicted_stage_hpf'

# Genotypes to analyze
GENOTYPES = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']

# Models to compare
MODELS_TO_TEST = {
    'ridge': {'alpha': 1.0},
}

# Test/train split (by embryos, not samples)
TEST_EMBRYO_FRACTION = 0.20
RANDOM_STATE = 42

# ============================================================================
# Data Loading
# ============================================================================

def load_and_prepare_data():
    """Load build06 data and prepare for analysis with time feature."""
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    print(f"\nLoading {DATA_FILE.name}...")
    df_raw = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # Filter to target genotypes
    print(f"\nFiltering to genotypes: {GENOTYPES}")
    df = filter_by_genotype(df_raw, GENOTYPES, verbose=True)

    # Identify embedding columns
    embedding_cols = sorted([col for col in df.columns if col.startswith(EMBEDDING_PATTERN)])

    if len(embedding_cols) == 0:
        raise ValueError(f"No embedding columns found with pattern '{EMBEDDING_PATTERN}'")

    # Add time column to features
    if TIME_COL not in df.columns:
        raise ValueError(f"Time column '{TIME_COL}' not found in data")

    feature_cols = embedding_cols + [TIME_COL]

    print(f"\nFound {len(embedding_cols)} embedding dimensions + 1 time feature")
    print(f"  Embeddings: {embedding_cols[0]} ... {embedding_cols[-1]}")
    print(f"  Time: {TIME_COL}")
    print(f"  Total features: {len(feature_cols)}")

    # Check curvature metrics
    missing_metrics = [m for m in CURVATURE_METRICS if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing curvature metrics: {missing_metrics}")

    print(f"\nCurvature metrics available:")
    for metric in CURVATURE_METRICS:
        n_valid = df[metric].notna().sum()
        print(f"  {metric}: {n_valid}/{len(df)} valid")

    return df, feature_cols, embedding_cols


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Run curvature prediction analysis with time feature."""
    print("\n" + "="*80)
    print("PREDICTING CURVATURE FROM EMBEDDINGS + TIME: GOAL 1B ANALYSIS")
    print("="*80)

    # Load and prepare data
    df, feature_cols, embedding_cols = load_and_prepare_data()

    # Ensure we have genotype info for stratified analysis
    if 'genotype' not in df.columns:
        raise ValueError("'genotype' column not found in data")

    # ========================================================================
    # Train models for each curvature metric
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: TRAINING MODELS (with time)")
    print("="*80)

    results_by_metric = {}

    for metric in CURVATURE_METRICS:
        print(f"\n{'─'*80}")
        print(f"Target: {metric}")
        print(f"{'─'*80}")

        # Prepare data for this metric - use our custom feature list
        df_metric = df.dropna(subset=feature_cols + [metric]).copy()

        print(f"\n  Features: {len(feature_cols)} ({len(embedding_cols)} embeddings + 1 time)")
        print(f"  Target: {metric}")
        print(f"  Samples after NaN removal: {len(df_metric)}")

        # Validate data
        validation = validate_data_completeness(
            df_metric,
            feature_cols,
            metric,
            verbose=True
        )

        if not validation['is_valid']:
            print(f"\n  ✗ Data validation failed for {metric}")
            continue

        # Train model with 80/20 holdout
        for model_type, model_params in MODELS_TO_TEST.items():
            result = train_regression_model_holdout(
                df_metric,
                feature_cols,
                metric,
                model_type=model_type,
                test_fraction=TEST_EMBRYO_FRACTION,
                scale_features=True,
                random_state=RANDOM_STATE,
                verbose=True,
                **model_params
            )

        results_by_metric[metric] = {
            'data': df_metric,
            'feature_cols': feature_cols,
            'embedding_cols': embedding_cols,
            'target_col': metric,
            'model': result
        }

    # ========================================================================
    # Save Results and Metrics
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: SAVING RESULTS")
    print("="*80)

    # Compile results
    results_list = []

    for metric, metric_results in results_by_metric.items():
        model_result = metric_results['model']
        pred_df = model_result['predictions'].copy()

        # Get genotypes for test embryos from original data
        test_embryos = model_result['split']['test_embryos']
        data = metric_results['data']
        genotypes_map = data[data['embryo_id'].isin(test_embryos)].groupby('embryo_id')['genotype'].first().to_dict()
        pred_df['genotype'] = pred_df['embryo_id'].map(genotypes_map)

        # Overall metrics
        overall_metrics = compute_regression_metrics(
            pred_df['actual'].values,
            pred_df['predicted'].values
        )

        results_list.append({
            'metric': metric,
            'split': 'overall',
            'n_samples': len(pred_df),
            'r2': overall_metrics['r2'],
            'mae': overall_metrics['mae']
        })

        # Per-genotype metrics
        for genotype in pred_df['genotype'].unique():
            subset = pred_df[pred_df['genotype'] == genotype]
            if len(subset) > 1:
                gen_metrics = compute_regression_metrics(
                    subset['actual'].values,
                    subset['predicted'].values
                )
                results_list.append({
                    'metric': metric,
                    'split': genotype,
                    'n_samples': len(subset),
                    'r2': gen_metrics['r2'],
                    'mae': gen_metrics['mae']
                })

    df_results = pd.DataFrame(results_list)

    # Save results table
    results_file = TABLE_DIR / 'model_results_with_time.csv'
    df_results.to_csv(results_file, index=False)
    print(f"\nSaved results: {results_file}")
    print(f"\n{'─'*60}")
    print(df_results.to_string(index=False))
    print(f"{'─'*60}")

    # ========================================================================
    # Generate Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*80)

    for metric, metric_results in results_by_metric.items():
        print(f"\n  Creating plots for {metric}...")

        model_result = metric_results['model']
        pred_df = model_result['predictions'].copy()

        # Get genotypes for test embryos
        test_embryos = model_result['split']['test_embryos']
        data = metric_results['data']
        genotypes_map = data[data['embryo_id'].isin(test_embryos)].groupby('embryo_id')['genotype'].first().to_dict()
        pred_df['genotype'] = pred_df['embryo_id'].map(genotypes_map)
        pred_df['residual'] = pred_df['actual'] - pred_df['predicted']

        # Predictions vs actual
        fig = plot_predictions_vs_actual(
            pred_df,
            title=f'{metric} (with time)',
            save_path=FIGURE_DIR / f'predictions_vs_actual_{metric}.png'
        )
        plt.close(fig)

        # Residuals
        fig = plot_residuals(
            pred_df,
            save_path=FIGURE_DIR / f'residuals_{metric}.png'
        )
        plt.close(fig)

        # Feature importance
        importance_df = get_feature_importance(
            model_result['model'],
            metric_results['feature_cols'],
            model_type='ridge'
        )

        if len(importance_df) > 0:
            # Top 15 feature importance bar plot
            fig = plot_feature_importance(
                importance_df,
                top_n=15,
                save_path=FIGURE_DIR / f'feature_importance_{metric}.png'
            )
            plt.close(fig)

            # Full histogram of all feature importances
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(importance_df['importance'].abs(), bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Absolute Feature Importance (Ridge Coefficients)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Distribution of Feature Importances (with time)\n{metric}', fontsize=14)
            ax.axvline(importance_df['importance'].abs().mean(), color='red', linestyle='--',
                      label=f'Mean: {importance_df["importance"].abs().mean():.4f}')
            ax.axvline(importance_df['importance'].abs().median(), color='orange', linestyle='--',
                      label=f'Median: {importance_df["importance"].abs().median():.4f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f'feature_importance_histogram_{metric}.png', dpi=150)
            plt.close(fig)
            print(f"    Saved: {FIGURE_DIR / f'feature_importance_histogram_{metric}.png'}")

            # Highlight time feature importance
            time_importance = importance_df[importance_df['feature'] == TIME_COL]['importance'].values
            if len(time_importance) > 0:
                time_rank = (importance_df['importance'].abs() >= abs(time_importance[0])).sum()
                print(f"    Time feature importance: {time_importance[0]:.4f} (rank {time_rank}/{len(importance_df)})")

            # Save feature importance
            importance_df.to_csv(
                TABLE_DIR / f'feature_importance_{metric}.csv',
                index=False
            )

    # ========================================================================
    # Summary Report
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    print(f"\n✓ Results saved to:")
    print(f"    Tables:  {TABLE_DIR}")
    print(f"    Figures: {FIGURE_DIR}")

    print(f"\n✓ Overall Results (with time feature):")
    for idx, (_, row) in enumerate(df_results[df_results['split'] == 'overall'].iterrows(), 1):
        print(f"  {idx}. {row['metric']:35s}: R²={row['r2']:.4f}, MAE={row['mae']:.6f}")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
