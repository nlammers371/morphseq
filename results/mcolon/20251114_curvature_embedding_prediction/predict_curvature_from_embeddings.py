#!/usr/bin/env python3
"""
Predict curvature from embeddings: Goal 1 Analysis

Quantifies how much of the curvature variation can be predicted from morphological
embeddings using ridge regression with 80/20 embryo holdout validation.

Key Question:
    How much can we predict curvature from embeddings?

Approach:
    - Load build06 df03 embeddings and curvature metrics
    - Split 80% embryos for train, 20% for test (no data leakage)
    - Train ridge regression and compare against gradient boosting
    - Extract feature importance to identify key embedding dimensions

Outputs:
    - Model performance metrics (R², MAE)
    - Predictions vs actual plots
    - Feature importance analysis
    - Per-genotype performance breakdown
    - Summary tables for all results
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
    plot_feature_importance,
)

# Create output directories
FIGURE_DIR = RESULTS_DIR / 'outputs' / 'figures'
TABLE_DIR = RESULTS_DIR / 'outputs' / 'tables'
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

# Genotypes to analyze
GENOTYPES = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']

# Models to compare (simplified to essential models)
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
    """Load build06 data and prepare for analysis."""
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

    print(f"\nFound {len(embedding_cols)} embedding dimensions")
    print(f"  First: {embedding_cols[0]}")
    print(f"  Last:  {embedding_cols[-1]}")

    # Check curvature metrics
    missing_metrics = [m for m in CURVATURE_METRICS if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing curvature metrics: {missing_metrics}")

    print(f"\nCurvature metrics available:")
    for metric in CURVATURE_METRICS:
        n_valid = df[metric].notna().sum()
        print(f"  {metric}: {n_valid}/{len(df)} valid")

    return df, embedding_cols


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Run curvature prediction analysis."""
    print("\n" + "="*80)
    print("PREDICTING CURVATURE FROM EMBEDDINGS: GOAL 1 ANALYSIS")
    print("="*80)

    # Load and prepare data
    df, embedding_cols = load_and_prepare_data()

    # Ensure we have genotype info for stratified analysis
    if 'genotype' not in df.columns:
        raise ValueError("'genotype' column not found in data")

    # ========================================================================
    # Train models for each curvature metric
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: TRAINING MODELS")
    print("="*80)

    results_by_metric = {}

    for metric in CURVATURE_METRICS:
        print(f"\n{'─'*80}")
        print(f"Target: {metric}")
        print(f"{'─'*80}")

        # Prepare data for this metric
        df_metric, feature_cols, target_col = prepare_features_and_target(
            df,
            feature_pattern=EMBEDDING_PATTERN,
            target_col=metric,
            remove_nan=True,
            verbose=True
        )

        # Validate data
        validation = validate_data_completeness(
            df_metric,
            feature_cols,
            target_col,
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
                target_col,
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
            'target_col': target_col,
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
    results_file = TABLE_DIR / 'model_results.csv'
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
            title=f'{metric}',
            save_path=FIGURE_DIR / f'predictions_vs_actual_{metric}.png'
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
            ax.set_title(f'Distribution of Feature Importances\n{metric}', fontsize=14)
            ax.axvline(importance_df['importance'].abs().mean(), color='red', linestyle='--',
                      label=f'Mean: {importance_df["importance"].abs().mean():.4f}')
            ax.axvline(importance_df['importance'].abs().median(), color='orange', linestyle='--',
                      label=f'Median: {importance_df["importance"].abs().median():.4f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f'feature_importance_histogram_{metric}.png', dpi=150)
            plt.close(fig)
            print(f"    Saved: {FIGURE_DIR / f'feature_importance_histogram_{metric}.png'}")

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

    print(f"\n✓ Overall Results:")
    for idx, (_, row) in enumerate(df_results[df_results['split'] == 'overall'].iterrows(), 1):
        print(f"  {idx}. {row['metric']:35s}: R²={row['r2']:.4f}, MAE={row['mae']:.6f}")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
