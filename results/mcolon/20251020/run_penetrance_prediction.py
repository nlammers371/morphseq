"""
Trajectory Prediction for Penetrance Classification

Uses dual-model approach:
- Model 2a: Train on homozygous data → learn "mutant trajectory"
- Model 2b: Train on WT data → learn "normal trajectory"
- Compare prediction errors to classify penetrance

Key insight: Penetrant embryos follow mutant trajectory, non-penetrant follow WT trajectory.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from penetrance_analysis.trajectory_prediction import (
    prepare_trajectory_data,
    train_trajectory_model,
    predict_all_trajectories,
    cross_validate_trajectory_model,
    classify_penetrance_by_dual_models,
    compute_per_embryo_prediction_metrics
)

from penetrance_analysis.visualization import (
    plot_dual_prediction_heatmaps,
    plot_trajectory_examples,
    plot_prediction_error_scatter,
    plot_penetrance_distribution
)

# Configuration
GENOTYPES = ['cep290_homozygous', 'tmem67_homozygous']
PREDICTION_HORIZONS = [2, 4, 6, 8]  # hpf ahead to predict
MIN_TIME = 30.0  # Filter to post-onset timepoints
MODEL_TYPE = 'random_forest'
ERROR_RATIO_THRESHOLD = 1.5

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'penetrance'
PLOT_DIR = BASE_DIR / 'plots' / 'penetrance'

# Output directories
TRAJECTORY_DATA_DIR = DATA_DIR / 'trajectory'
TRAJECTORY_PLOT_DIR = PLOT_DIR / 'trajectory'
TRAJECTORY_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAJECTORY_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_binned_data(genotype: str) -> pd.DataFrame:
    """Load binned data for a genotype."""
    genotype_family = genotype.split('_')[0]  # Extract 'cep290' or 'tmem67'

    # Load distances
    distances_file = DATA_DIR / f'{genotype_family}_distances.csv'
    df_distances = pd.read_csv(distances_file)

    # Load predictions
    predictions_file = DATA_DIR / f'{genotype_family}_predictions.csv'
    df_predictions = pd.read_csv(predictions_file)

    # Merge (rename pred_prob_mutant to predicted_probability for consistency)
    df_pred_subset = df_predictions[['embryo_id', 'time_bin', 'pred_prob_mutant']].copy()
    df_pred_subset.rename(columns={'pred_prob_mutant': 'predicted_probability'}, inplace=True)

    df_merged = df_distances.merge(
        df_pred_subset,
        on=['embryo_id', 'time_bin'],
        how='left'
    )

    # Filter to genotype and time range
    df_merged = df_merged[df_merged['genotype'] == genotype].copy()
    df_merged = df_merged[df_merged['time_bin'] >= MIN_TIME].copy()

    print(f"Loaded {len(df_merged)} timepoints for {genotype} (≥{MIN_TIME} hpf)")
    print(f"  Unique embryos: {df_merged['embryo_id'].nunique()}")
    print(f"  Time range: {df_merged['time_bin'].min():.1f} - {df_merged['time_bin'].max():.1f} hpf")

    return df_merged


def run_trajectory_prediction_for_genotype(genotype: str):
    """Run full trajectory prediction workflow for one genotype."""

    print(f"\n{'='*80}")
    print(f"TRAJECTORY PREDICTION: {genotype.upper()}")
    print(f"{'='*80}\n")

    genotype_family = genotype.split('_')[0]

    # -------------------------------------------------------------------
    # Step 1: Load binned data for both genotypes
    # -------------------------------------------------------------------
    print("Step 1: Loading binned data...")

    # Load homozygous data
    df_homo = load_binned_data(genotype)

    # Load WT data (need to construct genotype name)
    wt_genotype = f"{genotype_family}_wildtype"
    df_wt = load_binned_data(wt_genotype)

    # -------------------------------------------------------------------
    # Step 2: Prepare trajectory training data
    # -------------------------------------------------------------------
    print("\nStep 2: Preparing trajectory training data...")

    print(f"\n  Preparing HOMOZYGOUS training data...")
    df_homo_train = prepare_trajectory_data(
        df_binned=df_homo,
        genotype=genotype,
        prediction_horizons=PREDICTION_HORIZONS,
        min_timepoints=3
    )

    print(f"\n  Preparing WT training data...")
    df_wt_train = prepare_trajectory_data(
        df_binned=df_wt,
        genotype=wt_genotype,
        prediction_horizons=PREDICTION_HORIZONS,
        min_timepoints=3
    )

    # Save training data
    df_homo_train.to_csv(
        TRAJECTORY_DATA_DIR / f'{genotype_family}_homo_training_data.csv',
        index=False
    )
    df_wt_train.to_csv(
        TRAJECTORY_DATA_DIR / f'{genotype_family}_wt_training_data.csv',
        index=False
    )

    # -------------------------------------------------------------------
    # Step 3: Train models
    # -------------------------------------------------------------------
    print("\nStep 3: Training trajectory models...")

    print(f"\n  Training Model 2a: HOMOZYGOUS model...")
    homo_model, homo_features, homo_train_metrics = train_trajectory_model(
        df_train=df_homo_train,
        model_type=MODEL_TYPE
    )

    print(f"\n  Training Model 2b: WT model...")
    wt_model, wt_features, wt_train_metrics = train_trajectory_model(
        df_train=df_wt_train,
        model_type=MODEL_TYPE
    )

    # Cross-validate
    print(f"\n  Cross-validating HOMOZYGOUS model...")
    homo_cv_metrics = cross_validate_trajectory_model(
        df_train=df_homo_train,
        model_type=MODEL_TYPE,
        cv_method='embryo_loo'
    )

    print(f"\n  Cross-validating WT model...")
    wt_cv_metrics = cross_validate_trajectory_model(
        df_train=df_wt_train,
        model_type=MODEL_TYPE,
        cv_method='embryo_loo'
    )

    # -------------------------------------------------------------------
    # Step 4: Make predictions on homozygous embryos with BOTH models
    # -------------------------------------------------------------------
    print("\nStep 4: Making predictions...")

    print(f"\n  Predicting homozygous embryos with HOMO model...")
    homo_predictions = predict_all_trajectories(
        df_data=df_homo_train,
        model=homo_model,
        feature_cols=homo_features,
        prediction_horizons=PREDICTION_HORIZONS
    )

    print(f"\n  Predicting homozygous embryos with WT model...")
    wt_predictions = predict_all_trajectories(
        df_data=df_homo_train,  # Same data, different model
        model=wt_model,
        feature_cols=wt_features,
        prediction_horizons=PREDICTION_HORIZONS
    )

    # Save predictions
    homo_predictions.to_csv(
        TRAJECTORY_DATA_DIR / f'{genotype_family}_homo_model_predictions.csv',
        index=False
    )
    wt_predictions.to_csv(
        TRAJECTORY_DATA_DIR / f'{genotype_family}_wt_model_predictions.csv',
        index=False
    )

    # -------------------------------------------------------------------
    # Step 5: Classify penetrance
    # -------------------------------------------------------------------
    print("\nStep 5: Classifying penetrance...")

    classification = classify_penetrance_by_dual_models(
        homo_model_predictions=homo_predictions,
        wt_model_predictions=wt_predictions,
        error_ratio_threshold=ERROR_RATIO_THRESHOLD,
        min_predictions=3
    )

    # Save classification
    classification.to_csv(
        TRAJECTORY_DATA_DIR / f'{genotype_family}_penetrance_classification.csv',
        index=False
    )

    # Print summary
    print(f"\n  Classification Summary:")
    print(f"  {'-'*50}")
    penetrance_counts = classification['penetrance_status'].value_counts()
    total = len(classification)
    for status in ['penetrant', 'non-penetrant', 'intermediate']:
        count = penetrance_counts.get(status, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"    {status.capitalize():15s}: {count:3d} / {total:3d} ({pct:5.1f}%)")

    print(f"\n  Error Ratio Statistics:")
    print(f"  {'-'*50}")
    print(f"    Mean: {classification['error_ratio'].mean():.3f}")
    print(f"    Median: {classification['error_ratio'].median():.3f}")
    print(f"    Range: [{classification['error_ratio'].min():.3f}, {classification['error_ratio'].max():.3f}]")

    # -------------------------------------------------------------------
    # Step 6: Generate visualizations
    # -------------------------------------------------------------------
    print("\nStep 6: Generating visualizations...")

    # Plot 1: Dual prediction heatmaps
    print("  Creating dual prediction heatmaps...")
    fig1 = plot_dual_prediction_heatmaps(
        homo_predictions=homo_predictions,
        wt_predictions=wt_predictions,
        genotype=genotype_family
    )
    fig1.savefig(
        TRAJECTORY_PLOT_DIR / f'{genotype_family}_dual_heatmaps.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig1)

    # Plot 2: Trajectory examples
    print("  Creating trajectory examples...")
    fig2 = plot_trajectory_examples(
        df_binned=df_homo,
        homo_predictions=homo_predictions,
        wt_predictions=wt_predictions,
        classification=classification,
        genotype=genotype_family,
        n_examples=6
    )
    fig2.savefig(
        TRAJECTORY_PLOT_DIR / f'{genotype_family}_trajectory_examples.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig2)

    # Plot 3: Error scatter
    print("  Creating error scatter plot...")
    fig3 = plot_prediction_error_scatter(classification)
    fig3.savefig(
        TRAJECTORY_PLOT_DIR / f'{genotype_family}_error_scatter.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig3)

    # Plot 4: Penetrance distribution
    print("  Creating penetrance distribution...")
    fig4 = plot_penetrance_distribution(
        classification=classification,
        genotype=genotype_family
    )
    fig4.savefig(
        TRAJECTORY_PLOT_DIR / f'{genotype_family}_penetrance_distribution.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig4)

    print(f"\n✓ Completed trajectory prediction for {genotype}")

    return classification


def main():
    """Run trajectory prediction for all genotypes."""

    print("\n" + "="*80)
    print("TRAJECTORY PREDICTION FOR PENETRANCE CLASSIFICATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Genotypes: {GENOTYPES}")
    print(f"  Prediction horizons: {PREDICTION_HORIZONS} hpf")
    print(f"  Minimum time: ≥{MIN_TIME} hpf")
    print(f"  Model type: {MODEL_TYPE}")
    print(f"  Error ratio threshold: {ERROR_RATIO_THRESHOLD}")
    print(f"\nOutput directories:")
    print(f"  Data: {TRAJECTORY_DATA_DIR}")
    print(f"  Plots: {TRAJECTORY_PLOT_DIR}")

    # Run for each genotype
    results = {}
    for genotype in GENOTYPES:
        try:
            classification = run_trajectory_prediction_for_genotype(genotype)
            results[genotype] = classification
        except Exception as e:
            print(f"\n✗ ERROR processing {genotype}: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    if len(results) > 0:
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80 + "\n")

        for genotype, classification in results.items():
            genotype_family = genotype.split('_')[0]
            print(f"\n{genotype_family.upper()}:")
            print(f"  {'-'*50}")

            penetrance_counts = classification['penetrance_status'].value_counts()
            total = len(classification)

            for status in ['penetrant', 'non-penetrant', 'intermediate']:
                count = penetrance_counts.get(status, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"    {status.capitalize():15s}: {count:3d} ({pct:5.1f}%)")

        print("\n" + "="*80)
        print("All files saved to:")
        print(f"  Data: {TRAJECTORY_DATA_DIR}")
        print(f"  Plots: {TRAJECTORY_PLOT_DIR}")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
