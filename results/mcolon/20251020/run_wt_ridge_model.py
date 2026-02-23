#!/usr/bin/env python3
"""
WT Ridge Model: Train on Wild-Type, Test on Het and Homo

Trains Ridge regression models on WT embryos only to learn the typical
developmental trajectory from 30 hpf.

Then tests on:
1. WT (held-out) - should predict well (same distribution)
2. Het - should predict well (heterozygous = normal phenotype)
3. Homo - deviations indicate penetrant phenotype

This answers: "Can we detect mutant phenotype as deviation from WT trajectory?"
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
    plot_best_model_per_horizon
)

# ============================================================================
# Configuration
# ============================================================================

GENE = 'cep290'
START_TIME = 30.0  # hpf
MODEL_TYPE = 'ridge'
MODEL_PARAMS = {'alpha': 1.0}

# Training genotype (WT only)
TRAIN_GENOTYPES = ['cep290_wildtype']

# Test genotypes
TEST_GENOTYPES = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']

# Parallelization
N_JOBS = -1

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'penetrance'
PLOT_DIR = BASE_DIR / 'plots' / 'penetrance'

OUTPUT_DATA_DIR = DATA_DIR / 'wt_ridge'
OUTPUT_PLOT_DIR = PLOT_DIR / 'wt_ridge'

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DATA_DIR / 'predictions').mkdir(exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def load_and_prepare_genotype_data(genotype_name: str) -> pd.DataFrame:
    """Load and prepare data for one genotype."""
    genotype_family = genotype_name.split('_')[0]

    # Load raw experiments
    if genotype_family == 'cep290':
        experiments = config.CEP290_EXPERIMENTS
    elif genotype_family == 'tmem67':
        experiments = config.TMEM67_EXPERIMENTS
    else:
        raise ValueError(f"Unknown gene family: {genotype_family}")

    print(f"  Loading {genotype_name}...")

    df_raw = load_experiments(
        experiment_ids=experiments,
        build_dir=config.BUILD06_DIR,
        verbose=False
    )

    # Filter to specific genotype
    df_raw = df_raw[df_raw['genotype'] == genotype_name].copy()

    print(f"    Raw timepoints: {len(df_raw)}, Embryos: {df_raw['embryo_id'].nunique()}")

    # Bin by time
    df_binned = bin_embryos_by_time(df_raw, bin_width=2.0)

    # Rename z_mu_b columns to embedding_dim
    z_cols = [col for col in df_binned.columns if col.startswith('z_mu_b_') and col.endswith('_binned')]

    if len(z_cols) == 0:
        raise ValueError(f"No embedding columns found for {genotype_name}")

    rename_dict = {}
    for i, z_col in enumerate(sorted(z_cols)):
        rename_dict[z_col] = f'embedding_dim_{i}'

    df_binned = df_binned.rename(columns=rename_dict)

    # Load distances
    distances_file = DATA_DIR / f'{genotype_family}_distances.csv'
    if distances_file.exists():
        df_distances = pd.read_csv(distances_file)
        df_distances = df_distances[df_distances['genotype'] == genotype_name].copy()

        df_binned = df_binned.merge(
            df_distances[['embryo_id', 'time_bin', 'euclidean_distance']],
            on=['embryo_id', 'time_bin'],
            how='left'
        )

        df_binned = df_binned.rename(columns={'euclidean_distance': 'distance_from_wt'})
        df_binned = df_binned.dropna(subset=['distance_from_wt']).copy()

    print(f"    Binned timepoints: {len(df_binned)}, Embryos: {df_binned['embryo_id'].nunique()}")
    print(f"    Time range: {df_binned['time_bin'].min():.1f} - {df_binned['time_bin'].max():.1f} hpf")

    return df_binned


def train_on_genotype_test_on_all(
    train_df: pd.DataFrame,
    test_dfs: dict,
    model_type: str = 'ridge',
    start_time: float = 30.0,
    n_jobs: int = -1,
    **model_params
):
    """
    Train model on one genotype dataset, test on multiple.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (e.g., WT+Het combined)
    test_dfs : dict
        {genotype_name: df} for testing
    model_type : str
        Model type
    start_time : float
        Starting timepoint
    n_jobs : int
        Parallel jobs
    **model_params
        Model hyperparameters

    Returns
    -------
    dict
        Results per test genotype
    """
    from penetrance_analysis.simple_trajectory import (
        train_model_loeo_single_time,
        get_model
    )
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from joblib import Parallel, delayed

    # Create training pairs
    print("\nCreating training pairs...")
    train_pairs = create_start_time_pairs(train_df, start_time=start_time)

    if len(train_pairs) == 0:
        raise ValueError("No training pairs created!")

    target_times = sorted(train_pairs['target_time'].unique())
    print(f"  Training on {len(train_pairs)} pairs across {len(target_times)} horizons")

    # For each target time, train ONE model on ALL training data
    print(f"\nTraining {len(target_times)} models (one per horizon) in parallel...")

    def train_full_model_at_time(target_time):
        """Train model on ALL training data for one target time."""
        pairs_at_time = train_pairs[train_pairs['target_time'] == target_time]

        embedding_cols = [col for col in pairs_at_time.columns if col.startswith('embedding_dim_')]
        X = pairs_at_time[embedding_cols].values
        y = pairs_at_time['target_distance'].values

        # Train on ALL data (no LOEO for training set)
        model = get_model(model_type, **model_params)
        model.fit(X, y)

        return {
            'target_time': target_time,
            'model': model,
            'embedding_cols': embedding_cols
        }

    # Train models in parallel
    trained_models = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(train_full_model_at_time)(t) for t in target_times
    )

    models_by_time = {m['target_time']: m for m in trained_models}

    print(f"  ✓ Trained {len(models_by_time)} models")

    # Test on each genotype
    results_by_genotype = {}

    for genotype_name, test_df in test_dfs.items():
        print(f"\nTesting on {genotype_name}...")

        # Create test pairs
        test_pairs = create_start_time_pairs(test_df, start_time=start_time)

        if len(test_pairs) == 0:
            print(f"  WARNING: No test pairs for {genotype_name}")
            continue

        print(f"  Test pairs: {len(test_pairs)}")

        # Check if this is the same genotype as training (WT)
        is_training_genotype = (test_df is train_df)

        if is_training_genotype:
            print(f"  Using LOEO for WT test set (avoid data leakage)")

        # Predict for each horizon
        results_by_horizon = {}

        for target_time in sorted(test_pairs['target_time'].unique()):
            if target_time not in models_by_time:
                continue

            pairs_at_time = test_pairs[test_pairs['target_time'] == target_time]
            embedding_cols = [col for col in pairs_at_time.columns if col.startswith('embedding_dim_')]

            if is_training_genotype:
                # LOEO: For each WT test embryo, retrain model without that embryo
                train_pairs_at_time = train_pairs[train_pairs['target_time'] == target_time]

                predictions_list = []

                for test_embryo in pairs_at_time['embryo_id'].unique():
                    # Get test data for this embryo
                    test_embryo_data = pairs_at_time[pairs_at_time['embryo_id'] == test_embryo]

                    # Get training data WITHOUT this embryo
                    train_no_embryo = train_pairs_at_time[train_pairs_at_time['embryo_id'] != test_embryo]

                    if len(train_no_embryo) == 0:
                        continue

                    # Train model without this embryo
                    X_train = train_no_embryo[embedding_cols].values
                    y_train = train_no_embryo['target_distance'].values

                    model_loeo = get_model(model_type, **model_params)
                    model_loeo.fit(X_train, y_train)

                    # Predict for held-out embryo
                    X_test = test_embryo_data[embedding_cols].values
                    y_pred = model_loeo.predict(X_test)

                    # Store
                    results_embryo = test_embryo_data.copy()
                    results_embryo['predicted_distance'] = y_pred
                    predictions_list.append(results_embryo)

                if len(predictions_list) == 0:
                    continue

                results = pd.concat(predictions_list, ignore_index=True)

            else:
                # Not training genotype: use pre-trained model (no leakage)
                model_info = models_by_time[target_time]
                model = model_info['model']

                X_test = pairs_at_time[embedding_cols].values
                y_pred = model.predict(X_test)

                results = pairs_at_time.copy()
                results['predicted_distance'] = y_pred

            # Compute errors and metrics (same for both paths)
            y_test = results['target_distance'].values
            y_pred = results['predicted_distance'].values

            results['absolute_error'] = np.abs(y_pred - y_test)
            results['relative_error'] = np.abs(y_pred - y_test) / (y_test + 1e-8)
            results['model_type'] = model_type
            results['test_genotype'] = genotype_name

            metrics = {
                'model_type': model_type,
                'target_time': target_time,
                'horizon': results['delta_t'].iloc[0],
                'test_genotype': genotype_name,
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'n_predictions': len(results),
                'n_embryos': results['embryo_id'].nunique()
            }

            results_by_horizon[target_time] = {
                'model_type': model_type,
                'target_time': target_time,
                'horizon': metrics['horizon'],
                'predictions': results,
                'metrics': metrics
            }

        results_by_genotype[genotype_name] = results_by_horizon

        # Print summary
        all_maes = [r['metrics']['mae'] for r in results_by_horizon.values()]
        if len(all_maes) > 0:
            print(f"  Average MAE: {np.mean(all_maes):.4f}")

    return results_by_genotype


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("WT RIDGE MODEL: TRAIN ON WILD-TYPE, TEST ON HET AND HOMO")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Gene: {GENE.upper()}")
    print(f"  Model: {MODEL_TYPE.upper()} (alpha={MODEL_PARAMS['alpha']})")
    print(f"  Starting time: {START_TIME} hpf")
    print(f"  Training genotype: {TRAIN_GENOTYPES[0]}")
    print(f"  Test genotypes: {', '.join(TEST_GENOTYPES)}")

    # ------------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    all_data = {}
    for genotype in set(TRAIN_GENOTYPES + TEST_GENOTYPES):
        all_data[genotype] = load_and_prepare_genotype_data(genotype)

    # Get training data (WT only)
    print(f"\nUsing training genotype: {TRAIN_GENOTYPES[0]}")
    train_df = all_data[TRAIN_GENOTYPES[0]]
    print(f"  Training timepoints: {len(train_df)}")
    print(f"  Training embryos: {train_df['embryo_id'].nunique()}")

    # Prepare test sets
    test_dfs = {g: all_data[g] for g in TEST_GENOTYPES}

    # ------------------------------------------------------------------------
    # Step 2: Train on WT, test on all
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: TRAINING ON WT, TESTING ON ALL GENOTYPES")
    print("="*80)

    results_by_genotype = train_on_genotype_test_on_all(
        train_df=train_df,
        test_dfs=test_dfs,
        model_type=MODEL_TYPE,
        start_time=START_TIME,
        n_jobs=N_JOBS,
        **MODEL_PARAMS
    )

    # Save predictions
    for genotype, results_by_horizon in results_by_genotype.items():
        for target_time, result in results_by_horizon.items():
            pred_file = OUTPUT_DATA_DIR / 'predictions' / f'{genotype}_t{int(target_time)}_predictions.csv'
            result['predictions'].to_csv(pred_file, index=False)

    print(f"\n✓ Saved predictions for all genotypes")

    # ------------------------------------------------------------------------
    # Step 3: Compare performance across genotypes
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: COMPARING PERFORMANCE ACROSS GENOTYPES")
    print("="*80)

    # Aggregate metrics
    all_metrics = []
    for genotype, results_by_horizon in results_by_genotype.items():
        for target_time, result in results_by_horizon.items():
            all_metrics.append(result['metrics'])

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.sort_values(['test_genotype', 'horizon'])

    # Save
    comparison_file = OUTPUT_DATA_DIR / 'performance_by_genotype.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nSaved comparison to {comparison_file}")

    # Print summary
    print(f"\nPerformance Summary (Average MAE):")
    for genotype in TEST_GENOTYPES:
        subset = comparison_df[comparison_df['test_genotype'] == genotype]
        if len(subset) > 0:
            avg_mae = subset['mae'].mean()
            avg_r2 = subset['r2'].mean()
            print(f"  {genotype:25s}: MAE={avg_mae:.4f}, R²={avg_r2:.4f}")

    # ------------------------------------------------------------------------
    # Step 4: Visualizations
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*80)

    # Plot 1: Error vs horizon for each genotype
    print(f"\n  Creating error vs horizon plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for genotype in TEST_GENOTYPES:
        subset = comparison_df[comparison_df['test_genotype'] == genotype].sort_values('horizon')

        axes[0].plot(
            subset['horizon'],
            subset['mae'],
            marker='o',
            linewidth=2,
            label=genotype,
            alpha=0.8
        )

        axes[1].plot(
            subset['horizon'],
            subset['r2'],
            marker='o',
            linewidth=2,
            label=genotype,
            alpha=0.8
        )

    axes[0].set_xlabel('Prediction Horizon (hours ahead)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MAE', fontsize=12, fontweight='bold')
    axes[0].set_title('WT Ridge Model: Error by Genotype', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Prediction Horizon (hours ahead)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('R²', fontsize=12, fontweight='bold')
    axes[1].set_title('WT Ridge Model: R² by Genotype', fontsize=13, fontweight='bold')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT_DIR / 'error_vs_horizon_by_genotype.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Trajectory examples for each genotype
    print(f"  Creating trajectory examples for all genotypes...")

    for genotype_name in TEST_GENOTYPES:
        if genotype_name not in results_by_genotype:
            continue

        genotype_results = results_by_genotype[genotype_name]

        # Get example embryos
        first_result = list(genotype_results.values())[0]
        all_embryos = first_result['predictions']['embryo_id'].unique()
        n_examples = min(6, len(all_embryos))

        if n_examples == 0:
            continue

        np.random.seed(42)
        selected_embryos = np.random.choice(all_embryos, n_examples, replace=False)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, embryo_id in enumerate(selected_embryos):
            ax = axes[idx]

            times = []
            actual = []
            predicted = []

            for target_time in sorted(genotype_results.keys()):
                preds = genotype_results[target_time]['predictions']
                emb_data = preds[preds['embryo_id'] == embryo_id]

                if len(emb_data) > 0:
                    times.append(target_time)
                    actual.append(emb_data['target_distance'].iloc[0])
                    predicted.append(emb_data['predicted_distance'].iloc[0])

            # Color by genotype
            if 'homozygous' in genotype_name:
                actual_color = 'red'
                pred_color = 'darkred'
            elif 'heterozygous' in genotype_name:
                actual_color = 'orange'
                pred_color = 'darkorange'
            else:  # wildtype
                actual_color = 'blue'
                pred_color = 'darkblue'

            ax.plot(times, actual, 'o-', color=actual_color, linewidth=3, label='Actual', markersize=6)
            ax.plot(times, predicted, 's--', color=pred_color, linewidth=2, label='WT Model Prediction', alpha=0.7)
            ax.axvline(START_TIME, color='gray', linestyle=':', alpha=0.5)

            ax.set_xlabel('Time (hpf)', fontsize=10)
            ax.set_ylabel('Distance from WT', fontsize=10)
            ax.set_title(f'{embryo_id}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=0.3)

        # Genotype label in title
        genotype_label = genotype_name.replace('cep290_', '').replace('_', ' ').title()
        plt.suptitle(f'{genotype_label} Embryos: Actual vs WT Model Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save with genotype-specific filename
        short_name = genotype_name.replace('cep290_', '')
        fig.savefig(OUTPUT_PLOT_DIR / f'{short_name}_trajectory_examples.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    Saved {short_name}_trajectory_examples.png")

    # ------------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  - WT model predicts WT trajectory (baseline)")
    print(f"  - Het should match WT (heterozygous = normal phenotype)")
    print(f"  - Homo deviations indicate penetrant phenotype")
    print(f"\nOutputs saved to:")
    print(f"  Data: {OUTPUT_DATA_DIR}")
    print(f"  Plots: {OUTPUT_PLOT_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
