#!/usr/bin/env python3
"""
Homozygous Full Time Matrix: Predict All Timepoint Pairs

Trains Ridge models on HOMOZYGOUS embryos for ALL valid (start_time → target_time)
pairs across the full developmental timeline (10-120 hpf), not just from a single start time.

Automatically detects which timepoint pairs have sufficient data and creates
heatmap visualizations showing prediction accuracy, R², and variation.

This reveals:
- Which developmental stages are most/least predictable in Homo trajectory
- When prediction variation increases (uncertainty)
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
sys.path.insert(0, str(Path(__file__).parent.parent / '20251016'))

from utils.data_loading import load_experiments
from utils.binning import bin_embryos_by_time
import config

from penetrance_analysis.simple_trajectory import get_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed

# ============================================================================
# Configuration
# ============================================================================

GENE = 'cep290'
MODEL_TYPE = 'ridge'
MODEL_PARAMS = {'alpha': 1.0}

# Minimum embryos required for valid timepoint pair
MIN_EMBRYOS_PER_CELL = 3

# Genotypes
TRAIN_GENOTYPE = 'cep290_homozygous'
TEST_GENOTYPES = ['cep290_homozygous', 'cep290_heterozygous', 'cep290_wildtype']

# Parallelization
N_JOBS = -1

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'penetrance'
PLOT_DIR = BASE_DIR / 'plots' / 'penetrance'

OUTPUT_DATA_DIR = DATA_DIR / 'homo_full_time_matrix'
OUTPUT_PLOT_DIR = PLOT_DIR / 'homo_full_time_matrix'

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Loading
# ============================================================================

def load_genotype_data(genotype_name: str) -> pd.DataFrame:
    """Load and prepare data for one genotype."""
    genotype_family = genotype_name.split('_')[0]

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

    df_raw = df_raw[df_raw['genotype'] == genotype_name].copy()
    df_binned = bin_embryos_by_time(df_raw, bin_width=2.0)

    # Rename embeddings
    z_cols = [col for col in df_binned.columns if col.startswith('z_mu_b_') and col.endswith('_binned')]
    rename_dict = {z_col: f'embedding_dim_{i}' for i, z_col in enumerate(sorted(z_cols))}
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

    print(f"    {len(df_binned)} timepoints, {df_binned['embryo_id'].nunique()} embryos")
    print(f"    Time range: {df_binned['time_bin'].min():.1f} - {df_binned['time_bin'].max():.1f} hpf")

    return df_binned


# ============================================================================
# Timepoint Pair Detection
# ============================================================================

def detect_valid_timepoint_pairs(
    df: pd.DataFrame,
    min_embryos: int = 3
) -> List[Tuple[float, float]]:
    """
    Detect which (start_time, target_time) pairs have sufficient data.

    Parameters
    ----------
    df : pd.DataFrame
        Binned data with embryo trajectories
    min_embryos : int
        Minimum embryos that must have data at BOTH start and target

    Returns
    -------
    list of (start_time, target_time) tuples
        Valid timepoint pairs
    """
    print("\nDetecting valid timepoint pairs...")

    all_times = sorted(df['time_bin'].unique())
    print(f"  Available time bins: {len(all_times)} ({all_times[0]:.1f} - {all_times[-1]:.1f} hpf)")

    valid_pairs = []

    for i, start_time in enumerate(all_times):
        for j, target_time in enumerate(all_times):
            # Must predict forward in time
            if target_time <= start_time:
                continue

            # Find embryos with data at BOTH times
            embryos_at_start = set(df[df['time_bin'] == start_time]['embryo_id'].unique())
            embryos_at_target = set(df[df['time_bin'] == target_time]['embryo_id'].unique())
            shared_embryos = embryos_at_start & embryos_at_target

            if len(shared_embryos) >= min_embryos:
                valid_pairs.append((start_time, target_time))

    print(f"  Valid (start, target) pairs: {len(valid_pairs)}")
    print(f"  Coverage: {len(valid_pairs)} / {len(all_times) * (len(all_times) - 1) // 2} possible pairs")

    return valid_pairs


def create_pairs_for_timepoint_pair(
    df: pd.DataFrame,
    start_time: float,
    target_time: float,
    tolerance: float = 0.1
) -> pd.DataFrame:
    """
    Create training pairs for a specific (start, target) timepoint pair.

    Returns DataFrame with columns:
    - embryo_id
    - start_time, target_time
    - embedding_dim_0, ..., embedding_dim_N
    - target_distance
    """
    embedding_cols = [col for col in df.columns if col.startswith('embedding_dim_')]

    pairs = []

    for embryo_id in df['embryo_id'].unique():
        embryo_data = df[df['embryo_id'] == embryo_id]

        # Get start row
        start_mask = np.abs(embryo_data['time_bin'] - start_time) <= tolerance
        if not start_mask.any():
            continue
        start_row = embryo_data[start_mask].iloc[0]
        start_embedding = start_row[embedding_cols].values

        # Get target row
        target_mask = np.abs(embryo_data['time_bin'] - target_time) <= tolerance
        if not target_mask.any():
            continue
        target_row = embryo_data[target_mask].iloc[0]
        target_distance = target_row['distance_from_wt']

        if pd.isna(target_distance):
            continue

        # Create pair
        pair = {
            'embryo_id': embryo_id,
            'start_time': start_time,
            'target_time': target_time
        }
        for i, col in enumerate(embedding_cols):
            pair[col] = start_embedding[i]
        pair['target_distance'] = target_distance

        pairs.append(pair)

    return pd.DataFrame(pairs)


# ============================================================================
# Model Training
# ============================================================================

def train_and_test_single_cell(
    train_df: pd.DataFrame,
    test_dfs: Dict[str, pd.DataFrame],
    start_time: float,
    target_time: float,
    model_type: str = 'ridge',
    **model_params
) -> Dict:
    """
    Train model for one (start, target) cell and test on all genotypes.

    Returns results for all test genotypes for this cell.
    """
    # Create training pairs
    train_pairs = create_pairs_for_timepoint_pair(train_df, start_time, target_time)

    if len(train_pairs) == 0:
        return None

    embedding_cols = [col for col in train_pairs.columns if col.startswith('embedding_dim_')]

    results_by_genotype = {}

    for genotype_name, test_df in test_dfs.items():
        # Create test pairs
        test_pairs = create_pairs_for_timepoint_pair(test_df, start_time, target_time)

        if len(test_pairs) == 0:
            continue

        is_training_genotype = (genotype_name == TRAIN_GENOTYPE)

        if is_training_genotype:
            # LOEO for WT
            predictions_list = []

            for test_embryo in test_pairs['embryo_id'].unique():
                test_embryo_data = test_pairs[test_pairs['embryo_id'] == test_embryo]
                train_no_embryo = train_pairs[train_pairs['embryo_id'] != test_embryo]

                if len(train_no_embryo) == 0:
                    continue

                X_train = train_no_embryo[embedding_cols].values
                y_train = train_no_embryo['target_distance'].values

                model = get_model(model_type, **model_params)
                model.fit(X_train, y_train)

                X_test = test_embryo_data[embedding_cols].values
                y_pred = model.predict(X_test)

                results_embryo = test_embryo_data.copy()
                results_embryo['predicted_distance'] = y_pred
                predictions_list.append(results_embryo)

            if len(predictions_list) == 0:
                continue

            results = pd.concat(predictions_list, ignore_index=True)

        else:
            # Use full model for Het/Homo
            X_train = train_pairs[embedding_cols].values
            y_train = train_pairs['target_distance'].values

            model = get_model(model_type, **model_params)
            model.fit(X_train, y_train)

            X_test = test_pairs[embedding_cols].values
            y_pred = model.predict(X_test)

            results = test_pairs.copy()
            results['predicted_distance'] = y_pred

        # Compute metrics
        y_true = results['target_distance'].values
        y_pred = results['predicted_distance'].values
        errors = np.abs(y_pred - y_true)

        metrics = {
            'start_time': start_time,
            'target_time': target_time,
            'delta_t': target_time - start_time,
            'genotype': genotype_name,
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'error_std': errors.std(),
            'n_predictions': len(results),
            'n_embryos': results['embryo_id'].nunique()
        }

        results_by_genotype[genotype_name] = {
            'predictions': results,
            'metrics': metrics
        }

    return results_by_genotype if len(results_by_genotype) > 0 else None


# ============================================================================
# Visualization
# ============================================================================

def create_full_time_matrix_heatmaps(
    all_metrics: pd.DataFrame,
    save_path: Optional[Path] = None
):
    """
    Create 3×3 grid of heatmaps: 3 genotypes × 3 metrics.

    Heatmaps show upper-right triangle (target > start).
    Uses SHARED color scales within each metric column for comparability.
    """
    genotypes = sorted(all_metrics['genotype'].unique())
    metrics = ['mae', 'r2', 'error_std']
    metric_labels = {
        'mae': 'MAE (Lower=Better)',
        'r2': 'R² (Higher=Better)',
        'error_std': 'Error Std Dev\n(Lower=More Consistent)'
    }

    # COMPUTE SHARED COLOR SCALES for each metric (across ALL genotypes)
    shared_vmin = {}
    shared_vmax = {}

    for metric in metrics:
        if metric == 'r2':
            # R² has fixed scale
            shared_vmin[metric] = -1
            shared_vmax[metric] = 1
        else:
            # MAE and std: use data range across all genotypes
            metric_values = all_metrics[metric].dropna()
            if len(metric_values) > 0:
                shared_vmin[metric] = 0  # Start at 0 for errors
                shared_vmax[metric] = metric_values.quantile(0.95)  # Use 95th percentile to avoid outliers
            else:
                shared_vmin[metric] = 0
                shared_vmax[metric] = 1

    print(f"\nShared color scales:")
    for metric in metrics:
        print(f"  {metric}: {shared_vmin[metric]:.3f} - {shared_vmax[metric]:.3f}")

    # Create FULL time grid (all possible time bins across all genotypes)
    all_times = sorted(all_metrics['start_time'].unique() | all_metrics['target_time'].unique())
    print(f"  Full time grid: {len(all_times)} bins ({all_times[0]:.1f} - {all_times[-1]:.1f} hpf)")

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    for i, genotype in enumerate(genotypes):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            # Filter data
            subset = all_metrics[all_metrics['genotype'] == genotype]

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Create pivot table
            pivot = subset.pivot_table(
                index='start_time',
                columns='target_time',
                values=metric,
                aggfunc='mean'
            )

            # REINDEX to include ALL time bins (fills missing with NaN)
            pivot = pivot.reindex(index=all_times, columns=all_times)

            # Mask lower triangle (target <= start) AND missing data
            mask = np.zeros_like(pivot, dtype=bool)
            for r_idx, start in enumerate(pivot.index):
                for c_idx, target in enumerate(pivot.columns):
                    # Mask if: lower triangle OR missing data (NaN)
                    if target <= start or pd.isna(pivot.iloc[r_idx, c_idx]):
                        mask[r_idx, c_idx] = True

            # Choose colormap
            if metric == 'r2':
                cmap = 'RdYlGn'  # Red=bad, Green=good
            else:
                cmap = 'RdYlBu_r'  # Red=bad, Blue=good

            # Use SHARED color scale
            vmin = shared_vmin[metric]
            vmax = shared_vmax[metric]

            # Plot
            sns.heatmap(
                pivot,
                mask=mask,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': metric_labels[metric]},
                ax=ax,
                square=False,
                linewidths=0.5,
                linecolor='gray'
            )

            # Labels
            genotype_label = genotype.replace('cep290_', '').replace('_', ' ').title()
            ax.set_title(f'{genotype_label} - {metric_labels[metric]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Target Time (hpf)', fontsize=10)
            ax.set_ylabel('Start Time (hpf)', fontsize=10)

            # Rotate labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.suptitle('Full Time Matrix: Homo Ridge Model Predictions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*80)
    print("WT FULL TIME MATRIX: ALL TIMEPOINT PAIRS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Training genotype: {TRAIN_GENOTYPE}")
    print(f"  Test genotypes: {', '.join(TEST_GENOTYPES)}")
    print(f"  Model: {MODEL_TYPE} (alpha={MODEL_PARAMS['alpha']})")
    print(f"  Min embryos per cell: {MIN_EMBRYOS_PER_CELL}")

    # Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    all_data = {}
    for genotype in set([TRAIN_GENOTYPE] + TEST_GENOTYPES):
        all_data[genotype] = load_genotype_data(genotype)

    train_df = all_data[TRAIN_GENOTYPE]
    test_dfs = {g: all_data[g] for g in TEST_GENOTYPES}

    # Detect valid pairs
    print("\n" + "="*80)
    print("STEP 2: DETECTING VALID TIMEPOINT PAIRS")
    print("="*80)

    valid_pairs = detect_valid_timepoint_pairs(train_df, min_embryos=MIN_EMBRYOS_PER_CELL)

    if len(valid_pairs) == 0:
        print("ERROR: No valid timepoint pairs found!")
        return

    # Train and test
    print("\n" + "="*80)
    print(f"STEP 3: TRAINING {len(valid_pairs)} MODELS IN PARALLEL")
    print("="*80)

    print(f"  Parallelization: n_jobs={N_JOBS}")

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(train_and_test_single_cell)(
            train_df, test_dfs, start, target, MODEL_TYPE, **MODEL_PARAMS
        )
        for start, target in valid_pairs
    )

    # Filter out None results
    results = [r for r in results if r is not None]

    print(f"\n  ✓ Completed {len(results)} valid cells")

    # Aggregate metrics
    print("\n" + "="*80)
    print("STEP 4: AGGREGATING RESULTS")
    print("="*80)

    all_metrics = []
    for result in results:
        for genotype, genotype_result in result.items():
            all_metrics.append(genotype_result['metrics'])

    df_metrics = pd.DataFrame(all_metrics)

    print(f"\nTotal predictions: {len(df_metrics)}")
    print(f"  By genotype:")
    for genotype in TEST_GENOTYPES:
        count = len(df_metrics[df_metrics['genotype'] == genotype])
        print(f"    {genotype}: {count} cells")

    # Save
    metrics_file = OUTPUT_DATA_DIR / 'full_time_matrix_metrics.csv'
    df_metrics.to_csv(metrics_file, index=False)
    print(f"\nSaved metrics to {metrics_file}")

    # Visualize
    print("\n" + "="*80)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*80)

    print(f"  Creating 3×3 heatmap grid...")
    fig = create_full_time_matrix_heatmaps(
        df_metrics,
        save_path=OUTPUT_PLOT_DIR / 'full_time_matrix_3x3.png'
    )
    plt.close(fig)

    print(f"\n✓ Saved to {OUTPUT_PLOT_DIR / 'full_time_matrix_3x3.png'}")

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  Data: {OUTPUT_DATA_DIR}")
    print(f"  Plots: {OUTPUT_PLOT_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
