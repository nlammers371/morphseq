#!/usr/bin/env python3
"""
Validation of 4 Imputation Methods for Curvature Trajectory Data.

Tests linear, spline, forward-fill, and model-based (MICE) imputation methods
on synthetic holdout data using parallelized iterations.

Outputs: Summary statistics and recommendation for best imputation method.
"""

# Core imports (fast)
from pathlib import Path
import sys
import warnings
from typing import Dict

# Heavy imports (moved to functions):
# - numpy as np
# - pandas as pd
# - sklearn.impute.IterativeImputer
# - scipy.interpolate.interp1d

warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Dynamic import to handle module name starting with digit
import importlib.util
spec = importlib.util.spec_from_file_location(
    "load_data",
    SCRIPT_DIR / "load_data.py"
)
load_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data_module)

get_analysis_dataframe = load_data_module.get_analysis_dataframe
GENOTYPE_SHORT = load_data_module.GENOTYPE_SHORT
GENOTYPES = load_data_module.GENOTYPES

# ============================================================================
# Configuration
# ============================================================================

# Metric to analyze
METRIC_NAME = 'normalized_baseline_deviation'

# Synthetic holdout parameters
SYNTHETIC_HOLDOUT_FRACTION = 0.2
IMPUTATION_TEST_ITERATIONS = 20

# Output
OUTPUT_DIR = SCRIPT_DIR / 'outputs' / '07a' / 'validation'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================================
# Imputation Methods
# ============================================================================

def create_synthetic_missing_data(trajectories_df, fraction: float = 0.2,
                                  random_seed: int = 42):
    """
    Create synthetic missing data by removing random timepoints.

    Imports: numpy, pandas
    Parameters
    ----------
    trajectories_df : pd.DataFrame
        DataFrame with columns: embryo_id, hpf, metric_value
    fraction : float
        Fraction of timepoints to remove per embryo
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    data_with_missing : pd.DataFrame
        DataFrame with NaN values introduced
    holdout_truth : pd.DataFrame
        DataFrame with removed values (ground truth)
    """
    import numpy as np
    import pandas as pd

    np.random.seed(random_seed)
    data_with_missing = trajectories_df.copy()
    holdout_records = []

    for embryo_id in data_with_missing['embryo_id'].unique():
        embryo_mask = data_with_missing['embryo_id'] == embryo_id
        embryo_indices = np.where(embryo_mask)[0]

        n_remove = max(1, int(len(embryo_indices) * fraction))
        remove_indices = np.random.choice(embryo_indices, size=n_remove, replace=False)

        for idx in remove_indices:
            holdout_records.append({
                'embryo_id': data_with_missing.loc[idx, 'embryo_id'],
                'hpf': data_with_missing.loc[idx, 'hpf'],
                'metric_value': data_with_missing.loc[idx, 'metric_value']
            })
            data_with_missing.loc[idx, 'metric_value'] = np.nan

    holdout_truth = pd.DataFrame(holdout_records)
    return data_with_missing, holdout_truth


def impute_linear(trajectories_df):
    """Linear interpolation.
    Imports: numpy, scipy.interpolate.interp1d
    """
    import numpy as np
    from scipy.interpolate import interp1d

    df = trajectories_df.copy()
    for embryo_id in df['embryo_id'].unique():
        embryo_mask = df['embryo_id'] == embryo_id
        hpf_values = df.loc[embryo_mask, 'hpf'].values
        metric_values = df.loc[embryo_mask, 'metric_value'].values

        valid_mask = ~np.isnan(metric_values)
        if valid_mask.sum() < 2:
            continue

        f = interp1d(hpf_values[valid_mask], metric_values[valid_mask],
                     kind='linear', fill_value='extrapolate', bounds_error=False)
        imputed = f(hpf_values)
        df.loc[embryo_mask, 'metric_value'] = imputed

    return df


def impute_spline(trajectories_df):
    """Spline interpolation (k=3).
    Imports: numpy, scipy.interpolate.interp1d
    """
    import numpy as np
    from scipy.interpolate import interp1d

    df = trajectories_df.copy()
    for embryo_id in df['embryo_id'].unique():
        embryo_mask = df['embryo_id'] == embryo_id
        hpf_values = df.loc[embryo_mask, 'hpf'].values
        metric_values = df.loc[embryo_mask, 'metric_value'].values

        valid_mask = ~np.isnan(metric_values)
        if valid_mask.sum() < 4:  # Need at least 4 points for cubic spline
            # Fall back to linear if not enough points
            return impute_linear(df)

        try:
            f = interp1d(hpf_values[valid_mask], metric_values[valid_mask],
                        kind='cubic', fill_value='extrapolate', bounds_error=False)
            imputed = f(hpf_values)
            df.loc[embryo_mask, 'metric_value'] = imputed
        except ValueError:
            # Fall back to linear if spline fails
            return impute_linear(df)

    return df


def impute_forward_fill(trajectories_df):
    """Forward fill / LOCF (Last Observation Carried Forward).
    Imports: pandas
    """
    df = trajectories_df.copy()
    for embryo_id in df['embryo_id'].unique():
        embryo_mask = df['embryo_id'] == embryo_id
        embryo_data = df.loc[embryo_mask].sort_values('hpf').copy()
        embryo_data['metric_value'] = embryo_data['metric_value'].fillna(method='ffill')
        embryo_data['metric_value'] = embryo_data['metric_value'].fillna(method='bfill')
        df.loc[embryo_mask, 'metric_value'] = embryo_data['metric_value'].values

    return df


def impute_model_based(trajectories_df):
    """Model-based imputation using IterativeImputer (MICE-like).
    Imports: pandas, sklearn.experimental, sklearn.impute.IterativeImputer
    """
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    import pandas as pd

    # Pivot to embryo × timepoint matrix
    df = trajectories_df.copy()

    # Create wide format
    pivot_data = df.pivot_table(
        index='embryo_id',
        columns='hpf',
        values='metric_value',
        aggfunc='mean'
    )

    # Apply IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputed_matrix = imputer.fit_transform(pivot_data)

    # Convert back to long format
    imputed_df = pd.DataFrame(
        imputed_matrix,
        index=pivot_data.index,
        columns=pivot_data.columns
    ).reset_index().melt(
        id_vars='embryo_id',
        var_name='hpf',
        value_name='metric_value'
    )

    return imputed_df


def compute_imputation_rmse(holdout_truth, imputed_df):
    """Compute RMSE between holdout truth and imputed values.
    Imports: numpy
    """
    import numpy as np

    merged = holdout_truth.merge(
        imputed_df.rename(columns={'metric_value': 'imputed_value'}),
        on=['embryo_id', 'hpf'],
        how='inner'
    )

    if len(merged) == 0:
        return np.inf

    rmse = np.sqrt(np.mean((merged['metric_value'] - merged['imputed_value']) ** 2))
    return rmse


# ============================================================================
# Parallel Validation
# ============================================================================

def _test_single_iteration(args):
    """Helper function for parallel iteration testing.
    Imports: numpy
    """
    import numpy as np

    trajectories_df, iteration, seed = args
    methods = {
        'linear': impute_linear,
        'spline': impute_spline,
        'forward_fill': impute_forward_fill,
        'model_based': impute_model_based
    }

    iteration_results = {method: None for method in methods}

    # Create synthetic missing data with unique seed per iteration
    data_with_missing, holdout_truth = create_synthetic_missing_data(
        trajectories_df, fraction=SYNTHETIC_HOLDOUT_FRACTION,
        random_seed=seed
    )

    for method_name, impute_func in methods.items():
        try:
            imputed_df = impute_func(data_with_missing)
            rmse = compute_imputation_rmse(holdout_truth, imputed_df)
            iteration_results[method_name] = rmse
        except Exception as e:
            iteration_results[method_name] = np.inf

    return iteration_results


def validate_imputation_methods(trajectories_df, n_iterations: int = 20):
    """
    Parallelize test 4 imputation methods on synthetic holdout data.

    Imports: numpy, multiprocessing
    Returns
    -------
    results : Dict
        Keys: method names ('linear', 'spline', 'forward_fill', 'model_based')
        Values: dict with 'rmse_mean', 'rmse_std', 'runtime'
    """
    import numpy as np
    from multiprocessing import Pool
    import os

    methods = {
        'linear': impute_linear,
        'spline': impute_spline,
        'forward_fill': impute_forward_fill,
        'model_based': impute_model_based
    }

    print("\n" + "="*80)
    print("VALIDATING IMPUTATION METHODS")
    print("="*80)

    # Prepare arguments for parallel processing
    n_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 workers
    task_args = [(trajectories_df, i, 42 + i) for i in range(n_iterations)]

    print(f"\nRunning {n_iterations} iterations with {n_workers} workers...")

    # Run in parallel
    rmse_results = {method: [] for method in methods}
    with Pool(n_workers) as pool:
        iteration_results_list = pool.map(_test_single_iteration, task_args)

    # Aggregate results
    for iteration, iteration_results in enumerate(iteration_results_list):
        for method_name, rmse in iteration_results.items():
            rmse_results[method_name].append(rmse)

        if (iteration + 1) % max(1, n_iterations // 5) == 0:
            print(f"  Completed {iteration + 1}/{n_iterations} iterations")

    # Compute statistics
    results = {}
    for method_name in methods:
        rmse_vals = [x for x in rmse_results[method_name] if not np.isinf(x)]
        results[method_name] = {
            'rmse_mean': np.mean(rmse_vals) if rmse_vals else np.inf,
            'rmse_std': np.std(rmse_vals) if rmse_vals else np.nan,
            'n_valid': len(rmse_vals),
            'rmse_all': rmse_vals
        }

    # Print summary
    print("\n" + "-"*80)
    print("Imputation Method Comparison:")
    print("-"*80)
    for method_name, stats in results.items():
        print(f"{method_name:15s}: RMSE = {stats['rmse_mean']:.6f} ± {stats['rmse_std']:.6f}")

    # Recommend best method (lowest RMSE)
    best_method = min(results.items(), key=lambda x: x[1]['rmse_mean'])[0]
    print(f"\nRecommended method: {best_method}")
    print("="*80)

    return results, best_method


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Load data
    print("\nLoading curvature data...")
    df, metadata = get_analysis_dataframe(normalize=True)

    # Prepare trajectories
    print(f"\nPreparing trajectories for metric: {METRIC_NAME}")
    trajectories_df = df[['embryo_id', 'predicted_stage_hpf', METRIC_NAME, 'genotype']].copy()
    trajectories_df = trajectories_df.rename(columns={
        'predicted_stage_hpf': 'hpf',
        METRIC_NAME: 'metric_value'
    })
    trajectories_df = trajectories_df.dropna(subset=['metric_value'])

    print(f"  Total records: {len(trajectories_df)}")
    print(f"  Unique embryos: {trajectories_df['embryo_id'].nunique()}")

    # Validate imputation methods
    results, best_method = validate_imputation_methods(
        trajectories_df, n_iterations=IMPUTATION_TEST_ITERATIONS
    )

    # Save results to file
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_file = OUTPUT_DIR / 'imputation_validation_summary.txt'
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IMPUTATION METHOD VALIDATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Metric: {METRIC_NAME}\n")
        f.write(f"Iterations: {IMPUTATION_TEST_ITERATIONS}\n")
        f.write(f"Holdout fraction: {SYNTHETIC_HOLDOUT_FRACTION}\n\n")
        f.write("-"*80 + "\n")
        f.write("Method Comparison:\n")
        f.write("-"*80 + "\n")
        for method_name, stats in results.items():
            f.write(f"\n{method_name}:\n")
            f.write(f"  RMSE mean: {stats['rmse_mean']:.6f}\n")
            f.write(f"  RMSE std:  {stats['rmse_std']:.6f}\n")
            f.write(f"  Valid runs: {stats['n_valid']}/{IMPUTATION_TEST_ITERATIONS}\n")
        f.write(f"\n" + "-"*80 + "\n")
        f.write(f"RECOMMENDED METHOD: {best_method}\n")
        f.write(f"RMSE: {results[best_method]['rmse_mean']:.6f} ± {results[best_method]['rmse_std']:.6f}\n")
        f.write("="*80 + "\n")

    print(f"\nSaved validation results to: {output_file}")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nBest imputation method: {best_method}")
    print(f"RMSE: {results[best_method]['rmse_mean']:.6f} ± {results[best_method]['rmse_std']:.6f}")
