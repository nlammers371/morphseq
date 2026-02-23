"""
Leave-One-Embryo-Out (LOEO) Trajectory Prediction for Penetrance Analysis

Tests if embeddings at time i can predict morphological distance at time i+k,
using rigorous cross-validation to assess embedding quality and detect penetrance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def create_trajectory_pairs(
    df_binned: pd.DataFrame,
    genotype: str,
    min_delta_t: int = 2
) -> pd.DataFrame:
    """
    Create all valid (i → i+k) prediction pairs for a genotype.

    For each embryo:
        For each timepoint i:
            For each future timepoint j > i (where j - i >= min_delta_t):
                Create pair: (embedding at i) → (distance at j)

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned data with columns:
        - embryo_id, time_bin, genotype
        - embedding_dim_0, embedding_dim_1, ... (embedding features)
        - distance_from_wt (Euclidean distance from WT reference)
    genotype : str
        Genotype name for filtering
    min_delta_t : int
        Minimum time difference (in hpf) for predictions

    Returns
    -------
    pd.DataFrame
        Trajectory pairs with columns:
        - embryo_id
        - from_time, to_time, delta_t
        - embedding_dim_0, embedding_dim_1, ... (features at from_time)
        - actual_distance (distance at to_time)
    """
    # Filter to genotype
    df = df_binned[df_binned['genotype'] == genotype].copy()

    # Get embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('embedding_dim_')]

    if len(embedding_cols) == 0:
        raise ValueError(f"No embedding columns found in df_binned")

    pairs = []

    # For each embryo
    for embryo_id in df['embryo_id'].unique():
        embryo_data = df[df['embryo_id'] == embryo_id].sort_values('time_bin')

        times = embryo_data['time_bin'].values
        n_times = len(times)

        # For each starting time
        for i in range(n_times):
            from_time = times[i]

            # Get embedding at from_time
            from_row = embryo_data[embryo_data['time_bin'] == from_time].iloc[0]
            embedding_features = from_row[embedding_cols].values

            # For each future time
            for j in range(i + 1, n_times):
                to_time = times[j]
                delta_t = to_time - from_time

                if delta_t < min_delta_t:
                    continue

                # Get distance at to_time
                to_row = embryo_data[embryo_data['time_bin'] == to_time].iloc[0]
                actual_distance = to_row['distance_from_wt']

                # Create pair
                pair = {
                    'embryo_id': embryo_id,
                    'from_time': from_time,
                    'to_time': to_time,
                    'delta_t': delta_t,
                    'actual_distance': actual_distance
                }

                # Add embedding features
                for k, col in enumerate(embedding_cols):
                    pair[col] = embedding_features[k]

                pairs.append(pair)

    df_pairs = pd.DataFrame(pairs)

    print(f"  Created {len(df_pairs)} trajectory pairs for {genotype}")

    if len(df_pairs) == 0:
        print(f"    WARNING: No trajectory pairs created!")
        print(f"    This genotype may have no embryos with valid distance data.")
        return df_pairs

    print(f"    Embryos: {df_pairs['embryo_id'].nunique()}")
    print(f"    Time range: {df_pairs['from_time'].min():.1f} - {df_pairs['to_time'].max():.1f} hpf")
    print(f"    Delta_t range: {df_pairs['delta_t'].min():.1f} - {df_pairs['delta_t'].max():.1f} hpf")

    return df_pairs


def train_loeo_and_full_model(
    df_pairs: pd.DataFrame,
    model_name: str,
    model_type: str = 'random_forest',
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train model using leave-one-embryo-out cross-validation.

    Also trains a full model on all embryos for cross-genotype testing.

    Parameters
    ----------
    df_pairs : pd.DataFrame
        Trajectory pairs from create_trajectory_pairs()
    model_name : str
        Name for tracking (e.g., 'cep290_wt_model')
    model_type : str
        'random_forest', 'linear', or 'gradient_boosting'
    n_estimators : int
        Number of trees (for ensemble models)
    max_depth : int, optional
        Maximum tree depth
    random_state : int
        Random seed

    Returns
    -------
    dict
        {
            'loeo_predictions': pd.DataFrame,   # LOEO predictions
            'full_model': trained model,        # Model trained on all embryos
            'feature_cols': List[str],          # Feature column names
            'metrics': Dict                     # Overall performance metrics
        }
    """
    # Get feature columns (embeddings)
    feature_cols = [col for col in df_pairs.columns if col.startswith('embedding_dim_')]
    target_col = 'actual_distance'

    print(f"\n  Training {model_name}...")
    print(f"    Model type: {model_type}")
    print(f"    Features: {len(feature_cols)} embedding dimensions")
    print(f"    Training examples: {len(df_pairs)}")

    embryo_ids = df_pairs['embryo_id'].unique()
    n_embryos = len(embryo_ids)
    print(f"    LOEO folds: {n_embryos} embryos")

    loeo_predictions_list = []

    # LOEO cross-validation
    for i, held_out_embryo in enumerate(embryo_ids):
        if (i + 1) % 5 == 0 or (i + 1) == n_embryos:
            print(f"      Fold {i+1}/{n_embryos} (held out: {held_out_embryo})", end='\r')

        # Split data
        train_data = df_pairs[df_pairs['embryo_id'] != held_out_embryo]
        test_data = df_pairs[df_pairs['embryo_id'] == held_out_embryo]

        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        # Train model
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        elif model_type == 'linear':
            model = LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Store results
        results = test_data.copy()
        results['predicted_distance'] = predictions
        results['absolute_error'] = np.abs(predictions - y_test)
        results['relative_error'] = np.abs(predictions - y_test) / (y_test + 1e-8)
        results['cv_fold_embryo_id'] = held_out_embryo
        results['model_name'] = model_name

        loeo_predictions_list.append(results)

    print()  # New line after progress

    # Combine LOEO predictions
    loeo_predictions = pd.concat(loeo_predictions_list, ignore_index=True)

    # Train full model on ALL embryos (for cross-genotype testing)
    print(f"    Training full model on all {n_embryos} embryos...")
    X_full = df_pairs[feature_cols].values
    y_full = df_pairs[target_col].values

    if model_type == 'random_forest':
        full_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        full_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    elif model_type == 'linear':
        full_model = LinearRegression()

    full_model.fit(X_full, y_full)

    # Compute metrics
    metrics = compute_overall_metrics(loeo_predictions, model_name)

    print(f"    LOEO Performance:")
    print(f"      Mean Absolute Error: {metrics['mean_abs_error']:.4f}")
    print(f"      R²: {metrics['r2_overall']:.4f}")

    return {
        'loeo_predictions': loeo_predictions,
        'full_model': full_model,
        'feature_cols': feature_cols,
        'metrics': metrics
    }


def test_model_on_genotype(
    model: Any,
    feature_cols: List[str],
    df_test_pairs: pd.DataFrame,
    model_name: str,
    test_genotype: str
) -> pd.DataFrame:
    """
    Test a trained model on a different genotype.

    Parameters
    ----------
    model : sklearn model
        Trained model (from train_loeo_and_full_model['full_model'])
    feature_cols : List[str]
        Feature column names
    df_test_pairs : pd.DataFrame
        Test trajectory pairs
    model_name : str
        Name of the model being tested
    test_genotype : str
        Name of the test genotype

    Returns
    -------
    pd.DataFrame
        Predictions with columns:
        - All columns from df_test_pairs
        - predicted_distance
        - absolute_error
        - relative_error
        - model_name
        - test_genotype
    """
    X_test = df_test_pairs[feature_cols].values
    y_test = df_test_pairs['actual_distance'].values

    predictions = model.predict(X_test)

    results = df_test_pairs.copy()
    results['predicted_distance'] = predictions
    results['absolute_error'] = np.abs(predictions - y_test)
    results['relative_error'] = np.abs(predictions - y_test) / (y_test + 1e-8)
    results['model_name'] = model_name
    results['test_genotype'] = test_genotype

    metrics = compute_overall_metrics(results, model_name)

    print(f"    Cross-genotype test: {model_name} on {test_genotype}")
    print(f"      Mean Absolute Error: {metrics['mean_abs_error']:.4f}")
    print(f"      R²: {metrics['r2_overall']:.4f}")

    return results


def compute_overall_metrics(
    predictions: pd.DataFrame,
    model_name: str
) -> Dict[str, float]:
    """
    Compute overall model performance metrics.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions from train_loeo_and_full_model or test_model_on_genotype
    model_name : str
        Model name for reference

    Returns
    -------
    dict
        {
            'model_name': str,
            'mean_abs_error': float,
            'median_abs_error': float,
            'std_abs_error': float,
            'mean_rel_error': float,
            'r2_overall': float,
            'rmse': float,
            'n_predictions': int,
            'n_embryos': int,
            'time_range_min': float,
            'time_range_max': float
        }
    """
    y_true = predictions['actual_distance'].values
    y_pred = predictions['predicted_distance'].values

    metrics = {
        'model_name': model_name,
        'mean_abs_error': mean_absolute_error(y_true, y_pred),
        'median_abs_error': np.median(np.abs(y_true - y_pred)),
        'std_abs_error': np.std(np.abs(y_true - y_pred)),
        'mean_rel_error': predictions['relative_error'].mean(),
        'r2_overall': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'n_predictions': len(predictions),
        'n_embryos': predictions['embryo_id'].nunique(),
        'time_range_min': predictions['from_time'].min(),
        'time_range_max': predictions['to_time'].max()
    }

    return metrics


def compute_per_embryo_metrics(
    predictions: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute metrics separately for each embryo.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions DataFrame

    Returns
    -------
    pd.DataFrame
        Per-embryo metrics:
        - embryo_id
        - mean_abs_error
        - median_abs_error
        - std_abs_error
        - mean_rel_error
        - r2
        - n_predictions
    """
    metrics_list = []

    for embryo_id in predictions['embryo_id'].unique():
        embryo_preds = predictions[predictions['embryo_id'] == embryo_id]

        y_true = embryo_preds['actual_distance'].values
        y_pred = embryo_preds['predicted_distance'].values

        # R² (handle case where variance is 0)
        if np.var(y_true) > 1e-8:
            r2 = r2_score(y_true, y_pred)
        else:
            r2 = np.nan

        metrics = {
            'embryo_id': embryo_id,
            'mean_abs_error': mean_absolute_error(y_true, y_pred),
            'median_abs_error': np.median(np.abs(y_true - y_pred)),
            'std_abs_error': np.std(np.abs(y_true - y_pred)),
            'mean_rel_error': embryo_preds['relative_error'].mean(),
            'r2': r2,
            'n_predictions': len(embryo_preds)
        }

        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)


def compute_error_vs_horizon(
    predictions: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute mean error vs prediction horizon (delta_t).

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions DataFrame

    Returns
    -------
    pd.DataFrame
        Metrics by delta_t:
        - delta_t
        - mean_abs_error
        - std_abs_error
        - median_abs_error
        - mean_rel_error
        - std_rel_error
        - n_predictions
    """
    metrics_list = []

    for delta_t in sorted(predictions['delta_t'].unique()):
        subset = predictions[predictions['delta_t'] == delta_t]

        metrics = {
            'delta_t': delta_t,
            'mean_abs_error': subset['absolute_error'].mean(),
            'std_abs_error': subset['absolute_error'].std(),
            'median_abs_error': subset['absolute_error'].median(),
            'mean_rel_error': subset['relative_error'].mean(),
            'std_rel_error': subset['relative_error'].std(),
            'n_predictions': len(subset)
        }

        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)


def classify_penetrance_dual_model(
    wt_model_predictions: pd.DataFrame,
    homo_model_predictions: pd.DataFrame,
    error_ratio_threshold: float = 1.5,
    min_predictions: int = 3
) -> pd.DataFrame:
    """
    Classify penetrance by comparing WT vs Homo model performance on Homo embryos.

    For each Homo embryo:
        error_ratio = mean_error_wt_model / mean_error_homo_model

        If error_ratio > threshold:
            → PENETRANT (WT model fails, Homo model succeeds → follows mutant trajectory)
        If error_ratio < 1/threshold:
            → NON-PENETRANT (WT model succeeds, Homo model fails → follows WT trajectory)
        Else:
            → INTERMEDIATE

    Parameters
    ----------
    wt_model_predictions : pd.DataFrame
        Predictions from WT model tested on Homo embryos
    homo_model_predictions : pd.DataFrame
        Predictions from Homo model tested on Homo embryos (LOEO)
    error_ratio_threshold : float
        Threshold for classification (default: 1.5)
    min_predictions : int
        Minimum number of predictions required per embryo

    Returns
    -------
    pd.DataFrame
        Classification results:
        - embryo_id
        - mean_error_wt_model
        - mean_error_homo_model
        - error_ratio
        - penetrance_status ('penetrant', 'non-penetrant', 'intermediate')
        - n_predictions_wt
        - n_predictions_homo
    """
    # Compute per-embryo mean errors
    wt_errors = wt_model_predictions.groupby('embryo_id').agg({
        'absolute_error': ['mean', 'count']
    })
    wt_errors.columns = ['mean_error_wt_model', 'n_predictions_wt']

    homo_errors = homo_model_predictions.groupby('embryo_id').agg({
        'absolute_error': ['mean', 'count']
    })
    homo_errors.columns = ['mean_error_homo_model', 'n_predictions_homo']

    # Merge
    comparison = pd.merge(wt_errors, homo_errors, left_index=True, right_index=True, how='outer')

    # Filter by minimum predictions
    comparison = comparison[
        (comparison['n_predictions_wt'] >= min_predictions) &
        (comparison['n_predictions_homo'] >= min_predictions)
    ]

    # Compute error ratio
    comparison['error_ratio'] = comparison['mean_error_wt_model'] / comparison['mean_error_homo_model']

    # Classify
    comparison['penetrance_status'] = 'intermediate'
    comparison.loc[comparison['error_ratio'] > error_ratio_threshold, 'penetrance_status'] = 'penetrant'
    comparison.loc[comparison['error_ratio'] < (1.0 / error_ratio_threshold), 'penetrance_status'] = 'non-penetrant'

    # Reset index to make embryo_id a column
    comparison = comparison.reset_index()

    # Print summary
    print(f"\n  Penetrance Classification Summary:")
    print(f"  {'-'*50}")
    status_counts = comparison['penetrance_status'].value_counts()
    total = len(comparison)
    for status in ['penetrant', 'non-penetrant', 'intermediate']:
        count = status_counts.get(status, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"    {status.capitalize():15s}: {count:3d} / {total:3d} ({pct:5.1f}%)")

    print(f"\n  Error Ratio Statistics:")
    print(f"  {'-'*50}")
    print(f"    Mean: {comparison['error_ratio'].mean():.3f}")
    print(f"    Median: {comparison['error_ratio'].median():.3f}")
    print(f"    Range: [{comparison['error_ratio'].min():.3f}, {comparison['error_ratio'].max():.3f}]")

    return comparison
