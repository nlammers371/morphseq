"""
Simple trajectory prediction from fixed starting timepoint.

Trains SEPARATE models for EACH future timepoint to predict developmental
trajectories from a single timepoint embedding, using leave-one-embryo-out
validation with parallel CPU processing.

Key concept: Train model_32hpf, model_34hpf, model_36hpf, ... separately
to test how far ahead we can predict from 30 hpf snapshot.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Will skip XGBoost model.")


def create_start_time_pairs(
    df_binned: pd.DataFrame,
    start_time: float = 30.0,
    tolerance: float = 0.1
) -> pd.DataFrame:
    """
    Create training pairs: (embedding at start_time) → (distance at future times).

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned data with embeddings and distances
    start_time : float
        Starting timepoint (hpf)
    tolerance : float
        Time bin tolerance for matching start_time

    Returns
    -------
    pd.DataFrame
        Training pairs with columns:
        - embryo_id
        - start_time (always same value)
        - target_time (future timepoint)
        - delta_t (target_time - start_time)
        - embedding_dim_0, embedding_dim_1, ... (features)
        - target_distance (distance at target_time)
    """
    # Get embedding columns
    embedding_cols = [col for col in df_binned.columns if col.startswith('embedding_dim_')]

    if len(embedding_cols) == 0:
        raise ValueError("No embedding columns found in df_binned")

    pairs = []

    # Get unique embryos
    embryos = df_binned['embryo_id'].unique()

    for embryo_id in embryos:
        embryo_data = df_binned[df_binned['embryo_id'] == embryo_id].sort_values('time_bin')

        # Find start time row
        start_mask = np.abs(embryo_data['time_bin'] - start_time) <= tolerance

        if not start_mask.any():
            continue  # Skip embryos without start timepoint

        start_row = embryo_data[start_mask].iloc[0]
        start_embedding = start_row[embedding_cols].values

        # Get all future timepoints
        future_data = embryo_data[embryo_data['time_bin'] > start_time]

        for _, future_row in future_data.iterrows():
            target_time = future_row['time_bin']
            target_distance = future_row['distance_from_wt']

            # Skip if distance is NaN
            if pd.isna(target_distance):
                continue

            # Create pair
            pair = {
                'embryo_id': embryo_id,
                'start_time': start_time,
                'target_time': target_time,
                'delta_t': target_time - start_time
            }

            # Add embedding features
            for i, col in enumerate(embedding_cols):
                pair[col] = start_embedding[i]

            pair['target_distance'] = target_distance

            pairs.append(pair)

    df_pairs = pd.DataFrame(pairs)

    print(f"\nCreated {len(df_pairs)} prediction pairs from start_time={start_time} hpf")
    if len(df_pairs) > 0:
        print(f"  Embryos: {df_pairs['embryo_id'].nunique()}")
        print(f"  Target times: {sorted(df_pairs['target_time'].unique())}")
        print(f"  Target time range: {df_pairs['target_time'].min():.1f} - {df_pairs['target_time'].max():.1f} hpf")
        print(f"  Prediction horizon range: {df_pairs['delta_t'].min():.1f} - {df_pairs['delta_t'].max():.1f} hours")

    return df_pairs


def get_model(model_type: str, **kwargs) -> Any:
    """
    Get sklearn model instance by name.

    Parameters
    ----------
    model_type : str
        One of: 'linear', 'ridge', 'lasso', 'random_forest',
                'gradient_boosting', 'xgboost', 'svr', 'mlp'
    **kwargs
        Model-specific hyperparameters

    Returns
    -------
    model
        Sklearn-compatible model instance
    """
    if model_type == 'linear':
        return LinearRegression(**kwargs)

    elif model_type == 'ridge':
        return Ridge(alpha=kwargs.get('alpha', 1.0), random_state=42)

    elif model_type == 'lasso':
        return Lasso(alpha=kwargs.get('alpha', 0.1), random_state=42, max_iter=10000)

    elif model_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=42,
            n_jobs=1  # Individual RF models use 1 job (parallelism at horizon level)
        )

    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 3),
            random_state=42
        )

    elif model_type == 'xgboost':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        return XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 3),
            random_state=42,
            n_jobs=1  # Individual XGB models use 1 job
        )

    elif model_type == 'svr':
        return SVR(
            kernel=kwargs.get('kernel', 'rbf'),
            C=kwargs.get('C', 1.0),
            epsilon=kwargs.get('epsilon', 0.1)
        )

    elif model_type == 'mlp':
        return MLPRegressor(
            hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_model_loeo_single_time(
    df_pairs: pd.DataFrame,
    model_type: str,
    target_time: float,
    verbose: bool = False,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Train model for SINGLE target timepoint using LOEO.

    This trains ONE model to predict distance at ONE specific future time
    from the 30 hpf embedding.

    Parameters
    ----------
    df_pairs : pd.DataFrame
        Training pairs FILTERED to single target_time
    model_type : str
        Model type name
    target_time : float
        The specific target time this model predicts (for metadata)
    verbose : bool
        Print progress (use False for parallel execution)
    **model_kwargs
        Model hyperparameters

    Returns
    -------
    dict
        {
            'model_type': str,
            'target_time': float,
            'horizon': float (delta_t),
            'predictions': pd.DataFrame,  # All LOEO predictions
            'metrics': dict                # Overall metrics
        }
    """
    # Get feature columns (embeddings ONLY - no temporal features needed!)
    feature_cols = [col for col in df_pairs.columns if col.startswith('embedding_dim_')]
    target_col = 'target_distance'

    if verbose:
        print(f"    Training {model_type} for target_time={target_time} hpf (LOEO)...")
        print(f"      Features: {len(feature_cols)} embedding dimensions")
        print(f"      Training pairs: {len(df_pairs)}")

    embryo_ids = df_pairs['embryo_id'].unique()
    n_embryos = len(embryo_ids)

    if verbose:
        print(f"      LOEO folds: {n_embryos} embryos")

    predictions_list = []

    # LOEO cross-validation
    for i, held_out_embryo in enumerate(embryo_ids):
        # Split data
        train_data = df_pairs[df_pairs['embryo_id'] != held_out_embryo]
        test_data = df_pairs[df_pairs['embryo_id'] == held_out_embryo]

        # Skip if test set empty
        if len(test_data) == 0:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        # Train model
        model = get_model(model_type, **model_kwargs)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Store results
        results = test_data.copy()
        results['predicted_distance'] = y_pred
        results['absolute_error'] = np.abs(y_pred - y_test)
        results['relative_error'] = np.abs(y_pred - y_test) / (y_test + 1e-8)
        results['model_type'] = model_type
        results['cv_fold_embryo_id'] = held_out_embryo

        predictions_list.append(results)

    # Combine predictions
    if len(predictions_list) == 0:
        # No predictions made
        return {
            'model_type': model_type,
            'target_time': target_time,
            'horizon': df_pairs['delta_t'].iloc[0] if len(df_pairs) > 0 else np.nan,
            'predictions': pd.DataFrame(),
            'metrics': {
                'model_type': model_type,
                'target_time': target_time,
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'n_predictions': 0,
                'n_embryos': 0
            }
        }

    predictions = pd.concat(predictions_list, ignore_index=True)

    # Compute overall metrics
    y_true = predictions['target_distance'].values
    y_pred = predictions['predicted_distance'].values

    metrics = {
        'model_type': model_type,
        'target_time': target_time,
        'horizon': predictions['delta_t'].iloc[0],
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'n_predictions': len(predictions),
        'n_embryos': n_embryos
    }

    if verbose:
        print(f"      LOEO Performance:")
        print(f"        MAE:  {metrics['mae']:.4f}")
        print(f"        RMSE: {metrics['rmse']:.4f}")
        print(f"        R²:   {metrics['r2']:.4f}")

    return {
        'model_type': model_type,
        'target_time': target_time,
        'horizon': metrics['horizon'],
        'predictions': predictions,
        'metrics': metrics
    }


def train_all_horizons_parallel(
    df_pairs: pd.DataFrame,
    model_type: str,
    n_jobs: int = -1,
    verbose: bool = True,
    **model_kwargs
) -> Dict[float, Dict]:
    """
    Train separate model for each target timepoint in parallel.

    This is the KEY function that implements the correct approach:
    - One model per target time (32, 34, 36, ... 120 hpf)
    - Each model only sees ONE target time
    - Parallelized across target times using joblib

    Parameters
    ----------
    df_pairs : pd.DataFrame
        All training pairs (will be split by target_time)
    model_type : str
        Model type name
    n_jobs : int
        Number of parallel jobs (-1 = use all CPUs)
    verbose : bool
        Print progress
    **model_kwargs
        Model hyperparameters

    Returns
    -------
    dict
        {
            32.0: {model_type, target_time, horizon, predictions, metrics},
            34.0: {model_type, target_time, horizon, predictions, metrics},
            ...
        }
    """
    target_times = sorted(df_pairs['target_time'].unique())
    n_horizons = len(target_times)

    if verbose:
        print(f"\n  Training {model_type} for {n_horizons} horizons in parallel (n_jobs={n_jobs})...")
        print(f"    Target times: {target_times}")

    # Train in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(train_model_loeo_single_time)(
            df_pairs[df_pairs['target_time'] == t],
            model_type,
            target_time=t,
            verbose=False,
            **model_kwargs
        )
        for t in target_times
    )

    # Convert to dict
    results_dict = {r['target_time']: r for r in results}

    if verbose:
        # Compute overall stats
        all_maes = [r['metrics']['mae'] for r in results if not np.isnan(r['metrics']['mae'])]
        if len(all_maes) > 0:
            print(f"    Average MAE across all horizons: {np.mean(all_maes):.4f}")
            print(f"    MAE range: {np.min(all_maes):.4f} - {np.max(all_maes):.4f}")

    return results_dict


def aggregate_metrics_across_horizons(
    results_by_horizon: Dict[float, Dict]
) -> pd.DataFrame:
    """
    Aggregate metrics across all horizons into a DataFrame.

    Parameters
    ----------
    results_by_horizon : dict
        Results from train_all_horizons_parallel()

    Returns
    -------
    pd.DataFrame
        Metrics for each horizon with columns:
        - target_time
        - horizon (delta_t)
        - mae, rmse, r2
        - n_predictions, n_embryos
    """
    metrics_list = []

    for target_time in sorted(results_by_horizon.keys()):
        result = results_by_horizon[target_time]
        metrics_list.append(result['metrics'])

    return pd.DataFrame(metrics_list)


def compare_models_across_horizons(
    results_dict: Dict[str, Dict[float, Dict]]
) -> pd.DataFrame:
    """
    Compare all models across all horizons.

    Parameters
    ----------
    results_dict : dict
        {
            'linear': {32.0: result, 34.0: result, ...},
            'ridge': {32.0: result, 34.0: result, ...},
            ...
        }

    Returns
    -------
    pd.DataFrame
        Long-form table with columns:
        - model_type
        - target_time
        - horizon
        - mae, rmse, r2
        - n_predictions, n_embryos
    """
    all_metrics = []

    for model_type, results_by_horizon in results_dict.items():
        df_metrics = aggregate_metrics_across_horizons(results_by_horizon)
        all_metrics.append(df_metrics)

    df_all = pd.concat(all_metrics, ignore_index=True)

    # Sort by model_type and horizon
    df_all = df_all.sort_values(['model_type', 'horizon'])

    return df_all


def get_best_model_per_horizon(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each horizon, identify which model performs best.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        From compare_models_across_horizons()

    Returns
    -------
    pd.DataFrame
        Best model per horizon with columns:
        - horizon
        - best_model
        - mae
        - improvement_over_second (%)
    """
    best_per_horizon = []

    for horizon in sorted(comparison_df['horizon'].unique()):
        subset = comparison_df[comparison_df['horizon'] == horizon].copy()
        subset = subset.sort_values('mae')

        if len(subset) == 0:
            continue

        best = subset.iloc[0]
        second_best_mae = subset.iloc[1]['mae'] if len(subset) > 1 else np.nan

        improvement = ((second_best_mae - best['mae']) / second_best_mae * 100) if not np.isnan(second_best_mae) else np.nan

        best_per_horizon.append({
            'horizon': horizon,
            'target_time': best['target_time'],
            'best_model': best['model_type'],
            'mae': best['mae'],
            'r2': best['r2'],
            'improvement_over_second_pct': improvement
        })

    return pd.DataFrame(best_per_horizon)
