"""
Trajectory prediction for penetrance detection.

This module uses temporal trajectory prediction to classify penetrance:
- Train model on homozygous data → learn "mutant trajectory"
- Train model on WT data → learn "normal trajectory"
- Compare prediction errors to classify penetrance

Key insight: Penetrant embryos follow mutant trajectory, non-penetrant follow WT trajectory.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, List, Optional
import warnings


def prepare_trajectory_data(
    df_binned: pd.DataFrame,
    genotype: str,
    prediction_horizons: List[int] = [2, 4, 6, 8],
    z_cols: Optional[List[str]] = None,
    time_col: str = 'time_bin',
    distance_col: str = 'euclidean_distance',
    min_timepoints: int = 3
) -> pd.DataFrame:
    """
    Prepare training data for trajectory prediction.

    For each embryo at time i, create features to predict distance at i+k.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data with embeddings and distances
    genotype : str
        Genotype to filter
    prediction_horizons : list
        List of time gaps to predict (k values)
    z_cols : list, optional
        Embedding column names
    time_col : str
        Time column name
    distance_col : str
        Distance column name
    min_timepoints : int
        Minimum timepoints per embryo

    Returns
    -------
    pd.DataFrame
        Training data with columns:
        - embryo_id
        - time_i: Starting time
        - delta_t: Time gap
        - embedding features (z0_binned, z1_binned, ...)
        - target: distance at time i+k
    """
    # Filter to genotype
    df_genotype = df_binned[df_binned['genotype'] == genotype].copy()

    # Auto-detect embedding columns
    if z_cols is None:
        z_cols = [c for c in df_genotype.columns if c.endswith('_binned')]

    # Filter embryos with sufficient timepoints
    embryo_counts = df_genotype.groupby('embryo_id').size()
    valid_embryos = embryo_counts[embryo_counts >= min_timepoints].index
    df_genotype = df_genotype[df_genotype['embryo_id'].isin(valid_embryos)]

    print(f"  Using {len(valid_embryos)} embryos with ≥{min_timepoints} timepoints")

    training_data = []

    for embryo_id, embryo_data in df_genotype.groupby('embryo_id'):
        embryo_data = embryo_data.sort_values(time_col)

        timepoints = embryo_data[time_col].values
        embeddings = embryo_data[z_cols].values
        distances = embryo_data[distance_col].values if distance_col in embryo_data.columns else None

        # For each timepoint i, try to predict i+k for each horizon k
        for idx, time_i in enumerate(timepoints[:-1]):  # Exclude last point (no future)
            current_embedding = embeddings[idx]

            # Try each prediction horizon
            for delta_t in prediction_horizons:
                # Find closest future timepoint at approximately time_i + delta_t
                future_times = timepoints[idx+1:]
                target_time = time_i + delta_t

                # Find timepoint closest to target
                if len(future_times) == 0:
                    continue

                closest_idx = np.argmin(np.abs(future_times - target_time))
                actual_future_time = future_times[closest_idx]

                # Only use if within tolerance (±1 hpf)
                if abs(actual_future_time - target_time) > 1.0:
                    continue

                future_distance = distances[idx + 1 + closest_idx] if distances is not None else np.nan

                # Create training example
                example = {
                    'embryo_id': embryo_id,
                    'time_i': time_i,
                    'delta_t': delta_t,
                    'actual_time_future': actual_future_time,
                    **{f'z{i}': current_embedding[i] for i in range(len(current_embedding))},
                    'target_distance': future_distance
                }

                training_data.append(example)

    df_train = pd.DataFrame(training_data)

    # Remove rows with NaN targets
    df_train = df_train.dropna(subset=['target_distance'])

    print(f"  Generated {len(df_train)} training examples from {len(valid_embryos)} embryos")

    return df_train


def train_trajectory_model(
    df_train: pd.DataFrame,
    model_type: str = 'random_forest',
    **model_kwargs
) -> Tuple:
    """
    Train trajectory prediction model.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data from prepare_trajectory_data()
    model_type : str
        Model type: 'linear', 'random_forest', 'gradient_boosting'
    **model_kwargs
        Additional arguments for model

    Returns
    -------
    model : trained model
    feature_cols : list of feature column names
    metrics : dict of training metrics
    """
    # Extract features
    feature_cols = ['time_i', 'delta_t'] + [c for c in df_train.columns if c.startswith('z')]
    X = df_train[feature_cols].values
    y = df_train['target_distance'].values

    # Choose model
    if model_type == 'linear':
        model = Ridge(alpha=1.0, **model_kwargs)
    elif model_type == 'random_forest':
        default_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
        default_params.update(model_kwargs)
        model = RandomForestRegressor(**default_params)
    elif model_type == 'gradient_boosting':
        default_params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        default_params.update(model_kwargs)
        model = GradientBoostingRegressor(**default_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train
    print(f"  Training {model_type} model on {len(X)} examples...")
    model.fit(X, y)

    # Evaluate on training data
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_train': len(X)
    }

    print(f"  Training MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return model, feature_cols, metrics


def predict_with_model(
    model,
    df_predict: pd.DataFrame,
    feature_cols: List[str]
) -> np.ndarray:
    """
    Make predictions using trained model.

    Parameters
    ----------
    model : trained model
        Trajectory prediction model
    df_predict : pd.DataFrame
        Data to predict (must have feature_cols)
    feature_cols : list
        Feature column names used in training

    Returns
    -------
    np.ndarray
        Predicted distances
    """
    X = df_predict[feature_cols].values
    predictions = model.predict(X)
    return predictions


def cross_validate_trajectory_model(
    df_train: pd.DataFrame,
    model_type: str = 'random_forest',
    cv_method: str = 'embryo_loo',
    **model_kwargs
) -> Dict[str, float]:
    """
    Cross-validate trajectory model.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data
    model_type : str
        Model type
    cv_method : str
        'embryo_loo': Leave-one-embryo-out
        'standard': Standard k-fold CV
    **model_kwargs
        Model parameters

    Returns
    -------
    dict
        Cross-validation metrics
    """
    feature_cols = ['time_i', 'delta_t'] + [c for c in df_train.columns if c.startswith('z')]
    X = df_train[feature_cols].values
    y = df_train['target_distance'].values

    if cv_method == 'embryo_loo':
        # Leave-one-embryo-out
        embryo_ids = df_train['embryo_id'].unique()
        print(f"  Cross-validating with {len(embryo_ids)}-fold leave-one-embryo-out...")

        all_predictions = []
        all_actuals = []

        for test_embryo in embryo_ids:
            # Split data
            train_mask = df_train['embryo_id'] != test_embryo
            test_mask = df_train['embryo_id'] == test_embryo

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            # Train model
            if model_type == 'linear':
                model = Ridge(alpha=1.0, **model_kwargs)
            elif model_type == 'random_forest':
                default_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
                default_params.update(model_kwargs)
                model = RandomForestRegressor(**default_params)
            elif model_type == 'gradient_boosting':
                default_params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
                default_params.update(model_kwargs)
                model = GradientBoostingRegressor(**default_params)

            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

    else:
        # Standard cross-validation
        print(f"  Cross-validating with 5-fold CV...")
        # Not implemented for simplicity
        raise NotImplementedError("Use embryo_loo for now")

    mae = mean_absolute_error(all_actuals, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    r2 = r2_score(all_actuals, all_predictions)

    print(f"  CV MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return {
        'cv_mae': mae,
        'cv_rmse': rmse,
        'cv_r2': r2
    }


def predict_all_trajectories(
    df_data: pd.DataFrame,
    model,
    feature_cols: List[str],
    prediction_horizons: List[int] = [2, 4, 6, 8]
) -> pd.DataFrame:
    """
    Predict trajectories for all embryos in dataset.

    Parameters
    ----------
    df_data : pd.DataFrame
        Data prepared with prepare_trajectory_data()
    model : trained model
        Trajectory prediction model
    feature_cols : list
        Feature columns
    prediction_horizons : list
        Prediction horizons used

    Returns
    -------
    pd.DataFrame
        Predictions with columns: embryo_id, time_i, delta_t, predicted_distance, actual_distance, error
    """
    # Make predictions
    predictions = predict_with_model(model, df_data, feature_cols)

    # Create results dataframe
    results = df_data[['embryo_id', 'time_i', 'delta_t', 'actual_time_future', 'target_distance']].copy()
    results['predicted_distance'] = predictions
    results['prediction_error'] = np.abs(results['target_distance'] - results['predicted_distance'])
    results.rename(columns={'target_distance': 'actual_distance'}, inplace=True)

    return results


def compute_per_embryo_prediction_metrics(
    predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate prediction metrics per embryo.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Results from predict_all_trajectories()

    Returns
    -------
    pd.DataFrame
        Per-embryo metrics: mean_error, std_error, n_predictions
    """
    embryo_metrics = predictions_df.groupby('embryo_id').agg({
        'prediction_error': ['mean', 'std', 'count'],
        'actual_distance': ['mean', 'std'],
        'predicted_distance': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    embryo_metrics.columns = [
        'embryo_id',
        'mean_prediction_error',
        'std_prediction_error',
        'n_predictions',
        'mean_actual_distance',
        'std_actual_distance',
        'mean_predicted_distance',
        'std_predicted_distance'
    ]

    return embryo_metrics


def classify_penetrance_by_dual_models(
    homo_model_predictions: pd.DataFrame,
    wt_model_predictions: pd.DataFrame,
    error_ratio_threshold: float = 1.5,
    min_predictions: int = 3
) -> pd.DataFrame:
    """
    Classify penetrance by comparing homo vs WT model performance.

    Parameters
    ----------
    homo_model_predictions : pd.DataFrame
        Predictions from homozygous-trained model
    wt_model_predictions : pd.DataFrame
        Predictions from WT-trained model
    error_ratio_threshold : float
        Threshold for error_wt / error_homo to call penetrant
    min_predictions : int
        Minimum predictions required per embryo

    Returns
    -------
    pd.DataFrame
        Classification with columns:
        - embryo_id
        - mean_error_homo, mean_error_wt
        - error_ratio
        - penetrance_status: 'penetrant', 'non-penetrant', 'intermediate'
        - confidence: How far from threshold
    """
    # Per-embryo metrics
    homo_metrics = compute_per_embryo_prediction_metrics(homo_model_predictions)
    wt_metrics = compute_per_embryo_prediction_metrics(wt_model_predictions)

    # Merge
    classification = homo_metrics[['embryo_id', 'mean_prediction_error', 'n_predictions']].copy()
    classification.rename(columns={'mean_prediction_error': 'mean_error_homo'}, inplace=True)

    wt_errors = wt_metrics[['embryo_id', 'mean_prediction_error']].copy()
    wt_errors.rename(columns={'mean_prediction_error': 'mean_error_wt'}, inplace=True)

    classification = classification.merge(wt_errors, on='embryo_id')

    # Filter by minimum predictions
    classification = classification[classification['n_predictions'] >= min_predictions]

    # Compute error ratio
    classification['error_ratio'] = classification['mean_error_wt'] / classification['mean_error_homo']

    # Classify
    def classify_status(ratio, threshold):
        if ratio > threshold:
            return 'penetrant'
        elif ratio < 1.0 / threshold:
            return 'non-penetrant'
        else:
            return 'intermediate'

    classification['penetrance_status'] = classification['error_ratio'].apply(
        lambda r: classify_status(r, error_ratio_threshold)
    )

    # Compute confidence (distance from threshold)
    classification['confidence'] = np.abs(np.log(classification['error_ratio'] / error_ratio_threshold))

    return classification


def detect_penetrance_onset(
    predictions_df: pd.DataFrame,
    error_ratio_threshold: float = 1.5,
    window_size: int = 2
) -> pd.DataFrame:
    """
    Detect when penetrance emerges (when error ratio crosses threshold).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Combined predictions from both models with error_ratio column
    error_ratio_threshold : float
        Threshold for penetrance
    window_size : int
        Number of consecutive timepoints above threshold required

    Returns
    -------
    pd.DataFrame
        Per-embryo onset times
    """
    # Compute error ratio per prediction
    # (This requires both model predictions in same df - handle merging upstream)

    onset_results = []

    for embryo_id, embryo_data in predictions_df.groupby('embryo_id'):
        embryo_data = embryo_data.sort_values('time_i')

        # Find first sustained crossing
        crosses_threshold = embryo_data['error_ratio'] > error_ratio_threshold

        onset_time = np.nan
        for idx in range(len(crosses_threshold) - window_size + 1):
            if np.all(crosses_threshold.iloc[idx:idx+window_size]):
                onset_time = embryo_data['time_i'].iloc[idx]
                break

        onset_results.append({
            'embryo_id': embryo_id,
            'onset_time': onset_time,
            'has_onset': not np.isnan(onset_time)
        })

    return pd.DataFrame(onset_results)
