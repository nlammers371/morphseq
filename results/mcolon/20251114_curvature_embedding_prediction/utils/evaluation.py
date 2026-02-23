"""
Evaluation metrics and result aggregation for regression models.

Provides utilities for comparing models, computing feature importance,
and aggregating cross-validation results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multioutput: bool = False
) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    multioutput : bool
        If True, handle multiple outputs (n_samples, n_targets)

    Returns
    -------
    dict
        Metrics: 'r2', 'rmse', 'mae', 'mape'
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
    }

    # MAPE (avoid division by zero)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    metrics['mape'] = mape

    return metrics


def aggregate_loeo_results(
    fold_results_dict: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Aggregate per-embryo cross-validation results into a summary.

    Parameters
    ----------
    fold_results_dict : dict
        From train_regression_model_loeo()['fold_results']
        {embryo_id: {'r2': ..., 'rmse': ..., 'mae': ..., 'n_samples': ...}}

    Returns
    -------
    pd.DataFrame
        Summary statistics for each fold
    """
    df_list = []
    for embryo_id, metrics in fold_results_dict.items():
        row = {'embryo_id': embryo_id, **metrics}
        df_list.append(row)

    df_summary = pd.DataFrame(df_list)

    return df_summary


def get_feature_importance(
    model: Any,
    feature_cols: List[str],
    model_type: str = 'gradient_boosting'
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.

    Parameters
    ----------
    model : sklearn model
        Fitted regression model
    feature_cols : list of str
        Feature column names
    model_type : str
        Type of model ('gradient_boosting', 'random_forest', etc.)

    Returns
    -------
    pd.DataFrame
        Feature importance with columns: 'feature', 'importance'
    """
    importances = []

    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        for feat, imp in zip(feature_cols, model.feature_importances_):
            importances.append({
                'feature': feat,
                'importance': imp
            })
    elif hasattr(model, 'coef_'):
        # Linear models - use absolute coefficient values
        for feat, coef in zip(feature_cols, np.abs(model.coef_)):
            importances.append({
                'feature': feat,
                'importance': coef
            })
    else:
        # Model doesn't support feature importance
        return pd.DataFrame()

    df_imp = pd.DataFrame(importances)

    # Normalize to sum to 1
    if len(df_imp) > 0:
        total = df_imp['importance'].sum()
        if total > 0:
            df_imp['importance_normalized'] = df_imp['importance'] / total
        else:
            df_imp['importance_normalized'] = 0.0

        df_imp = df_imp.sort_values('importance', ascending=False)

    return df_imp


def compare_multiple_models(
    results_dict: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare multiple trained models side-by-side.

    Parameters
    ----------
    results_dict : dict
        {model_name: result_dict_from_training, ...}

    Returns
    -------
    pd.DataFrame
        Comparison table with model performance metrics
    """
    comparison_rows = []

    for model_name, result in results_dict.items():
        row = {
            'model': model_name,
            'r2': result['metrics']['r2'],
            'rmse': result['metrics']['rmse'],
            'mae': result['metrics']['mae'],
            'n_samples': result['metrics']['n_samples']
        }
        comparison_rows.append(row)

    df_comparison = pd.DataFrame(comparison_rows)

    # Sort by MAE (lower is better)
    df_comparison = df_comparison.sort_values('mae')

    return df_comparison


def compute_residual_statistics(
    predictions_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute statistics on prediction residuals.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        From train_regression_model_loeo()['predictions']

    Returns
    -------
    dict
        Residual statistics
    """
    residuals = predictions_df['residual'].values

    stats = {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals),
        'abs_mean_residual': np.mean(np.abs(residuals)),
    }

    return stats


def compute_prediction_error_by_group(
    predictions_df: pd.DataFrame,
    group_col: str
) -> pd.DataFrame:
    """
    Compute prediction errors stratified by a grouping column.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        With 'actual', 'predicted' columns and group_col
    group_col : str
        Column name for grouping (e.g., 'genotype', 'embryo_id')

    Returns
    -------
    pd.DataFrame
        Error metrics per group
    """
    if group_col not in predictions_df.columns:
        # Just return overall metrics
        return compute_regression_metrics(
            predictions_df['actual'].values,
            predictions_df['predicted'].values
        )

    results_by_group = []

    for group_val in predictions_df[group_col].unique():
        mask = predictions_df[group_col] == group_val
        subset = predictions_df[mask]

        metrics = compute_regression_metrics(
            subset['actual'].values,
            subset['predicted'].values
        )

        metrics['group'] = group_val
        metrics['n_samples'] = len(subset)

        results_by_group.append(metrics)

    df_results = pd.DataFrame(results_by_group)

    return df_results
