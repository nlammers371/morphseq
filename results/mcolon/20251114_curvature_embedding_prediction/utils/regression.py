"""
Regression models for continuous target prediction from embeddings.

Supports:
- Ridge regression (regularized linear)
- Gradient Boosting (ensemble)
- Leave-one-embryo-out cross-validation

This module is designed to be migrated to src/analyze/difference_detection/classification/
after validation in this analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')


def get_model(model_type: str, **kwargs) -> Any:
    """
    Get sklearn model instance by name.

    Parameters
    ----------
    model_type : str
        One of: 'linear', 'ridge', 'lasso', 'gradient_boosting', 'random_forest'
    **kwargs
        Model-specific hyperparameters

    Returns
    -------
    model
        Sklearn-compatible regression model instance
    """
    if model_type == 'linear':
        return LinearRegression(**kwargs)

    elif model_type == 'ridge':
        alpha = kwargs.get('alpha', 1.0)
        return Ridge(alpha=alpha, random_state=42)

    elif model_type == 'lasso':
        alpha = kwargs.get('alpha', 0.1)
        return Lasso(alpha=alpha, random_state=42, max_iter=10000)

    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 3),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42,
            verbose=0
        )

    elif model_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_regression_model_loeo(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    embryo_id_col: str = 'embryo_id',
    model_type: str = 'ridge',
    scale_features: bool = True,
    verbose: bool = True,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Train regression model with leave-one-embryo-out cross-validation.

    This prevents data leakage by ensuring all samples from a given embryo
    are either in train OR test set, never split across both.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with features, target, and embryo IDs
    feature_cols : list of str
        Column names for features (X)
    target_col : str
        Column name for target (y)
    embryo_id_col : str
        Column name identifying embryos
    model_type : str
        Type of model ('ridge', 'gradient_boosting', 'linear', etc.)
    scale_features : bool
        Whether to standardize features
    verbose : bool
        Print progress information
    **model_kwargs
        Model-specific hyperparameters

    Returns
    -------
    dict
        {
            'model': Final model trained on all data
            'scalers': dict of feature and target scalers
            'predictions': pd.DataFrame with 'actual', 'predicted', 'embryo_id'
            'metrics': dict with 'r2', 'rmse', 'mae'
            'fold_results': dict with per-embryo metrics
            'feature_cols': list of feature columns
        }
    """
    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    embryo_ids = df[embryo_id_col].values

    # Setup scalers
    scaler_X = StandardScaler() if scale_features else None
    scaler_y = StandardScaler()

    # Scale features and target
    if scale_features:
        X_scaled = scaler_X.fit_transform(X)
    else:
        X_scaled = X.copy()

    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # LOEO cross-validation
    unique_embryos = np.unique(embryo_ids)
    y_pred_list = []
    y_true_list = []
    embryo_pred_list = []
    fold_results = {}

    if verbose:
        print(f"\n  Training {model_type} model with LOEO CV...")
        print(f"    Features: {len(feature_cols)}")
        print(f"    Samples: {len(df)}")
        print(f"    Embryos: {len(unique_embryos)}")

    for test_embryo in unique_embryos:
        # Split on embryo
        train_mask = embryo_ids != test_embryo
        test_mask = embryo_ids == test_embryo

        X_train = X_scaled[train_mask]
        X_test = X_scaled[test_mask]
        y_train = y_scaled[train_mask]
        y_test = y_scaled[test_mask]

        # Train model
        model = get_model(model_type, **model_kwargs)
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred_scaled = model.predict(X_test)

        # Inverse transform to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        # Store results
        y_pred_list.extend(y_pred)
        y_true_list.extend(y_test_orig)
        embryo_pred_list.extend([test_embryo] * len(y_pred))

        # Compute fold metrics
        fold_r2 = r2_score(y_test_orig, y_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        fold_mae = mean_absolute_error(y_test_orig, y_pred)

        fold_results[test_embryo] = {
            'r2': fold_r2,
            'rmse': fold_rmse,
            'mae': fold_mae,
            'n_samples': len(y_pred)
        }

    # Aggregate results
    y_pred_array = np.array(y_pred_list)
    y_true_array = np.array(y_true_list)

    r2 = r2_score(y_true_array, y_pred_array)
    rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))
    mae = mean_absolute_error(y_true_array, y_pred_array)

    if verbose:
        print(f"\n  LOEO Results ({model_type}):")
        print(f"    R²:   {r2:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE:  {mae:.4f}")

    # Train final model on all data for inference
    model_final = get_model(model_type, **model_kwargs)
    model_final.fit(X_scaled, y_scaled)

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'actual': y_true_array,
        'predicted': y_pred_array,
        'embryo_id': embryo_pred_list,
        'residual': y_true_array - y_pred_array
    })

    return {
        'model': model_final,
        'scalers': {
            'X': scaler_X,
            'y': scaler_y
        },
        'predictions': predictions_df,
        'metrics': {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_true_array)
        },
        'fold_results': fold_results,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'model_type': model_type,
    }


def predict_with_trained_model(
    model_dict: Dict[str, Any],
    X_new: np.ndarray
) -> np.ndarray:
    """
    Make predictions with a trained model and apply inverse scaling.

    Parameters
    ----------
    model_dict : dict
        Result dict from train_regression_model_loeo()
    X_new : np.ndarray
        New feature data (n_samples, n_features)

    Returns
    -------
    np.ndarray
        Predictions in original target scale
    """
    model = model_dict['model']
    scaler_X = model_dict['scalers']['X']
    scaler_y = model_dict['scalers']['y']

    # Scale features
    if scaler_X is not None:
        X_new_scaled = scaler_X.transform(X_new)
    else:
        X_new_scaled = X_new

    # Predict on scaled values
    y_pred_scaled = model.predict(X_new_scaled)

    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    return y_pred


def train_regression_model_holdout(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    embryo_id_col: str = 'embryo_id',
    model_type: str = 'ridge',
    test_fraction: float = 0.20,
    scale_features: bool = True,
    random_state: int = 42,
    verbose: bool = True,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Train regression model with 80/20 embryo holdout test set.

    This prevents data leakage by ensuring all samples from a given embryo
    are either in train OR test set, never split across both.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with features, target, and embryo IDs
    feature_cols : list of str
        Column names for features (X)
    target_col : str
        Column name for target (y)
    embryo_id_col : str
        Column name identifying embryos
    model_type : str
        Type of model ('ridge', 'linear', etc.)
    test_fraction : float
        Fraction of embryos to reserve for testing (default 0.20)
    scale_features : bool
        Whether to standardize features
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print progress information
    **model_kwargs
        Model-specific hyperparameters

    Returns
    -------
    dict
        {
            'model': Trained model on train set
            'scalers': dict of feature and target scalers
            'predictions': pd.DataFrame with 'actual', 'predicted'
            'metrics': dict with 'r2', 'mae'
            'feature_cols': list of feature columns
            'split': dict with train/test embryo lists
        }
    """
    np.random.seed(random_state)

    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    embryo_ids = df[embryo_id_col].values

    # Setup scalers
    scaler_X = StandardScaler() if scale_features else None
    scaler_y = StandardScaler()

    # Scale features and target
    if scale_features:
        X_scaled = scaler_X.fit_transform(X)
    else:
        X_scaled = X.copy()

    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Split on embryos
    unique_embryos = np.unique(embryo_ids)
    n_test_embryos = max(1, int(len(unique_embryos) * test_fraction))
    test_embryos = np.random.choice(unique_embryos, size=n_test_embryos, replace=False)
    train_embryos = np.setdiff1d(unique_embryos, test_embryos)

    # Create masks
    train_mask = np.isin(embryo_ids, train_embryos)
    test_mask = np.isin(embryo_ids, test_embryos)

    X_train = X_scaled[train_mask]
    X_test = X_scaled[test_mask]
    y_train = y_scaled[train_mask]
    y_test = y_scaled[test_mask]

    if verbose:
        print(f"\n  Training {model_type.upper()} with holdout test set...")
        print(f"    Features: {len(feature_cols)}")
        print(f"    Train samples: {len(X_train)} from {len(train_embryos)} embryos")
        print(f"    Test samples: {len(X_test)} from {len(test_embryos)} embryos")

    # Train model on train set
    model = get_model(model_type, **model_kwargs)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred_scaled = model.predict(X_test)

    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Compute metrics
    r2 = r2_score(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)

    if verbose:
        print(f"    R²:  {r2:.4f}")
        print(f"    MAE: {mae:.4f}")

    # Build predictions dataframe
    test_embryo_ids = embryo_ids[test_mask]
    predictions_df = pd.DataFrame({
        'actual': y_test_orig,
        'predicted': y_pred,
        'embryo_id': test_embryo_ids,
    })

    return {
        'model': model,
        'scalers': {'X': scaler_X, 'y': scaler_y},
        'predictions': predictions_df,
        'metrics': {'r2': r2, 'mae': mae},
        'feature_cols': feature_cols,
        'split': {'train_embryos': train_embryos, 'test_embryos': test_embryos},
    }
