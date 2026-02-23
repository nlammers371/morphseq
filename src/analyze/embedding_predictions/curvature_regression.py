"""
Ridge-regression utilities for linking morphological embeddings to curvature metrics.

The goal is to quantify how well continuous curvature readouts (e.g., mean
curvature per micron, baseline deviation) can be predicted from VAE latent
embeddings.  This module keeps the orchestration lightweight so it can be
imported from notebooks, scripts, or CLI helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze.trajectory_analysis.io import load_experiment_dataframe


DEFAULT_LATENT_PREFIX = "z_mu"
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0)


@dataclass
class CurvatureRegressionOutputs:
    """Container for regression artifacts."""

    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame


def _select_latent_columns(
    df: pd.DataFrame,
    latent_cols: Optional[Sequence[str]],
    latent_prefix: str
) -> List[str]:
    if latent_cols:
        missing = [col for col in latent_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Latent columns not found in dataframe: {missing}")
        return list(latent_cols)

    auto = [c for c in df.columns if c.startswith(latent_prefix)]
    if not auto:
        raise ValueError(
            f"No latent columns detected using prefix '{latent_prefix}'. "
            "Provide `latent_cols` explicitly."
        )
    return auto


def _resolve_column_name(df: pd.DataFrame, requested: str) -> str:
    """
    Handle columns that may have suffixes after merges (e.g., foo -> foo_x).
    """
    if requested in df.columns:
        return requested

    suffix_matches = [col for col in df.columns if col.startswith(f"{requested}_")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        preferred = [col for col in suffix_matches if col.endswith("_x")]
        if len(preferred) == 1:
            return preferred[0]
        raise ValueError(
            f"Ambiguous matches for '{requested}': {suffix_matches}. "
            "Specify the exact column name via --target-metric."
        )
    raise ValueError(f"Column '{requested}' not found in dataframe.")


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Normalize boolean-ish strings to actual bool dtype."""
    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["1", "true", "t", "yes"])


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return np.nan

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.allclose(np.std(y_true), 0) or np.allclose(np.std(y_pred), 0):
        return np.nan

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr)


def _build_cv(
    groups: Optional[np.ndarray],
    n_splits: int
) -> Tuple[Iterable[Tuple[np.ndarray, np.ndarray]], str]:
    """
    Prepare the cross-validator along with a description of how splits were built.
    """
    if groups is not None:
        unique = np.unique(groups)
        usable_splits = min(n_splits, len(unique))
        if usable_splits < 2:
            raise ValueError(
                "Not enough unique groups to run grouped cross-validation. "
                f"Need at least 2, found {len(unique)}."
            )
        return GroupKFold(n_splits=usable_splits), f"GroupKFold({usable_splits})"

    if n_splits < 2:
        raise ValueError("Need at least 2 splits for KFold CV.")

    return KFold(n_splits=n_splits, shuffle=True, random_state=42), f"KFold({n_splits}, shuffle=True)"


def prepare_curvature_dataframe(
    experiment_id: str,
    format_version: str = "df03",
    df: Optional[pd.DataFrame] = None,
    curvature_success_col: str = "curvature_success",
    require_success: bool = True,
    additional_filters: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load and lightly filter the merged curvature + embedding dataframe.
    """
    if df is None:
        df = load_experiment_dataframe(experiment_id, format_version=format_version)

    df_clean = df.copy()

    if curvature_success_col in df_clean.columns and require_success:
        df_clean = df_clean[_coerce_bool(df_clean[curvature_success_col])]

    if additional_filters:
        for col, value in additional_filters.items():
            if col not in df_clean.columns:
                raise ValueError(f"Filter column '{col}' not present in dataframe.")
            df_clean = df_clean[df_clean[col] == value]

    df_clean = df_clean.reset_index(drop=True)
    return df_clean


def predict_curvature_from_embeddings(
    experiment_id: str,
    target_metric: str,
    format_version: str = "df03",
    df: Optional[pd.DataFrame] = None,
    latent_cols: Optional[Sequence[str]] = None,
    latent_prefix: str = DEFAULT_LATENT_PREFIX,
    group_col: str = "embryo_id",
    metadata_cols: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    inner_cv_splits: int = 4,
    require_success: bool = True,
    curvature_success_col: str = "curvature_success",
    additional_filters: Optional[Dict[str, str]] = None,
) -> CurvatureRegressionOutputs:
    """
    Fit ridge regression models that predict curvature metrics from latent embeddings.
    """
    df_full = prepare_curvature_dataframe(
        experiment_id=experiment_id,
        format_version=format_version,
        df=df,
        curvature_success_col=curvature_success_col,
        require_success=require_success,
        additional_filters=additional_filters,
    )

    try:
        target_col = _resolve_column_name(df_full, target_metric)
    except ValueError as exc:
        raise ValueError(
            f"Target metric '{target_metric}' not found. "
            f"Available columns include: {df_full.columns.tolist()[:50]}"
        ) from exc

    latent_cols = _select_latent_columns(df_full, latent_cols, latent_prefix)

    df_model = df_full.dropna(subset=[target_col]).copy()
    if df_model.empty:
        raise ValueError(f"No rows left after filtering for target '{target_metric}'.")

    X = df_model[latent_cols].to_numpy(dtype=float)
    y = df_model[target_col].to_numpy(dtype=float)

    mask = np.isfinite(y).astype(bool)
    if not mask.all():
        X = X[mask]
        y = y[mask]
        df_model = df_model.loc[mask].reset_index(drop=True)

    groups = df_model[group_col].to_numpy() if group_col in df_model.columns else None

    if metadata_cols is None:
        metadata_cols = [
            col for col in [
                "embryo_id",
                "snip_id",
                "predicted_stage_hpf",
                "genotype_output",
                "genotype_metadata",
                "chem_perturbation_metadata",
            ] if col in df_model.columns
        ]

    cross_val, cv_description = _build_cv(groups, n_splits)

    fold_records: List[Dict[str, float]] = []
    prediction_rows: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cross_val.split(X, y, groups), start=1):
        inner_splits = min(inner_cv_splits, len(np.unique(train_idx)))
        if inner_splits < 2:
            inner_splits = 2

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", RidgeCV(alphas=alpha_grid, cv=inner_splits))
        ])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        alpha = float(pipeline.named_steps["ridge"].alpha_)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(mean_squared_error(y_test, y_pred, squared=False))
        pearson = _pearson_r(y_test, y_pred)

        fold_records.append({
            "fold": fold_idx,
            "alpha": alpha,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "pearson_r": pearson,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        })

        meta_subset = df_model.iloc[test_idx][metadata_cols] if metadata_cols else None
        for row_idx, y_true_val, y_pred_val in zip(test_idx, y_test, y_pred):
            row_dict: Dict[str, object] = {
                "fold": fold_idx,
                "target_metric": target_metric,
                "y_true": float(y_true_val),
                "y_pred": float(y_pred_val),
            }
            if meta_subset is not None:
                # Align meta rows with order of test_idx
                meta_row = df_model.iloc[row_idx]
                for col in metadata_cols:
                    row_dict[col] = meta_row[col]
            prediction_rows.append(row_dict)

    df_folds = pd.DataFrame(fold_records)
    df_predictions = pd.DataFrame(prediction_rows)

    summary_data = {
        "experiment_id": experiment_id,
        "target_metric": target_metric,
        "n_samples": int(len(df_model)),
        "n_embryos": int(df_model[group_col].nunique()) if group_col in df_model.columns else np.nan,
        "cv_strategy": cv_description,
        "alpha_median": float(df_folds["alpha"].median()),
        "alpha_iqr": float(df_folds["alpha"].quantile(0.75) - df_folds["alpha"].quantile(0.25)),
        "r2_mean": float(df_folds["r2"].mean()),
        "r2_std": float(df_folds["r2"].std(ddof=0)),
        "mae_mean": float(df_folds["mae"].mean()),
        "rmse_mean": float(df_folds["rmse"].mean()),
        "pearson_mean": float(df_folds["pearson_r"].mean()),
    }
    df_summary = pd.DataFrame([summary_data])

    return CurvatureRegressionOutputs(
        summary=df_summary,
        fold_metrics=df_folds,
        predictions=df_predictions,
    )


def save_regression_outputs(
    outputs: CurvatureRegressionOutputs,
    output_dir: Path,
    target_metric: str,
) -> Dict[str, Path]:
    """
    Persist summary tables and diagnostic plots to disk.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_target = target_metric.replace("/", "_")

    summary_path = output_dir / f"{safe_target}_summary.csv"
    folds_path = output_dir / f"{safe_target}_fold_metrics.csv"
    preds_path = output_dir / f"{safe_target}_predictions.csv"

    outputs.summary.to_csv(summary_path, index=False)
    outputs.fold_metrics.to_csv(folds_path, index=False)
    outputs.predictions.to_csv(preds_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(outputs.predictions["y_true"], outputs.predictions["y_pred"], alpha=0.4, s=15)
    lims = [
        min(outputs.predictions["y_true"].min(), outputs.predictions["y_pred"].min()),
        max(outputs.predictions["y_true"].max(), outputs.predictions["y_pred"].max()),
    ]
    ax.plot(lims, lims, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Observed {target_metric}")
    ax.set_ylabel(f"Predicted {target_metric}")
    ax.set_title(f"Ridge CV predictions ({target_metric})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()

    plot_path = output_dir / f"{safe_target}_pred_vs_true.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    return {
        "summary": summary_path,
        "fold_metrics": folds_path,
        "predictions": preds_path,
        "scatter_plot": plot_path,
    }
