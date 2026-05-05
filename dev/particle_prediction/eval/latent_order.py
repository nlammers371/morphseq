"""Latent feature-correlation order summaries by temperature and stage bin."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def rms_offdiag_corr(R: np.ndarray) -> float:
    """Return RMS off-diagonal correlation."""

    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square 2D array")
    if R.shape[0] < 2:
        return np.nan
    mask = ~np.eye(R.shape[0], dtype=bool)
    return float(np.sqrt(np.mean(R[mask] ** 2)))


def mean_abs_offdiag_corr(R: np.ndarray) -> float:
    """Return mean absolute off-diagonal correlation."""

    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square 2D array")
    if R.shape[0] < 2:
        return np.nan
    mask = ~np.eye(R.shape[0], dtype=bool)
    return float(np.mean(np.abs(R[mask])))


def effective_rank(R: np.ndarray, eps: float = 1e-12) -> float:
    """Return entropy-based effective rank of a correlation matrix."""

    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square 2D array")
    if R.shape[0] < 1 or not np.isfinite(R).all():
        return np.nan

    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.clip(eigvals, 0, None)
    eigval_sum = eigvals.sum()
    if eigval_sum <= 0:
        return np.nan
    q = eigvals / eigval_sum
    entropy = -np.sum(q * np.log(q + eps))
    return float(np.exp(entropy))


def top_k_eigen_fraction(R: np.ndarray, k: int = 3) -> float:
    """Return the fraction of correlation-matrix spectrum captured by top k eigenvalues."""

    if k < 1:
        raise ValueError("k must be positive")

    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square 2D array")
    if R.shape[0] < 1 or not np.isfinite(R).all():
        return np.nan

    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.clip(eigvals, 0, None)
    eigval_sum = eigvals.sum()
    if eigval_sum <= 0:
        return np.nan
    eigvals = np.sort(eigvals)[::-1]
    return float(eigvals[:k].sum() / eigval_sum)


def latent_corr_order_summary(Z: np.ndarray) -> dict[str, float | int]:
    """Summarize order in a latent feature-feature correlation matrix."""

    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array of shape (n_samples, n_latent_features)")

    n_samples, n_latent_features = Z.shape
    if n_samples < 2 or n_latent_features < 1:
        return {
            "n_samples": int(n_samples),
            "n_latent_features": int(n_latent_features),
            "rms_offdiag_corr": np.nan,
            "mean_abs_offdiag_corr": np.nan,
            "effective_rank": np.nan,
            "top1_eigen_fraction": np.nan,
            "top3_eigen_fraction": np.nan,
            "top5_eigen_fraction": np.nan,
        }
    if n_latent_features == 1:
        return {
            "n_samples": int(n_samples),
            "n_latent_features": int(n_latent_features),
            "rms_offdiag_corr": np.nan,
            "mean_abs_offdiag_corr": np.nan,
            "effective_rank": 1.0,
            "top1_eigen_fraction": 1.0,
            "top3_eigen_fraction": 1.0,
            "top5_eigen_fraction": 1.0,
        }

    R = np.corrcoef(Z, rowvar=False)
    return {
        "n_samples": int(n_samples),
        "n_latent_features": int(n_latent_features),
        "rms_offdiag_corr": rms_offdiag_corr(R),
        "mean_abs_offdiag_corr": mean_abs_offdiag_corr(R),
        "effective_rank": effective_rank(R),
        "top1_eigen_fraction": top_k_eigen_fraction(R, k=1),
        "top3_eigen_fraction": top_k_eigen_fraction(R, k=3),
        "top5_eigen_fraction": top_k_eigen_fraction(R, k=5),
    }


def summarize_latent_order_by_group(
    Z: np.ndarray,
    temp: Sequence[object],
    stage_bin: Sequence[object],
) -> pd.DataFrame:
    """Return one latent-order summary row per temperature x stage-bin group."""

    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array of shape (n_samples, n_latent_features)")
    if len(temp) != Z.shape[0] or len(stage_bin) != Z.shape[0]:
        raise ValueError("temp and stage_bin must have one value per row of Z")

    group_df = pd.DataFrame({"temp": list(temp), "stage_bin": list(stage_bin)})
    finite_rows = np.isfinite(Z).all(axis=1)
    group_df = group_df.loc[finite_rows].reset_index(drop=True)
    Z = Z[finite_rows]

    rows: list[dict[str, object]] = []
    for (temp_value, stage_bin_value), group in group_df.groupby(["temp", "stage_bin"], sort=True):
        summary = latent_corr_order_summary(Z[group.index.to_numpy()])
        rows.append(
            {
                "temp": temp_value,
                "stage_bin": stage_bin_value,
                **summary,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "temp",
            "stage_bin",
            "n_samples",
            "n_latent_features",
            "rms_offdiag_corr",
            "mean_abs_offdiag_corr",
            "effective_rank",
            "top1_eigen_fraction",
            "top3_eigen_fraction",
            "top5_eigen_fraction",
        ],
    )


def infer_latent_columns(df: pd.DataFrame, prefix: str = "z_mu_b") -> list[str]:
    """Infer latent-feature columns from the standard z_mu_b prefix."""

    columns = [column for column in df.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"No latent columns with prefix {prefix!r} found")
    return sorted(columns, key=lambda column: (len(column), column))


def summarize_latent_order_dataframe(
    df: pd.DataFrame,
    latent_cols: Sequence[str] | None = None,
    temp_col: str = "temp",
    stage_bin_col: str = "stage_bin",
) -> pd.DataFrame:
    """Return latent-order summaries from a DataFrame with group labels and latent columns.

    Use tight stage bins, or stage-residualize the latent features before calling this,
    if the goal is temperature-dependent morphology structure rather than developmental
    progression.
    """

    if temp_col not in df.columns:
        raise ValueError(f"temp_col {temp_col!r} is not present in df")
    if stage_bin_col not in df.columns:
        raise ValueError(f"stage_bin_col {stage_bin_col!r} is not present in df")

    resolved_latent_cols = list(latent_cols) if latent_cols is not None else infer_latent_columns(df)
    missing = [column for column in resolved_latent_cols if column not in df.columns]
    if missing:
        raise ValueError(f"latent columns are not present in df: {missing}")

    return summarize_latent_order_by_group(
        Z=df[resolved_latent_cols].to_numpy(dtype=np.float64),
        temp=df[temp_col].to_numpy(),
        stage_bin=df[stage_bin_col].to_numpy(),
    )


__all__ = [
    "effective_rank",
    "infer_latent_columns",
    "latent_corr_order_summary",
    "mean_abs_offdiag_corr",
    "rms_offdiag_corr",
    "summarize_latent_order_by_group",
    "summarize_latent_order_dataframe",
    "top_k_eigen_fraction",
]
