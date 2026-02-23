"""
Phase 0 Step 6: Classification + AUROC Localization.

Provides:
  6.1 Univariate AUROC per bin (fast, interpretable)
  6.2 Multivariate logistic regression across bins (grouped CV)

All CV uses embryo-level grouping (GroupKFold by embryo_id).
Class imbalance handled via sklearn class_weight='balanced'.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


def compute_auroc_per_bin(
    sbin_df: pd.DataFrame,
    feature_col: str = "cost_mean",
    exclude_outliers: bool = True,
) -> pd.DataFrame:
    """
    6.1: Univariate AUROC per S-bin.

    For each bin k, compute AUROC of feature_col for discriminating label_int.

    Returns DataFrame with columns: k_bin, S_lo, S_hi, auroc, n_samples.
    """
    if exclude_outliers:
        df = sbin_df[~sbin_df["qc_outlier_flag"]].copy()
    else:
        df = sbin_df.copy()

    results = []
    for k_bin, group in df.groupby("k_bin"):
        y = group["label_int"].values
        x = group[feature_col].values

        if len(np.unique(y)) < 2:
            auroc = np.nan
        elif np.std(x) < 1e-12:
            auroc = 0.5
        else:
            auroc = roc_auc_score(y, x)

        results.append({
            "k_bin": k_bin,
            "S_lo": group["S_lo"].iloc[0],
            "S_hi": group["S_hi"].iloc[0],
            "feature": feature_col,
            "auroc": auroc,
            "n_samples": len(group),
        })

    return pd.DataFrame(results)


def compute_auroc_all_features(
    sbin_df: pd.DataFrame,
    feature_cols: List[str],
    exclude_outliers: bool = True,
) -> pd.DataFrame:
    """Compute AUROC per bin for multiple features, return concatenated DataFrame."""
    dfs = []
    for col in feature_cols:
        if col in sbin_df.columns:
            df = compute_auroc_per_bin(sbin_df, feature_col=col, exclude_outliers=exclude_outliers)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _pivot_sbin_to_sample_features(
    sbin_df: pd.DataFrame,
    feature_cols: List[str],
    exclude_outliers: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pivot S-bin table to per-sample feature matrix.

    Returns (X, y, groups) where X has shape (N_samples, K * len(feature_cols)).
    """
    if exclude_outliers:
        df = sbin_df[~sbin_df["qc_outlier_flag"]].copy()
    else:
        df = sbin_df.copy()

    K = df["k_bin"].nunique()
    sample_ids = df["sample_id"].unique()
    N = len(sample_ids)

    X = np.zeros((N, K * len(feature_cols)), dtype=np.float64)
    y = np.zeros(N, dtype=int)
    groups = []

    for i, sid in enumerate(sample_ids):
        sample_rows = df[df["sample_id"] == sid].sort_values("k_bin")
        y[i] = sample_rows["label_int"].iloc[0]
        groups.append(sample_rows["embryo_id"].iloc[0])

        for j, col in enumerate(feature_cols):
            vals = sample_rows[col].values
            X[i, j * K:(j + 1) * K] = vals

    return X, y, np.array(groups)


def run_grouped_cv_logistic(
    sbin_df: pd.DataFrame,
    feature_cols: List[str],
    n_folds: int = 5,
    exclude_outliers: bool = True,
) -> Dict:
    """
    6.2: Multivariate logistic regression across bins with grouped CV.

    Returns dict with:
      - auroc_mean, auroc_std, auroc_folds
      - coef_mean: mean coefficient profile over bins
      - model_label: description of features used
    """
    X, y, groups = _pivot_sbin_to_sample_features(
        sbin_df, feature_cols, exclude_outliers=exclude_outliers,
    )

    n_unique_groups = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique_groups)
    if actual_folds < 2:
        logger.warning(f"Only {n_unique_groups} unique groups, cannot run CV")
        return {"auroc_mean": np.nan, "auroc_std": np.nan, "auroc_folds": [],
                "coef_mean": np.zeros(X.shape[1]), "model_label": "+".join(feature_cols)}

    gkf = GroupKFold(n_splits=actual_folds)
    aurocs = []
    coefs = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_val)[:, 1]
        aurocs.append(roc_auc_score(y_val, y_prob))
        coefs.append(clf.coef_[0])

    result = {
        "auroc_mean": float(np.mean(aurocs)) if aurocs else np.nan,
        "auroc_std": float(np.std(aurocs)) if aurocs else np.nan,
        "auroc_folds": aurocs,
        "coef_mean": np.mean(coefs, axis=0) if coefs else np.zeros(X.shape[1]),
        "model_label": "+".join(feature_cols),
        "n_folds_completed": len(aurocs),
    }

    logger.info(
        f"Logistic CV ({result['model_label']}): "
        f"AUROC={result['auroc_mean']:.4f}±{result['auroc_std']:.4f} "
        f"({result['n_folds_completed']} folds)"
    )
    return result


def run_phase0_classification(
    sbin_df: pd.DataFrame,
    n_folds: int = 5,
) -> Dict:
    """
    Run full Phase 0 classification suite.

    Runs with/without outliers for comparison, and compares:
      - cost-only
      - dynamics-only (if available)
      - cost + dynamics (if available)
    """
    has_dynamics = "disp_mag_mean" in sbin_df.columns

    results = {}

    for label_suffix, exclude in [("filtered", True), ("all", False)]:
        # AUROC per bin
        auroc_cost = compute_auroc_per_bin(sbin_df, "cost_mean", exclude_outliers=exclude)
        results[f"auroc_per_bin_cost_{label_suffix}"] = auroc_cost

        if has_dynamics:
            for feat in ["disp_mag_mean", "disp_par_mean", "disp_perp_mean"]:
                if feat in sbin_df.columns:
                    results[f"auroc_per_bin_{feat}_{label_suffix}"] = compute_auroc_per_bin(
                        sbin_df, feat, exclude_outliers=exclude,
                    )

        # Logistic CV: cost-only
        results[f"logistic_cost_{label_suffix}"] = run_grouped_cv_logistic(
            sbin_df, ["cost_mean"], n_folds=n_folds, exclude_outliers=exclude,
        )

        if has_dynamics:
            # dynamics-only
            results[f"logistic_dynamics_{label_suffix}"] = run_grouped_cv_logistic(
                sbin_df, ["disp_mag_mean", "disp_par_mean", "disp_perp_mean"],
                n_folds=n_folds, exclude_outliers=exclude,
            )
            # cost + dynamics
            results[f"logistic_all_{label_suffix}"] = run_grouped_cv_logistic(
                sbin_df, ["cost_mean", "disp_mag_mean", "disp_par_mean", "disp_perp_mean"],
                n_folds=n_folds, exclude_outliers=exclude,
            )

    return results


__all__ = [
    "compute_auroc_per_bin",
    "compute_auroc_all_features",
    "run_grouped_cv_logistic",
    "run_phase0_classification",
]
