"""
Phase 0 Step 8: Nulls + Stability (selection-aware).

8.1 Permutation null: embryo-level label permutation.
    Repeats the SAME selection procedure under permuted labels.
    p-value = fraction of null scores >= observed.

8.2 Bootstrap stability: embryo-level bootstrap within class.
    Recomputes AUROC_k curves and interval endpoints.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from roi_config import Phase0NullConfig
from p0_classification import compute_auroc_per_bin, run_grouped_cv_logistic
from p0_interval_search import search_all_intervals, select_best_interval

logger = logging.getLogger(__name__)


def _permute_labels_by_embryo(
    sbin_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Permute labels at the embryo_id level (maintains within-embryo consistency)."""
    df = sbin_df.copy()
    embryo_labels = df.groupby("embryo_id")["label_int"].first()
    permuted = rng.permutation(embryo_labels.values)
    label_map = dict(zip(embryo_labels.index, permuted))
    df["label_int"] = df["embryo_id"].map(label_map)
    return df


def run_permutation_null_auroc_max(
    sbin_df: pd.DataFrame,
    observed_max_auroc: float,
    feature_col: str = "cost_mean",
    n_permute: int = 200,
    random_seed: int = 42,
    exclude_outliers: bool = True,
) -> Dict:
    """
    Selection-aware permutation null for max AUROC_k.

    For each permutation:
      - Permute labels at embryo level
      - Compute AUROC per bin
      - Record max AUROC across bins (selection-aware)

    Returns dict with null distribution and p-value.
    """
    rng = np.random.default_rng(random_seed)
    null_max_aurocs = np.zeros(n_permute)

    for perm_i in range(n_permute):
        perm_df = _permute_labels_by_embryo(sbin_df, rng)
        auroc_df = compute_auroc_per_bin(perm_df, feature_col, exclude_outliers=exclude_outliers)
        null_max_aurocs[perm_i] = auroc_df["auroc"].max()

        if (perm_i + 1) % 50 == 0:
            logger.info(f"Permutation {perm_i + 1}/{n_permute}")

    pvalue = float(np.mean(null_max_aurocs >= observed_max_auroc))

    result = {
        "test": "permutation_max_auroc",
        "statistic": "max_auroc_k",
        "observed": observed_max_auroc,
        "null_distribution": null_max_aurocs,
        "null_mean": float(np.mean(null_max_aurocs)),
        "null_std": float(np.std(null_max_aurocs)),
        "pvalue": pvalue,
        "n_permute": n_permute,
    }

    logger.info(
        f"Permutation null (max AUROC): observed={observed_max_auroc:.4f}, "
        f"null={result['null_mean']:.4f}±{result['null_std']:.4f}, p={pvalue:.4f}"
    )
    return result


def run_permutation_null_interval(
    sbin_df: pd.DataFrame,
    observed_interval_auroc: float,
    feature_cols: List[str] = ["cost_mean"],
    interval_config=None,
    K: int = 10,
    n_folds: int = 5,
    n_permute: int = 200,
    random_seed: int = 42,
    exclude_outliers: bool = True,
) -> Dict:
    """
    Selection-aware permutation null for best-interval AUROC.

    For each permutation:
      - Permute labels at embryo level
      - Run full interval search
      - Record best-interval AUROC (selection-aware)
    """
    from roi_config import Phase0IntervalConfig
    if interval_config is None:
        interval_config = Phase0IntervalConfig()

    rng = np.random.default_rng(random_seed)
    null_aurocs = np.zeros(n_permute)

    for perm_i in range(n_permute):
        perm_df = _permute_labels_by_embryo(sbin_df, rng)
        interval_df = search_all_intervals(
            perm_df, feature_cols, K=K, n_folds=n_folds,
            exclude_outliers=exclude_outliers,
        )
        selected = select_best_interval(interval_df, interval_config, K=K)
        null_aurocs[perm_i] = selected["auroc"]

        if (perm_i + 1) % 50 == 0:
            logger.info(f"Permutation (interval) {perm_i + 1}/{n_permute}")

    pvalue = float(np.mean(null_aurocs >= observed_interval_auroc))

    result = {
        "test": "permutation_best_interval",
        "statistic": "best_interval_auroc",
        "observed": observed_interval_auroc,
        "null_distribution": null_aurocs,
        "null_mean": float(np.mean(null_aurocs)),
        "null_std": float(np.std(null_aurocs)),
        "pvalue": pvalue,
        "n_permute": n_permute,
    }

    logger.info(
        f"Permutation null (interval): observed={observed_interval_auroc:.4f}, "
        f"null={result['null_mean']:.4f}±{result['null_std']:.4f}, p={pvalue:.4f}"
    )
    return result


def run_bootstrap_stability(
    sbin_df: pd.DataFrame,
    feature_col: str = "cost_mean",
    feature_cols_interval: List[str] = ["cost_mean"],
    interval_config=None,
    K: int = 10,
    n_folds: int = 5,
    n_boot: int = 200,
    random_seed: int = 42,
    exclude_outliers: bool = True,
) -> Dict:
    """
    Bootstrap stability analysis (embryo-level within-class resampling).

    Recomputes:
      - AUROC_k curves → confidence bands
      - Selected interval endpoints → distribution
      - Interval overlap stability metric
    """
    from roi_config import Phase0IntervalConfig
    if interval_config is None:
        interval_config = Phase0IntervalConfig()

    if exclude_outliers:
        df = sbin_df[~sbin_df["qc_outlier_flag"]].copy()
    else:
        df = sbin_df.copy()

    rng = np.random.default_rng(random_seed)

    # Get unique embryos per class
    embryo_labels = df.groupby("embryo_id")["label_int"].first()
    embryos_0 = embryo_labels[embryo_labels == 0].index.tolist()
    embryos_1 = embryo_labels[embryo_labels == 1].index.tolist()

    boot_auroc_curves = []  # (n_boot, K)
    boot_interval_starts = []
    boot_interval_ends = []

    for boot_i in range(n_boot):
        # Bootstrap embryos within each class
        boot_e0 = rng.choice(embryos_0, size=len(embryos_0), replace=True)
        boot_e1 = rng.choice(embryos_1, size=len(embryos_1), replace=True)
        boot_embryos = list(boot_e0) + list(boot_e1)

        # Build bootstrap S-bin table
        boot_rows = []
        for eid in boot_embryos:
            boot_rows.append(df[df["embryo_id"] == eid])
        if not boot_rows:
            continue
        boot_df = pd.concat(boot_rows, ignore_index=True)

        # AUROC per bin
        auroc_df = compute_auroc_per_bin(boot_df, feature_col, exclude_outliers=False)
        auroc_curve = auroc_df.sort_values("k_bin")["auroc"].values
        boot_auroc_curves.append(auroc_curve)

        # Interval search
        try:
            interval_df = search_all_intervals(
                boot_df, feature_cols_interval, K=K, n_folds=n_folds,
                exclude_outliers=False,
            )
            selected = select_best_interval(interval_df, interval_config, K=K)
            boot_interval_starts.append(selected["bin_start"])
            boot_interval_ends.append(selected["bin_end"])
        except Exception:
            pass

        if (boot_i + 1) % 50 == 0:
            logger.info(f"Bootstrap {boot_i + 1}/{n_boot}")

    # Summarize
    auroc_matrix = np.array(boot_auroc_curves)  # (n_boot, K)
    starts = np.array(boot_interval_starts)
    ends = np.array(boot_interval_ends)

    result = {
        "auroc_matrix": auroc_matrix,
        "auroc_mean": np.mean(auroc_matrix, axis=0) if len(auroc_matrix) > 0 else np.zeros(K),
        "auroc_ci_lo": np.percentile(auroc_matrix, 2.5, axis=0) if len(auroc_matrix) > 0 else np.zeros(K),
        "auroc_ci_hi": np.percentile(auroc_matrix, 97.5, axis=0) if len(auroc_matrix) > 0 else np.zeros(K),
        "interval_starts": starts,
        "interval_ends": ends,
        "interval_start_mean": float(np.mean(starts)) if len(starts) > 0 else np.nan,
        "interval_end_mean": float(np.mean(ends)) if len(ends) > 0 else np.nan,
        "interval_start_std": float(np.std(starts)) if len(starts) > 0 else np.nan,
        "interval_end_std": float(np.std(ends)) if len(ends) > 0 else np.nan,
        "n_boot": n_boot,
        "n_boot_completed": len(auroc_matrix),
    }

    logger.info(
        f"Bootstrap stability: {result['n_boot_completed']}/{n_boot} completed, "
        f"interval start={result['interval_start_mean']:.1f}±{result['interval_start_std']:.1f}"
    )
    return result


__all__ = [
    "run_permutation_null_auroc_max",
    "run_permutation_null_interval",
    "run_bootstrap_stability",
]
