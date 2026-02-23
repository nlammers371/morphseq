"""
Phase 0 Step 7: 1D Patch (Interval) Search on S.

Finds the contiguous S-bin interval [a..b] that best discriminates
WT from cep290. Includes sanity checks (only-interval, drop-interval).

Two selection rules:
  A) Parsimony: smallest interval within ε of best AUROC
  B) Penalized: maximize AUROC - gamma*(len/K)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from roi_config import Phase0IntervalConfig, Phase0IntervalSelectionRule

logger = logging.getLogger(__name__)


def _score_interval(
    sbin_df: pd.DataFrame,
    bin_start: int,
    bin_end: int,
    feature_cols: List[str],
    n_folds: int = 5,
    exclude_outliers: bool = True,
) -> float:
    """
    Score a contiguous interval [bin_start, bin_end) via grouped-CV AUROC.

    Uses mean feature over bins in the interval as the per-sample feature.
    """
    if exclude_outliers:
        df = sbin_df[~sbin_df["qc_outlier_flag"]].copy()
    else:
        df = sbin_df.copy()

    # Filter to bins in interval
    interval_df = df[(df["k_bin"] >= bin_start) & (df["k_bin"] < bin_end)]
    if len(interval_df) == 0:
        return 0.5

    # Aggregate: mean feature per sample over bins in interval
    agg_cols = {col: "mean" for col in feature_cols if col in interval_df.columns}
    agg_cols["label_int"] = "first"
    agg_cols["embryo_id"] = "first"
    sample_df = interval_df.groupby("sample_id").agg(agg_cols).reset_index()

    X = sample_df[list(agg_cols.keys() - {"label_int", "embryo_id"})].values
    y = sample_df["label_int"].values
    groups = sample_df["embryo_id"].values

    if len(np.unique(y)) < 2:
        return 0.5

    n_unique = len(np.unique(groups))
    actual_folds = min(n_folds, n_unique)
    if actual_folds < 2:
        return 0.5

    gkf = GroupKFold(n_splits=actual_folds)
    aurocs = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        y_train, y_val = y[train_idx], y[val_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(X[train_idx], y_train)
        aurocs.append(roc_auc_score(y_val, clf.predict_proba(X[val_idx])[:, 1]))

    return float(np.mean(aurocs)) if aurocs else 0.5


def search_all_intervals(
    sbin_df: pd.DataFrame,
    feature_cols: List[str] = ["cost_mean"],
    K: int = 10,
    n_folds: int = 5,
    min_bins: int = 1,
    max_bins: Optional[int] = None,
    exclude_outliers: bool = True,
) -> pd.DataFrame:
    """
    Brute-force search over all contiguous S-bin intervals.

    Returns DataFrame with columns: bin_start, bin_end, n_bins, auroc.
    """
    if max_bins is None:
        max_bins = K

    rows = []
    total_intervals = sum(1 for a in range(K) for b in range(a + min_bins, min(a + max_bins, K) + 1))
    logger.info(f"Searching {total_intervals} intervals (K={K}, bins={min_bins}..{max_bins})")

    count = 0
    for a in range(K):
        for b in range(a + min_bins, min(a + max_bins, K) + 1):
            auroc = _score_interval(
                sbin_df, a, b, feature_cols,
                n_folds=n_folds, exclude_outliers=exclude_outliers,
            )
            rows.append({
                "bin_start": a,
                "bin_end": b,
                "n_bins": b - a,
                "S_lo": a / K,
                "S_hi": b / K,
                "auroc": auroc,
            })
            count += 1
            if count % 10 == 0:
                logger.info(f"  interval {count}/{total_intervals}")

    df = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
    logger.info(f"Interval search complete: best AUROC={df['auroc'].iloc[0]:.4f}")
    return df


def select_best_interval(
    interval_df: pd.DataFrame,
    config: Phase0IntervalConfig = Phase0IntervalConfig(),
    K: int = 10,
) -> Dict:
    """
    Apply deterministic selection rule to choose best interval.

    Returns dict with selected interval info.
    """
    df = interval_df.copy()

    if config.selection_rule == Phase0IntervalSelectionRule.PARSIMONY:
        # Smallest interval within ε of best AUROC
        best_auroc = df["auroc"].max()
        threshold = best_auroc - config.epsilon_auroc
        candidates = df[df["auroc"] >= threshold]
        # Among candidates, pick smallest interval; ties broken by highest AUROC
        selected = candidates.sort_values(
            ["n_bins", "auroc"], ascending=[True, False]
        ).iloc[0]

    elif config.selection_rule == Phase0IntervalSelectionRule.PENALIZED:
        # Maximize AUROC - gamma * (len/K)
        df["score"] = df["auroc"] - config.gamma_penalty * (df["n_bins"] / K)
        selected = df.sort_values("score", ascending=False).iloc[0]

    else:
        raise ValueError(f"Unknown selection rule: {config.selection_rule}")

    result = {
        "bin_start": int(selected["bin_start"]),
        "bin_end": int(selected["bin_end"]),
        "n_bins": int(selected["n_bins"]),
        "S_lo": float(selected["S_lo"]),
        "S_hi": float(selected["S_hi"]),
        "auroc": float(selected["auroc"]),
        "selection_rule": config.selection_rule.value,
    }

    logger.info(
        f"Selected interval: bins [{result['bin_start']}, {result['bin_end']}), "
        f"S=[{result['S_lo']:.2f}, {result['S_hi']:.2f}), "
        f"AUROC={result['auroc']:.4f}"
    )
    return result


def run_sanity_checks(
    sbin_df: pd.DataFrame,
    selected: Dict,
    feature_cols: List[str] = ["cost_mean"],
    n_folds: int = 5,
    K: int = 10,
    exclude_outliers: bool = True,
) -> Dict:
    """
    Sanity checks on the selected interval.

    - Only-interval: keep bins in I, zero others → AUROC should stay high
    - Drop-interval: zero bins in I, keep others → AUROC should drop
    """
    bin_start = selected["bin_start"]
    bin_end = selected["bin_end"]

    # Score: only-interval
    only_auroc = _score_interval(
        sbin_df, bin_start, bin_end, feature_cols,
        n_folds=n_folds, exclude_outliers=exclude_outliers,
    )

    # Score: complement (drop-interval)
    # Use bins OUTSIDE the interval
    complement_df = sbin_df[
        (sbin_df["k_bin"] < bin_start) | (sbin_df["k_bin"] >= bin_end)
    ].copy()

    if len(complement_df) > 0 and complement_df["k_bin"].nunique() > 0:
        # Re-score the complement as one big interval
        drop_auroc = _score_interval(
            sbin_df.copy(),
            0, K, feature_cols,
            n_folds=n_folds, exclude_outliers=exclude_outliers,
        )
        # Actually, we need to zero the interval bins and score the full set
        # Simpler: score only complement bins
        comp_bins = sorted(complement_df["k_bin"].unique())
        if len(comp_bins) > 0:
            drop_auroc = _score_interval(
                complement_df, comp_bins[0], comp_bins[-1] + 1, feature_cols,
                n_folds=n_folds, exclude_outliers=exclude_outliers,
            )
        else:
            drop_auroc = 0.5
    else:
        drop_auroc = 0.5

    # Full model (all bins) for reference
    full_auroc = _score_interval(
        sbin_df, 0, K, feature_cols,
        n_folds=n_folds, exclude_outliers=exclude_outliers,
    )

    checks = {
        "only_interval_auroc": only_auroc,
        "drop_interval_auroc": drop_auroc,
        "full_auroc": full_auroc,
        "only_vs_full_diff": only_auroc - full_auroc,
        "drop_vs_full_diff": drop_auroc - full_auroc,
        "pass_only_high": only_auroc >= full_auroc - 0.05,
        "pass_drop_low": drop_auroc < only_auroc - 0.02,
    }

    logger.info(
        f"Sanity checks: only={only_auroc:.4f}, drop={drop_auroc:.4f}, full={full_auroc:.4f}"
    )
    return checks


__all__ = [
    "search_all_intervals",
    "select_best_interval",
    "run_sanity_checks",
]
