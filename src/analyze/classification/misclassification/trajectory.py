from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Keep sklearn linear algebra stable in restricted/OpenMP-limited environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from analyze.utils.resampling.lightweight_numpy_resampling import run_lite

STAGE_HARD = "hard"
STAGE_SOFT = "soft"
STAGE_DELTA = "delta"
STAGE_RESIDUAL = "residual"
STAGE_RESIDUAL_DTW = "residual_dtw"
VALID_STAGES = {
    STAGE_HARD,
    STAGE_SOFT,
    STAGE_DELTA,
    STAGE_RESIDUAL,
    STAGE_RESIDUAL_DTW,
}


@dataclass(frozen=True)
class StageGeometryResult:
    stage: str
    stage_table: pd.DataFrame
    feature_columns: list[str]
    class_labels: list[str]
    time_bins: list[int]
    pca_scores: np.ndarray
    distance_matrix: np.ndarray | None
    metrics_by_k: list[dict[str, Any]]
    explained_variance_ratio: list[float]
    baseline_mu: pd.DataFrame | None


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    m = int(mask.sum())
    if m == 0:
        return out
    p_sub = p[mask]
    order = np.argsort(p_sub)
    ranked = p_sub[order]
    q = ranked * m / (np.arange(m) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    sub_out = np.empty_like(q)
    sub_out[order] = np.clip(q, 0.0, 1.0)
    out[np.where(mask)[0]] = sub_out
    return out


def _compute_wrong_rate_significance(
    work: pd.DataFrame,
    *,
    embryo_ids: list[str],
    class_labels: list[str],
    n_permutations: int,
    random_state: int,
    q_threshold: float,
    window_min: float | None = None,
    window_max: float | None = None,
) -> pd.DataFrame:
    """Permutation-based wrong-rate significance per embryo.

    Null: shuffle pred_class within each time_bin, keep true_class fixed.
    """
    if n_permutations <= 0:
        raise ValueError(f"n_permutations must be >0, got {n_permutations}")

    work_sig = work.copy()
    window_col = "time_bin_center" if "time_bin_center" in work_sig.columns else "time_bin"
    if window_min is not None:
        work_sig = work_sig[work_sig[window_col].astype(float) >= float(window_min)].copy()
    if window_max is not None:
        work_sig = work_sig[work_sig[window_col].astype(float) <= float(window_max)].copy()

    if work_sig.empty:
        n_embryos = len(embryo_ids)
        return pd.DataFrame(
            {
                "embryo_id": embryo_ids,
                "wrong_rate_null_mean": np.full(n_embryos, np.nan),
                "wrong_rate_null_std": np.full(n_embryos, np.nan),
                "wrong_rate_z": np.full(n_embryos, np.nan),
                "wrong_rate_exceed_count": np.zeros(n_embryos, dtype=int),
                "pval_wrong_rate_perm": np.full(n_embryos, np.nan),
                "qval_wrong_rate_perm": np.full(n_embryos, np.nan),
                "is_wrong_significant_perm": np.zeros(n_embryos, dtype=bool),
                "wrong_rate_sig_tier": np.array(["no_window_rows"] * n_embryos, dtype=object),
                "wrong_rate_window_null_mean": np.full(n_embryos, np.nan),
                "wrong_rate_window_null_std": np.full(n_embryos, np.nan),
                "wrong_rate_window_z": np.full(n_embryos, np.nan),
                "wrong_rate_window_exceed_count": np.zeros(n_embryos, dtype=int),
                "pval_wrong_rate_window_perm": np.full(n_embryos, np.nan),
                "qval_wrong_rate_window_perm": np.full(n_embryos, np.nan),
                "is_wrong_significant_in_window_perm": np.zeros(n_embryos, dtype=bool),
                "wrong_rate_window_sig_tier": np.array(["no_window_rows"] * n_embryos, dtype=object),
                "wrong_rate_window_n_bins": np.zeros(n_embryos, dtype=int),
                "wrong_rate_n_permutations": int(n_permutations),
            }
        )

    sorted_df = work_sig.sort_values(["time_bin"], kind="mergesort").copy()
    sorted_df["embryo_idx"] = pd.Categorical(sorted_df["embryo_id"], categories=embryo_ids).codes.astype(np.int32)

    pred_codes = pd.Categorical(sorted_df["pred_class"], categories=class_labels).codes.astype(np.int16)
    true_codes = pd.Categorical(sorted_df["true_class"], categories=class_labels).codes.astype(np.int16)
    if (pred_codes < 0).any() or (true_codes < 0).any():
        raise ValueError("Found labels outside class_labels during permutation setup")

    embryo_idx = sorted_df["embryo_idx"].to_numpy(dtype=np.int32)
    n_embryos = len(embryo_ids)

    obs_wrong = (pred_codes != true_codes).astype(np.int8)
    n_windows = np.bincount(embryo_idx, minlength=n_embryos).astype(float)
    wrong_count_obs = np.bincount(embryo_idx, weights=obs_wrong.astype(np.int32), minlength=n_embryos).astype(float)
    obs_wrong_rate = wrong_count_obs / np.maximum(n_windows, 1.0)

    bin_slices: list[tuple[int, int]] = []
    pos = 0
    for _, grp in sorted_df.groupby("time_bin", sort=True):
        n = len(grp)
        bin_slices.append((pos, pos + n))
        pos += n

    def kernel(rng: np.random.Generator, n_iters: int):
        exceed = np.zeros(n_embryos, dtype=np.int64)
        null_sum = np.zeros(n_embryos, dtype=np.float64)
        null_sum_sq = np.zeros(n_embryos, dtype=np.float64)

        pred_perm = pred_codes.copy()
        for _ in range(n_iters):
            np.copyto(pred_perm, pred_codes)
            for start, stop in bin_slices:
                pred_perm[start:stop] = rng.permutation(pred_perm[start:stop])

            wrong_perm = (pred_perm != true_codes).astype(np.int8)
            wrong_count = np.bincount(
                embryo_idx,
                weights=wrong_perm.astype(np.int32),
                minlength=n_embryos,
            ).astype(np.float64)
            wr = wrong_count / np.maximum(n_windows, 1.0)
            exceed += (wr >= obs_wrong_rate)
            null_sum += wr
            null_sum_sq += wr ** 2

        null_mean = null_sum / n_iters
        null_var = null_sum_sq / n_iters - null_mean ** 2
        null_std = np.sqrt(np.maximum(null_var, 0.0))
        pval = (exceed + 1) / (n_iters + 1)
        z = (obs_wrong_rate - null_mean) / (null_std + 1e-9)
        return {
            "stat_name": "wrong_rate",
            "exceed_count": exceed,
            "null_mean": null_mean,
            "null_std": null_std,
            "obs_stat": obs_wrong_rate,
            "pval": pval,
            "z_score": z,
        }

    run = run_lite(
        test_name="trajectory_wrong_rate_permutation",
        n_iters=int(n_permutations),
        seed=int(random_state),
        spec={
            "type": "permute_labels",
            "within": "time_bin",
            "label": "pred_class",
            "derived": "is_wrong_perm",
        },
        kernel=kernel,
        collect_samples=False,
        metadata={"class_labels": class_labels},
    )

    pvals = run.summary["pval"].copy()
    invalid = n_windows <= 0
    pvals[invalid] = np.nan
    qvals = _bh_fdr(pvals)
    is_sig = (qvals <= float(q_threshold)) & (run.summary["z_score"] > 0)
    is_sig = np.where(np.isfinite(qvals), is_sig, False)

    tier = np.full(n_embryos, "ns", dtype=object)
    tier[~np.isfinite(qvals)] = "no_window_rows"
    tier[qvals <= 0.1] = "q<=0.10"
    tier[qvals <= 0.05] = "q<=0.05"
    tier[qvals <= 0.01] = "q<=0.01"

    return pd.DataFrame(
        {
            "embryo_id": embryo_ids,
            "wrong_rate_null_mean": run.summary["null_mean"],
            "wrong_rate_null_std": run.summary["null_std"],
            "wrong_rate_z": run.summary["z_score"],
            "wrong_rate_exceed_count": run.summary["exceed_count"],
            "pval_wrong_rate_perm": pvals,
            "qval_wrong_rate_perm": qvals,
            "is_wrong_significant_perm": is_sig.astype(bool),
            "wrong_rate_sig_tier": tier.astype(str),
            "wrong_rate_window_null_mean": run.summary["null_mean"],
            "wrong_rate_window_null_std": run.summary["null_std"],
            "wrong_rate_window_z": run.summary["z_score"],
            "wrong_rate_window_exceed_count": run.summary["exceed_count"],
            "pval_wrong_rate_window_perm": pvals,
            "qval_wrong_rate_window_perm": qvals,
            "is_wrong_significant_in_window_perm": is_sig.astype(bool),
            "wrong_rate_window_sig_tier": tier.astype(str),
            "wrong_rate_window_n_bins": n_windows.astype(int),
            "wrong_rate_n_permutations": int(n_permutations),
        }
    )


def compute_rolling_window_wrong_rate_significance(
    df: pd.DataFrame,
    *,
    class_labels: list[str],
    window_hpf: float = 5.0,
    centers_hpf: list[float] | None = None,
    n_permutations: int = 300,
    random_state: int = 42,
    q_threshold: float = 0.10,
) -> pd.DataFrame:
    """Rolling-window permutation significance for embryo wrong-rate.

    Returns one row per embryo per rolling-window center.
    """
    if window_hpf <= 0:
        raise ValueError(f"window_hpf must be >0, got {window_hpf}")

    _validate_base_columns(df)
    _validate_label_set(df, class_labels)
    _validate_time_order(df)

    work = df.copy()
    work["embryo_id"] = work["embryo_id"].astype(str)
    work["true_class"] = work["true_class"].astype(str)
    work["pred_class"] = work["pred_class"].astype(str)

    center_col = "time_bin_center" if "time_bin_center" in work.columns else "time_bin"
    if centers_hpf is None:
        centers = sorted(work[center_col].astype(float).unique().tolist())
    else:
        centers = sorted(float(x) for x in centers_hpf)

    embryo_ids = sorted(work["embryo_id"].unique().tolist())
    half = float(window_hpf) / 2.0

    rows = []
    for i, center in enumerate(centers):
        sub = _compute_wrong_rate_significance(
            work,
            embryo_ids=embryo_ids,
            class_labels=class_labels,
            n_permutations=int(n_permutations),
            random_state=int(random_state) + i,
            q_threshold=float(q_threshold),
            window_min=float(center - half),
            window_max=float(center + half),
        )
        sub["window_center_hpf"] = float(center)
        sub["window_start_hpf"] = float(center - half)
        sub["window_end_hpf"] = float(center + half)
        rows.append(sub)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out

    q_global = _bh_fdr(out["pval_wrong_rate_window_perm"].to_numpy(dtype=float))
    out["qval_wrong_rate_window_global_perm"] = q_global
    out["is_wrong_significant_in_window_global_perm"] = (
        (q_global <= float(q_threshold))
        & (out["wrong_rate_window_z"].to_numpy(dtype=float) > 0)
    )
    out["is_wrong_significant_in_window_global_perm"] = (
        out["is_wrong_significant_in_window_global_perm"].fillna(False).astype(bool)
    )
    return out


def _compute_destination_confusion_significance(
    work: pd.DataFrame,
    *,
    embryo_ids: list[str],
    class_labels: list[str],
    source_class: str,
    target_class: str,
    n_permutations: int,
    random_state: int,
    q_threshold: float,
    window_min: float | None = None,
    window_max: float | None = None,
) -> pd.DataFrame:
    """Permutation significance for destination-specific confusion rate.

    Statistic per embryo:
      rate = P(pred_class == target_class | true_class == source_class)
    evaluated within the selected window.
    """
    if n_permutations <= 0:
        raise ValueError(f"n_permutations must be >0, got {n_permutations}")
    if source_class not in set(class_labels):
        raise ValueError(f"source_class='{source_class}' not in class_labels")
    if target_class not in set(class_labels):
        raise ValueError(f"target_class='{target_class}' not in class_labels")

    work_sig = work.copy()
    window_col = "time_bin_center" if "time_bin_center" in work_sig.columns else "time_bin"
    if window_min is not None:
        work_sig = work_sig[work_sig[window_col].astype(float) >= float(window_min)].copy()
    if window_max is not None:
        work_sig = work_sig[work_sig[window_col].astype(float) <= float(window_max)].copy()

    n_embryos = len(embryo_ids)
    if work_sig.empty:
        return pd.DataFrame(
            {
                "embryo_id": embryo_ids,
                "dest_source_class": [source_class] * n_embryos,
                "dest_target_class": [target_class] * n_embryos,
                "dest_confusion_rate_obs": np.full(n_embryos, np.nan),
                "dest_confusion_n_source_rows": np.zeros(n_embryos, dtype=int),
                "dest_confusion_n_hits_obs": np.zeros(n_embryos, dtype=int),
                "dest_confusion_null_mean": np.full(n_embryos, np.nan),
                "dest_confusion_null_std": np.full(n_embryos, np.nan),
                "dest_confusion_z": np.full(n_embryos, np.nan),
                "dest_confusion_exceed_count": np.zeros(n_embryos, dtype=int),
                "pval_dest_confusion_perm": np.full(n_embryos, np.nan),
                "qval_dest_confusion_perm": np.full(n_embryos, np.nan),
                "is_dest_confusion_significant_perm": np.zeros(n_embryos, dtype=bool),
                "dest_confusion_sig_tier": np.array(["no_window_rows"] * n_embryos, dtype=object),
                "dest_confusion_n_permutations": int(n_permutations),
            }
        )

    sorted_df = work_sig.sort_values(["time_bin"], kind="mergesort").copy()
    sorted_df["embryo_idx"] = pd.Categorical(sorted_df["embryo_id"], categories=embryo_ids).codes.astype(np.int32)

    pred_codes = pd.Categorical(sorted_df["pred_class"], categories=class_labels).codes.astype(np.int16)
    true_codes = pd.Categorical(sorted_df["true_class"], categories=class_labels).codes.astype(np.int16)
    if (pred_codes < 0).any() or (true_codes < 0).any():
        raise ValueError("Found labels outside class_labels during destination confusion setup")

    source_code = int(class_labels.index(source_class))
    target_code = int(class_labels.index(target_class))

    embryo_idx = sorted_df["embryo_idx"].to_numpy(dtype=np.int32)
    is_source = (true_codes == source_code).astype(np.int8)
    is_hit_obs = ((true_codes == source_code) & (pred_codes == target_code)).astype(np.int8)

    n_source_rows = np.bincount(embryo_idx, weights=is_source.astype(np.int32), minlength=n_embryos).astype(float)
    n_hits_obs = np.bincount(embryo_idx, weights=is_hit_obs.astype(np.int32), minlength=n_embryos).astype(float)
    obs_rate = n_hits_obs / np.maximum(n_source_rows, 1.0)

    bin_slices: list[tuple[int, int]] = []
    pos = 0
    for _, grp in sorted_df.groupby("time_bin", sort=True):
        n = len(grp)
        bin_slices.append((pos, pos + n))
        pos += n

    def kernel(rng: np.random.Generator, n_iters: int):
        exceed = np.zeros(n_embryos, dtype=np.int64)
        null_sum = np.zeros(n_embryos, dtype=np.float64)
        null_sum_sq = np.zeros(n_embryos, dtype=np.float64)
        pred_perm = pred_codes.copy()

        for _ in range(n_iters):
            np.copyto(pred_perm, pred_codes)
            for start, stop in bin_slices:
                pred_perm[start:stop] = rng.permutation(pred_perm[start:stop])

            is_hit_perm = ((true_codes == source_code) & (pred_perm == target_code)).astype(np.int8)
            n_hits_perm = np.bincount(
                embryo_idx,
                weights=is_hit_perm.astype(np.int32),
                minlength=n_embryos,
            ).astype(np.float64)
            rate_perm = n_hits_perm / np.maximum(n_source_rows, 1.0)

            exceed += (rate_perm >= obs_rate)
            null_sum += rate_perm
            null_sum_sq += rate_perm ** 2

        null_mean = null_sum / n_iters
        null_var = null_sum_sq / n_iters - null_mean ** 2
        null_std = np.sqrt(np.maximum(null_var, 0.0))
        pval = (exceed + 1) / (n_iters + 1)
        z = (obs_rate - null_mean) / (null_std + 1e-9)
        return {
            "stat_name": "dest_confusion_rate",
            "exceed_count": exceed,
            "null_mean": null_mean,
            "null_std": null_std,
            "obs_stat": obs_rate,
            "pval": pval,
            "z_score": z,
        }

    run = run_lite(
        test_name=f"trajectory_dest_confusion_{source_class}_to_{target_class}",
        n_iters=int(n_permutations),
        seed=int(random_state),
        spec={
            "type": "permute_labels",
            "within": "time_bin",
            "label": "pred_class",
            "stat": "P(pred=target|true=source)",
            "source_class": source_class,
            "target_class": target_class,
        },
        kernel=kernel,
        collect_samples=False,
        metadata={"class_labels": class_labels},
    )

    pvals = run.summary["pval"].copy()
    invalid = n_source_rows <= 0
    pvals[invalid] = np.nan
    qvals = _bh_fdr(pvals)

    is_sig = (qvals <= float(q_threshold)) & (run.summary["z_score"] > 0)
    is_sig = np.where(np.isfinite(qvals), is_sig, False)

    tier = np.full(n_embryos, "ns", dtype=object)
    tier[~np.isfinite(qvals)] = "no_source_rows"
    tier[qvals <= 0.1] = "q<=0.10"
    tier[qvals <= 0.05] = "q<=0.05"
    tier[qvals <= 0.01] = "q<=0.01"

    return pd.DataFrame(
        {
            "embryo_id": embryo_ids,
            "dest_source_class": [source_class] * n_embryos,
            "dest_target_class": [target_class] * n_embryos,
            "dest_confusion_rate_obs": obs_rate,
            "dest_confusion_n_source_rows": n_source_rows.astype(int),
            "dest_confusion_n_hits_obs": n_hits_obs.astype(int),
            "dest_confusion_null_mean": run.summary["null_mean"],
            "dest_confusion_null_std": run.summary["null_std"],
            "dest_confusion_z": run.summary["z_score"],
            "dest_confusion_exceed_count": run.summary["exceed_count"],
            "pval_dest_confusion_perm": pvals,
            "qval_dest_confusion_perm": qvals,
            "is_dest_confusion_significant_perm": is_sig.astype(bool),
            "dest_confusion_sig_tier": tier.astype(str),
            "dest_confusion_n_permutations": int(n_permutations),
        }
    )


def compute_rolling_window_destination_confusion_significance(
    df: pd.DataFrame,
    *,
    class_labels: list[str],
    source_class: str,
    target_class: str,
    window_hpf: float = 5.0,
    centers_hpf: list[float] | None = None,
    n_permutations: int = 300,
    random_state: int = 42,
    q_threshold: float = 0.10,
) -> pd.DataFrame:
    """Rolling-window destination confusion significance over time."""
    if window_hpf <= 0:
        raise ValueError(f"window_hpf must be >0, got {window_hpf}")

    _validate_base_columns(df)
    _validate_label_set(df, class_labels)
    _validate_time_order(df)

    work = df.copy()
    work["embryo_id"] = work["embryo_id"].astype(str)
    work["true_class"] = work["true_class"].astype(str)
    work["pred_class"] = work["pred_class"].astype(str)

    center_col = "time_bin_center" if "time_bin_center" in work.columns else "time_bin"
    if centers_hpf is None:
        centers = sorted(work[center_col].astype(float).unique().tolist())
    else:
        centers = sorted(float(x) for x in centers_hpf)

    embryo_ids = sorted(work["embryo_id"].unique().tolist())
    half = float(window_hpf) / 2.0

    rows = []
    for i, center in enumerate(centers):
        sub = _compute_destination_confusion_significance(
            work,
            embryo_ids=embryo_ids,
            class_labels=class_labels,
            source_class=source_class,
            target_class=target_class,
            n_permutations=int(n_permutations),
            random_state=int(random_state) + i,
            q_threshold=float(q_threshold),
            window_min=float(center - half),
            window_max=float(center + half),
        )
        sub["window_center_hpf"] = float(center)
        sub["window_start_hpf"] = float(center - half)
        sub["window_end_hpf"] = float(center + half)
        rows.append(sub)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out

    q_global = _bh_fdr(out["pval_dest_confusion_perm"].to_numpy(dtype=float))
    out["qval_dest_confusion_global_perm"] = q_global
    out["is_dest_confusion_significant_global_perm"] = (
        (q_global <= float(q_threshold))
        & (out["dest_confusion_z"].to_numpy(dtype=float) > 0)
    )
    out["is_dest_confusion_significant_global_perm"] = (
        out["is_dest_confusion_significant_global_perm"].fillna(False).astype(bool)
    )
    return out


def _kmeans_numpy(
    x: np.ndarray,
    *,
    k: int,
    random_state: int,
    n_init: int = 20,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    """Simple KMeans with multiple restarts using NumPy only."""
    if k < 2:
        raise ValueError("k must be >= 2")
    n = int(x.shape[0])
    if k > n:
        raise ValueError("k cannot exceed number of samples")

    rng = np.random.default_rng(int(random_state))
    best_labels = None
    best_inertia = np.inf

    for _ in range(max(1, int(n_init))):
        init_idx = rng.choice(n, size=k, replace=False)
        centers = x[init_idx].copy()
        labels = np.full(n, -1, dtype=int)

        for _iter in range(int(max_iter)):
            sq = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(sq, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels

            for j in range(k):
                members = x[labels == j]
                if len(members) == 0:
                    centers[j] = x[rng.integers(0, n)]
                else:
                    centers[j] = members.mean(axis=0)

        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    if best_labels is None:
        raise RuntimeError("KMeans failed to produce labels")
    return best_labels, float(best_inertia)



def _mode_str(x: pd.Series) -> str:
    s = x.astype(str)
    mode = s.mode()
    if mode.empty:
        return ""
    return str(mode.iloc[0])



def _validate_base_columns(df: pd.DataFrame) -> None:
    required = {"embryo_id", "time_bin", "true_class", "pred_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for col in ["embryo_id", "time_bin", "true_class", "pred_class"]:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values")



def _validate_label_set(df: pd.DataFrame, class_labels: list[str]) -> None:
    class_set = set(class_labels)
    if not class_set:
        raise ValueError("class_labels must be non-empty")
    bad_true = set(df["true_class"].astype(str).unique()) - class_set
    bad_pred = set(df["pred_class"].astype(str).unique()) - class_set
    if bad_true:
        raise ValueError(f"true_class labels outside class_labels: {sorted(bad_true)}")
    if bad_pred:
        raise ValueError(f"pred_class labels outside class_labels: {sorted(bad_pred)}")



def _validate_time_order(df: pd.DataFrame) -> None:
    if not pd.api.types.is_integer_dtype(df["time_bin"]):
        raise ValueError(f"time_bin must be integer dtype; got {df['time_bin'].dtype}")
    for embryo_id, grp in df.sort_values(["embryo_id", "time_bin"]).groupby("embryo_id"):
        bins = grp["time_bin"].to_numpy(dtype=int)
        if len(bins) > 1 and not np.all(np.diff(bins) > 0):
            raise ValueError(
                f"embryo {embryo_id}: time_bin not strictly increasing (duplicates/unsorted): {bins.tolist()}"
            )



def _probability_columns_for_labels(class_labels: list[str]) -> list[str]:
    return [f"pred_proba_{label}" for label in class_labels]



def _validate_probabilities(df: pd.DataFrame, class_labels: list[str], sum_tol: float) -> None:
    # Require exact set match between present pred_proba_* columns and class_labels.
    present_prob_cols = [c for c in df.columns if c.startswith("pred_proba_")]
    present_labels = {c.replace("pred_proba_", "", 1) for c in present_prob_cols}
    expected_labels = set(class_labels)
    missing = expected_labels - present_labels
    extra = present_labels - expected_labels
    if missing or extra:
        raise ValueError(
            "pred_proba_* columns mismatch class_labels. "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    required = _probability_columns_for_labels(class_labels) + ["p_true", "p_pred"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required Stage 1 probability columns: {missing_required}")

    for col in required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric, got {df[col].dtype}")
        x = df[col].to_numpy(dtype=float)
        if np.isnan(x).any():
            raise ValueError(f"Column '{col}' contains NaN values")
        if np.any((x < -1e-6) | (x > 1 + 1e-6)):
            mn, mx = float(np.min(x)), float(np.max(x))
            raise ValueError(f"Column '{col}' out of [0,1] tolerance: min={mn:.4f}, max={mx:.4f}")

    prob_sum = df[_probability_columns_for_labels(class_labels)].sum(axis=1).to_numpy(dtype=float)
    bad = np.where(np.abs(prob_sum - 1.0) > float(sum_tol))[0]
    if len(bad) > 0:
        i = int(bad[0])
        raise ValueError(
            "pred_proba row sums must be ~1.0. "
            f"first_bad_row={i}, sum={prob_sum[i]:.6f}, tol={sum_tol}"
        )



def validate_predictions_for_stage(
    df: pd.DataFrame,
    *,
    class_labels: list[str],
    stage_mode: str,
    prob_sum_tol: float = 1e-3,
) -> None:
    if stage_mode not in VALID_STAGES:
        raise ValueError(f"Unknown stage_mode='{stage_mode}'. Valid: {sorted(VALID_STAGES)}")

    _validate_base_columns(df)
    _validate_label_set(df, class_labels)
    _validate_time_order(df)

    has_dupes = df.duplicated(subset=["embryo_id", "time_bin"]).any()
    if has_dupes:
        raise ValueError("Found duplicate (embryo_id, time_bin) rows; expected one row per embryo-time")

    if stage_mode != STAGE_HARD:
        _validate_probabilities(df, class_labels, sum_tol=prob_sum_tol)



def _compute_prob_matrix(df: pd.DataFrame, class_labels: list[str]) -> np.ndarray:
    return df[_probability_columns_for_labels(class_labels)].to_numpy(dtype=float)



def _compute_stage_values(
    df: pd.DataFrame,
    *,
    class_labels: list[str],
    stage_mode: str,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    n = len(df)
    k = len(class_labels)

    if stage_mode == STAGE_HARD:
        pred = df["pred_class"].astype(str).to_numpy()
        values = np.zeros((n, k), dtype=float)
        for j, label in enumerate(class_labels):
            values[:, j] = (pred == str(label)).astype(float)
        return values, None

    probs = _compute_prob_matrix(df, class_labels)

    if stage_mode == STAGE_SOFT:
        return probs, None

    if stage_mode == STAGE_DELTA:
        true = df["true_class"].astype(str).to_numpy()
        onehot = np.zeros_like(probs)
        for j, label in enumerate(class_labels):
            onehot[:, j] = (true == str(label)).astype(float)
        return probs - onehot, None

    if stage_mode in {STAGE_RESIDUAL, STAGE_RESIDUAL_DTW}:
        prob_cols = _probability_columns_for_labels(class_labels)
        baseline_mu = (
            df.groupby(["true_class", "time_bin"], sort=True)[prob_cols]
            .mean()
            .reset_index()
            .rename(columns={c: f"{c}_mu" for c in prob_cols})
        )
        merged = df.merge(baseline_mu, on=["true_class", "time_bin"], how="left")
        values = np.zeros_like(probs)
        for j, col in enumerate(prob_cols):
            values[:, j] = merged[col].to_numpy(dtype=float) - merged[f"{col}_mu"].to_numpy(dtype=float)
        return values, baseline_mu

    raise ValueError(f"Unsupported stage_mode={stage_mode}")



def build_stage_feature_matrix(
    df: pd.DataFrame,
    *,
    class_labels: list[str],
    stage_mode: str,
    wrong_rate_n_permutations: int = 400,
    wrong_rate_q_threshold: float = 0.10,
    wrong_rate_window_min: float | None = None,
    wrong_rate_window_max: float | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int], pd.DataFrame, pd.DataFrame | None]:
    """Build stage feature matrix.

    Returns
    -------
    X : np.ndarray
        Flattened features (n_embryos, n_classes * n_time_bins)
    tensor : np.ndarray
        Tensor in class-first layout (n_embryos, n_classes, n_time_bins)
    feature_columns : list[str]
        Names aligned with X columns
    time_bins : list[int]
        Canonical sorted time bins
    meta : pd.DataFrame
        Per-embryo metadata
    baseline_mu : pd.DataFrame | None
        Stage-3 baseline table if residual mode, else None
    """
    validate_predictions_for_stage(df, class_labels=class_labels, stage_mode=stage_mode)

    work = df.copy()
    work["embryo_id"] = work["embryo_id"].astype(str)
    work["true_class"] = work["true_class"].astype(str)
    work["pred_class"] = work["pred_class"].astype(str)

    values, baseline_mu = _compute_stage_values(work, class_labels=class_labels, stage_mode=stage_mode)
    for j, label in enumerate(class_labels):
        work[f"__value_{label}"] = values[:, j]

    embryo_ids = sorted(work["embryo_id"].unique().tolist())
    time_bins = sorted(work["time_bin"].astype(int).unique().tolist())

    blocks = []
    feature_columns: list[str] = []
    for label in class_labels:
        piv = work.pivot(index="embryo_id", columns="time_bin", values=f"__value_{label}")
        piv = piv.reindex(index=embryo_ids, columns=time_bins)
        piv = piv.apply(pd.to_numeric, errors="coerce")

        # Fill missing embryo-time entries by column mean, then zero fallback.
        piv = piv.fillna(piv.mean(axis=0))
        piv = piv.fillna(0.0)

        blocks.append(piv.to_numpy(dtype=float))
        feature_columns.extend([f"{label}|tb{tb}" for tb in time_bins])

    tensor = np.stack(blocks, axis=1)  # (n_embryos, n_classes, n_time_bins)
    x = tensor.reshape(len(embryo_ids), -1)

    true_class_mode = work.groupby("embryo_id", sort=True)["true_class"].agg(_mode_str)
    n_observed_bins = work.groupby("embryo_id", sort=True)["time_bin"].nunique()

    meta = pd.DataFrame({
        "embryo_id": embryo_ids,
        "true_class": [true_class_mode.loc[e] for e in embryo_ids],
        "n_observed_bins": [int(n_observed_bins.loc[e]) for e in embryo_ids],
        "n_missing_bins": [int(len(time_bins) - n_observed_bins.loc[e]) for e in embryo_ids],
    })

    wrong_frac = (
        (work["pred_class"] != work["true_class"])
        .groupby(work["embryo_id"], sort=True)
        .mean()
    )
    meta["wrong_frac"] = [float(wrong_frac.loc[e]) for e in embryo_ids]

    med = float(np.nanmedian(meta["wrong_frac"].to_numpy(dtype=float)))
    q75 = float(np.nanquantile(meta["wrong_frac"].to_numpy(dtype=float), 0.75))
    meta["is_wrong_more_often"] = meta["wrong_frac"] > med
    meta["is_wrong_top_quartile"] = meta["wrong_frac"] >= q75

    sig = _compute_wrong_rate_significance(
        work,
        embryo_ids=embryo_ids,
        class_labels=class_labels,
        n_permutations=int(wrong_rate_n_permutations),
        random_state=int(random_state),
        q_threshold=float(wrong_rate_q_threshold),
        window_min=wrong_rate_window_min,
        window_max=wrong_rate_window_max,
    )
    meta = meta.merge(sig, on="embryo_id", how="left", validate="one_to_one")

    return x, tensor, feature_columns, time_bins, meta, baseline_mu



def _pairwise_multivariate_dtw(
    series: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Pairwise constrained DTW for multivariate series.

    Parameters
    ----------
    series : np.ndarray
        Shape (n_embryos, n_time, n_dim)
    window : int
        Sakoe-Chiba half-window (in bins).
    """
    n_embryos, t_len, _ = series.shape
    dist = np.zeros((n_embryos, n_embryos), dtype=float)

    def dtw(a: np.ndarray, b: np.ndarray) -> float:
        w = max(int(window), abs(len(a) - len(b)))
        inf = np.inf
        dp = np.full((len(a) + 1, len(b) + 1), inf, dtype=float)
        dp[0, 0] = 0.0
        for i in range(1, len(a) + 1):
            j_start = max(1, i - w)
            j_end = min(len(b), i + w)
            for j in range(j_start, j_end + 1):
                cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[len(a), len(b)])

    for i in range(n_embryos):
        for j in range(i + 1, n_embryos):
            d = dtw(series[i], series[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist



def run_stage_geometry(
    df: pd.DataFrame,
    *,
    class_labels: list[str],
    stage_mode: str,
    k_values: tuple[int, ...] = (2, 3, 4, 5),
    pca_components: int = 3,
    random_state: int = 42,
    kmeans_n_init: int = 20,
    dtw_window: int = 1,
    wrong_rate_n_permutations: int = 400,
    wrong_rate_q_threshold: float = 0.10,
    wrong_rate_window_min: float | None = None,
    wrong_rate_window_max: float | None = None,
) -> StageGeometryResult:
    x, tensor, feature_columns, time_bins, stage_table, baseline_mu = build_stage_feature_matrix(
        df,
        class_labels=class_labels,
        stage_mode=stage_mode,
        wrong_rate_n_permutations=int(wrong_rate_n_permutations),
        wrong_rate_q_threshold=float(wrong_rate_q_threshold),
        wrong_rate_window_min=wrong_rate_window_min,
        wrong_rate_window_max=wrong_rate_window_max,
        random_state=int(random_state),
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)

    distance_matrix: np.ndarray | None = None
    if stage_mode == STAGE_RESIDUAL_DTW:
        series = np.transpose(tensor, (0, 2, 1))
        distance_matrix = _pairwise_multivariate_dtw(series, window=dtw_window)
        n_components = max(1, min(int(pca_components), len(stage_table) - 1))
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=int(random_state),
            n_init=4,
            normalized_stress="auto",
        )
        z = mds.fit_transform(distance_matrix)
        explained = [np.nan] * n_components
    else:
        n_components = max(1, min(int(pca_components), x_scaled.shape[0], x_scaled.shape[1]))
        pca = PCA(n_components=n_components, random_state=int(random_state))
        z = pca.fit_transform(x_scaled)
        explained = pca.explained_variance_ratio_.tolist()

    for i in range(z.shape[1]):
        stage_table[f"PC{i+1}"] = z[:, i]

    metrics_by_k: list[dict[str, Any]] = []
    for k in sorted(set(int(kv) for kv in k_values if int(kv) >= 2)):
        if k > len(stage_table):
            continue
        labels, inertia = _kmeans_numpy(
            z,
            k=k,
            random_state=int(random_state),
            n_init=int(kmeans_n_init),
        )
        stage_table[f"cluster_k{k}"] = labels.astype(int)

        row: dict[str, Any] = {
            "k": k,
            "inertia": float(inertia),
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
        }
        if len(np.unique(labels)) > 1 and len(labels) > k:
            row["silhouette"] = float(silhouette_score(z, labels))
            row["davies_bouldin"] = float(davies_bouldin_score(z, labels))
        metrics_by_k.append(row)

    if "PC1" in stage_table.columns and stage_table["wrong_frac"].nunique() > 1 and stage_table["PC1"].nunique() > 1:
        corr = np.corrcoef(
            stage_table["PC1"].to_numpy(dtype=float),
            stage_table["wrong_frac"].to_numpy(dtype=float),
        )[0, 1]
        stage_table["corr_pc1_wrong_frac"] = float(corr)
    else:
        stage_table["corr_pc1_wrong_frac"] = np.nan

    return StageGeometryResult(
        stage=stage_mode,
        stage_table=stage_table,
        feature_columns=feature_columns,
        class_labels=class_labels,
        time_bins=time_bins,
        pca_scores=z,
        distance_matrix=distance_matrix,
        metrics_by_k=metrics_by_k,
        explained_variance_ratio=explained,
        baseline_mu=baseline_mu,
    )
