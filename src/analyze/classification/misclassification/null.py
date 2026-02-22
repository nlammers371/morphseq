from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analyze.utils.resampling.lightweight_numpy_resampling import LiteRun, run_lite

WRONG_RATE_SPEC = {
    "type": "permute_labels",
    "within": "time_bin",
    "label": "pred_class",
    "derived": "is_wrong_perm",
}
STREAK_SPEC = {
    "type": "simulate",
    "dist": "bernoulli",
    "p": "baseline_wrong_rate(true_class,time_bin)",
    "gaps": "observed_only",
}
TOP_CONFUSED_SPEC = {
    "type": "simulate",
    "dist": "multinomial",
    "probs": "pooled_q_C_excluding_self",
}


def _vectorized_longest_run(matrix: np.ndarray) -> np.ndarray:
    """Longest run of True per row in boolean matrix of shape (n_sim, T)."""
    n_sim, _ = matrix.shape
    padded = np.concatenate(
        [
            np.zeros((n_sim, 1), dtype=np.int8),
            matrix.astype(np.int8),
            np.zeros((n_sim, 1), dtype=np.int8),
        ],
        axis=1,
    )
    diff = np.diff(padded, axis=1)
    out = np.zeros(n_sim, dtype=int)
    for i in range(n_sim):
        starts = np.where(diff[i] == 1)[0]
        ends = np.where(diff[i] == -1)[0]
        if len(starts) == 0:
            out[i] = 0
        else:
            out[i] = int(np.max(ends - starts))
    return out


def null_test_wrong_rate(
    *,
    embryo_predictions: pd.DataFrame,
    per_embryo_metrics: pd.DataFrame,
    class_labels: list[str],
    n_permutations: int = 1_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, LiteRun]:
    """Permute pred_class within time_bin and evaluate embryo-level wrong_rate excess."""
    df_sorted = embryo_predictions.sort_values("time_bin").copy()

    pred_codes = pd.Categorical(df_sorted["pred_class"], categories=class_labels).codes.astype(np.int16)
    true_codes = pd.Categorical(df_sorted["true_class"], categories=class_labels).codes.astype(np.int16)
    if (pred_codes < 0).any():
        raise ValueError("pred_class contains labels not in class_labels")
    if (true_codes < 0).any():
        raise ValueError("true_class contains labels not in class_labels")

    embryo_idx_arr = df_sorted["embryo_idx"].to_numpy(dtype=int)
    n_embryos = len(per_embryo_metrics)

    if not per_embryo_metrics["embryo_idx"].is_monotonic_increasing:
        raise ValueError("per_embryo_metrics must be sorted by embryo_idx")
    if not (per_embryo_metrics["embryo_idx"].to_numpy() == np.arange(n_embryos)).all():
        raise ValueError("per_embryo_metrics embryo_idx must be 0..n-1 with no gaps")

    n_windows_arr = per_embryo_metrics["n_windows"].to_numpy(dtype=float)
    obs_wrong_rate = per_embryo_metrics["wrong_rate"].to_numpy(dtype=float)

    bin_slices: dict[int, tuple[int, int]] = {}
    pos = 0
    for tb, grp in df_sorted.groupby("time_bin", sort=True):
        n = len(grp)
        bin_slices[int(tb)] = (pos, pos + n)
        pos += n

    def kernel(rng: np.random.Generator, n_iters: int) -> dict[str, Any]:
        exceed = np.zeros(n_embryos, dtype=np.int64)
        null_sum = np.zeros(n_embryos, dtype=np.float64)
        null_sum_sq = np.zeros(n_embryos, dtype=np.float64)

        pred_perm = pred_codes.copy()

        for _ in range(n_iters):
            np.copyto(pred_perm, pred_codes)
            for tb in sorted(bin_slices):
                start, stop = bin_slices[tb]
                pred_perm[start:stop] = rng.permutation(pred_perm[start:stop])

            is_wrong_perm = (pred_perm != true_codes).astype(np.int8)

            wrong_count = np.bincount(
                embryo_idx_arr,
                weights=is_wrong_perm.astype(np.int32),
                minlength=n_embryos,
            ).astype(np.float64)

            wr = wrong_count / n_windows_arr
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
        test_name="misclassification_wrong_rate",
        n_iters=n_permutations,
        seed=random_state,
        spec=WRONG_RATE_SPEC,
        kernel=kernel,
        collect_samples=False,
        metadata={"class_labels": class_labels},
    )

    out = per_embryo_metrics.copy()
    out["wrong_rate_null_mean"] = run.summary["null_mean"]
    out["wrong_rate_null_std"] = run.summary["null_std"]
    out["wrong_rate_z"] = run.summary["z_score"]
    out["wrong_rate_exceed_count"] = run.summary["exceed_count"]
    out["pval_wrong_rate"] = run.summary["pval"]
    return out, run


def null_test_streak(
    *,
    per_embryo_metrics: pd.DataFrame,
    baseline_ct_df: pd.DataFrame,
    embryo_time_bins: pd.DataFrame,
    n_sim: int = 10_000,
    random_state: int = 42,
    p_clip: tuple[float, float] = (1e-6, 1 - 1e-6),
) -> tuple[pd.DataFrame, LiteRun]:
    """Bernoulli varying-p(t) null for longest wrong streak."""
    per = per_embryo_metrics.copy()

    base = baseline_ct_df.copy()
    base_key = base.set_index(["true_class", "time_bin"])["baseline_wrong_rate"]

    em_to_bins = (
        embryo_time_bins.sort_values(["embryo_id", "time_bin"]).groupby("embryo_id")["time_bin"].apply(list)
    )

    n_embryos = len(per)
    obs = per["longest_wrong_streak"].to_numpy(dtype=float)

    def kernel(rng: np.random.Generator, n_iters: int) -> dict[str, Any]:
        exceed = np.zeros(n_embryos, dtype=np.int64)
        null_mean = np.zeros(n_embryos, dtype=np.float64)
        null_std = np.zeros(n_embryos, dtype=np.float64)

        for i, row in per.iterrows():
            eid = row["embryo_id"]
            true_class = row["true_class"]
            bins = em_to_bins.get(eid, [])
            if len(bins) == 0:
                continue

            p_vec = []
            for tb in bins:
                key = (true_class, int(tb))
                if key not in base_key.index:
                    raise ValueError(f"Missing baseline rate for key={key}")
                p_vec.append(float(base_key.loc[key]))
            p_vec_arr = np.clip(np.asarray(p_vec, dtype=float), p_clip[0], p_clip[1])

            sim_matrix = rng.random((n_iters, len(p_vec_arr))) < p_vec_arr
            streak_null = _vectorized_longest_run(sim_matrix)

            exceed_i = int(np.sum(streak_null >= obs[i]))
            exceed[i] = exceed_i
            null_mean[i] = float(np.mean(streak_null))
            null_std[i] = float(np.std(streak_null))

        pval = (exceed + 1) / (n_iters + 1)
        z = (obs - null_mean) / (null_std + 1e-9)
        return {
            "stat_name": "longest_wrong_streak",
            "exceed_count": exceed,
            "null_mean": null_mean,
            "null_std": null_std,
            "obs_stat": obs,
            "pval": pval,
            "z_score": z,
        }

    run = run_lite(
        test_name="misclassification_streak",
        n_iters=n_sim,
        seed=random_state,
        spec=STREAK_SPEC,
        kernel=kernel,
        collect_samples=False,
        metadata={"p_clip": p_clip},
    )

    per["streak_null_mean"] = run.summary["null_mean"]
    per["streak_null_std"] = run.summary["null_std"]
    per["streak_z"] = run.summary["z_score"]
    per["streak_exceed_count"] = run.summary["exceed_count"]
    per["pval_streak"] = run.summary["pval"]
    return per, run


def null_test_top_confused_frac(
    *,
    per_embryo_metrics: pd.DataFrame,
    embryo_predictions: pd.DataFrame,
    class_labels: list[str],
    n_sim: int = 10_000,
    random_state: int = 42,
    require_n_wrong_min: int = 3,
    loo_min_class_size: int = 10,
) -> tuple[pd.DataFrame, LiteRun]:
    """Multinomial null for top-confused fraction."""
    per = per_embryo_metrics.copy()
    preds = embryo_predictions.copy()
    preds["is_wrong"] = (preds["pred_class"] != preds["true_class"]).astype(bool)

    wrong = preds[preds["is_wrong"]].copy()
    class_sizes = per.groupby("true_class")["embryo_id"].nunique().to_dict()

    n_embryos = len(per)
    obs = per["top_confused_frac"].to_numpy(dtype=float)

    q_meta: dict[str, Any] = {}
    q_excl_meta: dict[str, Any] = {}

    def kernel(rng: np.random.Generator, n_iters: int) -> dict[str, Any]:
        exceed = np.zeros(n_embryos, dtype=np.int64)
        null_mean = np.zeros(n_embryos, dtype=np.float64)
        null_std = np.zeros(n_embryos, dtype=np.float64)
        pval = np.ones(n_embryos, dtype=np.float64)
        z = np.zeros(n_embryos, dtype=np.float64)

        skip = np.zeros(n_embryos, dtype=bool)
        skip_reason = np.array([""] * n_embryos, dtype=object)
        used_loo = np.zeros(n_embryos, dtype=bool)

        for i, row in per.iterrows():
            eid = row["embryo_id"]
            true_class = row["true_class"]
            n_wrong = int(row["n_wrong"])

            if n_wrong < require_n_wrong_min:
                skip[i] = True
                skip_reason[i] = "n_wrong<require_n_wrong_min"
                continue

            support = [k for k in class_labels if k != true_class]
            wrong_true_class = wrong[wrong["true_class"] == true_class]

            if len(wrong_true_class) == 0:
                skip[i] = True
                skip_reason[i] = "no_pooled_wrong_rows_for_class"
                continue

            do_loo = int(class_sizes.get(true_class, 0)) >= loo_min_class_size
            used_loo[i] = do_loo

            if do_loo:
                pool_rows = wrong_true_class[wrong_true_class["embryo_id"] != eid]
            else:
                pool_rows = wrong_true_class

            counts = (
                pool_rows["pred_class"].value_counts().reindex(support, fill_value=0).to_numpy(dtype=float)
            )
            total = float(np.sum(counts))
            if total <= 0:
                skip[i] = True
                skip_reason[i] = "no_pooled_wrong_rows_for_class"
                continue

            q = counts / total
            q_meta[true_class] = {
                "support": support,
                "probs": q.tolist(),
            }
            q_excl_meta[str(eid)] = {
                "support": support,
                "probs": q.tolist(),
                "used_loo": bool(do_loo),
            }

            draws = rng.multinomial(n_wrong, q, size=n_iters)
            top_frac_null = draws.max(axis=1) / float(n_wrong)

            exceed_i = int(np.sum(top_frac_null >= obs[i]))
            exceed[i] = exceed_i
            null_mean[i] = float(np.mean(top_frac_null))
            null_std[i] = float(np.std(top_frac_null))
            pval[i] = float((exceed_i + 1) / (n_iters + 1))
            z[i] = float((obs[i] - null_mean[i]) / (null_std[i] + 1e-9))

        summary = {
            "stat_name": "top_confused_frac",
            "exceed_count": exceed,
            "null_mean": null_mean,
            "null_std": null_std,
            "obs_stat": obs,
            "pval": pval,
            "z_score": z,
            "top_confused_test_skipped": skip,
            "top_confused_skip_reason": skip_reason,
            "top_confused_used_loo": used_loo,
        }
        return summary

    run = run_lite(
        test_name="misclassification_top_confused",
        n_iters=n_sim,
        seed=random_state,
        spec=TOP_CONFUSED_SPEC,
        kernel=kernel,
        collect_samples=False,
        metadata={
            "q_C_per_class": q_meta,
            "q_C_excluding_self_per_embryo": q_excl_meta,
            "require_n_wrong_min": require_n_wrong_min,
            "loo_min_class_size": loo_min_class_size,
        },
    )

    per["top_confused_null_mean"] = run.summary["null_mean"]
    per["top_confused_null_std"] = run.summary["null_std"]
    per["top_confused_z"] = run.summary["z_score"]
    per["top_confused_exceed_count"] = run.summary["exceed_count"]
    per["pval_top_confused_frac"] = run.summary["pval"]
    per["top_confused_test_skipped"] = run.summary["top_confused_test_skipped"]
    per["top_confused_skip_reason"] = run.summary["top_confused_skip_reason"]
    per["top_confused_used_loo"] = run.summary["top_confused_used_loo"]
    return per, run
