from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd


def _longest_run_true(x: np.ndarray) -> int:
    if x.size == 0:
        return 0
    max_streak = 0
    cur = 0
    for v in x:
        if v:
            cur += 1
            if cur > max_streak:
                max_streak = cur
        else:
            cur = 0
    return int(max_streak)


def compute_per_embryo_metrics(
    embryo_predictions: pd.DataFrame,
    *,
    embryo_id_col: str = "embryo_id",
    time_bin_col: str = "time_bin",
    pred_col: str = "pred_class",
    true_col: str = "true_class",
    p_true_col: str = "p_true",
    p_pred_col: str = "p_pred",
    allow_mode_true_class: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-embryo wrongness metrics and baseline tables."""
    df = embryo_predictions.copy()
    df["_is_wrong"] = (df[pred_col] != df[true_col]).astype(np.int8)

    baseline_ct = (
        df.groupby([true_col, time_bin_col], sort=True)["_is_wrong"]
        .mean()
        .reset_index()
        .rename(
            columns={
                true_col: "true_class",
                time_bin_col: "time_bin",
                "_is_wrong": "baseline_wrong_rate",
            }
        )
    )

    baseline_c = (
        df.groupby(true_col, sort=True)["_is_wrong"]
        .mean()
        .reset_index()
        .rename(columns={true_col: "true_class", "_is_wrong": "baseline_wrong_rate_class"})
    )

    df = df.sort_values([embryo_id_col, time_bin_col], kind="mergesort")

    df["_embryo_idx"] = pd.Categorical(df[embryo_id_col]).codes
    if (df["_embryo_idx"] < 0).any():
        raise ValueError("Unexpected -1 embryo_idx values")

    true_per_embryo = df.groupby(embryo_id_col)[true_col].nunique()
    multi = true_per_embryo[true_per_embryo > 1]
    if len(multi) > 0:
        if not allow_mode_true_class:
            raise ValueError(
                f"Embryos with >1 true_class detected: {multi.index.tolist()[:5]} (n={len(multi)})"
            )
        warnings.warn(
            "Embryos with >1 true_class detected; using per-embryo mode true_class.",
            stacklevel=2,
        )

    true_mode = df.groupby(embryo_id_col)[true_col].agg(lambda s: s.mode().iloc[0])

    baseline_ct_keyed = baseline_ct.set_index(["true_class", "time_bin"])["baseline_wrong_rate"]
    baseline_c_keyed = baseline_c.set_index("true_class")["baseline_wrong_rate_class"]

    rows = []
    for embryo_id, g in df.groupby(embryo_id_col, sort=False):
        true_class = str(true_mode.loc[embryo_id])
        embryo_idx = int(g["_embryo_idx"].iloc[0])
        n_windows = int(len(g))

        is_wrong = g["_is_wrong"].to_numpy(dtype=bool)
        n_wrong = int(is_wrong.sum())
        wrong_rate = float(n_wrong / n_windows) if n_windows > 0 else 0.0

        longest_wrong_streak = _longest_run_true(is_wrong)
        longest_correct_streak = _longest_run_true(~is_wrong)

        preds = g[pred_col].astype(str).to_numpy()
        flip_rate = float(np.mean(preds[1:] != preds[:-1])) if len(preds) > 1 else 0.0

        p_true = g[p_true_col].to_numpy(dtype=float)
        p_pred = g[p_pred_col].to_numpy(dtype=float)

        mean_p_true = float(np.mean(p_true)) if n_windows > 0 else np.nan
        min_p_true = float(np.min(p_true)) if n_windows > 0 else np.nan

        if n_wrong > 0:
            margins = (p_true - p_pred)[is_wrong]
            mean_margin = float(np.mean(margins)) if margins.size else np.nan
            wrong_preds = preds[is_wrong]
            vc = pd.Series(wrong_preds).value_counts()
            top_confused_as = str(vc.index[0])
            top_confused_frac = float(vc.iloc[0] / n_wrong)
            # transitions among wrong-only sequence
            n_class_switches = int(np.sum(wrong_preds[1:] != wrong_preds[:-1])) if len(wrong_preds) > 1 else 0
        else:
            mean_margin = np.nan
            top_confused_as = ""
            top_confused_frac = 0.0
            n_class_switches = 0

        exp_vals = []
        for tb in g[time_bin_col].to_numpy(dtype=int):
            key = (true_class, int(tb))
            if key not in baseline_ct_keyed.index:
                raise ValueError(f"Missing baseline_wrong_rate for (true_class, time_bin)={key}")
            exp_vals.append(float(baseline_ct_keyed.loc[key]))
        expected_wrong_rate = float(np.mean(exp_vals)) if exp_vals else np.nan

        rows.append(
            {
                "embryo_id": str(embryo_id),
                "true_class": true_class,
                "embryo_idx": embryo_idx,
                "n_windows": n_windows,
                "n_wrong": n_wrong,
                "wrong_rate": wrong_rate,
                "longest_wrong_streak": longest_wrong_streak,
                "longest_correct_streak": longest_correct_streak,
                "flip_rate": flip_rate,
                "mean_p_true": mean_p_true,
                "min_p_true": min_p_true,
                "mean_margin": mean_margin,
                "top_confused_as": top_confused_as,
                "top_confused_frac": top_confused_frac,
                "n_class_switches": n_class_switches,
                "expected_wrong_rate": expected_wrong_rate,
                "baseline_wrong_rate_class": float(baseline_c_keyed.loc[true_class]),
            }
        )

    per_embryo = pd.DataFrame(rows).sort_values("embryo_idx").reset_index(drop=True)

    if not per_embryo["embryo_idx"].is_monotonic_increasing:
        raise ValueError("per_embryo_metrics must be sorted by embryo_idx")
    if not (per_embryo["embryo_idx"].to_numpy() == np.arange(len(per_embryo))).all():
        raise ValueError("per_embryo_metrics embryo_idx must be 0..n-1 with no gaps")

    return per_embryo, baseline_ct, baseline_c
