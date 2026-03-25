"""
Envelope construction utilities for the WT quantile-envelope penetrance pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CATEGORY_COL,
    EMBRYO_BIN_AGG,
    EMBRYO_COL,
    GENOTYPE_COL,
    LOESS_CANDIDATE_FRACS,
    LOESS_FALLBACK_FRAC,
    LOESS_FRAC_OVERRIDE,
    METRIC_NAME,
    METRIC_NONNEG,
    PAIR_COL,
    QUANTILE_HIGH,
    QUANTILE_LOW,
    ROBUST_SMOOTHING_MIN_POINTS,
    ROBUST_SMOOTHING_MIN_RESID_FRACTION,
    ROBUST_SMOOTHING_SIGMA_THRESHOLD,
    ROBUST_SMOOTHING_WINDOW,
    TIME_COL,
)
from smoothing import detect_curve_inliers, loess_smooth, select_quantile_curve_smoother


def aggregate_embryo_bins(df: pd.DataFrame, agg: str = EMBRYO_BIN_AGG) -> pd.DataFrame:
    """
    Collapse frame rows into one embryo_id × time_bin summary row.
    """
    agg_fn = {"median": "median", "mean": "mean"}.get(agg)
    if agg_fn is None:
        raise ValueError(f"agg must be 'median' or 'mean', got {agg!r}")

    meta_cols = [
        c for c in [GENOTYPE_COL, CATEGORY_COL, "cluster_subcategories", "experiment_id", "experiment_date", PAIR_COL]
        if c in df.columns
    ]
    group_cols = [EMBRYO_COL, "time_bin"]

    agg_spec: dict[str, str] = {
        TIME_COL: "median",
        METRIC_NAME: agg_fn,
    }
    for col in meta_cols:
        agg_spec[col] = "first"

    embryo_bins = (
        df.groupby(group_cols, as_index=False)
        .agg(agg_spec)
        .sort_values([EMBRYO_COL, "time_bin"])
        .reset_index(drop=True)
    )

    frame_counts = (
        df.groupby(group_cols)
        .size()
        .rename("n_frames_in_bin")
        .reset_index()
    )
    return embryo_bins.merge(frame_counts, on=group_cols, how="left")


def compute_raw_wt_quantiles(
    wt_df: pd.DataFrame,
    *,
    min_units: int,
    unit_label: str,
) -> pd.DataFrame:
    """
    Compute raw WT quantiles for the rows provided.
    """
    rows = []
    for tb, grp in wt_df.groupby("time_bin"):
        vals = grp[METRIC_NAME].dropna().to_numpy()
        n_units = len(vals)
        n_embryos = grp[EMBRYO_COL].nunique()
        supported = n_units >= min_units

        raw_low = float(np.quantile(vals, QUANTILE_LOW)) if n_units > 0 else np.nan
        raw_high = float(np.quantile(vals, QUANTILE_HIGH)) if n_units > 0 else np.nan
        rows.append(
            {
                "time_bin": tb,
                "raw_low": raw_low,
                "raw_high": raw_high,
                "n_wt_units": n_units,
                "n_wt_embryos": n_embryos,
                "supported": supported,
            }
        )

    df_q = pd.DataFrame(rows).sort_values("time_bin").reset_index(drop=True)
    print(
        f"\nRaw quantiles ({unit_label}): {df_q['supported'].sum()}/{len(df_q)} bins "
        f"supported (>= {min_units} {unit_label}s)"
    )
    return df_q


def validate_envelope(lower_sm, upper_sm, supported_mask, metric_nonneg=True):
    """Ensure lower < upper and lower >= 0 (if nonnegative)."""
    lower_sm = lower_sm.copy()
    upper_sm = upper_sm.copy()

    valid = np.where(supported_mask)[0]

    if metric_nonneg:
        neg = valid[lower_sm[valid] < 0]
        if len(neg) > 0:
            print(f"  WARNING: Clipping {len(neg)} bins where smoothed lower < 0")
            lower_sm[neg] = 0.0

    crossed = valid[lower_sm[valid] >= upper_sm[valid]]
    if len(crossed) > 0:
        print(f"  WARNING: {len(crossed)} bins where lower >= upper; nudging upper up")
        for i in crossed:
            upper_sm[i] = lower_sm[i] + 1e-6

    return lower_sm, upper_sm


def compute_wt_envelope(
    wt_df: pd.DataFrame,
    *,
    min_units: int,
    unit_label: str,
):
    """
    Compute raw WT quantiles and a smoothed diagnostic envelope for the given rows.
    """
    print("\n=== Step 1: Raw WT quantiles ===")
    df_q = compute_raw_wt_quantiles(wt_df, min_units=min_units, unit_label=unit_label)

    times = df_q["time_bin"].to_numpy(dtype=float)
    supported = df_q["supported"].to_numpy(dtype=bool)

    lower_result = upper_result = None
    lower_fit_mask = supported.copy()
    upper_fit_mask = supported.copy()
    lower_fit_diag: dict = {}
    upper_fit_diag: dict = {}

    if LOESS_FRAC_OVERRIDE is not None:
        print(f"\n=== Step 2: Smoothing (OVERRIDE frac={LOESS_FRAC_OVERRIDE}) ===")
        frac_low = frac_high = LOESS_FRAC_OVERRIDE
        valid = ~np.isnan(df_q["raw_low"].to_numpy())
        sm_low = np.full(len(times), np.nan)
        sm_high = np.full(len(times), np.nan)
        sm_low[valid] = loess_smooth(times[valid], df_q.loc[valid, "raw_low"].to_numpy(), frac_low)
        sm_high[valid] = loess_smooth(times[valid], df_q.loc[valid, "raw_high"].to_numpy(), frac_high)
    else:
        print("\n=== Step 2: Smoothing frac selection by shape stability ===")
        if supported.sum() >= max(ROBUST_SMOOTHING_MIN_POINTS, 3):
            supported_idx = np.where(supported)[0]
            low_inliers, lower_fit_diag = detect_curve_inliers(
                times[supported],
                df_q.loc[supported, "raw_low"].to_numpy(),
                window=ROBUST_SMOOTHING_WINDOW,
                min_points=ROBUST_SMOOTHING_MIN_POINTS,
                sigma_threshold=ROBUST_SMOOTHING_SIGMA_THRESHOLD,
                min_resid_fraction=ROBUST_SMOOTHING_MIN_RESID_FRACTION,
            )
            high_inliers, upper_fit_diag = detect_curve_inliers(
                times[supported],
                df_q.loc[supported, "raw_high"].to_numpy(),
                window=ROBUST_SMOOTHING_WINDOW,
                min_points=ROBUST_SMOOTHING_MIN_POINTS,
                sigma_threshold=ROBUST_SMOOTHING_SIGMA_THRESHOLD,
                min_resid_fraction=ROBUST_SMOOTHING_MIN_RESID_FRACTION,
            )
            lower_fit_mask = np.zeros(len(df_q), dtype=bool)
            upper_fit_mask = np.zeros(len(df_q), dtype=bool)
            lower_fit_mask[supported_idx] = low_inliers
            upper_fit_mask[supported_idx] = high_inliers

        lower_result = select_quantile_curve_smoother(
            times,
            df_q["raw_low"].to_numpy(),
            candidate_fracs=LOESS_CANDIDATE_FRACS,
            nonnegative=METRIC_NONNEG,
            fallback_frac=LOESS_FALLBACK_FRAC,
            fit_mask=lower_fit_mask,
        )
        upper_result = select_quantile_curve_smoother(
            times,
            df_q["raw_high"].to_numpy(),
            candidate_fracs=LOESS_CANDIDATE_FRACS,
            nonnegative=METRIC_NONNEG,
            fallback_frac=LOESS_FALLBACK_FRAC,
            fit_mask=upper_fit_mask,
        )
        lower_result.fit_diagnostics = lower_fit_diag
        upper_result.fit_diagnostics = upper_fit_diag

        for name, result in [("lower", lower_result), ("upper", upper_result)]:
            fallback = " [FALLBACK]" if result.used_fallback else ""
            print(f"  [{name}] selected frac={result.selected_frac}{fallback}")
            excluded_bins = [int(v) for v in result.fit_diagnostics.get("excluded_x", [])]
            if excluded_bins:
                print(f"    excluded outlier bins from smoothing: {excluded_bins}")
            for frac, diag in result.diagnostics.items():
                status = "ok" if diag["passed"] else f"fail:{diag['failed_checks']}"
                print(f"    frac={frac}: {status}")

        frac_low = lower_result.selected_frac
        frac_high = upper_result.selected_frac
        sm_low = lower_result.smoothed_y
        sm_high = upper_result.smoothed_y

    print("\n=== Step 3: Envelope validation ===")
    sm_low, sm_high = validate_envelope(sm_low, sm_high, supported, metric_nonneg=METRIC_NONNEG)

    df_env = df_q.copy()
    df_env["smoothed_low"] = sm_low
    df_env["smoothed_high"] = sm_high
    df_env["lower_frac"] = frac_low
    df_env["upper_frac"] = frac_high
    df_env["smooth_fit_low"] = lower_fit_mask
    df_env["smooth_fit_high"] = upper_fit_mask
    df_env["smooth_excluded_low"] = supported & ~lower_fit_mask
    df_env["smooth_excluded_high"] = supported & ~upper_fit_mask
    return df_env, lower_result, upper_result
