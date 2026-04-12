"""
Threshold application and penetrance-calling utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import METRIC_NAME, UPPER_BOUND_ONLY


def mark_penetrant(
    df: pd.DataFrame,
    df_env: pd.DataFrame,
    *,
    call_mode: str = "raw",
    raw_lower_col: str = "raw_low",
    raw_upper_col: str = "raw_high",
    smoothed_lower_col: str = "smoothed_low",
    smoothed_upper_col: str = "smoothed_high",
    lower_excluded_col: str = "smooth_excluded_low",
    upper_excluded_col: str = "smooth_excluded_high",
    upper_bound_only: bool = UPPER_BOUND_ONLY,
) -> pd.DataFrame:
    """
    Mark rows as penetrant using raw, smoothed, or hybrid thresholds.
    """
    valid_modes = {"raw", "smoothed", "hybrid"}
    if call_mode not in valid_modes:
        raise ValueError(f"call_mode must be one of {sorted(valid_modes)}, got {call_mode!r}")

    lookup_cols = [
        raw_lower_col,
        raw_upper_col,
        smoothed_lower_col,
        smoothed_upper_col,
        "supported",
    ]
    for optional_col in [lower_excluded_col, upper_excluded_col]:
        if optional_col in df_env.columns:
            lookup_cols.append(optional_col)

    env_lookup = df_env.set_index("time_bin")[lookup_cols]

    out = df.copy()
    out = out.join(env_lookup, on="time_bin")

    low_excluded = out.get(lower_excluded_col, pd.Series(False, index=out.index)).fillna(False).astype(bool)
    high_excluded = out.get(upper_excluded_col, pd.Series(False, index=out.index)).fillna(False).astype(bool)
    supported = out["supported"].fillna(False).astype(bool)

    if call_mode == "raw":
        active_low = out[raw_lower_col]
        active_high = out[raw_upper_col]
        low_source = np.where(supported, "raw", "unsupported")
        high_source = np.where(supported, "raw", "unsupported")
    elif call_mode == "smoothed":
        active_low = out[smoothed_lower_col]
        active_high = out[smoothed_upper_col]
        low_source = np.full(len(out), "smoothed", dtype=object)
        high_source = np.full(len(out), "smoothed", dtype=object)
    else:
        use_smoothed_low = (~supported) | low_excluded
        use_smoothed_high = (~supported) | high_excluded
        active_low = np.where(use_smoothed_low, out[smoothed_lower_col], out[raw_lower_col])
        active_high = np.where(use_smoothed_high, out[smoothed_upper_col], out[raw_upper_col])
        low_source = np.where(use_smoothed_low, "smoothed", "raw")
        high_source = np.where(use_smoothed_high, "smoothed", "raw")

    out["threshold_low"] = active_low
    out["threshold_high"] = active_high
    out["threshold_source_low"] = low_source
    out["threshold_source_high"] = high_source
    out["threshold_call_mode"] = call_mode

    if upper_bound_only:
        outside = out[METRIC_NAME] > out["threshold_high"]
        valid_call = ~pd.isna(out["threshold_high"])
    else:
        outside = (out[METRIC_NAME] < out["threshold_low"]) | (out[METRIC_NAME] > out["threshold_high"])
        valid_call = ~(pd.isna(out["threshold_low"]) | pd.isna(out["threshold_high"]))

    out["penetrant"] = np.where(valid_call, np.where(outside, 1.0, 0.0), np.nan)
    return out
