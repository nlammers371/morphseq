"""Shared helpers for time-unit derivation."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def ensure_time_int_column(
    df: pd.DataFrame,
    *,
    time_col: str = "time_int",
    stage_name: str = "table",
) -> pd.DataFrame:
    """Ensure time_int exists and is integer-valued."""
    out = df.copy()
    has_time = time_col in out.columns

    if not has_time:
        raise ValueError(f"Expected '{time_col}' in {stage_name}")

    time_vals = pd.to_numeric(out[time_col], errors="coerce")
    if time_vals.isna().any():
        raise ValueError(f"Column '{time_col}' contains non-numeric values in {stage_name}")
    if (time_vals % 1 != 0).any():
        raise ValueError(f"Column '{time_col}' must contain integer values in {stage_name}")

    time_int = time_vals.astype(int)

    out[time_col] = time_int
    return out


def add_elapsed_time_columns(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    experiment_time_col: str = "experiment_time_s",
    frame_interval_col: str = "frame_interval_s",
    out_seconds: str = "elapsed_time_s",
    out_minutes: str = "elapsed_time_min",
    out_hours: str = "elapsed_time_hr",
) -> pd.DataFrame:
    """Add elapsed-time columns using experiment_time_s first, then interval fallback."""
    out = df.copy()
    missing_group = [col for col in group_cols if col not in out.columns]
    if missing_group:
        raise ValueError(f"Missing group columns for elapsed-time computation: {missing_group}")

    out = ensure_time_int_column(
        out,
        stage_name="elapsed_time_inputs",
    )

    out[out_seconds] = pd.NA
    use_cols = list(group_cols)
    order_cols = use_cols + ["time_int"]
    out = out.sort_values(order_cols).copy()

    time_numeric = pd.to_numeric(out["time_int"], errors="coerce")
    time_origin = out.groupby(use_cols)["time_int"].transform("min")
    elapsed_from_frame = (time_numeric - time_origin).astype(float)

    if frame_interval_col in out.columns:
        interval = pd.to_numeric(out[frame_interval_col], errors="coerce")
        elapsed_from_interval = elapsed_from_frame * interval
    else:
        elapsed_from_interval = elapsed_from_frame

    if experiment_time_col in out.columns:
        experiment_time = pd.to_numeric(out[experiment_time_col], errors="coerce")
        group_min = out.groupby(use_cols)[experiment_time_col].transform("min")
        group_min = pd.to_numeric(group_min, errors="coerce")
        elapsed_from_experiment = experiment_time - group_min
        out[out_seconds] = elapsed_from_experiment.where(elapsed_from_experiment.notna(), elapsed_from_interval)
    else:
        out[out_seconds] = elapsed_from_interval

    out[out_seconds] = pd.to_numeric(out[out_seconds], errors="coerce").fillna(elapsed_from_interval)
    out[out_minutes] = out[out_seconds] / 60.0
    out[out_hours] = out[out_seconds] / 3600.0
    return out


def add_frame_interval_unit_columns(
    df: pd.DataFrame,
    *,
    interval_col: str = "frame_interval_s",
    out_minutes: str = "frame_interval_min",
    out_hours: str = "frame_interval_hr",
) -> pd.DataFrame:
    """Ensure frame interval seconds exists as numeric and derive min/hr columns."""
    out = df.copy()
    if interval_col not in out.columns:
        out[interval_col] = pd.NA

    seconds = pd.to_numeric(out[interval_col], errors="coerce")
    out[interval_col] = seconds
    out[out_minutes] = seconds / 60.0
    out[out_hours] = seconds / 3600.0
    return out


def add_experiment_time_cols(
    df: pd.DataFrame,
    *,
    experiment_time_col: str = "experiment_time_s",
    out_minutes: str = "experiment_time_min",
    out_hours: str = "experiment_time_hr",
) -> pd.DataFrame:
    """Ensure experiment time in seconds exists as numeric and derive min/hr columns."""
    out = df.copy()
    if experiment_time_col not in out.columns:
        out[experiment_time_col] = pd.NA

    seconds = pd.to_numeric(out[experiment_time_col], errors="coerce")
    out[experiment_time_col] = seconds
    out[out_minutes] = seconds / 60.0
    out[out_hours] = seconds / 3600.0
    return out
