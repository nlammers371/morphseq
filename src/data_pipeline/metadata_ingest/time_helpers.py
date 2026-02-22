"""Shared helpers for frame/time aliasing and time-unit derivation."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def ensure_frame_time_alias(
    df: pd.DataFrame,
    *,
    frame_col: str = "frame_index",
    time_col: str = "time_int",
    stage_name: str = "table",
    require_match_when_both_present: bool = True,
) -> pd.DataFrame:
    """Ensure frame/time columns exist, are integer-valued, and agree."""
    out = df.copy()
    has_frame = frame_col in out.columns
    has_time = time_col in out.columns

    if not has_frame and not has_time:
        raise ValueError(f"Expected {frame_col} or {time_col} in {stage_name}")
    if not has_frame:
        out[frame_col] = out[time_col]
    if not has_time:
        out[time_col] = out[frame_col]

    frame_vals = pd.to_numeric(out[frame_col], errors="coerce")
    time_vals = pd.to_numeric(out[time_col], errors="coerce")
    if frame_vals.isna().any():
        raise ValueError(f"Column '{frame_col}' contains non-numeric values in {stage_name}")
    if time_vals.isna().any():
        raise ValueError(f"Column '{time_col}' contains non-numeric values in {stage_name}")
    if (frame_vals % 1 != 0).any():
        raise ValueError(f"Column '{frame_col}' must contain integer values in {stage_name}")
    if (time_vals % 1 != 0).any():
        raise ValueError(f"Column '{time_col}' must contain integer values in {stage_name}")

    frame_int = frame_vals.astype(int)
    time_int = time_vals.astype(int)
    if require_match_when_both_present and (frame_int != time_int).any():
        mismatch = out.loc[
            frame_int != time_int,
            [c for c in ["experiment_id", "well_id", "channel_id", frame_col, time_col] if c in out.columns],
        ]
        raise ValueError(
            f"Detected rows where {frame_col} != {time_col} in {stage_name}: "
            f"{mismatch.head(10).to_dict(orient='records')}"
        )

    out[frame_col] = frame_int
    out[time_col] = time_int
    return out


def add_elapsed_time_columns(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    frame_col: str = "frame_index",
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

    out = ensure_frame_time_alias(
        out,
        frame_col=frame_col,
        time_col="time_int",
        stage_name="elapsed_time_inputs",
        require_match_when_both_present=False,
    )

    out[out_seconds] = pd.NA
    use_cols = list(group_cols)
    order_cols = use_cols + [frame_col]
    out = out.sort_values(order_cols).copy()

    frame_numeric = pd.to_numeric(out[frame_col], errors="coerce")
    frame_origin = out.groupby(use_cols)[frame_col].transform("min")
    elapsed_from_frame = (frame_numeric - frame_origin).astype(float)

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

