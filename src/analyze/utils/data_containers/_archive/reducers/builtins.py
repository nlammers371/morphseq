from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...specs import InputRef, ReducerSpec
from .registry import register_reducer


def _target_series(resolved: dict[str, Any]) -> pd.Series:
    target = resolved["target"]
    if isinstance(target, pd.Series):
        return target
    return pd.Series(target)


def _mean_equal_bin(group_df: pd.DataFrame, resolved: dict[str, Any]) -> dict[str, Any]:
    series = _target_series(resolved)
    return {"value": float(series.astype(float).mean(skipna=True))}


def _max(group_df: pd.DataFrame, resolved: dict[str, Any]) -> dict[str, Any]:
    series = _target_series(resolved)
    return {"value": float(series.astype(float).max(skipna=True))}


def _top2(group_df: pd.DataFrame, resolved: dict[str, Any]) -> dict[str, Any]:
    series = _target_series(resolved).astype(float).dropna().sort_values(ascending=False)
    if series.empty:
        return {"value": float("nan")}
    return {"value": float(series.iloc[:2].mean())}


def _auc(group_df: pd.DataFrame, resolved: dict[str, Any]) -> dict[str, Any]:
    series = _target_series(resolved).astype(float)
    times = pd.Series(resolved["bin_center_time"]).astype(float)
    valid = series.notna() & times.notna()
    if valid.sum() < 2:
        return {"value": float("nan")}
    series = series.loc[valid].sort_index()
    times = times.loc[valid].sort_index()
    order = np.argsort(times.to_numpy())
    x = times.to_numpy()[order]
    y = series.to_numpy()[order]
    return {"value": float(np.trapz(y, x=x))}


def _mean_frame_weighted(group_df: pd.DataFrame, resolved: dict[str, Any]) -> dict[str, Any]:
    series = _target_series(resolved).astype(float)
    weights = pd.Series(resolved["n_frames"]).astype(float)
    valid = series.notna() & weights.notna()
    if valid.sum() == 0:
        return {"value": float("nan")}
    series = series.loc[valid]
    weights = weights.loc[valid]
    return {"value": float(np.average(series.to_numpy(), weights=weights.to_numpy()))}


def _mean_time_weighted(group_df: pd.DataFrame, resolved: dict[str, Any]) -> dict[str, Any]:
    series = _target_series(resolved).astype(float)
    weights = pd.Series(resolved["bin_width_seconds"]).astype(float)
    valid = series.notna() & weights.notna()
    if valid.sum() == 0:
        return {"value": float("nan")}
    series = series.loc[valid]
    weights = weights.loc[valid]
    return {"value": float(np.average(series.to_numpy(), weights=weights.to_numpy()))}


def _make_builtin(name: str, consumes: list[InputRef], math_min_bins: int, func) -> ReducerSpec:
    return ReducerSpec(
        name=name,
        consumes=tuple(consumes),
        output_schema=("value",),
        math_min_bins=math_min_bins,
        func=func,
        notes=f"Built-in reducer {name}",
    )


def ensure_builtin_reducers_registered() -> None:
    register_reducer(_make_builtin("mean_equal_bin", [InputRef("binned", "target")], 1, _mean_equal_bin), overwrite=True)
    register_reducer(_make_builtin("max", [InputRef("binned", "target")], 1, _max), overwrite=True)
    register_reducer(_make_builtin("top2", [InputRef("binned", "target")], 2, _top2), overwrite=True)
    register_reducer(
        _make_builtin(
            "auc",
            [InputRef("binned", "target"), InputRef("bin_meta", "bin_center_time")],
            2,
            _auc,
        ),
        overwrite=True,
    )
    register_reducer(
        _make_builtin(
            "mean_frame_weighted",
            [InputRef("binned", "target"), InputRef("bin_meta", "n_frames")],
            1,
            _mean_frame_weighted,
        ),
        overwrite=True,
    )
    register_reducer(
        _make_builtin(
            "mean_time_weighted",
            [InputRef("binned", "target"), InputRef("bin_meta", "bin_width_seconds")],
            1,
            _mean_time_weighted,
        ),
        overwrite=True,
    )
