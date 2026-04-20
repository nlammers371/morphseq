from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..specs import InputRef, ReducerSpec
from .registry import get_reducer, register_reducer


def _merge_consumes(base_consumes: tuple[InputRef, ...], extra: tuple[InputRef, ...]) -> tuple[InputRef, ...]:
    out: list[InputRef] = []
    seen: set[tuple[str, str]] = set()
    for ref in (*base_consumes, *extra):
        key = (ref.level, ref.key)
        if key not in seen:
            out.append(ref)
            seen.add(key)
    return tuple(out)


def make_centered_reducer(
    *,
    name: str,
    base_reducer: str | ReducerSpec = "mean_equal_bin",
    baseline_value: float,
    math_min_bins: int | None = None,
    register: bool = True,
) -> ReducerSpec:
    """Create reducer that subtracts a fixed baseline from the base reducer output."""
    base = get_reducer(base_reducer)
    min_bins = base.math_min_bins if math_min_bins is None else math_min_bins

    def _func(group_df: pd.DataFrame, resolved: dict[str, Any]) -> dict[str, Any]:
        if base.func is None:
            raise ValueError(f"Base reducer {base.name!r} has no callable implementation")
        out = base.func(group_df, resolved)
        return {"value": float(out["value"]) - float(baseline_value)}

    reducer = ReducerSpec(
        name=name,
        consumes=base.consumes,
        output_schema=("value",),
        math_min_bins=min_bins,
        func=_func,
        notes=f"Centered reducer from {base.name} with baseline={baseline_value}",
    )
    return register_reducer(reducer, overwrite=True) if register else reducer


def make_group_centered_reducer(
    *,
    name: str,
    group_key: str,
    baseline_by_group: dict[str, float],
    base_reducer: str | ReducerSpec = "mean_equal_bin",
    fallback_baseline: float | None = None,
    math_min_bins: int | None = None,
    register: bool = True,
) -> ReducerSpec:
    """Create reducer that subtracts group-specific baseline (e.g., per genotype centering)."""
    base = get_reducer(base_reducer)
    min_bins = base.math_min_bins if math_min_bins is None else math_min_bins
    consumes = _merge_consumes(base.consumes, (InputRef("embryo_meta", group_key),))

    def _func(group_df: pd.DataFrame, resolved: dict[str, Any]) -> dict[str, Any]:
        if base.func is None:
            raise ValueError(f"Base reducer {base.name!r} has no callable implementation")
        out = base.func(group_df, resolved)
        group_series = pd.Series(resolved[group_key]).astype(str)
        group_value = str(group_series.iloc[0])
        if group_value in baseline_by_group:
            baseline = baseline_by_group[group_value]
        elif fallback_baseline is not None:
            baseline = fallback_baseline
        else:
            raise KeyError(f"No baseline defined for group {group_value!r}")
        return {"value": float(out["value"]) - float(baseline)}

    reducer = ReducerSpec(
        name=name,
        consumes=consumes,
        output_schema=("value",),
        math_min_bins=min_bins,
        func=_func,
        notes=f"Group-centered reducer from {base.name} using embryo_meta:{group_key}",
    )
    return register_reducer(reducer, overwrite=True) if register else reducer


def make_group_difference_reducer(
    *,
    name: str,
    group_key: str,
    reference_group: str,
    mean_by_group: dict[str, float],
    base_reducer: str | ReducerSpec = "mean_equal_bin",
    math_min_bins: int | None = None,
    register: bool = True,
) -> ReducerSpec:
    """Create reducer that returns difference to a reference group mean (e.g., group - WT)."""
    if reference_group not in mean_by_group:
        raise KeyError(f"reference_group {reference_group!r} is missing from mean_by_group")
    reference_mean = float(mean_by_group[reference_group])
    reducer = make_group_centered_reducer(
        name=name,
        group_key=group_key,
        baseline_by_group={k: reference_mean for k in mean_by_group},
        base_reducer=base_reducer,
        fallback_baseline=reference_mean,
        math_min_bins=math_min_bins,
        register=False,
    )
    reducer.notes = (
        f"Group-difference reducer from {get_reducer(base_reducer).name}: "
        f"value - mean({reference_group})"
    )
    return register_reducer(reducer, overwrite=True) if register else reducer
