"""Internal helpers shared across emergence modules."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def get_onset(onset_matrix: pd.DataFrame, a: str, b: str) -> float:
    """Return the onset for pair (a, b), or NaN if not found."""

    try:
        v = onset_matrix.loc[a, b]
    except KeyError:
        return float("nan")
    return float("nan") if pd.isna(v) else float(v)


def symmetric_onset(onset_matrix: pd.DataFrame, a: str, b: str) -> float:
    """Return finite onset for (a, b) or (b, a), preferring a→b."""

    v = get_onset(onset_matrix, a, b)
    if math.isfinite(v):
        return v
    return get_onset(onset_matrix, b, a)


def nanmedian(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.median(finite))


def nanmin(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else float("nan")


def nanmax(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else float("nan")
