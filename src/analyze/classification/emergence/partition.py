"""Recursive within-block partitioning for emergence trees."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import pandas as pd

from .types import ResolutionNode
from .utils import nanmedian, symmetric_onset


def _all_bipartitions(members: list[str]) -> list[tuple[list[str], list[str]]]:
    """Enumerate all non-trivial bipartitions of members."""

    seen: set[frozenset[str]] = set()
    result = []
    for size in range(1, len(members)):
        for combo in combinations(members, size):
            key = frozenset(combo)
            if key in seen:
                continue
            complement_key = frozenset(members) - key
            seen.add(key)
            seen.add(complement_key)
            b1 = list(combo)
            b2 = [m for m in members if m not in key]
            result.append((b1, b2))
    return result


def _score_partition(
    b1: list[str],
    b2: list[str],
    onset_matrix: pd.DataFrame,
    min_cross_support: float,
) -> tuple[bool, float, float, int]:
    """Score a bipartition for use as a split."""

    cross_onsets = []
    for a in b1:
        for b in b2:
            cross_onsets.append(symmetric_onset(onset_matrix, a, b))

    n_total_cross = len(cross_onsets)
    finite_cross = [v for v in cross_onsets if math.isfinite(v)]
    n_finite_cross = len(finite_cross)

    if n_total_cross == 0:
        return False, float("nan"), 0.0, 0

    cross_support = n_finite_cross / n_total_cross
    cross_median = nanmedian(finite_cross)

    if cross_support < min_cross_support:
        return False, cross_median, cross_support, 0
    if not math.isfinite(cross_median) or cross_median <= 0:
        return False, cross_median, cross_support, 0

    internal_finite = 0
    for a, b in combinations(b1, 2):
        if math.isfinite(symmetric_onset(onset_matrix, a, b)):
            internal_finite += 1
    for a, b in combinations(b2, 2):
        if math.isfinite(symmetric_onset(onset_matrix, a, b)):
            internal_finite += 1

    return True, cross_median, cross_support, internal_finite


def _is_monotone_feasible(
    members: list[str],
    onset_matrix: pd.DataFrame,
    min_cross_support: float,
    floor: float,
    _cache: dict | None = None,
) -> bool:
    """Return True if members can be recursively resolved with split times >= floor."""

    if _cache is None:
        _cache = {}
    if len(members) <= 2:
        return True

    key = (frozenset(members), round(floor, 1))
    if key in _cache:
        return _cache[key]

    result = False
    for b1, b2 in _all_bipartitions(members):
        accepted, cross_median, _, _ = _score_partition(b1, b2, onset_matrix, min_cross_support)
        if not accepted:
            continue
        if cross_median < floor:
            continue
        if (
            _is_monotone_feasible(b1, onset_matrix, min_cross_support, cross_median, _cache)
            and _is_monotone_feasible(b2, onset_matrix, min_cross_support, cross_median, _cache)
        ):
            result = True
            break

    _cache[key] = result
    return result


def _find_best_split(
    members: list[str],
    onset_matrix: pd.DataFrame,
    min_cross_support: float,
    floor: float = 0.0,
    _cache: dict | None = None,
) -> tuple[list[str], list[str], float] | None:
    """Find the best monotone-feasible bipartition of members."""

    if _cache is None:
        _cache = {}

    best: tuple[list[str], list[str], float] | None = None
    best_score: tuple[int, float, float] = (int(-1e9), -1.0, -1.0)

    for b1, b2 in _all_bipartitions(members):
        accepted, cross_median, cross_support, internal_finite = _score_partition(
            b1, b2, onset_matrix, min_cross_support
        )
        if not accepted:
            continue
        if cross_median < floor:
            continue
        if not (
            _is_monotone_feasible(b1, onset_matrix, min_cross_support, cross_median, _cache)
            and _is_monotone_feasible(b2, onset_matrix, min_cross_support, cross_median, _cache)
        ):
            continue

        score = (-internal_finite, cross_support, cross_median)
        if (
            best is None
            or score[0] > best_score[0]
            or (score[0] == best_score[0] and score[1] > best_score[1])
            or (score[0] == best_score[0] and score[1] == best_score[1] and score[2] > best_score[2])
        ):
            best = (b1, b2, cross_median)
            best_score = score

    return best


def resolve_block(
    members: Sequence[str],
    onset_matrix: pd.DataFrame,
    *,
    min_cross_support: float = 0.5,
    _floor: float = 0.0,
) -> ResolutionNode:
    """Recursively resolve a block of co-emerging classes."""

    mlist = list(members)

    if len(mlist) == 0:
        return ResolutionNode(members=[], split_time=None, children=[], unresolved=False)
    if len(mlist) == 1:
        return ResolutionNode(members=mlist, split_time=None, children=[], unresolved=False)

    cache: dict = {}
    best = _find_best_split(mlist, onset_matrix, min_cross_support, floor=_floor, _cache=cache)
    if best is None:
        return ResolutionNode(members=mlist, split_time=None, children=[], unresolved=True)

    b1, b2, split_time = best
    child1 = resolve_block(b1, onset_matrix, min_cross_support=min_cross_support, _floor=split_time)
    child2 = resolve_block(b2, onset_matrix, min_cross_support=min_cross_support, _floor=split_time)
    return ResolutionNode(members=mlist, split_time=split_time, children=[child1, child2], unresolved=False)
