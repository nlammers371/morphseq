"""
emergence.py
------------
Reference-rooted phenotype emergence timeline.

Two-layer model:
  1. Emergence from reference: when does each non-reference class first
     become distinguishable from the reference set?
  2. Within-block resolution: for classes that emerge at the same time,
     when do they become distinguishable from *each other*?

Public API
----------
    validate_reference(onset_matrix, reference) -> ReferenceValidation
    compute_emergence_scores(onset_matrix, reference) -> list[EmergenceScore]
    form_emergence_blocks(scores, *, bin_width) -> list[EmergenceBlock]
    resolve_block(members, onset_matrix, *, min_cross_support) -> ResolutionNode
    build_emergence_timeline(onset_matrix, reference, *, bin_width, min_cross_support)
        -> EmergenceTimeline
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReferenceValidation:
    """Result of Step 1: reference coherence check."""
    reference: list[str]
    status: Literal["valid", "ambiguous", "invalid"]
    coherence_score: float          # n_nan_pairs / n_total_pairs; 1.0 for singletons
    offending_pairs: list[tuple[str, str, float]]  # (a, b, onset_hpf) for finite internal pairs
    n_internal_pairs: int


@dataclass
class EmergenceScore:
    """Per-class emergence timing relative to the reference set (Step 2)."""
    class_name: str
    emergence_time: float           # median raw onset to any ref member (NaN if none resolved)
    emergence_min: float            # min raw onset to any ref member
    emergence_max: float            # max raw onset to any ref member
    n_resolved_refs: int            # number of ref members with finite onset
    n_total_refs: int
    per_ref_onsets: dict[str, float | None]


@dataclass
class EmergenceBlock:
    """A group of non-reference classes that emerge at the same time bin (Step 3)."""
    block_id: int                   # stable, assigned in order of emergence (NaN block last)
    members: list[str]
    bin_key: float                  # floor(median_emergence / bin_width) * bin_width — grouping only
    emergence_time: float           # median of raw member emergence_times (for display)
    emergence_min: float            # min raw member emergence_time
    emergence_max: float            # max raw member emergence_time


@dataclass
class ResolutionNode:
    """Recursive tree node for within-block resolution (Step 4).

    Leaf:     len(children) == 0
    Internal: len(children) == 2, split_time is set
    Composite leaf (unresolved): unresolved=True, len(children) == 0
    """
    members: list[str]
    split_time: float | None        # median cross-partition onset; None for leaves
    children: list[ResolutionNode]  # empty for leaves, exactly 2 for internal nodes
    unresolved: bool                # True if multi-member but no coherent split found


@dataclass
class EmergenceTimeline:
    """Full emergence timeline for a set of classes given a reference (Step 5)."""
    reference_validation: ReferenceValidation
    scores: list[EmergenceScore]               # sorted by emergence_time (NaN last)
    blocks: list[EmergenceBlock]               # sorted by emergence_time (NaN block last)
    block_resolutions: dict[int, ResolutionNode]  # block_id → resolution tree
    all_classes: list[str]
    reference: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_onset(onset_matrix: pd.DataFrame, a: str, b: str) -> float:
    """Return the onset for pair (a, b), or NaN if not found."""
    try:
        v = onset_matrix.loc[a, b]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return float("nan")
        return float(v)
    except KeyError:
        return float("nan")


def _symmetric_onset(onset_matrix: pd.DataFrame, a: str, b: str) -> float:
    """Return finite onset for (a,b) or (b,a), preferring a→b."""
    v = _get_onset(onset_matrix, a, b)
    if math.isfinite(v):
        return v
    return _get_onset(onset_matrix, b, a)


def _nanmedian(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.median(finite))


def _nanmin(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else float("nan")


def _nanmax(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else float("nan")


# ---------------------------------------------------------------------------
# Step 1: Reference validation
# ---------------------------------------------------------------------------

def validate_reference(
    onset_matrix: pd.DataFrame,
    reference: Sequence[str],
) -> ReferenceValidation:
    """Check internal coherence of the reference set.

    Parameters
    ----------
    onset_matrix : symmetric DataFrame (index = columns = class names)
        Entry [i, j] = onset hpf when i becomes distinguishable from j, or NaN.
    reference : sequence of class names forming the reference set

    Returns
    -------
    ReferenceValidation with status in {"valid", "ambiguous", "invalid"}
    """
    ref = list(reference)
    pairs = list(combinations(ref, 2))
    n_total = len(pairs)

    if n_total == 0:
        return ReferenceValidation(
            reference=ref,
            status="valid",
            coherence_score=1.0,
            offending_pairs=[],
            n_internal_pairs=0,
        )

    offending: list[tuple[str, str, float]] = []
    for a, b in pairs:
        v = _symmetric_onset(onset_matrix, a, b)
        if math.isfinite(v):
            offending.append((a, b, v))

    n_nan = n_total - len(offending)
    coherence = n_nan / n_total

    if len(offending) == 0:
        status: Literal["valid", "ambiguous", "invalid"] = "valid"
    elif coherence >= 0.5:
        status = "ambiguous"
    else:
        status = "invalid"

    return ReferenceValidation(
        reference=ref,
        status=status,
        coherence_score=coherence,
        offending_pairs=offending,
        n_internal_pairs=n_total,
    )


# ---------------------------------------------------------------------------
# Step 2: Emergence scoring
# ---------------------------------------------------------------------------

def compute_emergence_scores(
    onset_matrix: pd.DataFrame,
    reference: Sequence[str],
) -> list[EmergenceScore]:
    """Score each non-reference class by its emergence time relative to R.

    emergence_time = median{ onset(c, r) : r ∈ R, finite }

    Returns list sorted by emergence_time (NaN last).
    """
    ref = list(reference)
    ref_set = set(ref)
    all_classes = list(onset_matrix.index)
    non_ref = [c for c in all_classes if c not in ref_set]

    scores: list[EmergenceScore] = []
    for c in non_ref:
        per_ref: dict[str, float | None] = {}
        for r in ref:
            v = _symmetric_onset(onset_matrix, c, r)
            per_ref[r] = v if math.isfinite(v) else None

        finite_onsets = [v for v in per_ref.values() if v is not None]
        n_resolved = len(finite_onsets)

        scores.append(EmergenceScore(
            class_name=c,
            emergence_time=_nanmedian(finite_onsets) if finite_onsets else float("nan"),
            emergence_min=_nanmin(finite_onsets) if finite_onsets else float("nan"),
            emergence_max=_nanmax(finite_onsets) if finite_onsets else float("nan"),
            n_resolved_refs=n_resolved,
            n_total_refs=len(ref),
            per_ref_onsets=per_ref,
        ))

    scores.sort(key=lambda s: (math.isnan(s.emergence_time), s.emergence_time))
    return scores


# ---------------------------------------------------------------------------
# Step 3: Block formation
# ---------------------------------------------------------------------------

def form_emergence_blocks(
    scores: Sequence[EmergenceScore],
    *,
    bin_width: float = 4.0,
) -> list[EmergenceBlock]:
    """Group emergence scores into time bins.

    Grouping key: floor(emergence_time / bin_width) * bin_width
    Display time: median of raw emergence_times of block members (NOT floored).
    Classes with NaN emergence → single "unresolved" block at the end.

    Returns list sorted by emergence_time (NaN block last).
    """
    finite_scores = [s for s in scores if math.isfinite(s.emergence_time)]
    nan_scores = [s for s in scores if not math.isfinite(s.emergence_time)]

    # Group finite scores by bin_key
    bin_groups: dict[float, list[EmergenceScore]] = {}
    for s in finite_scores:
        bin_key = math.floor(s.emergence_time / bin_width) * bin_width
        bin_groups.setdefault(bin_key, []).append(s)

    blocks: list[EmergenceBlock] = []
    block_id = 0

    for bin_key in sorted(bin_groups.keys()):
        group = bin_groups[bin_key]
        raw_times = [s.emergence_time for s in group]
        blocks.append(EmergenceBlock(
            block_id=block_id,
            members=[s.class_name for s in group],
            bin_key=bin_key,
            emergence_time=float(np.median(raw_times)),
            emergence_min=min(raw_times),
            emergence_max=max(raw_times),
        ))
        block_id += 1

    if nan_scores:
        blocks.append(EmergenceBlock(
            block_id=block_id,
            members=[s.class_name for s in nan_scores],
            bin_key=float("nan"),
            emergence_time=float("nan"),
            emergence_min=float("nan"),
            emergence_max=float("nan"),
        ))

    return blocks


# ---------------------------------------------------------------------------
# Step 4: Recursive block resolution
# ---------------------------------------------------------------------------

def _all_bipartitions(members: list[str]) -> list[tuple[list[str], list[str]]]:
    """Enumerate all non-trivial bipartitions of members.

    For n members, returns all (B1, B2) pairs where:
    - B1 is non-empty, B2 is non-empty
    - B1 ∪ B2 = members (no overlap)
    - To avoid duplicates: |B1| <= |B2| (so {a}|{b,c} and {b,c}|{a} counted once)
    - When |B1| == |B2|: first element of members is fixed in B1 (canonical form)
    """
    n = len(members)
    seen: set[frozenset[str]] = set()
    result = []
    for size in range(1, n):
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
    """Score a bipartition for use as a split.

    Returns
    -------
    (accepted, cross_median, cross_support, internal_finite_count)
    accepted: True if meets acceptance criteria
    cross_median: median of finite cross-partition onsets
    cross_support: n_finite_cross / n_total_cross
    internal_finite_count: total finite onsets within b1 + b2 (lower = more coherent children)
    """
    cross_onsets = []
    for a in b1:
        for b in b2:
            v = _symmetric_onset(onset_matrix, a, b)
            cross_onsets.append(v)

    n_total_cross = len(cross_onsets)
    finite_cross = [v for v in cross_onsets if math.isfinite(v)]
    n_finite_cross = len(finite_cross)

    if n_total_cross == 0:
        return False, float("nan"), 0.0, 0

    cross_support = n_finite_cross / n_total_cross
    cross_median = _nanmedian(finite_cross)

    # Acceptance criteria
    if cross_support < min_cross_support:
        return False, cross_median, cross_support, 0
    if not math.isfinite(cross_median) or cross_median <= 0:
        return False, cross_median, cross_support, 0

    # Count internal finite onsets (lower = children more coherent)
    internal_finite = 0
    for a, b in combinations(b1, 2):
        v = _symmetric_onset(onset_matrix, a, b)
        if math.isfinite(v):
            internal_finite += 1
    for a, b in combinations(b2, 2):
        v = _symmetric_onset(onset_matrix, a, b)
        if math.isfinite(v):
            internal_finite += 1

    return True, cross_median, cross_support, internal_finite


def _min_descendant_split(
    members: list[str],
    onset_matrix: pd.DataFrame,
    min_cross_support: float,
    floor: float,
) -> float:
    """Return the minimum split time in the recursively resolved subtree for members.

    Returns float('inf') if the subtree is a leaf or unresolved (no splits).
    Returns the actual minimum split time otherwise.
    Used to check monotone-feasibility before committing to a partition.
    """
    if len(members) <= 1:
        return float("inf")

    partitions = _all_bipartitions(members)
    for b1, b2 in partitions:
        accepted, cross_median, cross_support, _ = _score_partition(
            b1, b2, onset_matrix, min_cross_support
        )
        if not accepted:
            continue
        if cross_median < floor:
            continue
        # This child could split at cross_median; recurse to find min in its subtrees
        min_b1 = _min_descendant_split(b1, onset_matrix, min_cross_support, cross_median)
        min_b2 = _min_descendant_split(b2, onset_matrix, min_cross_support, cross_median)
        # The minimum split in this subtree is cross_median or anything lower in children
        return min(cross_median, min_b1, min_b2)

    return float("inf")  # no feasible split found → leaf/unresolved


def _is_monotone_feasible(
    members: list[str],
    onset_matrix: pd.DataFrame,
    min_cross_support: float,
    floor: float,
    _cache: dict | None = None,
) -> bool:
    """Return True if members can be recursively resolved with all split times >= floor.

    A group is monotone-feasible from floor if:
    - it is a singleton or unresolvable (trivially feasible), OR
    - there exists a partition with cross_median >= floor where both children
      are also monotone-feasible from cross_median.

    Results are memoized in _cache keyed by (frozenset(members), floor).
    """
    if _cache is None:
        _cache = {}

    if len(members) <= 1:
        return True

    key = (frozenset(members), round(floor, 1))
    if key in _cache:
        return _cache[key]

    partitions = _all_bipartitions(members)
    result = False
    for b1, b2 in partitions:
        accepted, cross_median, cross_support, _ = _score_partition(
            b1, b2, onset_matrix, min_cross_support
        )
        if not accepted:
            continue
        if cross_median < floor:
            continue
        if (_is_monotone_feasible(b1, onset_matrix, min_cross_support, cross_median, _cache)
                and _is_monotone_feasible(b2, onset_matrix, min_cross_support, cross_median, _cache)):
            result = True
            break

    # If no partition passed, the subtree becomes an unresolved leaf —
    # no split times at all, so monotonicity is trivially satisfied.
    if not result:
        result = True
    _cache[key] = result
    return result


def _find_best_split(
    members: list[str],
    onset_matrix: pd.DataFrame,
    min_cross_support: float,
    floor: float = 0.0,
    _cache: dict | None = None,
) -> tuple[list[str], list[str], float] | None:
    """Find the best monotone-feasible bipartition of members.

    Acceptance criteria:
    1. Support threshold met (cross_support >= min_cross_support)
    2. cross_median >= floor (parent's split time — ensures monotonicity)
    3. Both children are monotone-feasible from cross_median

    Scoring rule (among feasible candidates, in priority order):
    1. Primary: lower internal_finite_count (children more coherent — cleaner split)
    2. Tiebreak: higher cross_support (more evidence)
    3. Tertiary: higher cross_median (later split time = more informative)

    Returns (B1, B2, split_time) or None if no feasible partition found.
    """
    if _cache is None:
        _cache = {}

    partitions = _all_bipartitions(members)

    best: tuple[list[str], list[str], float] | None = None
    # Score: (-internal_finite, cross_support, cross_median) — all maximized
    best_score: tuple[int, float, float] = (int(-1e9), -1.0, -1.0)

    for b1, b2 in partitions:
        accepted, cross_median, cross_support, internal_finite = _score_partition(
            b1, b2, onset_matrix, min_cross_support
        )
        if not accepted:
            continue
        if cross_median < floor:
            continue
        # Monotonicity check: both children must be resolvable with all times >= cross_median
        if not (_is_monotone_feasible(b1, onset_matrix, min_cross_support, cross_median, _cache)
                and _is_monotone_feasible(b2, onset_matrix, min_cross_support, cross_median, _cache)):
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
    """Recursively resolve a block of co-emerging classes.

    Uses only within-block pairwise onsets (ignores reference).
    Only accepts partitions whose split_time >= _floor, ensuring the result
    tree is monotone (parent split times always <= child split times).

    Parameters
    ----------
    members : classes in the block
    onset_matrix : full onset matrix (only within-block pairs used)
    min_cross_support : minimum fraction of cross-partition pairs that must be finite
    _floor : internal — minimum acceptable split_time for this level

    Returns
    -------
    ResolutionNode (leaf, internal, or unresolved composite)
    """
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

    return ResolutionNode(
        members=mlist,
        split_time=split_time,
        children=[child1, child2],
        unresolved=False,
    )


# ---------------------------------------------------------------------------
# Step 5: Pipeline
# ---------------------------------------------------------------------------

def build_emergence_timeline(
    onset_matrix: pd.DataFrame,
    reference: Sequence[str],
    *,
    bin_width: float = 4.0,
    min_cross_support: float = 0.5,
) -> EmergenceTimeline:
    """Build a reference-rooted emergence timeline.

    Parameters
    ----------
    onset_matrix : symmetric DataFrame with class names as index and columns.
        Entry [i, j] = onset hpf when i becomes distinguishable from j, or NaN.
    reference : class names forming the reference/baseline set
    bin_width : width of grouping bins in hpf (default 4.0)
    min_cross_support : minimum cross-partition support for block resolution

    Returns
    -------
    EmergenceTimeline
    """
    ref = list(reference)
    ref_set = set(ref)
    all_classes = list(onset_matrix.index)

    # Step 1: validate reference
    ref_validation = validate_reference(onset_matrix, ref)

    # Steps 2-3: score and block (always proceed, regardless of reference status)
    scores = compute_emergence_scores(onset_matrix, ref)
    blocks = form_emergence_blocks(scores, bin_width=bin_width)

    # Step 4: resolve each multi-member block
    block_resolutions: dict[int, ResolutionNode] = {}
    for block in blocks:
        node = resolve_block(block.members, onset_matrix, min_cross_support=min_cross_support)
        block_resolutions[block.block_id] = node

    return EmergenceTimeline(
        reference_validation=ref_validation,
        scores=scores,
        blocks=blocks,
        block_resolutions=block_resolutions,
        all_classes=all_classes,
        reference=ref,
    )
