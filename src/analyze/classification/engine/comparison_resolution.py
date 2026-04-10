"""Comparison resolution: type definitions, mode detection, and expansion.

Pure functions that turn user-facing comparison specs into a canonical
list of ``ResolvedComparison`` objects.  No side effects, no data access
beyond ``available_labels``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, TypedDict, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ClassLabel = str
PooledGroup = tuple[str, ...]  # >= 2 elements, sorted+deduped
ComparisonGroup = Union[ClassLabel, PooledGroup]
UserComparisonSpec = Union[
    ClassLabel,
    PooledGroup,
    list[Union[ClassLabel, PooledGroup]],
]


class ComparisonRow(TypedDict):
    positive: ComparisonGroup
    negative: ComparisonGroup


ComparisonScheme = Union[
    Literal["all_vs_rest", "all_pairs"],
    pd.DataFrame,
    list[ComparisonRow],
    None,
]

# ---------------------------------------------------------------------------
# ResolvedComparison
# ---------------------------------------------------------------------------

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_id(label: str) -> str:
    return _SAFE_RE.sub("_", label)


@dataclass(frozen=True)
class ResolvedComparison:
    """Canonical representation of one positive-vs-negative comparison."""

    positive_members: tuple[str, ...]  # sorted
    negative_members: tuple[str, ...]  # sorted
    positive_label: str
    negative_label: str
    comparison_id: str  # filesystem-safe

    @property
    def is_pooled_positive(self) -> bool:
        return len(self.positive_members) > 1

    @property
    def is_pooled_negative(self) -> bool:
        return len(self.negative_members) > 1

    @property
    def all_members(self) -> frozenset[str]:
        return frozenset(self.positive_members) | frozenset(self.negative_members)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_group_input(val: UserComparisonSpec | None, param_name: str) -> None:
    """Validate that *val* is a legal ``UserComparisonSpec``."""
    if val is None:
        return
    if isinstance(val, str):
        return
    if isinstance(val, tuple):
        if len(val) < 2:
            raise TypeError(
                f"{param_name}: pooled tuple must have >= 2 elements, got {val!r}"
            )
        for elem in val:
            if not isinstance(elem, str):
                raise TypeError(
                    f"{param_name}: pooled tuple elements must be str, got {type(elem).__name__}"
                )
        return
    if isinstance(val, list):
        for item in val:
            if isinstance(item, str):
                continue
            if isinstance(item, tuple):
                _validate_group_input(item, param_name)
            else:
                raise TypeError(
                    f"{param_name}: list elements must be str or tuple[str, ...], "
                    f"got {type(item).__name__}"
                )
        return
    raise TypeError(
        f"{param_name}: expected str, tuple[str, ...], or list, got {type(val).__name__}"
    )


def _as_group_list(val: UserComparisonSpec) -> list[ComparisonGroup]:
    """Wrap a scalar (str / tuple) in a list; pass lists through."""
    if isinstance(val, (str, tuple)):
        return [val]
    return list(val)


def _canonicalize_group(group: ComparisonGroup) -> ComparisonGroup:
    """Sort + dedupe pooled tuples; pass strings through."""
    if isinstance(group, str):
        return group
    deduped = tuple(sorted(set(group)))
    if len(deduped) < 2:
        raise ValueError(
            f"Pooled group collapsed to <2 elements after dedup: {group!r}"
        )
    return deduped


def _group_members(group: ComparisonGroup) -> set[str]:
    if isinstance(group, str):
        return {group}
    return set(group)


def _make_label(group: ComparisonGroup) -> str:
    if isinstance(group, str):
        return group
    return "+".join(group)


def _to_resolved(pos_group: ComparisonGroup, neg_group: ComparisonGroup) -> ResolvedComparison:
    pos_group = _canonicalize_group(pos_group)
    neg_group = _canonicalize_group(neg_group)
    pos_label = _make_label(pos_group)
    neg_label = _make_label(neg_group)
    pos_id = _sanitize_id(pos_label)
    neg_id = _sanitize_id(neg_label)
    cid = f"{pos_id}__vs__{neg_id}"
    pos_members = tuple(sorted(_group_members(pos_group)))
    neg_members = tuple(sorted(_group_members(neg_group)))
    return ResolvedComparison(
        positive_members=pos_members,
        negative_members=neg_members,
        positive_label=pos_label,
        negative_label=neg_label,
        comparison_id=cid,
    )


def _validate_design_table(df: pd.DataFrame) -> None:
    """Validate a manual-design DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    for col in ("positive", "negative"):
        if col not in df.columns:
            raise ValueError(f"Design table missing required column: {col!r}")
    extra = set(df.columns) - {"positive", "negative"}
    if extra:
        raise ValueError(f"Design table has unexpected columns: {extra}")
    for idx, row in df.iterrows():
        for col in ("positive", "negative"):
            val = row[col]
            if isinstance(val, str):
                continue
            if isinstance(val, tuple):
                if len(val) < 2 or not all(isinstance(e, str) for e in val):
                    raise ValueError(
                        f"Design table row {idx}, column {col!r}: "
                        f"tuple must have >=2 str elements, got {val!r}"
                    )
                continue
            raise TypeError(
                f"Design table row {idx}, column {col!r}: "
                f"expected str or tuple[str,...], got {type(val).__name__}"
            )


def _check_labels_exist(
    pairs: list[tuple[ComparisonGroup, ComparisonGroup]],
    available_labels: set[str],
    class_col: str,
) -> None:
    """Verify all referenced labels exist in the data."""
    referenced: set[str] = set()
    for pos, neg in pairs:
        referenced |= _group_members(pos)
        referenced |= _group_members(neg)
    missing = referenced - available_labels
    if missing:
        raise ValueError(
            f"Labels not found in {class_col!r}: {sorted(missing)}. "
            f"Available: {sorted(available_labels)}"
        )


def _check_overlap(pairs: list[tuple[ComparisonGroup, ComparisonGroup]]) -> None:
    """Ensure no label appears on both sides of a comparison."""
    for pos, neg in pairs:
        overlap = _group_members(pos) & _group_members(neg)
        if overlap:
            raise ValueError(
                f"Label(s) {sorted(overlap)} appear on both positive and negative "
                f"sides of comparison: {_make_label(pos)} vs {_make_label(neg)}"
            )


def _dedup_pairs(
    pairs: list[tuple[ComparisonGroup, ComparisonGroup]],
) -> list[tuple[ComparisonGroup, ComparisonGroup]]:
    """Remove duplicate pairs, preserving insertion order."""
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    result: list[tuple[ComparisonGroup, ComparisonGroup]] = []
    for pos, neg in pairs:
        pos_c = _canonicalize_group(pos)
        neg_c = _canonicalize_group(neg)
        pos_key = (pos_c,) if isinstance(pos_c, str) else pos_c
        neg_key = (neg_c,) if isinstance(neg_c, str) else neg_c
        key = (pos_key, neg_key)
        if key not in seen:
            seen.add(key)
            result.append((pos_c, neg_c))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_comparisons(
    positive: UserComparisonSpec | None,
    negative: UserComparisonSpec | None,
    comparisons: ComparisonScheme,
    available_labels: set[str],
    class_col: str,
) -> list[ResolvedComparison]:
    """Resolve user-facing comparison spec into canonical ``ResolvedComparison`` list.

    Pure function — no data access beyond *available_labels*.
    """
    # -- Step 1: Validate input types -----------------------------------------
    _validate_group_input(positive, "positive")
    _validate_group_input(negative, "negative")

    # -- Mutual-exclusion checks ---------------------------------------------
    is_design = isinstance(comparisons, (pd.DataFrame, list)) and comparisons is not None
    is_scheme = isinstance(comparisons, str)

    if is_design:
        if positive is not None or negative is not None:
            raise ValueError(
                "Cannot combine comparisons=DataFrame/list[dict] with "
                "positive or negative arguments."
            )
    if is_scheme:
        if negative is not None:
            raise ValueError(
                f"Cannot combine comparisons={comparisons!r} with negative argument."
            )
        if positive is not None and isinstance(positive, (str, tuple)):
            raise ValueError(
                f"Cannot combine comparisons={comparisons!r} with scalar positive. "
                "Use a list to scope: positive=['A', 'B']."
            )
    if negative is not None and positive is None and comparisons is not None:
        raise ValueError("Cannot set negative without positive when comparisons is specified.")

    # -- Step 2: Mode detection and expansion --------------------------------
    pairs: list[tuple[ComparisonGroup, ComparisonGroup]]

    if isinstance(comparisons, list) and comparisons is not None:
        # list[dict] → DataFrame
        df_design = pd.DataFrame(comparisons)
        _validate_design_table(df_design)
        pairs = [
            (row["positive"], row["negative"])
            for _, row in df_design.iterrows()
        ]

    elif isinstance(comparisons, pd.DataFrame):
        _validate_design_table(comparisons)
        pairs = [
            (row["positive"], row["negative"])
            for _, row in comparisons.iterrows()
        ]

    elif comparisons == "all_pairs":
        if positive is not None:
            scope = _as_group_list(positive)
            for g in scope:
                if isinstance(g, tuple):
                    raise ValueError(
                        "all_pairs mode does not support pooled tuples in scope. "
                        f"Got: {g!r}"
                    )
            scope_labels = sorted(str(g) for g in scope)
        else:
            scope_labels = sorted(available_labels)
        pairs = [(a, b) for a, b in combinations(scope_labels, 2)]

    elif comparisons == "all_vs_rest" or comparisons is None:
        if positive is None and negative is not None:
            # Negative-only convenience mode: all remaining labels vs supplied
            # negative group(s). Lists enumerate negatives; tuples pool them.
            neg_list = _as_group_list(negative)
            pairs = []
            for neg in neg_list:
                neg_members = _group_members(neg)
                remaining = sorted(available_labels - neg_members)
                if not remaining:
                    raise ValueError(
                        f"No labels left for positives after removing negative group {neg_members}"
                    )
                for pos in remaining:
                    pairs.append((pos, neg))
        elif positive is not None and negative is not None:
            # Explicit Cartesian: positive × negative
            pos_list = _as_group_list(positive)
            neg_list = _as_group_list(negative)
            pairs = [(p, n) for p in pos_list for n in neg_list]
        elif positive is not None:
            # Scoped all-vs-rest
            scope = _as_group_list(positive)
            pairs = []
            for pos in scope:
                pos_mems = _group_members(pos)
                rest_labels = sorted(available_labels - pos_mems)
                if not rest_labels:
                    raise ValueError(
                        f"No labels left for 'rest' after removing {pos_mems}"
                    )
                neg: ComparisonGroup
                if len(rest_labels) == 1:
                    neg = rest_labels[0]
                else:
                    neg = tuple(rest_labels)
                pairs.append((pos, neg))
        else:
            # Default: every class vs all others
            pairs = []
            for label in sorted(available_labels):
                rest_labels = sorted(available_labels - {label})
                if not rest_labels:
                    continue
                neg = rest_labels[0] if len(rest_labels) == 1 else tuple(rest_labels)
                pairs.append((label, neg))
    else:
        raise ValueError(
            f"Invalid comparisons value: {comparisons!r}. "
            "Expected 'all_vs_rest', 'all_pairs', DataFrame, list[dict], or None."
        )

    # -- Step 3: Validation ---------------------------------------------------
    _check_labels_exist(pairs, available_labels, class_col)
    _check_overlap(pairs)

    # -- Step 4: Deduplication -----------------------------------------------
    pairs = _dedup_pairs(pairs)

    # -- Step 5: Convert to ResolvedComparison -------------------------------
    resolved = [_to_resolved(p, n) for p, n in pairs]

    # Collision detection: ensure comparison_ids are unique
    ids_seen: dict[str, int] = {}
    for rc in resolved:
        if rc.comparison_id in ids_seen:
            first = ids_seen[rc.comparison_id]
            raise ValueError(
                f"comparison_id collision: {rc.comparison_id!r} is produced by "
                f"both comparison #{first + 1} and #{resolved.index(rc) + 1}. "
                f"Labels that differ only in special characters will collide."
            )
        ids_seen[rc.comparison_id] = resolved.index(rc)

    return resolved


def check_min_samples(
    resolved: list[ResolvedComparison],
    label_counts: dict[str, int],
    min_samples_per_group: int,
    min_samples_per_member: int,
) -> None:
    """Validate sample counts against group-level and per-member thresholds.

    Parameters
    ----------
    resolved
        Output of ``resolve_comparisons``.
    label_counts
        Mapping ``class_label -> n_unique_units`` from the data.
    min_samples_per_group
        Total unique units required across all members of a side.
    min_samples_per_member
        Each individual member in a pooled group must have at least this many.
    """
    for rc in resolved:
        for side_name, members in [
            ("positive", rc.positive_members),
            ("negative", rc.negative_members),
        ]:
            group_total = sum(label_counts.get(m, 0) for m in members)
            if group_total < min_samples_per_group:
                raise ValueError(
                    f"Comparison {rc.comparison_id!r}, {side_name} side: "
                    f"group total {group_total} < min_samples_per_group={min_samples_per_group}. "
                    f"Members: {members}"
                )
            if len(members) > 1:
                for m in members:
                    count = label_counts.get(m, 0)
                    if count < min_samples_per_member:
                        raise ValueError(
                            f"Comparison {rc.comparison_id!r}, {side_name} side: "
                            f"member {m!r} has {count} units < "
                            f"min_samples_per_member={min_samples_per_member}."
                        )
