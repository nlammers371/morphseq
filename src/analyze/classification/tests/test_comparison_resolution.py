"""Tests for engine/comparison_resolution.py."""

import pandas as pd
import pytest

from analyze.classification.engine.comparison_resolution import (
    ResolvedComparison,
    check_min_samples,
    resolve_comparisons,
)

LABELS_3 = {"A", "B", "C"}
LABELS_4 = {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# resolve_comparisons — all_vs_rest
# ---------------------------------------------------------------------------


def test_resolve_all_vs_rest_default():
    """Omit all optional params → each class vs all others."""
    result = resolve_comparisons(
        positive=None, negative=None, comparisons=None,
        available_labels=LABELS_3, class_col="genotype",
    )
    assert len(result) == 3
    ids = {r.comparison_id for r in result}
    assert len(ids) == 3
    for r in result:
        assert isinstance(r, ResolvedComparison)
        assert r.all_members == LABELS_3


def test_resolve_all_vs_rest_scoped():
    """positive=['A','B'] → only those two get comparisons."""
    result = resolve_comparisons(
        positive=["A", "B"], negative=None, comparisons=None,
        available_labels=LABELS_3, class_col="cls",
    )
    assert len(result) == 2
    pos_labels = {r.positive_label for r in result}
    assert pos_labels == {"A", "B"}
    for r in result:
        assert "C" in r.negative_members


# ---------------------------------------------------------------------------
# resolve_comparisons — all_pairs
# ---------------------------------------------------------------------------


def test_resolve_all_pairs_scoped():
    """all_pairs within scope → C(n,2) pairs."""
    result = resolve_comparisons(
        positive=["A", "B", "C"], negative=None, comparisons="all_pairs",
        available_labels=LABELS_3, class_col="cls",
    )
    assert len(result) == 3  # C(3,2)
    for r in result:
        assert len(r.positive_members) == 1
        assert len(r.negative_members) == 1


def test_resolve_all_pairs_default_scope():
    """all_pairs with no scope → all labels."""
    result = resolve_comparisons(
        positive=None, negative=None, comparisons="all_pairs",
        available_labels=LABELS_4, class_col="cls",
    )
    assert len(result) == 6  # C(4,2)


def test_resolve_all_pairs_rejects_pooled_in_scope():
    with pytest.raises(ValueError, match="pooled tuples"):
        resolve_comparisons(
            positive=[("A", "B"), "C"], negative=None, comparisons="all_pairs",
            available_labels=LABELS_3, class_col="cls",
        )


# ---------------------------------------------------------------------------
# resolve_comparisons — explicit (Cartesian)
# ---------------------------------------------------------------------------


def test_resolve_explicit_cartesian():
    """positive=['A','B'], negative='C' → 2 comparisons."""
    result = resolve_comparisons(
        positive=["A", "B"], negative="C", comparisons=None,
        available_labels=LABELS_3, class_col="cls",
    )
    assert len(result) == 2
    for r in result:
        assert r.negative_members == ("C",)


def test_resolve_explicit_pooled():
    """Pooled positive tuple → single comparison with pooled pos."""
    result = resolve_comparisons(
        positive=("A", "B"), negative="C", comparisons=None,
        available_labels=LABELS_3, class_col="cls",
    )
    assert len(result) == 1
    r = result[0]
    assert r.positive_members == ("A", "B")
    assert r.is_pooled_positive
    assert not r.is_pooled_negative
    assert r.positive_label == "A+B"


# ---------------------------------------------------------------------------
# resolve_comparisons — explicit_design
# ---------------------------------------------------------------------------


def test_resolve_explicit_design_dataframe():
    df = pd.DataFrame({"positive": ["A", "B"], "negative": ["C", "C"]})
    result = resolve_comparisons(
        positive=None, negative=None, comparisons=df,
        available_labels=LABELS_3, class_col="cls",
    )
    assert len(result) == 2


def test_resolve_explicit_design_list_dict():
    rows = [
        {"positive": "A", "negative": "B"},
        {"positive": ("B", "C"), "negative": "A"},
    ]
    result = resolve_comparisons(
        positive=None, negative=None, comparisons=rows,
        available_labels=LABELS_3, class_col="cls",
    )
    assert len(result) == 2
    assert result[1].is_pooled_positive


# ---------------------------------------------------------------------------
# Mutual-exclusion errors
# ---------------------------------------------------------------------------


def test_mutual_exclusion_design_plus_positive():
    df = pd.DataFrame({"positive": ["A"], "negative": ["B"]})
    with pytest.raises(ValueError, match="Cannot combine"):
        resolve_comparisons(
            positive="A", negative=None, comparisons=df,
            available_labels=LABELS_3, class_col="cls",
        )


def test_mutual_exclusion_design_plus_negative():
    df = pd.DataFrame({"positive": ["A"], "negative": ["B"]})
    with pytest.raises(ValueError, match="Cannot combine"):
        resolve_comparisons(
            positive=None, negative="B", comparisons=df,
            available_labels=LABELS_3, class_col="cls",
        )


def test_mutual_exclusion_scheme_plus_negative():
    with pytest.raises(ValueError, match="Cannot combine"):
        resolve_comparisons(
            positive=["A"], negative="B", comparisons="all_pairs",
            available_labels=LABELS_3, class_col="cls",
        )


def test_mutual_exclusion_scheme_plus_scalar_positive():
    with pytest.raises(ValueError, match="scalar positive"):
        resolve_comparisons(
            positive="A", negative=None, comparisons="all_pairs",
            available_labels=LABELS_3, class_col="cls",
        )


def test_negative_only_default_resolves_all_others_vs_negative():
    result = resolve_comparisons(
        positive=None, negative="B", comparisons=None,
        available_labels=LABELS_3, class_col="cls",
    )
    assert len(result) == 2
    assert {r.positive_label for r in result} == {"A", "C"}
    assert {r.negative_label for r in result} == {"B"}


def test_negative_only_pooled_tuple_stays_binary_with_pooled_negative():
    result = resolve_comparisons(
        positive=None, negative=("B", "C"), comparisons=None,
        available_labels=LABELS_4, class_col="cls",
    )
    assert len(result) == 2
    assert {r.positive_label for r in result} == {"A", "D"}
    assert {r.negative_label for r in result} == {"B+C"}
    assert all(r.is_pooled_negative for r in result)


def test_negative_only_list_enumerates_separate_negative_groups():
    result = resolve_comparisons(
        positive=None, negative=["B", "C"], comparisons=None,
        available_labels=LABELS_4, class_col="cls",
    )
    assert len(result) == 6
    assert {r.negative_label for r in result} == {"B", "C"}
    assert ("A", "B") in {(r.positive_label, r.negative_label) for r in result}
    assert ("A", "C") in {(r.positive_label, r.negative_label) for r in result}


def test_negative_only_rejected_when_scheme_is_specified():
    with pytest.raises(ValueError, match="Cannot combine comparisons='all_pairs' with negative argument"):
        resolve_comparisons(
            positive=None, negative="B", comparisons="all_pairs",
            available_labels=LABELS_3, class_col="cls",
        )


# ---------------------------------------------------------------------------
# Overlap & label existence
# ---------------------------------------------------------------------------


def test_overlap_check():
    with pytest.raises(ValueError, match="appear on both"):
        resolve_comparisons(
            positive="A", negative=("A", "B"), comparisons=None,
            available_labels=LABELS_3, class_col="cls",
        )


def test_label_existence_check():
    with pytest.raises(ValueError, match="not found"):
        resolve_comparisons(
            positive="TYPO", negative="A", comparisons=None,
            available_labels=LABELS_3, class_col="cls",
        )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_deduplication():
    rows = [
        {"positive": "A", "negative": "B"},
        {"positive": "A", "negative": "B"},  # duplicate
        {"positive": "B", "negative": "C"},
    ]
    result = resolve_comparisons(
        positive=None, negative=None, comparisons=rows,
        available_labels=LABELS_3, class_col="cls",
    )
    assert len(result) == 2


# ---------------------------------------------------------------------------
# check_min_samples
# ---------------------------------------------------------------------------


def test_check_min_samples_group_level():
    rc = resolve_comparisons(
        positive="A", negative="B", comparisons=None,
        available_labels={"A", "B"}, class_col="cls",
    )
    with pytest.raises(ValueError, match="group total"):
        check_min_samples(rc, {"A": 1, "B": 5}, min_samples_per_group=3, min_samples_per_member=1)


def test_check_min_samples_per_member():
    rc = resolve_comparisons(
        positive=("A", "B"), negative="C", comparisons=None,
        available_labels=LABELS_3, class_col="cls",
    )
    with pytest.raises(ValueError, match="min_samples_per_member"):
        check_min_samples(rc, {"A": 10, "B": 1, "C": 10}, min_samples_per_group=2, min_samples_per_member=2)


def test_check_min_samples_passes():
    rc = resolve_comparisons(
        positive="A", negative="B", comparisons=None,
        available_labels={"A", "B"}, class_col="cls",
    )
    check_min_samples(rc, {"A": 5, "B": 5}, min_samples_per_group=3, min_samples_per_member=2)


def test_check_min_samples_non_pooled_skips_per_member():
    """Single-label groups only check group minimum, not per-member."""
    rc = resolve_comparisons(
        positive="A", negative="B", comparisons=None,
        available_labels={"A", "B"}, class_col="cls",
    )
    # per_member=10 should NOT trigger for single-label groups
    check_min_samples(rc, {"A": 5, "B": 5}, min_samples_per_group=3, min_samples_per_member=10)


# ---------------------------------------------------------------------------
# ResolvedComparison properties
# ---------------------------------------------------------------------------


def test_resolved_comparison_properties():
    rc = resolve_comparisons(
        positive=("A", "B"), negative="C", comparisons=None,
        available_labels=LABELS_3, class_col="cls",
    )[0]
    assert rc.is_pooled_positive is True
    assert rc.is_pooled_negative is False
    assert rc.all_members == frozenset({"A", "B", "C"})
    assert "__vs__" in rc.comparison_id
