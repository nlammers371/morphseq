"""Tests for engine/analysis.py and engine/loop.py boundary functions."""

import numpy as np
import pandas as pd
import pytest

from analyze.classification.engine.analysis import (
    ClassificationAnalysis,
    _LazyLayers,
    _validate_scores,
)
from analyze.classification.engine.comparison_resolution import (
    ResolvedComparison,
    resolve_comparisons,
)
from analyze.classification.engine.data_prep import _build_binary_labels
from analyze.classification.engine.loop import _collect_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores(
    n_bins: int = 3,
    feature_sets: list[str] | None = None,
    comparison_ids: list[str] | None = None,
) -> pd.DataFrame:
    feature_sets = feature_sets or ["emb"]
    comparison_ids = comparison_ids or ["A__vs__B"]
    rows = []
    rng = np.random.default_rng(42)
    for fs in feature_sets:
        for cid in comparison_ids:
            for i in range(n_bins):
                rows.append({
                    "feature_set": fs,
                    "comparison_id": cid,
                    "positive_label": cid.split("__vs__")[0],
                    "negative_label": cid.split("__vs__")[1],
                    "time_bin_center": 26.0 + i * 4.0,
                    "time_bin": 24 + i * 4,
                    "bin_width": 4.0,
                    "auroc_obs": float(rng.uniform(0.5, 1.0)),
                    "pval": float(rng.uniform(0.0, 0.1)),
                    "n_positive": 10,
                    "n_negative": 10,
                    "auroc_null_mean": 0.5,
                    "auroc_null_std": 0.05,
                    "n_permutations": 100,
                })
    return pd.DataFrame(rows)


def _make_uns() -> dict:
    return {
        "schema_version": "classification_v1",
        "class_col": "genotype",
        "id_col": "embryo_id",
        "time_col": "predicted_stage_hpf",
        "bin_width": 4.0,
        "n_permutations": 100,
        "feature_sets": {
            "emb": {"spec": "z_mu_b", "columns": ["z_mu_b_0", "z_mu_b_1"]},
        },
        "comparisons": {
            "A__vs__B": {
                "positive_members": ["A"],
                "negative_members": ["B"],
                "positive_label": "A",
                "negative_label": "B",
            }
        },
    }


# ---------------------------------------------------------------------------
# _validate_scores
# ---------------------------------------------------------------------------


def test_validate_scores_missing_columns():
    df = pd.DataFrame({"feature_set": ["a"], "comparison_id": ["b"]})
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_scores(df)


def test_validate_scores_duplicates():
    df = _make_scores(n_bins=2)
    # duplicate a row
    df = pd.concat([df, df.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate keys"):
        _validate_scores(df)


def test_validate_scores_empty_ok():
    df = pd.DataFrame(columns=list(_LazyLayers._REGISTRY.keys()))
    # Just needs required columns, even if empty
    df = pd.DataFrame({col: pd.Series(dtype="object") for col in [
        "feature_set", "comparison_id", "positive_label",
        "negative_label", "time_bin_center", "auroc_obs",
    ]})
    _validate_scores(df)


# ---------------------------------------------------------------------------
# ClassificationAnalysis — save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    scores = _make_scores()
    uns = _make_uns()
    confusion = pd.DataFrame({
        "feature_set": ["emb"],
        "comparison_id": ["A__vs__B"],
        "time_bin_center": [26.0],
        "true_class": ["A"],
        "predicted_class": ["B"],
        "proportion": [0.2],
        "count": [2],
        "is_correct": [False],
    })
    layers = _LazyLayers()
    layers.store("confusion", confusion)

    ca = ClassificationAnalysis(scores=scores, uns=uns, layers=layers)
    out_path = ca.save(tmp_path / "test_run")

    loaded = ClassificationAnalysis.load(out_path)
    pd.testing.assert_frame_equal(loaded.scores, scores)
    assert loaded.uns["schema_version"] == "classification_v1"
    assert "confusion" in loaded.layers
    pd.testing.assert_frame_equal(loaded.layers["confusion"], confusion)


# ---------------------------------------------------------------------------
# _LazyLayers
# ---------------------------------------------------------------------------


def test_lazy_layers_missing():
    layers = _LazyLayers()
    with pytest.raises(KeyError, match="not computed"):
        layers["predictions"]


def test_lazy_layers_contains_no_disk_load(tmp_path):
    layers = _LazyLayers(tmp_path)
    assert "predictions" not in layers
    assert layers.cached() == []


def test_lazy_layers_store_and_retrieve():
    layers = _LazyLayers()
    df = pd.DataFrame({"x": [1]})
    layers.store("predictions", df)
    assert "predictions" in layers
    pd.testing.assert_frame_equal(layers["predictions"], df)


def test_lazy_layers_multiclass_missing_message(tmp_path):
    """When loaded from disk, multiclass_predictions gives actionable message."""
    layers = _LazyLayers(tmp_path)
    with pytest.raises(KeyError, match="save_multiclass_predictions=True"):
        layers["multiclass_predictions"]


# ---------------------------------------------------------------------------
# ClassificationAnalysis — subset
# ---------------------------------------------------------------------------


def test_subset_filters_correctly():
    scores = _make_scores(n_bins=5, feature_sets=["emb", "shape"])
    ca = ClassificationAnalysis(scores=scores, uns=_make_uns())

    sub = ca.subset(feature_set="emb")
    assert sub.feature_sets == ["emb"]
    assert len(sub.scores) == 5

    sub2 = ca.subset(time_range=(28.0, 34.0))
    assert all(sub2.scores["time_bin_center"] >= 28.0)
    assert all(sub2.scores["time_bin_center"] <= 34.0)


# ---------------------------------------------------------------------------
# ClassificationAnalysis — stack
# ---------------------------------------------------------------------------


def test_stack_merges_scores():
    s1 = _make_scores(n_bins=2, comparison_ids=["A__vs__B"])
    s2 = _make_scores(n_bins=2, comparison_ids=["C__vs__D"])
    ca1 = ClassificationAnalysis(scores=s1, uns={"comparisons": {"A__vs__B": {}}})
    ca2 = ClassificationAnalysis(scores=s2, uns={"comparisons": {"C__vs__D": {}}})
    merged = ca1.stack(ca2)
    assert len(merged.scores) == 4
    assert set(merged.comparison_ids) == {"A__vs__B", "C__vs__D"}


def test_stack_conflict_error():
    s1 = _make_scores(n_bins=2, comparison_ids=["A__vs__B"])
    s2 = _make_scores(n_bins=2, comparison_ids=["A__vs__B"])
    ca1 = ClassificationAnalysis(scores=s1, uns={"comparisons": {"A__vs__B": {"x": 1}}})
    ca2 = ClassificationAnalysis(scores=s2, uns={"comparisons": {"A__vs__B": {"x": 2}}})
    with pytest.raises(ValueError, match="Conflict"):
        ca1.stack(ca2)


# ---------------------------------------------------------------------------
# _build_binary_labels (from loop.py)
# ---------------------------------------------------------------------------


def _make_class_df() -> pd.DataFrame:
    rows = []
    for cls in ["A", "B", "C"]:
        for eid in range(3):
            rows.append({"embryo_id": f"{cls}_{eid}", "genotype": cls, "val": 1.0})
    return pd.DataFrame(rows)


def test_build_binary_labels_unpooled():
    df = _make_class_df()
    rc = resolve_comparisons(
        positive="A", negative="B", comparisons=None,
        available_labels={"A", "B", "C"}, class_col="genotype",
    )[0]
    result = _build_binary_labels(df, "genotype", rc)
    assert set(result["_y"].unique()) == {0, 1}
    assert len(result) == 6  # 3 A + 3 B, no C
    assert all(result.loc[result["genotype"] == "A", "_y"] == 1)
    assert all(result.loc[result["genotype"] == "B", "_y"] == 0)


def test_build_binary_labels_pooled():
    df = _make_class_df()
    rc = resolve_comparisons(
        positive=("A", "B"), negative="C", comparisons=None,
        available_labels={"A", "B", "C"}, class_col="genotype",
    )[0]
    result = _build_binary_labels(df, "genotype", rc)
    assert len(result) == 9  # all rows
    assert all(result.loc[result["genotype"] == "A", "_y"] == 1)
    assert all(result.loc[result["genotype"] == "B", "_y"] == 1)
    assert all(result.loc[result["genotype"] == "C", "_y"] == 0)


def test_build_binary_labels_filters_unrelated():
    df = _make_class_df()
    rc = resolve_comparisons(
        positive="A", negative="B", comparisons=None,
        available_labels={"A", "B", "C"}, class_col="genotype",
    )[0]
    result = _build_binary_labels(df, "genotype", rc)
    assert "C" not in result["genotype"].values


# ---------------------------------------------------------------------------
# _collect_scores schema (from loop.py)
# ---------------------------------------------------------------------------


def test_collect_scores_schema():
    rc = resolve_comparisons(
        positive="A", negative="B", comparisons=None,
        available_labels={"A", "B"}, class_col="cls",
    )[0]
    bin_results = [{
        "time_bin": 24,
        "time_bin_center": 26.0,
        "bin_width": 4.0,
        "auroc_obs": 0.85,
        "pval": 0.01,
        "n_positive": 10,
        "n_negative": 10,
        "auroc_null_mean": 0.5,
        "auroc_null_std": 0.05,
        "n_permutations": 100,
        "_null_array": np.zeros(100),
    }]
    rows = _collect_scores(bin_results, rc, "emb")
    assert len(rows) == 1
    row = rows[0]
    expected_keys = {
        "feature_set", "comparison_id", "positive_label", "negative_label",
        "time_bin_center", "time_bin", "bin_width",
        "auroc_obs", "pval", "n_positive", "n_negative",
        "auroc_null_mean", "auroc_null_std", "n_permutations",
    }
    assert set(row.keys()) == expected_keys
    assert "_null_array" not in row
    assert row["feature_set"] == "emb"
    assert row["comparison_id"] == rc.comparison_id
