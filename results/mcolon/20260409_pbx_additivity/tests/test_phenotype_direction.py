"""Tests for experiment-local phenotype_direction helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TEST_ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(TEST_ROOT))

from analyze.classification.engine.analysis import ClassifierDirections

from phenotype_direction import (
    CENTERING_VARIANTS,
    INTERCEPT_CENTERED,
    MIDPOINT_CENTERED,
    NEG_CENTROID_CENTERED,
    RAW_PROJECTION,
    center_metadata_row,
    compute_all_centered_scores,
    compute_center_stats,
    axis_alignment,
    cosine_alignment,
    project_binned_features,
    weighted_axis,
)


def _directions() -> ClassifierDirections:
    metadata = pd.DataFrame([
        {
            "feature_set": "emb",
            "comparison_id": "A_vs_B",
            "time_bin_center": 26.0,
            "auroc_obs": 0.6,
            "vector_id": "v1",
        },
        {
            "feature_set": "emb",
            "comparison_id": "A_vs_B",
            "time_bin_center": 30.0,
            "auroc_obs": 0.9,
            "vector_id": "v2",
        },
    ])
    return ClassifierDirections(
        metadata=metadata,
        vectors={
            "v1": np.array([1.0, 0.0]),
            "v2": np.array([0.0, 1.0]),
        },
        feature_names={"emb": ["z_mu_b_0", "z_mu_b_1"]},
    )


def test_alignment_helpers_distinguish_direction_and_axis():
    assert cosine_alignment(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == -1.0
    assert axis_alignment(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == 1.0


def test_weighted_axis_defaults_to_auroc_minus_half():
    directions = _directions()
    axis, weights = weighted_axis(
        directions,
        feature_set="emb",
        comparison_id="A_vs_B",
    )
    expected = np.array([0.1, 0.4])
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(axis, expected)
    np.testing.assert_allclose(weights["axis_weight"].to_numpy(), [0.1, 0.4])


def test_project_binned_features_uses_saved_feature_order():
    directions = _directions()
    df = pd.DataFrame([
        {
            "embryo_id": "e1",
            "genotype": "A",
            "predicted_stage_hpf": 24.0,
            "z_mu_b_0": 1.0,
            "z_mu_b_1": 2.0,
        },
        {
            "embryo_id": "e1",
            "genotype": "A",
            "predicted_stage_hpf": 25.0,
            "z_mu_b_0": 3.0,
            "z_mu_b_1": 4.0,
        },
    ])
    projected = project_binned_features(
        df,
        directions=directions,
        axis=np.array([10.0, 1.0]),
        feature_set="emb",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        class_col="genotype",
        bin_width=4.0,
    )
    assert len(projected) == 1
    assert projected.loc[0, "_time_bin"] == 24
    assert projected.loc[0, "phenotype_direction_score"] == 23.0


def _projected_bin() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"embryo_id": "e1", "time_bin_center": 26.0, "genotype": "mut", "raw_score": 2.0},
            {"embryo_id": "e2", "time_bin_center": 26.0, "genotype": "mut", "raw_score": 4.0},
            {"embryo_id": "e3", "time_bin_center": 26.0, "genotype": "wt", "raw_score": -1.0},
            {"embryo_id": "e4", "time_bin_center": 26.0, "genotype": "wt", "raw_score": 1.0},
        ]
    )


def test_compute_center_stats_returns_expected_summary():
    stats = compute_center_stats(
        _projected_bin(),
        intercept=-3.0,
        coef_norm=2.0,
        pos_label="mut",
        neg_label="wt",
    )
    assert stats["n_pos"] == 2
    assert stats["n_neg"] == 2
    assert stats["pos_mean"] == 3.0
    assert stats["neg_mean"] == 0.0
    assert stats["midpoint"] == 1.5
    assert stats["boundary"] == 1.5


def test_compute_all_centered_scores_populates_all_variants():
    out, stats = compute_all_centered_scores(
        _projected_bin(),
        intercept=-3.0,
        coef_norm=2.0,
        pos_label="mut",
        neg_label="wt",
    )
    assert CENTERING_VARIANTS == (
        INTERCEPT_CENTERED,
        NEG_CENTROID_CENTERED,
        MIDPOINT_CENTERED,
        RAW_PROJECTION,
    )
    np.testing.assert_allclose(out[RAW_PROJECTION].to_numpy(dtype=float), [2.0, 4.0, -1.0, 1.0])
    np.testing.assert_allclose(out[INTERCEPT_CENTERED].to_numpy(dtype=float), [0.5, 2.5, -2.5, -0.5])
    np.testing.assert_allclose(out[NEG_CENTROID_CENTERED].to_numpy(dtype=float), [2.0, 4.0, -1.0, 1.0])
    np.testing.assert_allclose(out[MIDPOINT_CENTERED].to_numpy(dtype=float), [0.5, 2.5, -2.5, -0.5])
    assert stats["midpoint"] == 1.5


def test_center_metadata_row_has_stable_schema():
    _, stats = compute_all_centered_scores(
        _projected_bin(),
        intercept=-3.0,
        coef_norm=2.0,
        pos_label="mut",
        neg_label="wt",
    )
    row = center_metadata_row(
        vector_id="v123",
        comparison_id="mut__vs__wt",
        time_bin_center=26.0,
        time_bin=24,
        positive_label="mut",
        negative_label="wt",
        center_stats=stats,
    )
    assert row == {
        "vector_id": "v123",
        "comparison_id": "mut__vs__wt",
        "time_bin_center": 26.0,
        "time_bin": 24,
        "positive_label": "mut",
        "negative_label": "wt",
        "coef_norm": 2.0,
        "intercept": -3.0,
        "boundary_score": 1.5,
        "neg_mean": 0.0,
        "pos_mean": 3.0,
        "midpoint": 1.5,
        "n_pos": 2,
        "n_neg": 2,
    }


def test_compute_center_stats_requires_both_labels():
    df = _projected_bin()
    df = df[df["genotype"] != "wt"].copy()
    try:
        compute_center_stats(
            df,
            intercept=0.0,
            coef_norm=1.0,
            pos_label="mut",
            neg_label="wt",
        )
    except ValueError as exc:
        assert "both labels" in str(exc)
    else:
        raise AssertionError("Expected ValueError when one label is absent.")
