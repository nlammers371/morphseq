"""Tests for experiment-local phenotype_direction helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyze.classification.engine.analysis import ClassifierDirections

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phenotype_direction import (
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
