"""Tests for morphology_geometry/vectors.py."""

from __future__ import annotations

import numpy as np
import pytest

from analyze.morphology_geometry.vectors import (
    axis_alignment,
    cosine_alignment,
    direction_matrix,
    weighted_axis,
)
from .conftest import COMPARISONS, N_FEATURES


# ---------------------------------------------------------------------------
# cosine_alignment
# ---------------------------------------------------------------------------

class TestCosineAlignment:
    def test_parallel_vectors(self):
        u = np.array([1.0, 0.0, 0.0])
        assert cosine_alignment(u, u) == pytest.approx(1.0)

    def test_anti_parallel_vectors(self):
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([-1.0, 0.0, 0.0])
        assert cosine_alignment(u, v) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        assert cosine_alignment(u, v) == pytest.approx(0.0)

    def test_allow_sign_flip_anti_parallel(self):
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([-1.0, 0.0, 0.0])
        assert cosine_alignment(u, v, allow_sign_flip=True) == pytest.approx(1.0)

    def test_unnormalized_vectors(self):
        u = np.array([3.0, 0.0])
        v = np.array([5.0, 0.0])
        assert cosine_alignment(u, v) == pytest.approx(1.0)

    def test_zero_vector_raises(self):
        u = np.array([1.0, 0.0])
        z = np.array([0.0, 0.0])
        with pytest.raises(ValueError, match="zero vectors"):
            cosine_alignment(u, z)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes differ"):
            cosine_alignment(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_nan_raises(self):
        u = np.array([1.0, np.nan])
        v = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="non-finite"):
            cosine_alignment(u, v)

    def test_result_clipped_to_minus1_plus1(self):
        # Floating point can push dot product slightly outside [-1, 1]
        u = np.ones(1000) / np.sqrt(1000)
        v = u + np.finfo(float).eps * 100
        # Just check it doesn't raise and returns in range
        result = cosine_alignment(u, v)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# axis_alignment
# ---------------------------------------------------------------------------

class TestAxisAlignment:
    def test_equals_abs_cosine(self):
        rng = np.random.default_rng(7)
        u = rng.standard_normal(5)
        v = rng.standard_normal(5)
        assert axis_alignment(u, v) == pytest.approx(abs(cosine_alignment(u, v)))

    def test_always_non_negative(self):
        u = np.array([1.0, 0.0])
        v = np.array([-1.0, 0.0])
        assert axis_alignment(u, v) >= 0.0


# ---------------------------------------------------------------------------
# direction_matrix
# ---------------------------------------------------------------------------

class TestDirectionMatrix:
    def test_full_shape(self, validated):
        meta, vecs, names = direction_matrix(validated)
        n_rows = len(COMPARISONS) * 3  # 3 bins
        assert vecs.shape == (n_rows, N_FEATURES)
        assert len(meta) == n_rows
        assert names == validated.feature_names

    def test_rows_aligned_with_metadata(self, validated):
        meta, vecs, _ = direction_matrix(validated)
        for i, row in meta.iterrows():
            vid = row["vector_id"]
            expected = validated.vectors[i]
            np.testing.assert_allclose(vecs[i], expected, atol=1e-12)

    def test_comparison_id_filter(self, validated):
        cid = COMPARISONS[0]
        meta, vecs, _ = direction_matrix(validated, comparison_id=cid)
        assert (meta["comparison_id"] == cid).all()
        assert vecs.shape == (3, N_FEATURES)  # 3 bins

    def test_unknown_comparison_id_raises(self, validated):
        with pytest.raises(ValueError, match="not found"):
            direction_matrix(validated, comparison_id="does_not_exist")

    def test_no_filter_returns_all(self, validated):
        meta, vecs, _ = direction_matrix(validated)
        assert len(meta) == len(validated.metadata)


# ---------------------------------------------------------------------------
# weighted_axis
# ---------------------------------------------------------------------------

class TestWeightedAxis:
    def test_returns_unit_norm(self, validated):
        axis, _ = weighted_axis(validated, comparison_id=COMPARISONS[0])
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-10)

    def test_axis_weight_column_present(self, validated):
        _, meta_w = weighted_axis(validated, comparison_id=COMPARISONS[0])
        assert "axis_weight" in meta_w.columns

    def test_uniform_mode(self, validated):
        axis, meta_w = weighted_axis(
            validated, comparison_id=COMPARISONS[0], weight_mode="uniform"
        )
        np.testing.assert_allclose(
            meta_w["axis_weight"].to_numpy(), np.ones(3), atol=1e-12
        )
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-10)

    def test_auroc_mode(self, validated):
        axis, _ = weighted_axis(
            validated, comparison_id=COMPARISONS[0], weight_mode="auroc"
        )
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-10)

    def test_auroc_minus_half_mode(self, validated):
        axis, meta_w = weighted_axis(
            validated, comparison_id=COMPARISONS[0], weight_mode="auroc_minus_half"
        )
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-10)
        assert (meta_w["axis_weight"] >= 0.0).all()

    def test_invalid_weight_mode_raises(self, validated):
        with pytest.raises(ValueError, match="weight_mode"):
            weighted_axis(validated, comparison_id=COMPARISONS[0], weight_mode="bad")

    def test_fallback_to_uniform_when_no_auroc(self, minimal_directions):
        """When has_auroc=False, auroc_minus_half falls back to uniform."""
        from analyze.morphology_geometry.validation import validate_classifier_directions
        minimal_directions.metadata.drop(columns=["auroc_obs"], inplace=True)
        vd = validate_classifier_directions(minimal_directions, feature_set="pca")
        assert vd.has_auroc is False
        axis, meta_w = weighted_axis(
            vd, comparison_id=COMPARISONS[0], weight_mode="auroc_minus_half"
        )
        np.testing.assert_allclose(
            meta_w["axis_weight"].to_numpy(), np.ones(3), atol=1e-12
        )

    def test_degenerate_weights_fall_back_to_uniform(self, minimal_directions):
        """All-zero auroc_obs → weights all zero → fallback to uniform."""
        from analyze.morphology_geometry.validation import validate_classifier_directions
        minimal_directions.metadata["auroc_obs"] = 0.5  # auroc_minus_half → 0 weights
        vd = validate_classifier_directions(minimal_directions, feature_set="pca")
        axis, meta_w = weighted_axis(
            vd, comparison_id=COMPARISONS[0], weight_mode="auroc_minus_half"
        )
        # Falls back to uniform; axis is still unit-norm
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-10)
