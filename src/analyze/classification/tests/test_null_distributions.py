"""Tests for engine/null.py — NullDistributions."""

import numpy as np
import pytest

from analyze.classification.engine.null import NullDistributions


def _make_null_dists(n_rows: int = 3, n_perms: int = 10) -> NullDistributions:
    rng = np.random.default_rng(0)
    return NullDistributions(
        null_auc=rng.random((n_rows, n_perms), dtype=np.float32),
        feature_set=np.array(["emb", "emb", "shape"]),
        comparison_id=np.array(["A__vs__B", "A__vs__C", "A__vs__B"]),
        time_bin_center=np.array([26.0, 26.0, 30.0]),
    )


def test_roundtrip(tmp_path):
    nd = _make_null_dists()
    path = tmp_path / "null_distributions.npz"
    nd.save(path)
    loaded = NullDistributions.load(path)
    np.testing.assert_array_almost_equal(loaded.null_auc, nd.null_auc, decimal=5)
    np.testing.assert_array_equal(loaded.feature_set, nd.feature_set)
    np.testing.assert_array_equal(loaded.comparison_id, nd.comparison_id)
    np.testing.assert_array_equal(loaded.time_bin_center, nd.time_bin_center)


def test_get():
    nd = _make_null_dists()
    row = nd.get("emb", "A__vs__B", 26.0)
    np.testing.assert_array_equal(row, nd.null_auc[0])


def test_get_missing_key():
    nd = _make_null_dists()
    with pytest.raises(KeyError, match="No null distribution"):
        nd.get("emb", "MISSING", 26.0)


def test_shape_mismatch():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="length"):
        NullDistributions(
            null_auc=rng.random((3, 10), dtype=np.float32),
            feature_set=np.array(["emb", "emb"]),  # wrong length
            comparison_id=np.array(["A", "B", "C"]),
            time_bin_center=np.array([26.0, 26.0, 30.0]),
        )
