"""Tests for _perturbation.py."""

import numpy as np
import pytest

from analyze.utils.resampling._spec import (
    ResampleSpec, IndicesParams, LabelsParams, GroupsParams,
)
from analyze.utils.resampling._perturbation import _perturb


# ── helpers ──────────────────────────────────────────────────────────

def _make_rng(seed=0):
    return np.random.default_rng(seed)


# ── indices ──────────────────────────────────────────────────────────

class TestPerturbIndices:
    def test_basic_subsample(self):
        data = {"labels": np.arange(10)}
        spec = ResampleSpec("indices", IndicesParams(replacement=False, frac=0.5))
        out = _perturb(data, spec, _make_rng())
        assert "indices" in out
        assert len(out["indices"]) == 5
        assert len(np.unique(out["indices"])) == 5  # no replacement

    def test_basic_bootstrap(self):
        data = {"labels": np.arange(10)}
        spec = ResampleSpec("indices", IndicesParams(replacement=True))
        out = _perturb(data, spec, _make_rng())
        assert len(out["indices"]) == 10

    def test_size_explicit(self):
        data = {"labels": np.arange(20)}
        spec = ResampleSpec("indices", IndicesParams(replacement=False, size=7))
        out = _perturb(data, spec, _make_rng())
        assert len(out["indices"]) == 7

    def test_population_from_n(self):
        data = {"n": 50}
        spec = ResampleSpec("indices", IndicesParams(replacement=False, frac=0.1))
        out = _perturb(data, spec, _make_rng())
        assert len(out["indices"]) == 5

    def test_unit_key_preserves_block_order(self):
        """Repeated units (bootstrap) preserve intra-unit row order."""
        data = {
            "labels": np.array([0, 0, 1, 1, 2, 2]),
            "embryo_id": np.array(["A", "A", "B", "B", "C", "C"]),
        }
        spec = ResampleSpec(
            "indices",
            IndicesParams(replacement=True),
            unit_key="embryo_id",
        )
        rng = _make_rng(42)
        out = _perturb(data, spec, rng)
        idx = out["indices"]
        # Each unit block should have consecutive indices
        # (e.g., [0,1] or [2,3] or [4,5])
        unit_col = data["embryo_id"]
        # Check that within each selected unit block, rows are in order
        i = 0
        while i < len(idx):
            u = unit_col[idx[i]]
            block_rows = np.where(unit_col == u)[0]
            block_len = len(block_rows)
            actual = idx[i:i + block_len]
            np.testing.assert_array_equal(actual, block_rows)
            i += block_len

    def test_within_key_stratified(self):
        data = {
            "labels": np.arange(20),
            "stratum": np.array(["A"] * 10 + ["B"] * 10),
        }
        spec = ResampleSpec(
            "indices",
            IndicesParams(replacement=False, frac=0.5),
            within_key="stratum",
        )
        out = _perturb(data, spec, _make_rng())
        idx = out["indices"]
        assert len(idx) == 10
        # Should have some from each stratum
        from_a = np.sum(idx < 10)
        from_b = np.sum(idx >= 10)
        assert from_a > 0
        assert from_b > 0


# ── labels ───────────────────────────────────────────────────────────

class TestPerturbLabels:
    def test_basic_shuffle(self):
        labels = np.array(["A", "A", "B", "B", "B"])
        data = {"labels": labels}
        spec = ResampleSpec("labels", LabelsParams())
        out = _perturb(data, spec, _make_rng())
        # Same elements, possibly different order
        assert sorted(out["labels"]) == sorted(labels)

    def test_within_strata(self):
        data = {
            "labels": np.array(["A", "A", "B", "B"]),
            "time": np.array(["T1", "T1", "T2", "T2"]),
        }
        spec = ResampleSpec("labels", LabelsParams(), within_key="time")
        out = _perturb(data, spec, _make_rng())
        # Labels within T1 should be a permutation of original T1 labels
        assert sorted(out["labels"][:2]) == sorted(data["labels"][:2])
        assert sorted(out["labels"][2:]) == sorted(data["labels"][2:])

    def test_unit_level_shuffle(self):
        data = {
            "labels": np.array(["A", "A", "B", "B", "C", "C"]),
            "unit": np.array(["u1", "u1", "u2", "u2", "u3", "u3"]),
        }
        spec = ResampleSpec("labels", LabelsParams(), unit_key="unit")
        out = _perturb(data, spec, _make_rng())
        # Labels within each unit should be constant
        for u in ["u1", "u2", "u3"]:
            mask = data["unit"] == u
            vals = out["labels"][mask]
            assert len(np.unique(vals)) == 1


# ── groups ───────────────────────────────────────────────────────────

class TestPerturbGroups:
    def test_pool_and_redistribute(self):
        data = {
            "X1": np.array([[1, 2], [3, 4]]),
            "X2": np.array([[5, 6], [7, 8], [9, 10]]),
        }
        spec = ResampleSpec("groups", GroupsParams())
        out = _perturb(data, spec, _make_rng())
        assert out["X1"].shape == (2, 2)
        assert out["X2"].shape == (3, 2)
        # All original values present
        combined_orig = np.vstack([data["X1"], data["X2"]])
        combined_new = np.vstack([out["X1"], out["X2"]])
        np.testing.assert_array_equal(
            np.sort(combined_orig, axis=0),
            np.sort(combined_new, axis=0),
        )

    def test_custom_keys(self):
        data = {"grp_a": np.array([1, 2]), "grp_b": np.array([3, 4, 5])}
        spec = ResampleSpec("groups", GroupsParams(a_key="grp_a", b_key="grp_b"))
        out = _perturb(data, spec, _make_rng())
        assert len(out["grp_a"]) == 2
        assert len(out["grp_b"]) == 3


# ── determinism ──────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_result(self):
        data = {"labels": np.arange(100)}
        spec = ResampleSpec("indices", IndicesParams(replacement=False, frac=0.5))
        out1 = _perturb(data, spec, _make_rng(42))
        out2 = _perturb(data, spec, _make_rng(42))
        np.testing.assert_array_equal(out1["indices"], out2["indices"])

    def test_different_seed_different_result(self):
        data = {"labels": np.arange(100)}
        spec = ResampleSpec("indices", IndicesParams(replacement=False, frac=0.5))
        out1 = _perturb(data, spec, _make_rng(42))
        out2 = _perturb(data, spec, _make_rng(99))
        assert not np.array_equal(out1["indices"], out2["indices"])
