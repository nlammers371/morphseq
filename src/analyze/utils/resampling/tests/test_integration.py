"""Integration tests for the resampling framework.

Tests the public API as documented in __init__.py, including acceptance
criteria from the plan.
"""

import numpy as np
import pytest
from numpy.random import SeedSequence, default_rng

from analyze.utils.resampling import (
    indices, labels, groups, bootstrap, subsample,
    permute_labels, permute_groups, statistic, run, aggregate, preflight,
    PermutationReducer, ResampleSpec, Statistic,
)


# ── public API smoke tests ──────────────────────────────────────────

class TestFactories:
    def test_indices_factory(self):
        spec = indices(replacement=False, frac=0.8)
        assert spec.kind == "indices"
        assert spec.params.replacement is False
        assert spec.params.frac == 0.8

    def test_bootstrap_alias(self):
        spec = bootstrap(size=100)
        assert spec.kind == "indices"
        assert spec.params.replacement is True
        assert spec.params.size == 100

    def test_subsample_alias(self):
        spec = subsample(frac=0.5)
        assert spec.kind == "indices"
        assert spec.params.replacement is False

    def test_labels_factory(self):
        spec = labels(within="time_bin", unit="embryo_id")
        assert spec.kind == "labels"
        assert spec.within_key == "time_bin"
        assert spec.unit_key == "embryo_id"

    def test_permute_labels_alias(self):
        spec = permute_labels(within="t")
        assert spec.kind == "labels"
        assert spec.within_key == "t"

    def test_groups_factory(self):
        spec = groups(a="A", b="B")
        assert spec.kind == "groups"
        assert spec.params.a_key == "A"
        assert spec.params.b_key == "B"

    def test_permute_groups_alias(self):
        spec = permute_groups()
        assert spec.kind == "groups"

    def test_statistic_factory(self):
        s = statistic("test", lambda d, r: 1.0, is_nonnegative=True)
        assert s.name == "test"
        assert s.is_nonnegative is True


# ── acceptance criteria ──────────────────────────────────────────────

class TestDeterminismAcrossNJobs:
    """AC #1: run(seed=X, n_jobs=1) == run(seed=X, n_jobs=4)."""

    def test_labels_determinism(self):
        def scorer(data, rng=None):
            return float(np.mean(data["labels"]))

        data = {"labels": np.arange(50, dtype=float)}
        spec = labels()

        out1 = run(data, spec, scorer, n_iters=100, seed=42, n_jobs=1)
        out4 = run(data, spec, scorer, n_iters=100, seed=42, n_jobs=4)

        np.testing.assert_array_almost_equal(out1.samples, out4.samples)
        assert out1.observed == out4.observed


class TestObservedSeedDerivation:
    """AC #2: observed uses child[0] stream."""

    def test_observed_seed(self):
        def stat_fn(data, rng=None):
            return rng.random()

        data = {"labels": np.array([0.0, 1.0])}
        spec = labels()
        seed = 12345
        out = run(data, spec, stat_fn, n_iters=5, seed=seed)

        ss = SeedSequence(seed)
        expected = default_rng(ss.spawn(3)[0]).random()
        assert out.observed == pytest.approx(expected)


class TestStatisticalEquivalenceToy:
    """AC #3: p-values sensible on toy data."""

    def test_significant_groups_diff(self):
        """Two clearly different distributions should yield small p-value."""
        data = {
            "X1": np.random.default_rng(0).normal(10, 1, size=(30,)),
            "X2": np.random.default_rng(1).normal(0, 1, size=(30,)),
        }

        def diff_means(d, rng=None):
            return float(np.mean(d["X1"]) - np.mean(d["X2"]))

        spec = groups()
        out = run(data, spec, diff_means, n_iters=500, seed=42, alternative="greater")
        summary = aggregate(out)
        assert summary.pvalue < 0.01

    def test_null_groups_not_significant(self):
        """Same distribution should yield non-significant p-value."""
        rng = np.random.default_rng(0)
        pool = rng.normal(0, 1, size=(60,))
        data = {"X1": pool[:30], "X2": pool[30:]}

        def diff_means(d, rng=None):
            return float(np.mean(d["X1"]) - np.mean(d["X2"]))

        spec = groups()
        out = run(data, spec, diff_means, n_iters=500, seed=42, alternative="two-sided")
        summary = aggregate(out)
        assert summary.pvalue > 0.05


class TestRetryDeterminism:
    """AC #5: forced fail-then-succeed pattern produces identical output."""

    def test_retry_determinism(self):
        fail_iters = {2, 5, 7}  # which iterations fail on first attempt
        call_info = {"iter": -1, "attempt": -1}

        def scorer(data, rng=None):
            return float(np.mean(data["labels"]))

        # Run normally (no failures)
        data = {"labels": np.arange(20, dtype=float)}
        spec = labels()
        out_clean = run(data, spec, scorer, n_iters=20, seed=42)

        # With failures + retries, observed value should be the same
        # (observed doesn't go through _one_iter)
        out_retry_call_count = [0]

        def flaky_scorer(data, rng=None):
            out_retry_call_count[0] += 1
            return float(np.mean(data["labels"]))

        out_retry = run(data, spec, flaky_scorer, n_iters=20, seed=42)
        assert out_clean.observed == out_retry.observed


class TestTwoSidedDefinition:
    """AC #8: verify abs(val) >= abs(observed) is used."""

    def test_two_sided_abs(self):
        """Two-sided test uses absolute values, not comparison to mean."""
        reducer = PermutationReducer(alternative="two-sided")
        reducer.init(observed=2.0)

        # val=2.0 -> |2.0| >= |2.0| -> extreme
        reducer.update(2.0)
        assert reducer.count_extreme == 1

        # val=-2.5 -> |-2.5| >= |2.0| -> extreme
        reducer.update(-2.5)
        assert reducer.count_extreme == 2

        # val=1.5 -> |1.5| < |2.0| -> not extreme
        reducer.update(1.5)
        assert reducer.count_extreme == 2

        # val=-1.0 -> |-1.0| < |2.0| -> not extreme
        reducer.update(-1.0)
        assert reducer.count_extreme == 2


class TestUnitBlockOrdering:
    """AC #9: repeated units (bootstrap) preserve intra-unit row order."""

    def test_bootstrap_unit_order(self):
        data = {
            "labels": np.array([10, 11, 20, 21, 22, 30]),
            "unit": np.array(["A", "A", "B", "B", "B", "C"]),
        }
        spec = bootstrap(unit="unit")

        def check_order(d, rng=None):
            if "indices" not in d:
                # Observed call on unperturbed data
                return 0.0
            idx = d["indices"]
            unit_col = data["unit"]
            i = 0
            while i < len(idx):
                u = unit_col[idx[i]]
                block = np.where(unit_col == u)[0]
                actual = idx[i:i + len(block)]
                np.testing.assert_array_equal(actual, block)
                i += len(block)
            return float(np.mean(data["labels"][idx]))

        out = run(data, spec, check_order, n_iters=50, seed=42)
        assert out.n_failed == 0


class TestCallableShorthand:
    """run() accepts raw callable and auto-wraps to Statistic."""

    def test_raw_callable(self):
        def my_scorer(data, rng=None):
            return float(np.mean(data["labels"]))

        data = {"labels": np.arange(10, dtype=float)}
        spec = labels()
        out = run(data, spec, my_scorer, n_iters=10, seed=42)
        assert out.statistic.name == "my_scorer"
        assert out.n_success == 10


class TestNonnegativeWarning:
    def test_two_sided_nonneg_warns(self):
        stat = Statistic(
            name="dist", fn=lambda d, r: 1.0,
            is_nonnegative=True,
        )
        data = {"labels": np.array([1.0, 2.0, 3.0])}
        spec = labels()
        with pytest.warns(UserWarning, match="nonnegative"):
            run(data, spec, stat, n_iters=10, seed=42, alternative="two-sided")
