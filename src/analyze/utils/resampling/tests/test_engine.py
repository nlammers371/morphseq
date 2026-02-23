"""Tests for _engine.py."""

import numpy as np
import pytest
from numpy.random import SeedSequence, default_rng

from analyze.utils.resampling._spec import (
    ResampleSpec, IndicesParams, LabelsParams, GroupsParams,
)
from analyze.utils.resampling._statistic import Statistic
from analyze.utils.resampling._engine import run, _resolve_alternative
from analyze.utils.resampling._reducer import PermutationReducer


# ── helpers ──────────────────────────────────────────────────────────

def _mean_stat(data, rng=None):
    """Return mean of labels."""
    return float(np.mean(data["labels"]))


def _rng_stat(data, rng=None):
    """Return rng.random() — for testing seed derivation."""
    return rng.random()


def _fail_once_stat():
    """Factory for a stat that fails on first call per iteration."""
    calls = {"count": 0}

    def fn(data, rng=None):
        calls["count"] += 1
        if calls["count"] % 2 == 1:
            raise RuntimeError("transient failure")
        return float(np.mean(data["labels"]))

    return fn


MEAN = Statistic(name="mean", fn=_mean_stat)
RNG_STAT = Statistic(name="rng_draw", fn=_rng_stat)


# ── resolve alternative ─────────────────────────────────────────────

class TestResolveAlternative:
    def test_caller_wins(self):
        stat = Statistic(name="x", fn=_mean_stat, default_alternative="less")
        assert _resolve_alternative("greater", stat) == "greater"

    def test_stat_default(self):
        stat = Statistic(name="x", fn=_mean_stat, default_alternative="less")
        assert _resolve_alternative(None, stat) == "less"

    def test_nonneg_infers_greater(self):
        stat = Statistic(name="x", fn=_mean_stat, is_nonnegative=True)
        assert _resolve_alternative(None, stat) == "greater"

    def test_fallback_two_sided(self):
        stat = Statistic(name="x", fn=_mean_stat)
        assert _resolve_alternative(None, stat) == "two-sided"


# ── basic run ────────────────────────────────────────────────────────

class TestRunBasic:
    def test_labels_permutation(self):
        data = {"labels": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        spec = ResampleSpec("labels", LabelsParams())
        out = run(data, spec, MEAN, n_iters=50, seed=42)
        assert out.n_success == 50
        assert out.n_failed == 0
        assert len(out.samples) == 50
        assert out.observed == pytest.approx(3.0)

    def test_groups_permutation(self):
        data = {
            "X1": np.array([1.0, 2.0]),
            "X2": np.array([5.0, 6.0, 7.0]),
        }

        def diff_means(d, rng=None):
            return float(np.mean(d["X1"]) - np.mean(d["X2"]))

        spec = ResampleSpec("groups", GroupsParams())
        out = run(data, spec, diff_means, n_iters=50, seed=42)
        assert out.n_success == 50
        assert out.statistic.name == "diff_means"  # auto-wrapped

    def test_indices_subsample(self):
        data = {"labels": np.arange(20, dtype=float)}
        spec = ResampleSpec("indices", IndicesParams(replacement=False, frac=0.5))
        out = run(data, spec, MEAN, n_iters=30, seed=42)
        assert out.n_success == 30

    def test_store_none(self):
        data = {"labels": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        spec = ResampleSpec("labels", LabelsParams())
        reducer = PermutationReducer()
        out = run(data, spec, MEAN, n_iters=50, seed=42, store="none", reducer=reducer)
        assert out.samples is None
        assert out.reducer_state is not None
        assert out.reducer_state["n_permutations"] == 50


# ── determinism ──────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_result(self):
        data = {"labels": np.arange(20, dtype=float)}
        spec = ResampleSpec("labels", LabelsParams())
        out1 = run(data, spec, MEAN, n_iters=50, seed=42)
        out2 = run(data, spec, MEAN, n_iters=50, seed=42)
        np.testing.assert_array_equal(out1.samples, out2.samples)
        assert out1.observed == out2.observed

    def test_observed_uses_child0(self):
        """Observed value comes from SeedSequence(seed).spawn(3)[0]."""
        seed = 123
        ss = SeedSequence(seed)
        obs_ss = ss.spawn(3)[0]
        expected = default_rng(obs_ss).random()

        data = {"labels": np.array([0.0, 1.0])}
        spec = ResampleSpec("labels", LabelsParams())
        out = run(data, spec, RNG_STAT, n_iters=5, seed=seed)
        assert out.observed == pytest.approx(expected)

    def test_different_seed_different_result(self):
        # Use RNG_STAT so samples vary with seed (mean is invariant to permutation)
        data = {"labels": np.array([0.0, 1.0])}
        spec = ResampleSpec("labels", LabelsParams())
        out1 = run(data, spec, RNG_STAT, n_iters=50, seed=42)
        out2 = run(data, spec, RNG_STAT, n_iters=50, seed=99)
        assert out1.samples != out2.samples


# ── retry logic ──────────────────────────────────────────────────────

class TestRetry:
    def test_retry_recovers(self):
        """Stat that fails on first attempt but succeeds on retry."""
        # Track calls: dry run = call 1, observed = call 2,
        # then iterations alternate fail/succeed.
        call_count = [0]

        def flaky(data, rng=None):
            call_count[0] += 1
            # First 2 calls are dry run + observed; let those pass.
            # After that, fail on every other call (first attempt of each iter).
            if call_count[0] > 2 and call_count[0] % 2 == 1:
                raise RuntimeError("transient")
            return float(np.mean(data["labels"]))

        data = {"labels": np.arange(5, dtype=float)}
        spec = ResampleSpec("labels", LabelsParams())
        stat = Statistic(name="flaky", fn=flaky)
        out = run(data, spec, stat, n_iters=10, seed=42, max_retries_per_iter=1)
        # All should succeed since we have 1 retry
        assert out.n_success == 10
        assert out.n_failed == 0


# ── reducer integration ─────────────────────────────────────────────

class TestReducerIntegration:
    def test_reducer_matches_stored(self):
        """Reducer p-value should match recomputed p-value from stored samples."""
        data = {"labels": np.array([1.0, 2.0, 3.0, 10.0, 20.0])}
        spec = ResampleSpec("labels", LabelsParams())
        reducer = PermutationReducer(alternative="greater")
        out = run(
            data, spec, MEAN, n_iters=200, seed=42,
            store="all", reducer=reducer, alternative="greater",
        )

        # Recompute from stored samples
        null = np.array(out.samples)
        k = np.sum(null >= out.observed)
        expected_p = (1 + k) / (200 + 1)

        assert out.reducer_state["pvalue"] == pytest.approx(expected_p)

    def test_reservoir_determinism(self):
        """Same seed produces same reservoir."""
        data = {"labels": np.arange(50, dtype=float)}
        spec = ResampleSpec("labels", LabelsParams())

        r1 = PermutationReducer(reservoir_size=20)
        out1 = run(data, spec, MEAN, n_iters=200, seed=42,
                    store="none", reducer=r1, alternative="two-sided")

        r2 = PermutationReducer(reservoir_size=20)
        out2 = run(data, spec, MEAN, n_iters=200, seed=42,
                    store="none", reducer=r2, alternative="two-sided")

        np.testing.assert_array_equal(
            out1.reducer_state["reservoir"],
            out2.reducer_state["reservoir"],
        )

    def test_reservoir_seed_coupling(self):
        """Different run seed -> different reservoir selection."""
        data = {"labels": np.array([0.0, 1.0])}
        spec = ResampleSpec("labels", LabelsParams())

        r1 = PermutationReducer(reservoir_size=20)
        out1 = run(data, spec, RNG_STAT, n_iters=200, seed=42,
                    store="none", reducer=r1, alternative="two-sided")

        r2 = PermutationReducer(reservoir_size=20)
        out2 = run(data, spec, RNG_STAT, n_iters=200, seed=99,
                    store="none", reducer=r2, alternative="two-sided")

        # Reservoirs should differ (different null values AND selection)
        assert not np.array_equal(
            out1.reducer_state["reservoir"],
            out2.reducer_state["reservoir"],
        )
