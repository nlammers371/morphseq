"""Tests for _aggregator.py."""

import numpy as np
import pytest

from analyze.utils.resampling._spec import (
    ResampleSpec, IndicesParams, LabelsParams, GroupsParams,
)
from analyze.utils.resampling._statistic import Statistic
from analyze.utils.resampling._engine import run
from analyze.utils.resampling._aggregator import (
    aggregate, BootstrapSummary, PermutationSummary,
)
from analyze.utils.resampling._reducer import PermutationReducer


def _mean_stat(data, rng=None):
    return float(np.mean(data["labels"]))


MEAN = Statistic(name="mean", fn=_mean_stat)


class TestBootstrapSummary:
    def test_basic(self):
        data = {"labels": np.arange(20, dtype=float)}
        spec = ResampleSpec("indices", IndicesParams(replacement=False, frac=0.8))
        out = run(data, spec, MEAN, n_iters=100, seed=42)
        summary = aggregate(out)
        assert isinstance(summary, BootstrapSummary)
        assert summary.ci_method == "percentile"
        assert summary.ci_is_exact is True
        assert summary.ci_low <= summary.observed <= summary.ci_high
        assert summary.n_success == 100

    def test_custom_alpha(self):
        data = {"labels": np.arange(20, dtype=float)}
        spec = ResampleSpec("indices", IndicesParams(replacement=True))
        out = run(data, spec, MEAN, n_iters=200, seed=42)
        s1 = aggregate(out, alpha=0.05)
        s2 = aggregate(out, alpha=0.01)
        # Wider CI for smaller alpha
        assert (s2.ci_high - s2.ci_low) >= (s1.ci_high - s1.ci_low)


class TestPermutationSummary:
    def test_labels_permutation(self):
        data = {"labels": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        spec = ResampleSpec("labels", LabelsParams())
        out = run(data, spec, MEAN, n_iters=100, seed=42, alternative="two-sided")
        summary = aggregate(out)
        assert isinstance(summary, PermutationSummary)
        assert summary.alternative == "two-sided"
        assert 0 < summary.pvalue <= 1
        assert summary.null_distribution is not None
        assert len(summary.null_distribution) == 100

    def test_groups_permutation(self):
        data = {
            "X1": np.array([10.0, 11.0, 12.0]),
            "X2": np.array([1.0, 2.0, 3.0]),
        }

        def diff_means(d, rng=None):
            return float(np.mean(d["X1"]) - np.mean(d["X2"]))

        spec = ResampleSpec("groups", GroupsParams())
        out = run(data, spec, diff_means, n_iters=200, seed=42, alternative="greater")
        summary = aggregate(out)
        assert isinstance(summary, PermutationSummary)
        assert summary.pvalue < 0.1  # should be significant

    def test_reducer_only_summary(self):
        data = {"labels": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        spec = ResampleSpec("labels", LabelsParams())
        reducer = PermutationReducer(alternative="two-sided")
        out = run(data, spec, MEAN, n_iters=100, seed=42,
                  store="none", reducer=reducer, alternative="two-sided")
        summary = aggregate(out)
        assert isinstance(summary, PermutationSummary)
        assert summary.null_distribution is None
        assert 0 < summary.pvalue <= 1

    def test_override_alternative_with_stored(self):
        # Use groups with clearly asymmetric distributions so greater/less differ
        data = {
            "X1": np.array([10.0, 11.0, 12.0]),
            "X2": np.array([1.0, 2.0, 3.0]),
        }

        def diff_means(d, rng=None):
            return float(np.mean(d["X1"]) - np.mean(d["X2"]))

        spec = ResampleSpec("groups", GroupsParams())
        out = run(data, spec, diff_means, n_iters=200, seed=42, alternative="greater")
        s1 = aggregate(out)
        s2 = aggregate(out, alternative="less")
        assert s1.alternative == "greater"
        assert s2.alternative == "less"
        # p-values should be different for asymmetric data
        assert s1.pvalue != s2.pvalue

    def test_override_alternative_reducer_only_fails(self):
        data = {"labels": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        spec = ResampleSpec("labels", LabelsParams())
        reducer = PermutationReducer()
        out = run(data, spec, MEAN, n_iters=100, seed=42,
                  store="none", reducer=reducer, alternative="greater")
        with pytest.raises(ValueError, match="Cannot override"):
            aggregate(out, alternative="less")


class TestToPermutationResult:
    def test_backward_compat(self):
        data = {
            "X1": np.array([10.0, 11.0, 12.0]),
            "X2": np.array([1.0, 2.0, 3.0]),
        }

        def diff_means(d, rng=None):
            return float(np.mean(d["X1"]) - np.mean(d["X2"]))

        spec = ResampleSpec("groups", GroupsParams())
        out = run(data, spec, diff_means, n_iters=50, seed=42, alternative="greater")
        summary = aggregate(out)
        pr = summary.to_permutation_result()

        assert pr.statistic_name == "diff_means"
        assert pr.observed == summary.observed
        assert pr.pvalue == summary.pvalue
        assert hasattr(pr, "null_mean")
