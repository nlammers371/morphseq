"""Tests for _preflight.py."""

import numpy as np
import pytest

from analyze.utils.resampling._spec import (
    ResampleSpec, IndicesParams, LabelsParams, GroupsParams,
)
from analyze.utils.resampling._statistic import Statistic
from analyze.utils.resampling._preflight import preflight
from analyze.utils.resampling._reducer import PermutationReducer


def _scalar_stat(data, rng=None):
    return 1.0


def _array_stat(data, rng=None):
    return np.array([1.0, 2.0])


def _dict_stat(data, rng=None):
    return {"a": 1.0, "b": 2.0}


SCALAR = Statistic(name="scalar", fn=_scalar_stat)
ARRAY = Statistic(name="array", fn=_array_stat)
DICT = Statistic(name="dict", fn=_dict_stat)


class TestGroupsRestrictions:
    def test_groups_rejects_within(self):
        spec = ResampleSpec("groups", GroupsParams(), within_key="t")
        data = {"X1": np.array([1]), "X2": np.array([2]), "t": np.array([0])}
        with pytest.raises(ValueError, match="not yet supported"):
            preflight(data, spec, SCALAR)

    def test_groups_rejects_unit(self):
        spec = ResampleSpec("groups", GroupsParams(), unit_key="u")
        data = {"X1": np.array([1]), "X2": np.array([2]), "u": np.array([0])}
        with pytest.raises(ValueError, match="not yet supported"):
            preflight(data, spec, SCALAR)


class TestRequiredKeys:
    def test_labels_mode_requires_labels(self):
        spec = ResampleSpec("labels", LabelsParams())
        with pytest.raises(ValueError, match="data\\['labels'\\]"):
            preflight({"n": 10}, spec, SCALAR)

    def test_groups_mode_requires_keys(self):
        spec = ResampleSpec("groups", GroupsParams(a_key="A", b_key="B"))
        with pytest.raises(ValueError, match="data\\['A'\\]"):
            preflight({}, spec, SCALAR)

    def test_unit_key_missing(self):
        spec = ResampleSpec("labels", LabelsParams(), unit_key="uid")
        data = {"labels": np.array([1, 2])}
        with pytest.raises(ValueError, match="unit_key"):
            preflight(data, spec, SCALAR)

    def test_within_key_missing(self):
        spec = ResampleSpec("labels", LabelsParams(), within_key="strat")
        data = {"labels": np.array([1, 2])}
        with pytest.raises(ValueError, match="within_key"):
            preflight(data, spec, SCALAR)

    def test_inconsistent_lengths(self):
        spec = ResampleSpec("labels", LabelsParams(), unit_key="uid")
        data = {"labels": np.array([1, 2, 3]), "uid": np.array([1, 2])}
        with pytest.raises(ValueError, match="inconsistent lengths"):
            preflight(data, spec, SCALAR)


class TestPopulationSize:
    def test_indices_needs_population(self):
        spec = ResampleSpec("indices", IndicesParams(replacement=False))
        with pytest.raises(ValueError, match="Cannot determine population"):
            preflight({}, spec, SCALAR)

    def test_indices_from_labels(self):
        spec = ResampleSpec("indices", IndicesParams(replacement=False))
        data = {"labels": np.arange(10)}
        preflight(data, spec, SCALAR)  # should not raise

    def test_indices_from_n(self):
        spec = ResampleSpec("indices", IndicesParams(replacement=False))
        data = {"n": 10}
        preflight(data, spec, SCALAR)  # should not raise


class TestUnitConsistency:
    def test_within_not_constant_per_unit(self):
        data = {
            "labels": np.array([1, 1, 2, 2]),
            "uid": np.array(["A", "A", "B", "B"]),
            "strat": np.array(["s1", "s2", "s1", "s1"]),  # A has mixed strata
        }
        spec = ResampleSpec(
            "labels", LabelsParams(), unit_key="uid", within_key="strat",
        )
        with pytest.raises(ValueError, match="not constant"):
            preflight(data, spec, SCALAR)


class TestLabelConsistency:
    def test_labels_not_constant_per_unit(self):
        data = {
            "labels": np.array([1, 2, 3, 3]),
            "uid": np.array(["A", "A", "B", "B"]),
        }
        spec = ResampleSpec("labels", LabelsParams(), unit_key="uid")
        with pytest.raises(ValueError, match="not constant"):
            preflight(data, spec, SCALAR)


class TestStrataViability:
    def test_all_singletons_error(self):
        data = {
            "labels": np.array([1, 2, 3]),
            "strat": np.array(["a", "b", "c"]),
        }
        spec = ResampleSpec("labels", LabelsParams(), within_key="strat")
        with pytest.raises(ValueError, match="singletons"):
            preflight(data, spec, SCALAR)

    def test_some_singletons_ok(self):
        data = {
            "labels": np.array([1, 2, 3, 4]),
            "strat": np.array(["a", "a", "b", "c"]),
        }
        spec = ResampleSpec("labels", LabelsParams(), within_key="strat")
        preflight(data, spec, SCALAR)  # should not raise


class TestDryRun:
    def test_failing_statistic(self):
        def bad_fn(data, rng=None):
            raise RuntimeError("boom")

        stat = Statistic(name="bad", fn=bad_fn)
        data = {"labels": np.arange(5)}
        spec = ResampleSpec("labels", LabelsParams())
        with pytest.raises(ValueError, match="dry run failed"):
            preflight(data, spec, stat)

    def test_non_numeric_output(self):
        def str_fn(data, rng=None):
            return "hello"

        stat = Statistic(name="str", fn=str_fn)
        data = {"labels": np.arange(5)}
        spec = ResampleSpec("labels", LabelsParams())
        with pytest.raises(ValueError, match="scalar, ndarray, or dict"):
            preflight(data, spec, stat)


class TestReducerCompat:
    def test_reducer_rejects_array(self):
        data = {"labels": np.arange(5)}
        spec = ResampleSpec("labels", LabelsParams())
        reducer = PermutationReducer()
        with pytest.raises(ValueError, match="scalar outputs only"):
            preflight(data, spec, ARRAY, reducer=reducer)

    def test_reducer_accepts_scalar(self):
        data = {"labels": np.arange(5)}
        spec = ResampleSpec("labels", LabelsParams())
        reducer = PermutationReducer()
        preflight(data, spec, SCALAR, reducer=reducer)  # should not raise


class TestStorePolicy:
    def test_indices_store_none_error(self):
        data = {"labels": np.arange(5)}
        spec = ResampleSpec("indices", IndicesParams(replacement=False))
        with pytest.raises(ValueError, match="store='all'"):
            preflight(data, spec, SCALAR, store="none")
