"""Tests for morphology_geometry/io.py."""

from __future__ import annotations

import numpy as np
import pytest

from analyze.morphology_geometry.io import load_classifier_directions
from analyze.morphology_geometry.validation import (
    ClassifierDirectionContractError,
    ValidatedDirections,
)
from .conftest import FEATURE_SET, BIN_WIDTH


class TestLoadClassifierDirections:
    def test_returns_validated_directions(self, minimal_directions, tmp_path):
        minimal_directions.save(tmp_path / "classifier_directions_vectors.npz")
        minimal_directions.metadata.to_parquet(
            tmp_path / "classifier_directions.parquet", index=False
        )
        vd = load_classifier_directions(tmp_path, feature_set=FEATURE_SET)
        assert isinstance(vd, ValidatedDirections)
        assert vd.feature_set == FEATURE_SET

    def test_round_trip_bin_width(self, minimal_directions, tmp_path):
        minimal_directions.save(tmp_path / "classifier_directions_vectors.npz")
        minimal_directions.metadata.to_parquet(
            tmp_path / "classifier_directions.parquet", index=False
        )
        vd = load_classifier_directions(
            tmp_path, feature_set=FEATURE_SET, expected_bin_width=BIN_WIDTH
        )
        assert vd.inferred_bin_width == pytest.approx(BIN_WIDTH)

    def test_wrong_feature_set_raises_contract_error(self, minimal_directions, tmp_path):
        minimal_directions.save(tmp_path / "classifier_directions_vectors.npz")
        minimal_directions.metadata.to_parquet(
            tmp_path / "classifier_directions.parquet", index=False
        )
        with pytest.raises(ClassifierDirectionContractError, match="not found"):
            load_classifier_directions(tmp_path, feature_set="nonexistent_fs")

    def test_missing_parquet_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="classifier_directions.parquet"):
            load_classifier_directions(tmp_path, feature_set=FEATURE_SET)

    def test_missing_npz_raises_file_not_found(self, minimal_directions, tmp_path):
        minimal_directions.metadata.to_parquet(
            tmp_path / "classifier_directions.parquet", index=False
        )
        with pytest.raises(FileNotFoundError, match="classifier_directions_vectors.npz"):
            load_classifier_directions(tmp_path, feature_set=FEATURE_SET)

    def test_required_comparison_ids_forwarded(self, minimal_directions, tmp_path):
        minimal_directions.save(tmp_path / "classifier_directions_vectors.npz")
        minimal_directions.metadata.to_parquet(
            tmp_path / "classifier_directions.parquet", index=False
        )
        with pytest.raises(ClassifierDirectionContractError, match="Required comparison_ids"):
            load_classifier_directions(
                tmp_path,
                feature_set=FEATURE_SET,
                required_comparison_ids=["does_not_exist"],
            )

    def test_vectors_aligned_with_metadata(self, minimal_directions, tmp_path):
        minimal_directions.save(tmp_path / "classifier_directions_vectors.npz")
        minimal_directions.metadata.to_parquet(
            tmp_path / "classifier_directions.parquet", index=False
        )
        vd = load_classifier_directions(tmp_path, feature_set=FEATURE_SET)
        assert vd.vectors.shape == (len(vd.metadata), len(vd.feature_names))
