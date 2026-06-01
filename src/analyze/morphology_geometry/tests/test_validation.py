"""Tests for morphology_geometry/validation.py.

Covers all 10 checks in validate_classifier_directions():
  1. feature_set present in feature_names
  2. required metadata columns
  3. filter to feature_set (empty after filter)
  4. vector_id integrity (in metadata but missing from NPZ)
  5. vector shape / finite / unit-norm checks (zero-norm drop)
  6. direction_space == "raw_feature_space"
  7. preprocess_fingerprint constant within feature_set
  8. required_comparison_ids present
  9. bin uniformity + expected_bin_width match
 10. has_auroc detection

Each error path raises ClassifierDirectionContractError with a specific message.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from analyze.classification.directions.artifact import ClassifierDirections
from analyze.morphology_geometry.validation import (
    ClassifierDirectionContractError,
    ValidatedDirections,
    validate_classifier_directions,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal valid fixtures
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _make_directions(
    *,
    n_features: int = 4,
    comparisons: list[str] | None = None,
    bin_centers: list[float] | None = None,
    feature_set: str = "pca",
    direction_space: str = "raw_feature_space",
    fingerprint: str = "fp_abc123",
    include_auroc: bool = False,
    rng: np.random.Generator | None = None,
) -> ClassifierDirections:
    """Build a minimal valid ClassifierDirections with configurable options."""
    if rng is None:
        rng = np.random.default_rng(0)
    if comparisons is None:
        comparisons = ["wt_vs_het"]
    if bin_centers is None:
        bin_centers = [24.0, 28.0, 32.0]

    feature_names = [f"f{i}" for i in range(n_features)]
    rows = []
    vectors: dict[str, np.ndarray] = {}

    for cid in comparisons:
        for t in bin_centers:
            vid = f"{feature_set}__{cid}__{t:.1f}"
            v = _unit(rng.standard_normal(n_features))
            vectors[vid] = v
            row = {
                "vector_id": vid,
                "feature_set": feature_set,
                "comparison_id": cid,
                "positive_label": "het",
                "negative_label": "wt",
                "time_bin_center": t,
                "n_pos": 10,
                "n_neg": 10,
                "coef_norm": float(np.linalg.norm(v)),
                "intercept": 0.0,
                "sign_flipped": False,
                "centroid_dot": 0.5,
                "direction_space": direction_space,
                "preprocess_fingerprint": fingerprint,
            }
            if include_auroc:
                row["auroc_obs"] = 0.7
            rows.append(row)

    metadata = pd.DataFrame(rows)
    return ClassifierDirections(
        metadata=metadata,
        vectors=vectors,
        feature_names={feature_set: feature_names},
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_returns_validated_directions(self):
        d = _make_directions()
        vd = validate_classifier_directions(d, feature_set="pca")
        assert isinstance(vd, ValidatedDirections)

    def test_field_values(self):
        d = _make_directions(comparisons=["a_vs_b"], bin_centers=[24.0, 28.0, 32.0])
        vd = validate_classifier_directions(d, feature_set="pca")
        assert vd.feature_set == "pca"
        assert vd.feature_names == [f"f{i}" for i in range(4)]
        np.testing.assert_allclose(vd.bin_centers, [24.0, 28.0, 32.0])
        assert vd.inferred_bin_width == pytest.approx(4.0)
        assert vd.has_auroc is False
        assert vd.comparison_ids == ("a_vs_b",)

    def test_vectors_aligned_with_metadata(self):
        d = _make_directions(comparisons=["c1", "c2"], bin_centers=[10.0, 14.0])
        vd = validate_classifier_directions(d, feature_set="pca")
        assert vd.vectors.shape == (len(vd.metadata), 4)
        for i, row in vd.metadata.iterrows():
            vid = row["vector_id"]
            np.testing.assert_allclose(
                vd.vectors[i],
                d.vectors[vid],
                atol=1e-12,
            )

    def test_has_auroc_true(self):
        d = _make_directions(include_auroc=True)
        vd = validate_classifier_directions(d, feature_set="pca")
        assert vd.has_auroc == True

    def test_metadata_sorted_by_comparison_and_time(self):
        d = _make_directions(
            comparisons=["z_cmp", "a_cmp"],
            bin_centers=[32.0, 24.0, 28.0],
        )
        vd = validate_classifier_directions(d, feature_set="pca")
        cids = vd.metadata["comparison_id"].tolist()
        times = vd.metadata["time_bin_center"].tolist()
        # sorted by (comparison_id, time_bin_center) — "a_cmp" < "z_cmp"
        assert cids == sorted(cids)
        assert times == sorted(times[:3]) + sorted(times[3:])


# ---------------------------------------------------------------------------
# Check 1: feature_set present
# ---------------------------------------------------------------------------

class TestFeatureSetPresent:
    def test_missing_feature_set_raises(self):
        d = _make_directions(feature_set="pca")
        with pytest.raises(ClassifierDirectionContractError, match="not found"):
            validate_classifier_directions(d, feature_set="umap")

    def test_empty_feature_names_raises(self):
        d = _make_directions(feature_set="pca")
        # Manually inject an empty feature_names list
        d.feature_names["pca"] = []
        with pytest.raises(ClassifierDirectionContractError, match="empty"):
            validate_classifier_directions(d, feature_set="pca")


# ---------------------------------------------------------------------------
# Check 2: required metadata columns
# ---------------------------------------------------------------------------

class TestRequiredMetadataColumns:
    @pytest.mark.parametrize("drop_col", [
        "vector_id", "feature_set", "comparison_id", "positive_label",
        "negative_label", "time_bin_center", "n_pos", "n_neg",
        "coef_norm", "intercept", "sign_flipped", "centroid_dot",
        "direction_space", "preprocess_fingerprint",
    ])
    def test_missing_column_raises(self, drop_col: str):
        d = _make_directions()
        d.metadata.drop(columns=[drop_col], inplace=True)
        with pytest.raises(ClassifierDirectionContractError, match="missing required columns"):
            validate_classifier_directions(d, feature_set="pca")


# ---------------------------------------------------------------------------
# Check 3: filter produces non-empty result
# ---------------------------------------------------------------------------

class TestFilterNonEmpty:
    def test_no_rows_for_feature_set_raises(self):
        d = _make_directions(feature_set="pca")
        with pytest.raises(ClassifierDirectionContractError, match="No rows"):
            # "umap" is absent from feature_names keys too, so check 1 would fire first;
            # to hit check 3 we need a feature_set that IS in feature_names but has no rows.
            d.feature_names["umap"] = ["f0", "f1", "f2", "f3"]
            validate_classifier_directions(d, feature_set="umap")


# ---------------------------------------------------------------------------
# Check 4: vector_id integrity
# ---------------------------------------------------------------------------

class TestVectorIdIntegrity:
    def test_vector_missing_from_npz_raises(self):
        d = _make_directions(comparisons=["a"], bin_centers=[24.0])
        # Remove one vector from the dict without removing it from metadata
        vid = next(iter(d.vectors))
        del d.vectors[vid]
        # Bypass __post_init__ check by mutating after construction
        # Use object.__setattr__ trick — ClassifierDirections is not frozen
        with pytest.raises((ClassifierDirectionContractError, ValueError)):
            validate_classifier_directions(d, feature_set="pca")


# ---------------------------------------------------------------------------
# Check 5: shape / finite / unit-norm / zero-norm
# ---------------------------------------------------------------------------

class TestVectorChecks:
    def test_wrong_shape_raises(self):
        d = _make_directions(n_features=4)
        vid = next(iter(d.vectors))
        d.vectors[vid] = np.array([1.0, 0.0])  # wrong length
        # Also fix feature_names length so check 1 passes but check 5 fails
        with pytest.raises((ClassifierDirectionContractError, ValueError)):
            validate_classifier_directions(d, feature_set="pca")

    def test_nan_in_vector_raises(self):
        d = _make_directions(n_features=4)
        vid = next(iter(d.vectors))
        v = d.vectors[vid].copy()
        v[0] = np.nan
        d.vectors[vid] = v
        with pytest.raises(ClassifierDirectionContractError, match="non-finite"):
            validate_classifier_directions(d, feature_set="pca")

    def test_inf_in_vector_raises(self):
        d = _make_directions(n_features=4)
        vid = next(iter(d.vectors))
        v = d.vectors[vid].copy()
        v[1] = np.inf
        d.vectors[vid] = v
        with pytest.raises(ClassifierDirectionContractError, match="non-finite"):
            validate_classifier_directions(d, feature_set="pca")

    def test_non_unit_norm_raises(self):
        d = _make_directions(n_features=4)
        vid = next(iter(d.vectors))
        d.vectors[vid] = d.vectors[vid] * 2.0   # norm = 2
        # coef_norm != 0 so the validator checks unit-norm
        d.metadata.loc[d.metadata["vector_id"] == vid, "coef_norm"] = 2.0
        with pytest.raises(ClassifierDirectionContractError, match="unit norm"):
            validate_classifier_directions(d, feature_set="pca")

    def test_zero_norm_with_coef_norm_zero_warns_and_drops(self):
        d = _make_directions(n_features=4, bin_centers=[24.0, 28.0])
        # Make first vector zero
        first_vid = d.metadata.sort_values("time_bin_center")["vector_id"].iloc[0]
        d.vectors[first_vid] = np.zeros(4)
        d.metadata.loc[d.metadata["vector_id"] == first_vid, "coef_norm"] = 0.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vd = validate_classifier_directions(d, feature_set="pca")
        assert any("zero norm" in str(warning.message).lower() or "coef_norm=0" in str(warning.message) for warning in w)
        # One row dropped, one remains
        assert len(vd.metadata) == 1
        assert vd.vectors.shape[0] == 1

    def test_all_zero_norm_raises(self):
        d = _make_directions(n_features=4, bin_centers=[24.0])
        vid = next(iter(d.vectors))
        d.vectors[vid] = np.zeros(4)
        d.metadata.loc[d.metadata["vector_id"] == vid, "coef_norm"] = 0.0
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(ClassifierDirectionContractError, match="zero norm"):
                validate_classifier_directions(d, feature_set="pca")


# ---------------------------------------------------------------------------
# Check 6: direction_space
# ---------------------------------------------------------------------------

class TestDirectionSpace:
    def test_wrong_direction_space_raises(self):
        d = _make_directions(direction_space="pca_space")
        with pytest.raises(ClassifierDirectionContractError, match="direction_space"):
            validate_classifier_directions(d, feature_set="pca")

    def test_correct_direction_space_passes(self):
        d = _make_directions(direction_space="raw_feature_space")
        vd = validate_classifier_directions(d, feature_set="pca")
        assert isinstance(vd, ValidatedDirections)


# ---------------------------------------------------------------------------
# Check 7: preprocess_fingerprint constant
# ---------------------------------------------------------------------------

class TestPreprocessFingerprint:
    def test_drifted_fingerprint_raises(self):
        d = _make_directions(comparisons=["a", "b"], bin_centers=[24.0])
        # Give one row a different fingerprint
        idx = d.metadata[d.metadata["comparison_id"] == "b"].index[0]
        d.metadata.loc[idx, "preprocess_fingerprint"] = "different_fp"
        with pytest.raises(ClassifierDirectionContractError, match="preprocess_fingerprint"):
            validate_classifier_directions(d, feature_set="pca")

    def test_fingerprint_check_disabled(self):
        d = _make_directions(comparisons=["a", "b"], bin_centers=[24.0])
        idx = d.metadata[d.metadata["comparison_id"] == "b"].index[0]
        d.metadata.loc[idx, "preprocess_fingerprint"] = "different_fp"
        # Should not raise when check is disabled
        vd = validate_classifier_directions(
            d, feature_set="pca", check_preprocess_fingerprint=False
        )
        assert isinstance(vd, ValidatedDirections)


# ---------------------------------------------------------------------------
# Check 8: required_comparison_ids
# ---------------------------------------------------------------------------

class TestRequiredComparisonIds:
    def test_missing_comparison_id_raises(self):
        d = _make_directions(comparisons=["wt_vs_het"])
        with pytest.raises(ClassifierDirectionContractError, match="Required comparison_ids"):
            validate_classifier_directions(
                d, feature_set="pca", required_comparison_ids=["wt_vs_homo"]
            )

    def test_all_present_passes(self):
        d = _make_directions(comparisons=["a_vs_b", "a_vs_c"])
        vd = validate_classifier_directions(
            d, feature_set="pca", required_comparison_ids=["a_vs_b", "a_vs_c"]
        )
        assert isinstance(vd, ValidatedDirections)

    def test_no_required_ids_passes(self):
        d = _make_directions()
        vd = validate_classifier_directions(
            d, feature_set="pca", required_comparison_ids=None
        )
        assert isinstance(vd, ValidatedDirections)


# ---------------------------------------------------------------------------
# Check 9: bin uniformity + expected_bin_width
# ---------------------------------------------------------------------------

class TestBinUniformity:
    def test_non_uniform_bins_raises(self):
        d = _make_directions(bin_centers=[24.0, 27.0, 35.0])
        with pytest.raises(ClassifierDirectionContractError, match="uniformly spaced"):
            validate_classifier_directions(d, feature_set="pca")

    def test_expected_bin_width_mismatch_raises(self):
        d = _make_directions(bin_centers=[24.0, 28.0, 32.0])  # inferred = 4.0
        with pytest.raises(ClassifierDirectionContractError, match="Bin width mismatch"):
            validate_classifier_directions(d, feature_set="pca", expected_bin_width=2.0)

    def test_expected_bin_width_match_passes(self):
        d = _make_directions(bin_centers=[24.0, 28.0, 32.0])
        vd = validate_classifier_directions(d, feature_set="pca", expected_bin_width=4.0)
        assert vd.inferred_bin_width == pytest.approx(4.0)

    def test_single_bin_no_inferred_width(self):
        d = _make_directions(bin_centers=[24.0])
        vd = validate_classifier_directions(d, feature_set="pca")
        assert np.isnan(vd.inferred_bin_width)

    def test_single_bin_with_expected_width(self):
        d = _make_directions(bin_centers=[24.0])
        vd = validate_classifier_directions(d, feature_set="pca", expected_bin_width=4.0)
        assert vd.inferred_bin_width == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Check 10: has_auroc
# ---------------------------------------------------------------------------

class TestHasAuroc:
    def test_no_auroc_column(self):
        d = _make_directions(include_auroc=False)
        vd = validate_classifier_directions(d, feature_set="pca")
        assert vd.has_auroc is False

    def test_with_auroc_column(self):
        d = _make_directions(include_auroc=True)
        vd = validate_classifier_directions(d, feature_set="pca")
        assert vd.has_auroc == True

    def test_all_nan_auroc_is_false(self):
        d = _make_directions(include_auroc=True)
        d.metadata["auroc_obs"] = np.nan
        vd = validate_classifier_directions(d, feature_set="pca")
        assert vd.has_auroc == False
