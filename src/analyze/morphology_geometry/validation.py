"""Geometry contract validator for ClassifierDirections artifacts.

This module is the front door of morphology_geometry. Every function in this
package accepts ValidatedDirections, not raw ClassifierDirections — so it is
impossible to run geometry math against an unchecked artifact.

validate_classifier_directions() is called once, in io.load_classifier_directions()
and at the top of run_morphology_geometry(). All failure modes raise
ClassifierDirectionContractError with a message that names the offending field
and the expected vs. observed value.

Public API
----------
ClassifierDirectionContractError  — exception raised on any schema violation
ValidatedDirections               — frozen view returned by the validator
validate_classifier_directions()  — the validator
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# The ONE classification import permitted anywhere in morphology_geometry.
from analyze.classification.directions.artifact import ClassifierDirections

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required metadata columns
# ---------------------------------------------------------------------------

_REQUIRED_METADATA_COLS = frozenset({
    "vector_id",
    "feature_set",
    "comparison_id",
    "positive_label",
    "negative_label",
    "time_bin_center",
    "n_pos",
    "n_neg",
    "coef_norm",
    "intercept",
    "sign_flipped",
    "centroid_dot",
    "direction_space",
    "preprocess_fingerprint",
})

_EXPECTED_DIRECTION_SPACE = "raw_feature_space"
_BIN_UNIFORMITY_TOL = 1e-6    # hpf
_UNIT_NORM_DEFAULT_TOL = 1e-6


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ClassifierDirectionContractError(ValueError):
    """Raised when a ClassifierDirections artifact violates the geometry contract.

    The message always names the offending field and the expected vs. observed
    value so the caller knows exactly what to fix.
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedDirections:
    """Validated, geometry-ready view of a ClassifierDirections artifact.

    All fields are computed once by validate_classifier_directions() and are
    guaranteed to satisfy the geometry contract. Downstream geometry functions
    accept this type, not raw ClassifierDirections.

    Attributes
    ----------
    metadata : pd.DataFrame
        Filtered to the requested feature_set, sorted by
        (comparison_id, time_bin_center). Index is reset.
    vectors : np.ndarray
        Shape (n_rows, n_features). Rows are aligned with metadata.
        All vectors are finite and unit-norm (within unit_norm_tol).
    feature_names : list[str]
        Authoritative column order for projection math. Length = n_features.
    feature_set : str
        The feature set this view covers.
    inferred_bin_width : float
        Bin width inferred from the uniform spacing of time_bin_center values.
    bin_centers : np.ndarray
        Sorted unique time_bin_center values, shape (n_bins,).
    preprocess_fingerprint : str
        SHA-256 fingerprint of (feature_names, direction_space, estimator_config).
        Constant across all rows in this feature_set.
    has_auroc : bool
        True if auroc_obs is present and non-NaN for at least one row.
        False when the artifact was produced by extract_classifier_directions.
    comparison_ids : tuple[str, ...]
        Sorted unique comparison_ids present in the filtered metadata.
    """

    metadata: pd.DataFrame
    vectors: np.ndarray
    feature_names: list[str]
    feature_set: str
    inferred_bin_width: float
    bin_centers: np.ndarray
    preprocess_fingerprint: str
    has_auroc: bool
    comparison_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_classifier_directions(
    directions: ClassifierDirections,
    *,
    feature_set: str,
    required_comparison_ids: Sequence[str] | None = None,
    expected_bin_width: float | None = None,
    unit_norm_tol: float = _UNIT_NORM_DEFAULT_TOL,
    check_preprocess_fingerprint: bool = True,
) -> ValidatedDirections:
    """Validate a ClassifierDirections artifact and return a ValidatedDirections view.

    Parameters
    ----------
    directions : ClassifierDirections
        Raw artifact to validate.
    feature_set : str
        Which feature set to filter and validate. Must be present in
        directions.feature_names.
    required_comparison_ids : sequence of str, optional
        If supplied, all listed comparison_ids must be present in the filtered
        metadata. Missing IDs raise ClassifierDirectionContractError.
    expected_bin_width : float, optional
        If supplied, the inferred bin width (from time_bin_center spacing) must
        match this value within _BIN_UNIFORMITY_TOL. Mismatch raises.
    unit_norm_tol : float
        Tolerance for |‖v‖ - 1| <= unit_norm_tol. Default 1e-6.
        Zero-norm vectors (coef_norm == 0) are warned and excluded from
        the returned vectors array.
    check_preprocess_fingerprint : bool
        If True (default), assert that preprocess_fingerprint is constant
        across all rows in the filtered metadata.

    Returns
    -------
    ValidatedDirections

    Raises
    ------
    ClassifierDirectionContractError on any schema violation.
    """
    # ------------------------------------------------------------------
    # 1. feature_set must be in feature_names
    # ------------------------------------------------------------------
    if feature_set not in directions.feature_names:
        raise ClassifierDirectionContractError(
            f"feature_set {feature_set!r} not found in ClassifierDirections.feature_names. "
            f"Available: {sorted(directions.feature_names)}"
        )
    feature_names = list(directions.feature_names[feature_set])
    if not feature_names:
        raise ClassifierDirectionContractError(
            f"feature_names[{feature_set!r}] is empty."
        )
    n_features = len(feature_names)

    # ------------------------------------------------------------------
    # 2. Required metadata columns
    # ------------------------------------------------------------------
    missing_cols = _REQUIRED_METADATA_COLS - set(directions.metadata.columns)
    if missing_cols:
        raise ClassifierDirectionContractError(
            f"ClassifierDirections metadata is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    # ------------------------------------------------------------------
    # 3. Filter to requested feature_set
    # ------------------------------------------------------------------
    meta = (
        directions.metadata[directions.metadata["feature_set"] == feature_set]
        .sort_values(["comparison_id", "time_bin_center"])
        .reset_index(drop=True)
        .copy()
    )
    if meta.empty:
        raise ClassifierDirectionContractError(
            f"No rows in ClassifierDirections metadata for feature_set={feature_set!r}."
        )

    # ------------------------------------------------------------------
    # 4. vector_id integrity: every metadata row must have a vector
    # ------------------------------------------------------------------
    meta_ids = set(meta["vector_id"].astype(str))
    npz_ids = set(directions.vectors.keys())

    missing_in_npz = meta_ids - npz_ids
    if missing_in_npz:
        raise ClassifierDirectionContractError(
            f"vector_ids in metadata but missing from NPZ: {sorted(missing_in_npz)}"
        )

    # ------------------------------------------------------------------
    # 5. Stacking: build the (n_rows, n_features) matrix in metadata order
    # ------------------------------------------------------------------
    raw_vectors: list[np.ndarray] = []
    drop_mask: list[bool] = []   # True = row should be dropped (zero-norm)

    for _, row in meta.iterrows():
        vid = str(row["vector_id"])
        vec = np.asarray(directions.vectors[vid], dtype=np.float64)

        # Shape check
        if vec.shape != (n_features,):
            raise ClassifierDirectionContractError(
                f"Vector {vid!r}: expected shape ({n_features},), got {vec.shape}. "
                f"feature_names for {feature_set!r} has {n_features} entries."
            )

        # Finite check
        if not np.all(np.isfinite(vec)):
            raise ClassifierDirectionContractError(
                f"Vector {vid!r} contains non-finite values (NaN or Inf)."
            )

        coef_norm = float(row["coef_norm"])
        norm = float(np.linalg.norm(vec))

        if coef_norm == 0.0:
            # Zero-norm vector: allowed only if coef_norm==0 in metadata
            warnings.warn(
                f"Vector {vid!r} has coef_norm=0 in metadata and zero norm. "
                "This row will be excluded from ValidatedDirections.vectors.",
                stacklevel=3,
            )
            drop_mask.append(True)
        elif abs(norm - 1.0) > unit_norm_tol:
            raise ClassifierDirectionContractError(
                f"Vector {vid!r}: expected unit norm, got ‖v‖={norm:.8f} "
                f"(|‖v‖ - 1| = {abs(norm - 1.0):.2e} > tol={unit_norm_tol:.2e})."
            )
        else:
            drop_mask.append(False)

        raw_vectors.append(vec)

    # Filter out zero-norm rows from both metadata and vectors
    keep = [not d for d in drop_mask]
    meta = meta[keep].reset_index(drop=True)
    vectors = np.stack([v for v, k in zip(raw_vectors, keep) if k], axis=0) if any(keep) else np.empty((0, n_features))

    if meta.empty:
        raise ClassifierDirectionContractError(
            f"All direction vectors for feature_set={feature_set!r} have zero norm. "
            "Cannot proceed with geometry analysis."
        )

    # ------------------------------------------------------------------
    # 6. direction_space must be the expected value
    # ------------------------------------------------------------------
    bad_spaces = meta[meta["direction_space"] != _EXPECTED_DIRECTION_SPACE]["direction_space"].unique()
    if len(bad_spaces) > 0:
        raise ClassifierDirectionContractError(
            f"direction_space must be {_EXPECTED_DIRECTION_SPACE!r}. "
            f"Found: {sorted(bad_spaces)}"
        )

    # ------------------------------------------------------------------
    # 7. preprocess_fingerprint must be constant within this feature_set
    # ------------------------------------------------------------------
    fingerprints = meta["preprocess_fingerprint"].unique()
    if check_preprocess_fingerprint and len(fingerprints) > 1:
        raise ClassifierDirectionContractError(
            f"preprocess_fingerprint is not constant across rows for "
            f"feature_set={feature_set!r}. Found {len(fingerprints)} distinct values: "
            f"{sorted(fingerprints)[:3]}{'...' if len(fingerprints) > 3 else ''}. "
            "This means different comparisons used different feature orderings or "
            "estimator configs — their directions cannot be meaningfully combined."
        )
    preprocess_fingerprint = str(fingerprints[0])

    # ------------------------------------------------------------------
    # 8. required_comparison_ids
    # ------------------------------------------------------------------
    present_ids = set(meta["comparison_id"].unique())
    if required_comparison_ids is not None:
        missing_ids = set(required_comparison_ids) - present_ids
        if missing_ids:
            raise ClassifierDirectionContractError(
                f"Required comparison_ids not found in the artifact for "
                f"feature_set={feature_set!r}: {sorted(missing_ids)}. "
                f"Available: {sorted(present_ids)}"
            )

    # ------------------------------------------------------------------
    # 9. Bin uniformity and optional bin-width check
    # ------------------------------------------------------------------
    bin_centers = np.sort(meta["time_bin_center"].unique())
    if len(bin_centers) < 2:
        # Only one bin — cannot infer bin width from spacing; use expected if given
        if expected_bin_width is not None:
            inferred_bin_width = float(expected_bin_width)
        else:
            inferred_bin_width = float("nan")
    else:
        diffs = np.diff(bin_centers)
        if np.any(np.abs(diffs - diffs[0]) > _BIN_UNIFORMITY_TOL):
            raise ClassifierDirectionContractError(
                f"time_bin_center values for feature_set={feature_set!r} are not "
                f"uniformly spaced. Differences: {diffs.tolist()}. "
                "This suggests mixed bin widths — geometry analysis requires a "
                "single consistent bin width."
            )
        inferred_bin_width = float(diffs[0])

    if expected_bin_width is not None and not np.isnan(inferred_bin_width):
        if abs(inferred_bin_width - expected_bin_width) > _BIN_UNIFORMITY_TOL:
            raise ClassifierDirectionContractError(
                f"Bin width mismatch for feature_set={feature_set!r}: "
                f"expected {expected_bin_width} hpf but inferred "
                f"{inferred_bin_width:.4f} hpf from time_bin_center spacing. "
                "Ensure run_morphology_geometry bin_width matches the classification run."
            )

    # ------------------------------------------------------------------
    # 10. has_auroc: check whether auroc_obs is populated
    # ------------------------------------------------------------------
    has_auroc = (
        "auroc_obs" in meta.columns
        and meta["auroc_obs"].notna().any()
    )

    return ValidatedDirections(
        metadata=meta,
        vectors=vectors,
        feature_names=feature_names,
        feature_set=feature_set,
        inferred_bin_width=inferred_bin_width,
        bin_centers=bin_centers,
        preprocess_fingerprint=preprocess_fingerprint,
        has_auroc=has_auroc,
        comparison_ids=tuple(sorted(present_ids)),
    )
