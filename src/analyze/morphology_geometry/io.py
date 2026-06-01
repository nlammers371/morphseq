"""I/O gateway for morphology_geometry.

This is the ONLY file in morphology_geometry that imports from
analyze.classification. All other modules in this package accept
ValidatedDirections and do not touch the classification internals.

Public API
----------
load_classifier_directions(path, *, feature_set, ...) -> ValidatedDirections
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# The ONE permitted classification import in this package.
from analyze.classification.directions.artifact import ClassifierDirections

from analyze.morphology_geometry.validation import (
    ValidatedDirections,
    validate_classifier_directions,
)


def load_classifier_directions(
    path: str | Path,
    *,
    feature_set: str,
    required_comparison_ids: Sequence[str] | None = None,
    expected_bin_width: float | None = None,
    unit_norm_tol: float = 1e-6,
    check_preprocess_fingerprint: bool = True,
) -> ValidatedDirections:
    """Load and validate a ClassifierDirections artifact from disk.

    Expects the directory at *path* to contain:
      - ``classifier_directions.parquet``
      - ``classifier_directions_vectors.npz``

    Parameters
    ----------
    path : str or Path
        Directory containing the two artifact files.
    feature_set : str
        Which feature set to filter and validate.
    required_comparison_ids : sequence of str, optional
        Comparison IDs that must be present; missing IDs raise
        ClassifierDirectionContractError.
    expected_bin_width : float, optional
        If given, the inferred bin width must match within 1e-6 hpf.
    unit_norm_tol : float
        Tolerance for unit-norm check. Default 1e-6.
    check_preprocess_fingerprint : bool
        Whether to enforce a constant preprocess_fingerprint. Default True.

    Returns
    -------
    ValidatedDirections

    Raises
    ------
    ClassifierDirectionContractError
        If the artifact does not satisfy the geometry contract.
    FileNotFoundError
        If the expected artifact files are absent from *path*.
    """
    p = Path(path)
    parquet_path = p / "classifier_directions.parquet"
    npz_path = p / "classifier_directions_vectors.npz"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"classifier_directions.parquet not found in {p}. "
            "Run run_classification(..., save_classifier_directions=True) or "
            "extract_classifier_directions(..., save_dir=...) first."
        )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"classifier_directions_vectors.npz not found in {p}."
        )

    raw = ClassifierDirections.load(parquet_path, npz_path)
    return validate_classifier_directions(
        raw,
        feature_set=feature_set,
        required_comparison_ids=required_comparison_ids,
        expected_bin_width=expected_bin_width,
        unit_norm_tol=unit_norm_tol,
        check_preprocess_fingerprint=check_preprocess_fingerprint,
    )
