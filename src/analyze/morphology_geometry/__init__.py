"""morphology_geometry — geometry analysis on classifier direction vectors.

Public API
----------
IO + validation
    load_classifier_directions   (path, *, feature_set, ...) -> ValidatedDirections
    validate_classifier_directions  (ClassifierDirections, ...) -> ValidatedDirections
    ValidatedDirections
    ClassifierDirectionContractError

Vectors
    cosine_alignment
    axis_alignment
    direction_matrix
    weighted_axis

Projection
    project_binned_features
"""

from analyze.morphology_geometry.io import load_classifier_directions
from analyze.morphology_geometry.projection import project_binned_features
from analyze.morphology_geometry.validation import (
    ClassifierDirectionContractError,
    ValidatedDirections,
    validate_classifier_directions,
)
from analyze.morphology_geometry.vectors import (
    axis_alignment,
    cosine_alignment,
    direction_matrix,
    weighted_axis,
)

__all__ = [
    # IO + validation
    "load_classifier_directions",
    "validate_classifier_directions",
    "ValidatedDirections",
    "ClassifierDirectionContractError",
    # vectors
    "cosine_alignment",
    "axis_alignment",
    "direction_matrix",
    "weighted_axis",
    # projection
    "project_binned_features",
]
