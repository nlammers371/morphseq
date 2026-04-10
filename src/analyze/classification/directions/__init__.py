"""classification/directions — direction vector production and artifact management.

This package owns everything required to FIT, ASSEMBLE, and SAVE
ClassifierDirections artifacts (signed unit logistic regression coefficient
vectors, one per (feature_set, comparison, time_bin)).

Public API
----------
ClassifierDirections          — the persisted artifact dataclass (artifact.py)
fit_classifier_direction      — fit one binary direction vector (fit.py)
build_classifier_directions_payload — assemble multiple fit dicts into the artifact (build_payload.py)
extract_classifier_directions — lightweight entry point, no AUROC/permutations (extract.py)
make_vector_id / parse_vector_id — vector ID conventions (ids.py)

Design constraints
------------------
- engine/loop.py calls this package; it does not define direction logic.
- morphology_geometry imports only ClassifierDirections (from artifact.py);
  it never imports fit.py, build_payload.py, extract.py, or ids.py.
- No utils.py, no helpers.py. Every file is named by the kind of thing it contains.
"""

from .artifact import ClassifierDirections
from .build_payload import build_classifier_directions_payload
from .extract import extract_classifier_directions
from .fit import (
    DIRECTION_SPACE_RAW,
    REFIT_SCOPE_FULL_BIN,
    VECTOR_KIND_SIGNED_UNIT_COEF,
    fit_classifier_direction,
)
from .ids import make_vector_id, parse_vector_id

__all__ = [
    "ClassifierDirections",
    "build_classifier_directions_payload",
    "extract_classifier_directions",
    "fit_classifier_direction",
    "make_vector_id",
    "parse_vector_id",
    "DIRECTION_SPACE_RAW",
    "REFIT_SCOPE_FULL_BIN",
    "VECTOR_KIND_SIGNED_UNIT_COEF",
]
