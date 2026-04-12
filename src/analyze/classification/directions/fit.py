"""Fit a single binary classifier direction vector.

This is the ONE place in the codebase where sklearn is called for direction
production. Both run_classification and extract_classifier_directions route
through fit_classifier_direction — guaranteeing identical sign convention,
unit normalization, and preprocess fingerprint for all direction artifacts.

Public API
----------
fit_classifier_direction(*, X, y_binary, feature_cols, random_state, class_weight)
    -> dict[str, Any] | None
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Constants (written into every direction artifact for auditability)
# ---------------------------------------------------------------------------

DIRECTION_SPACE_RAW = "raw_feature_space"
"""Directions are in the raw (unscaled) feature space."""

REFIT_SCOPE_FULL_BIN = "full_bin_after_cv"
"""Direction is fit on the full bin (not on individual CV folds)."""

VECTOR_KIND_SIGNED_UNIT_COEF = "signed_unit_coef"
"""Vector is the sign-corrected, unit-normalized logistic regression coefficient."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_logistic_classifier(
    n_classes: int,
    random_state: int,
    class_weight: Any | None = "balanced",
) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight=class_weight,
        random_state=random_state,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _estimator_config(clf: LogisticRegression) -> dict[str, Any]:
    params = clf.get_params(deep=False)
    keys = [
        "solver",
        "penalty",
        "C",
        "class_weight",
        "max_iter",
        "random_state",
        "fit_intercept",
        "multi_class",
        "l1_ratio",
        "tol",
    ]
    return {key: _json_safe(params.get(key)) for key in keys}


def _preprocess_fingerprint(
    *,
    feature_names: list[str],
    direction_space: str,
    estimator_config: dict[str, Any],
) -> str:
    payload = {
        "feature_names": list(feature_names),
        "direction_space": direction_space,
        "preprocess": {"kind": "identity", "version": "classification_identity_v1"},
        "estimator_config": estimator_config,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_classifier_direction(
    *,
    X: np.ndarray,
    y_binary: np.ndarray,
    feature_cols: list[str],
    random_state: int,
    class_weight: Any | None,
) -> dict[str, Any] | None:
    """Fit a binary logistic regression and extract a signed unit direction vector.

    Parameters
    ----------
    X : (n_samples, n_features) float array
    y_binary : (n_samples,) int array, values in {0, 1}
    feature_cols : list of column names, length n_features (authoritative order)
    random_state : int
    class_weight : passed to LogisticRegression

    Returns
    -------
    dict with keys:
        feature_names, unit_coef, coef_norm, intercept, sign_flipped,
        centroid_dot, direction_space, preprocess_fingerprint, estimator_config
    or None if fewer than 2 unique classes are present in y_binary.

    Notes
    -----
    Sign convention: the returned unit_coef always points from the negative
    class centroid toward the positive class centroid, i.e.
        dot(unit_coef, pos_mean - neg_mean) >= 0
    This is enforced by flipping coef if centroid_dot < 0.
    """
    if X.shape[1] != len(feature_cols):
        raise ValueError(
            "Classifier direction: X has "
            f"{X.shape[1]} columns but feature_cols has {len(feature_cols)} entries."
        )
    if len(np.unique(y_binary)) < 2:
        return None

    clf = _make_logistic_classifier(
        n_classes=2,
        random_state=random_state,
        class_weight=class_weight,
    )
    clf.fit(X, y_binary)
    coef = np.asarray(clf.coef_, dtype=float).ravel()
    intercept = float(np.asarray(clf.intercept_, dtype=float).ravel()[0])

    pos_mean = np.asarray(X[y_binary == 1], dtype=float).mean(axis=0)
    neg_mean = np.asarray(X[y_binary == 0], dtype=float).mean(axis=0)
    centroid_dot = float(np.dot(coef, pos_mean - neg_mean))
    sign_flipped = centroid_dot < 0.0
    if sign_flipped:
        coef = -coef
        intercept = -intercept
        centroid_dot = -centroid_dot

    coef_norm = float(np.linalg.norm(coef))
    unit_coef = (
        np.zeros_like(coef, dtype=float) if coef_norm == 0.0 else coef / coef_norm
    )
    estimator_config = _estimator_config(clf)
    feature_names = list(feature_cols)
    return {
        "feature_names": feature_names,
        "unit_coef": unit_coef,
        "coef_norm": coef_norm,
        "intercept": intercept,
        "sign_flipped": bool(sign_flipped),
        "centroid_dot": centroid_dot,
        "direction_space": DIRECTION_SPACE_RAW,
        "preprocess_fingerprint": _preprocess_fingerprint(
            feature_names=feature_names,
            direction_space=DIRECTION_SPACE_RAW,
            estimator_config=estimator_config,
        ),
        "estimator_config": estimator_config,
    }
