"""Assemble a ClassifierDirections artifact from a list of per-bin fit results.

This module is pure dict / dataframe plumbing — no sklearn, no fitting.
It takes the output of fit_classifier_direction() calls collected across
all (feature_set, comparison, time_bin) combinations and builds the
ClassifierDirections dataclass ready for save.

Both run_classification and extract_classifier_directions use this function,
guaranteeing a single artifact schema regardless of which entry point is used.

Public API
----------
build_classifier_directions_payload(fits, *, feature_sets) -> ClassifierDirections
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .artifact import ClassifierDirections
from .fit import REFIT_SCOPE_FULL_BIN, VECTOR_KIND_SIGNED_UNIT_COEF
from .ids import make_vector_id


def build_classifier_directions_payload(
    fits: list[dict[str, Any]],
    *,
    feature_sets: dict[str, list[str]],
) -> ClassifierDirections:
    """Assemble a ClassifierDirections artifact from per-bin fit dicts.

    Parameters
    ----------
    fits : list of dicts
        Each dict is the union of:
        - the return value of fit_classifier_direction() (keys: feature_names,
          unit_coef, coef_norm, intercept, sign_flipped, centroid_dot,
          direction_space, preprocess_fingerprint, estimator_config)
        - caller-supplied metadata (keys: feature_set, comparison_id,
          positive_label, negative_label, time_bin, time_bin_center,
          n_pos, n_neg, bin_width).
        Optionally: auroc_obs, pval (present from run_classification, absent
        from extract_classifier_directions).

    feature_sets : dict[str, list[str]]
        Maps feature_set name -> ordered feature column list. Used to populate
        ClassifierDirections.feature_names (the authoritative column order).

    Returns
    -------
    ClassifierDirections with metadata DataFrame, vectors dict, feature_names dict.

    Raises
    ------
    ValueError if a fit dict has inconsistent feature_names for its feature_set,
    or if a unit_coef vector has a length mismatch.
    """
    rows: list[dict[str, Any]] = []
    vectors: dict[str, np.ndarray] = {}
    feature_names_by_set: dict[str, list[str]] = {}

    for fit in fits:
        feature_set = str(fit["feature_set"])
        comparison_id = str(fit["comparison_id"])
        time_bin = int(fit["time_bin"])

        fit_feature_names = list(fit["feature_names"])
        previous = feature_names_by_set.get(feature_set)
        if previous is not None and previous != fit_feature_names:
            raise ValueError(
                f"Classifier direction feature order changed within feature set "
                f"{feature_set!r}. This likely means two different feature column "
                f"lists were used for the same feature_set across time bins."
            )
        feature_names_by_set[feature_set] = fit_feature_names

        unit_coef = np.asarray(fit["unit_coef"], dtype=float)
        if len(unit_coef) != len(fit_feature_names):
            raise ValueError(
                f"unit_coef length {len(unit_coef)} != feature_names length "
                f"{len(fit_feature_names)} for feature_set={feature_set!r}, "
                f"comparison_id={comparison_id!r}, time_bin={time_bin}."
            )

        vector_id = make_vector_id(
            feature_set=feature_set,
            comparison_id=comparison_id,
            time_bin=time_bin,
        )
        vectors[vector_id] = unit_coef

        estimator_config = dict(fit["estimator_config"])
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "comparison_id": comparison_id,
            "positive_label": str(fit["positive_label"]),
            "negative_label": str(fit["negative_label"]),
            "time_bin": time_bin,
            "time_bin_center": float(fit["time_bin_center"]),
            "bin_width": float(fit.get("bin_width", float("nan"))),
            "n_pos": int(fit["n_pos"]),
            "n_neg": int(fit["n_neg"]),
            "vector_id": vector_id,
            "vector_kind": VECTOR_KIND_SIGNED_UNIT_COEF,
            "coef_norm": float(fit["coef_norm"]),
            "intercept": float(fit["intercept"]),
            "sign_flipped": bool(fit["sign_flipped"]),
            "centroid_dot": float(fit["centroid_dot"]),
            "direction_space": str(fit["direction_space"]),
            "preprocess_fingerprint": str(fit["preprocess_fingerprint"]),
            "refit_scope": REFIT_SCOPE_FULL_BIN,
            # auroc_obs and pval are optional — use NaN when not supplied
            "auroc_obs": float(fit["auroc_obs"]) if "auroc_obs" in fit else float("nan"),
            "pval": float(fit["pval"]) if "pval" in fit else float("nan"),
        }
        for key, value in estimator_config.items():
            row[f"estimator_{key}"] = value
        rows.append(row)

    if not rows:
        # Return a valid but empty artifact
        metadata = pd.DataFrame(
            columns=[
                "feature_set", "comparison_id", "positive_label", "negative_label",
                "time_bin", "time_bin_center", "bin_width", "n_pos", "n_neg",
                "vector_id", "vector_kind", "coef_norm", "intercept",
                "sign_flipped", "centroid_dot", "direction_space",
                "preprocess_fingerprint", "refit_scope", "auroc_obs", "pval",
            ]
        )
        return ClassifierDirections(
            metadata=metadata,
            vectors={},
            feature_names={fs: list(cols) for fs, cols in feature_sets.items()},
        )

    metadata = pd.DataFrame(rows)
    return ClassifierDirections(
        metadata=metadata,
        vectors=vectors,
        feature_names={fs: list(cols) for fs, cols in feature_names_by_set.items()},
    )
