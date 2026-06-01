"""Shared pytest fixtures for morphology_geometry tests.

All fixtures are synthetic — no sklearn, no on-disk artifacts, no classification run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyze.classification.directions.artifact import ClassifierDirections
from analyze.morphology_geometry.validation import (
    ValidatedDirections,
    validate_classifier_directions,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES = 4
FEATURE_SET = "pca"
COMPARISONS = ["wt_vs_het", "wt_vs_homo"]
BIN_CENTERS = [24.0, 28.0, 32.0]
BIN_WIDTH = 4.0
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_directions() -> ClassifierDirections:
    """Minimal valid ClassifierDirections: 2 comparisons × 3 bins × 4 features.

    Vectors are deterministic (seeded), unit-norm, finite.
    auroc_obs is populated so has_auroc=True.
    """
    rng = np.random.default_rng(42)
    rows = []
    vectors: dict[str, np.ndarray] = {}

    for cid in COMPARISONS:
        for t in BIN_CENTERS:
            vid = f"{FEATURE_SET}__{cid}__{t:.1f}"
            v = _unit(rng.standard_normal(N_FEATURES))
            vectors[vid] = v
            rows.append({
                "vector_id": vid,
                "feature_set": FEATURE_SET,
                "comparison_id": cid,
                "positive_label": cid.split("_vs_")[1],
                "negative_label": "wt",
                "time_bin_center": t,
                "n_pos": 10,
                "n_neg": 10,
                "coef_norm": 1.0,
                "intercept": 0.0,
                "sign_flipped": False,
                "centroid_dot": 0.5,
                "direction_space": "raw_feature_space",
                "preprocess_fingerprint": "fp_test_abc",
                "auroc_obs": 0.65 + rng.uniform(-0.1, 0.1),
            })

    return ClassifierDirections(
        metadata=pd.DataFrame(rows),
        vectors=vectors,
        feature_names={FEATURE_SET: FEATURE_NAMES},
    )


@pytest.fixture()
def validated(minimal_directions: ClassifierDirections) -> ValidatedDirections:
    """ValidatedDirections built from minimal_directions."""
    return validate_classifier_directions(
        minimal_directions,
        feature_set=FEATURE_SET,
        expected_bin_width=BIN_WIDTH,
    )
