"""
Feature extraction configuration defaults.

This mirrors the QC defaults pattern so stage-specific entrypoints can merge
module defaults with workflow/config overrides in one place.
"""

from __future__ import annotations

DEFAULT_FEATURE_EXTRACTION_CONFIG = {
    "stage_predictions": {
        "start_age_col": "start_age_hpf",
        "time_col": "experiment_time_s",
        "temp_col": "temperature",
    },
    "consolidate_features": {},
}


def get_feature_defaults(stage: str) -> dict:
    return dict(DEFAULT_FEATURE_EXTRACTION_CONFIG.get(stage, {}))


def merge_feature_defaults(stage: str, overrides: dict | None = None) -> dict:
    effective = dict(DEFAULT_FEATURE_EXTRACTION_CONFIG.get(stage, {}))
    if overrides:
        effective.update(overrides)
    return effective
