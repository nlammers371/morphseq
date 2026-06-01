"""
Quality control configuration and default parameters.

This module defines all default parameters used in QC operations to ensure
consistency across the codebase. All QC functions should import from this
module rather than hardcoding defaults.
"""

DEFAULT_QC_CONFIG = {
    "segmentation_qc": {
        "edge_margin_pixels": 2,
        "max_mask_overlap_fraction": 0.3,
    },
    "viability_qc": {
        "min_mask_size_px": 100,
        "aspect_ratio_threshold": 8.0,
    },
    "death_detection": {
        "persistence_threshold": 0.80,
        "lead_time_hr": 4.0,
        "decline_rate_threshold": 0.05,
        "dead_fraction_threshold": 0.90,
    },
    "surface_area_qc": {
        "k_upper": 1.4,
        "k_lower": 0.7,
    },
    "auxiliary_mask_qc": {},
    "focus_qc": {},
    "motion_qc": {
        "ncc_min_threshold": 0.85,
        "bad_pair_frac_threshold": 0.10,
    },
}

# Backwards-compatible alias for callers already using the old name.
QC_DEFAULTS = DEFAULT_QC_CONFIG


def get_qc_defaults(stage: str) -> dict:
    return dict(DEFAULT_QC_CONFIG.get(stage, {}))


def merge_qc_defaults(stage: str, overrides: dict | None = None) -> dict:
    effective = dict(DEFAULT_QC_CONFIG.get(stage, {}))
    if overrides:
        effective.update(overrides)
    return effective


def get_dead_lead_time() -> float:
    return float(DEFAULT_QC_CONFIG["death_detection"]["lead_time_hr"])
