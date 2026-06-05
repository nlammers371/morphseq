"""Schema definition for quality control flags.

This module is the source of truth for the consolidated QC contract.
"""

from __future__ import annotations

SNIP_EXCLUSION_FLAGS = [
    "edge_flag",
    "discontinuous_mask_flag",
    "overlapping_mask_flag",
    "viability_flag",
    "dead_flag",
    "sa_outlier_flag",
    "focus_flag",
    "motion_flag",
]

SNIP_INFORMATIONAL_FLAGS = [
    "yolk_flag",
    "bubble_flag",
]

REQUIRED_COLUMNS_QC = [
    "snip_id",
    "use_snip",
    *SNIP_EXCLUSION_FLAGS,
    *SNIP_INFORMATIONAL_FLAGS,
    "death_inflection_time_int",
]
