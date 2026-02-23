"""
Schema definition for quality control flags.

This module defines required columns for the consolidated QC table,
which merges all quality control flags and computes the use_embryo gate.
"""

REQUIRED_COLUMNS_QC = [
    # Core ID
    'snip_id',
    'embryo_id',
    'experiment_id',
    'time_int',

    # Gating flag
    'use_embryo',           # Final gate for embeddings/analysis

    # Viability QC
    'dead_flag',
    'fraction_alive',
    'death_inflection_time_int',  # When embryo died
    'death_predicted_stage_hpf',  # Developmental stage at death (age at death)

    # Imaging QC (from auxiliary masks)
    'focus_flag',
    'bubble_flag',
    'yolk_flag',
    'edge_flag',
    'discontinuous_mask_flag',
    'overlapping_mask_flag',

    # Segmentation QC
    'mask_quality_flag',

    # Morphology QC
    'sa_outlier_flag',      # Surface area outlier
]

QC_FAIL_FLAGS = [
    'dead_flag',
    'sa_outlier_flag',
    'yolk_flag',
    'edge_flag',
    'discontinuous_mask_flag',
    'overlapping_mask_flag',
    'focus_flag',
    'bubble_flag',
]
