"""
Schema definition for consolidated snip features.

This module defines required columns for the consolidated feature table,
which combines all SAM2-derived features, developmental stage predictions,
and viability metrics.
"""

REQUIRED_COLUMNS_FEATURES = [
    # Core IDs
    'snip_id',
    'embryo_id',
    'experiment_id',        # For cross-experiment analysis
    'well_id',              # Well identifier
    'frame_index',

    # Calibration (document what was used for conversions)
    'micrometers_per_pixel',
    'frame_interval_s',     # For velocity calculations

    # Geometry features (μm-based)
    'area_um2',             # Critical - must use μm², not pixels
    'perimeter_um',
    'centroid_x_um',
    'centroid_y_um',

    # Developmental stage
    'predicted_stage_hpf',  # Critical for QC and downstream analysis

    # Fraction alive detection (used to determine dead_flag in QC)
    'fraction_alive',

    # Pose/kinematics
    'orientation_angle',
    'bbox_width_um',
    'bbox_height_um',
]

# Alias for backwards compatibility
REQUIRED_COLUMNS_CONSOLIDATED_FEATURES = REQUIRED_COLUMNS_FEATURES
