"""
Schema definition for scope (microscope) metadata.

This module defines required columns for microscope-extracted metadata,
including spatial and temporal calibration parameters.
"""

REQUIRED_COLUMNS_SCOPE_METADATA = [
    # Core identifiers
    'experiment_id',
    'well_id',
    'well_index',
    'image_id',
    'time_int',

    # Spatial calibration (extracted from microscope)
    'micrometers_per_pixel',
    'image_width_px',
    'image_height_px',
    'objective_magnification',

    # Temporal calibration
    'frame_interval_s',
    'absolute_start_time',
    'experiment_time_s',

    # Acquisition metadata
    'microscope_id',
    'channel',
    'z_position',
    'frame_index',
]
