"""
Schema definition for joined scope and plate metadata.

This module defines required columns for the consolidated metadata table
that combines plate layout annotations with microscope metadata.
"""

REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA = [
    # Core identifiers
    'experiment_id',
    'well_id',
    'well_index',
    'image_id',
    'time_int',
    'frame_index',
    # Note: embryo_id is not available at Phase 1 metadata alignment
    # It is generated during Phase 3 SAM2 segmentation

    # From plate_metadata
    'genotype',
    'treatment',
    'temperature_c',
    'embryos_per_well',

    # From scope_metadata
    'micrometers_per_pixel',
    'frame_interval_s',
    'absolute_start_time',
    'image_width_px',
    'image_height_px',
]
