"""
Schema definition for snip processing manifest.

This module defines required columns for the snip manifest table,
which tracks extracted and processed embryo crops.
"""

REQUIRED_COLUMNS_SNIP_MANIFEST = [
    # Core IDs
    'snip_id',
    'embryo_id',
    'experiment_id',
    'frame_index',
    'time_int',

    # File paths
    'source_image_path',    # Path to stitched FF image
    'cropped_snip_path',    # Path to extracted snip JPG

    # Extraction metadata
    'rotation_angle',       # PCA rotation applied (degrees)
    'crop_x_min',           # Crop bounding box
    'crop_y_min',
    'crop_x_max',
    'crop_y_max',
]
