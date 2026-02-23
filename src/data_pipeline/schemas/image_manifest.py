"""
Schema definition for experiment image manifest.

This module defines required fields for the experiment_image_manifest.json file,
which serves as the single source of truth for per-well, per-channel frame ordering.
"""

# Top-level experiment metadata fields
REQUIRED_EXPERIMENT_FIELDS = [
    'experiment_id',
    'microscope_id',
    'created_at',
    'total_wells',
    'total_channels',
    'total_frames',
]

# Per-well metadata fields
REQUIRED_WELL_FIELDS = [
    'well_id',
    'well_index',
    'embryos_per_well',
    'genotype',
    'treatment',
    'channels',  # List of channel objects
]

# Per-channel metadata fields (within each well)
REQUIRED_CHANNEL_FIELDS = [
    'channel_name',          # Normalized channel name (BF, GFP, etc.)
    'raw_channel_name',      # Original microscope channel name
    'frames',                # List of frame objects
]

# Per-frame metadata fields (within each channel)
REQUIRED_FRAME_FIELDS = [
    'frame_index',
    'time_int',
    'experiment_time_s',
    'image_id',
    'file_path',             # Relative path to stitched image
    'image_width_px',
    'image_height_px',
    'micrometers_per_pixel',
]
