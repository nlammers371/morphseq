"""
Schema definition for the physical frame manifest.

frame_manifest.parquet is the single physical inventory contract that downstream stages
(segmentation/tracking, snip processing) use to enumerate frames and retrieve calibration.
"""

REQUIRED_COLUMNS_FRAME_MANIFEST = [
    # Core IDs
    "experiment_id",
    "microscope_id",
    "well_index",
    "well_id",
    "channel_id",
    "frame_index",
    "time_int",
    "image_id",
    # Physical pointers + calibration
    "source_image_path",
    "source_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
]

# One row per frame.
UNIQUE_KEY_FRAME_MANIFEST = [
    "experiment_id",
    "well_id",
    "channel_id",
    "frame_index",
]

