"""
Schema definition for the canonical frame contract.

This is the plate-free handoff consumed by segmentation and feature
entrypoints. It carries the frame-level calibration and image location
needed downstream without introducing plate metadata joins.
"""

REQUIRED_COLUMNS_FRAME_CONTRACT = [
    "experiment_id",
    "microscope_id",
    "well_id",
    "well_index",
    "channel_id",
    "channel_name_raw",
    "time_int",
    "image_id",
    "stitched_image_path",
    "micrometers_per_pixel",
    "frame_interval_s",
    "absolute_start_time",
    "experiment_time_s",
    "image_width_px",
    "image_height_px",
    "objective_magnification",
]

UNIQUE_KEY_FRAME_CONTRACT = (
    "experiment_id",
    "well_id",
    "channel_id",
    "time_int",
)
