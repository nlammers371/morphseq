"""Schema definition for the canonical frame contract."""

REQUIRED_COLUMNS_FRAME_CONTRACT = [
    "experiment_id",
    "well_id",
    "well_index",
    "frame_index",
    "channel_id",
    "image_id",
    "time_int",
    "microscope_id",
    "channel_name_raw",
    "stitched_image_path",
    "micrometers_per_pixel",
    "frame_interval_s",
    "frame_interval_min",
    "frame_interval_hr",
    "absolute_start_time",
    "experiment_time_s",
    "elapsed_time_s",
    "elapsed_time_min",
    "elapsed_time_hr",
    "image_width_px",
    "image_height_px",
    "objective_magnification",
]

UNIQUE_KEY_FRAME_CONTRACT = [
    "experiment_id",
    "well_id",
    "channel_id",
    "frame_index",
]
