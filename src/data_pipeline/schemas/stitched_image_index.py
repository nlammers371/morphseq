"""Schema definition for stitched image index contract."""

REQUIRED_COLUMNS_STITCHED_IMAGE_INDEX = [
    "experiment_id",
    "well_id",
    "well_index",
    "time_int",
    "channel_id",
    "image_id",
    "time_int",
    "microscope_id",
    "stitched_image_path",
    "materialization_status",
    "source_artifact_path",
    "source_artifact_kind",
    "frame_interval_s",
    "frame_interval_min",
    "frame_interval_hr",
    "experiment_time_s",
    "elapsed_time_s",
    "elapsed_time_min",
    "elapsed_time_hr",
]

UNIQUE_KEY_STITCHED_IMAGE_INDEX = [
    "experiment_id",
    "well_id",
    "channel_id",
    "time_int",
]
