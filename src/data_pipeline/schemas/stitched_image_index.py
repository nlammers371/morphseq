"""Schema definition for stitched image index contract."""

REQUIRED_COLUMNS_STITCHED_IMAGE_INDEX = [
    "experiment_id",
    "microscope_id",
    "well_id",
    "well_index",
    "channel_id",
    "frame_index",
    "image_id",
    "stitched_image_path",
    "materialization_status",
    "source_artifact_path",
    "source_artifact_kind",
]

UNIQUE_KEY_STITCHED_IMAGE_INDEX = [
    "experiment_id",
    "well_id",
    "channel_id",
    "frame_index",
]
