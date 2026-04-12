"""
Schema definition for SAM2 segmentation tracking output.

This module defines required columns for the segmentation tracking table,
which contains embryo masks, tracking metadata, and SAM2-specific fields.
"""

REQUIRED_COLUMNS_SEGMENTATION_TRACKING = [
    # Schema (do not change without bumping schema_version)
    "schema_version",

    # Core IDs
    "experiment_id",
    "video_id",
    "well_id",              # Well identifier for grouping
    "well_index",
    "image_id",
    "embryo_id",
    "instance_id",
    "snip_id",
    "frame_index",
    "time_int",

    # Frame physical snapshot (downstream must not join frame manifests)
    "channel_id",
    "source_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
    "frame_snapshot_hash",

    # Mask data (legacy + canonical names)
    "mask_rle",
    "embryo_mask_rle",
    "area_px",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "mask_confidence",

    # Geometry
    "centroid_x_px",
    "centroid_y_px",

    # SAM2 metadata
    "is_seed_frame",

    # File references (legacy + canonical names)
    "source_image_path",
    "exported_mask_path",
    "embryo_mask_path",
]
