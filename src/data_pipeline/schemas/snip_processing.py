"""
Schema definition for snip processing manifest.

This module defines required columns for the snip manifest table,
which tracks extracted and processed embryo crops.
"""

REQUIRED_COLUMNS_SNIP_MANIFEST = [
    # Schema
    "schema_version",

    # Identity
    "snip_id",  # unique: embryo_id_{channel_id}_f{frame_index:04d}
    "experiment_id",
    "well_id",
    "well_index",
    "image_id",
    "frame_index",
    "channel_id",
    "embryo_id",
    "instance_id",

    # Inputs used
    "source_image_path",
    "embryo_mask_path",
    "yolk_mask_path",  # nullable
    "source_micrometers_per_pixel",
    "frame_snapshot_hash",

    # Outputs written
    "processed_snip_path",
    "raw_crop_path",  # nullable

    # Processing params
    "target_pixel_size_um",
    "output_height_px",
    "output_width_px",
    "blend_radius_um",

    # Background stats used
    "background_mean",
    "background_std",

    # Rotation
    "rotation_angle_rad",
    "rotation_angle_deg",
    "rotation_used_yolk",

    # Provenance
    "snip_processing_run_id",
    "snip_processing_version",
]
