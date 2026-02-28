"""
Schema definition for Phase 4 snip processing manifest.

The snip manifest is the authoritative inventory of exported embryo crops ("snips"),
including enough provenance to reproduce processing and join downstream tables.
"""

REQUIRED_COLUMNS_SNIP_MANIFEST = [
    # IDs / joins
    "snip_id",
    "mask_type",
    "experiment_id",
    "well_id",
    "well_index",
    "image_id",
    "embryo_id",
    "frame_index",

    # Inputs (relative to data_pipeline_output where possible)
    "source_image_path",
    "exported_mask_path",
    "yolk_mask_path",  # nullable

    # Outputs (relative to data_pipeline_output)
    "processed_snip_path",  # nullable if is_valid==False
    "raw_crop_path",  # nullable

    # Processing params
    "source_micrometers_per_pixel",
    "target_pixel_size_um",
    "output_height_px",
    "output_width_px",
    "blend_radius_um",

    # Background noise stats (augmentation)
    "background_mean",
    "background_std",
    "background_definition",

    # Rotation
    "rotation_angle_rad",  # nullable if is_valid==False
    "rotation_angle_deg",  # nullable if is_valid==False
    "rotation_source",  # "yolk_guided" | "embryo_only"

    # Provenance
    "pipeline_version",
    "snip_processing_config_hash",
    "processing_timestamp_utc",

    # File sizes
    "processed_file_size_bytes",  # nullable if is_valid==False
    "raw_file_size_bytes",  # nullable

    # Error handling
    "is_valid",
    "error_message",  # nullable
]

# Columns that must exist but may be null (e.g. optional inputs, optional raw crops,
# file sizes for optional artifacts, or per-row failures when is_valid==False).
NULLABLE_COLUMNS_SNIP_MANIFEST = [
    "yolk_mask_path",
    "raw_crop_path",
    "processed_snip_path",
    "processed_file_size_bytes",
    "raw_file_size_bytes",
    "rotation_angle_rad",
    "rotation_angle_deg",
    "error_message",
]
