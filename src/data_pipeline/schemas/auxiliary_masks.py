"""
Schema definition for auxiliary mask manifests.

This manifest is the contract for UNet-derived auxiliary masks. It records
the source frame, the family-specific output paths, and a small provenance
stamp so downstream consumers do not need to infer paths from conventions.
"""

REQUIRED_COLUMNS_AUXILIARY_MASKS = [
    "schema_version",
    "experiment_id",
    "well_id",
    "well_index",
    "time_int",
    "image_id",
    "source_image_path",
    "source_micrometers_per_pixel",
    "image_width_px",
    "image_height_px",
    "via_mask_path",
    "yolk_mask_path",
    "focus_mask_path",
    "bubble_mask_path",
    "auxiliary_mask_version",
    "materialization_status",
]
