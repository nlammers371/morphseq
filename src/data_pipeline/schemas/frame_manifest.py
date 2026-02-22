"""Schema definition for frame manifest contract."""

REQUIRED_COLUMNS_FRAME_MANIFEST = [
    "experiment_id",
    "microscope_id",
    "well_id",
    "well_index",
    "channel_id",
    "channel_name_raw",
    "time_int",
    "frame_index",
    "image_id",
    "stitched_image_path",
    "micrometers_per_pixel",
    "frame_interval_s",
    "absolute_start_time",
    "experiment_time_s",
    "image_width_px",
    "image_height_px",
    "objective_magnification",
    "genotype",
    "treatment",
    "medium",
    "temperature",
    "start_age_hpf",
    "embryos_per_well",
]

UNIQUE_KEY_FRAME_MANIFEST = [
    "experiment_id",
    "well_id",
    "channel_id",
    "time_int",
]
