"""Schema definition for per-snip developmental stage predictions (HPF)."""

REQUIRED_COLUMNS_STAGE_PREDICTIONS = [
    # IDs
    "experiment_id",
    "well_id",
    "well_index",
    "image_id",
    "embryo_id",
    "time_int",
    "snip_id",

    # Inputs used
    "elapsed_time_s",
    "start_age_hpf",
    "temperature",

    # Outputs
    "developmental_rate_hpf_per_h",
    "predicted_stage_hpf",
    "stage_confidence",

    # Provenance
    "stage_model",
    "pipeline_version",
]

UNIQUE_KEY_STAGE_PREDICTIONS = ["snip_id"]
