# MorphSeq Pipeline: Data Output Structure

**Status:** Target Output Spec (refactor)
**Audience:** Scientists and developers
**Last Updated:** 2026-02-10

**Note:** This is the intended end-state output layout for the refactor; the repo may still contain legacy outputs until the implementation is complete.

## 2026-02-10 - Addendum, highlighting what we need to change in the original doc
This addendum is additive only. The original output structure remains valid.

Scope of update:
- Keep downstream output logic unchanged.
- Clarify ingest/pre-segmentation handoff expectations only.

Clarifications:
1. Treat ingest as scope-first through metadata extraction and series mapping.
2. Keep stitched image materialization scope-specific.
3. Keep pre-segmentation canonical contracts as:
   - `stitched_image_index.csv`
   - `frame_manifest.csv` (canonical frame metadata table)
4. Canonical frame-level naming remains:
   - `channel_id`
   - `channel_name_raw`
   - `temperature`
   - required `micrometers_per_pixel`

## TL;DR
Pre-segmentation now has two key CSV contracts:

1. `stitched_image_index.csv` tells you what stitched files were materialized.
2. `frame_manifest.csv` is the canonical frame-level table for segmentation.

The old `experiment_image_manifest.json` is deprecated and removed.

---

## Directory Structure

```
{data_pipeline_root}/
│
├── inputs/
│   ├── raw_image_data/
│   │   ├── Keyence/{experiment_id}/
│   │   └── YX1/{experiment_id}/
│   ├── reference_data/
│   │   ├── surface_area_ref.csv
│   │   └── perturbation_catalog.csv
│   └── plate_metadata/
│       └── {experiment_id}_well_metadata.xlsx
│
├── experiment_metadata/
│   └── {experiment_id}/
│       ├── plate_metadata.csv
│       ├── scope_metadata_raw.csv
│       ├── series_well_mapping.csv
│       ├── series_well_mapping_provenance.json
│       ├── scope_metadata_mapped.csv
│       ├── stitched_image_index.csv
│       └── frame_manifest.csv
│
├── built_image_data/
│   └── {experiment_id}/
│       └── stitched_ff_images/
│           └── {well_index}/{channel_id}/{image_id}.tif
│
├── segmentation/
│   └── {experiment_id}/
│       ├── gdino_detections.json
│       ├── sam2_raw_output.json
│       ├── segmentation_tracking.csv
│       ├── mask_images/
│       └── unet_masks/
│
├── processed_snips/
│   └── {experiment_id}/
│       ├── raw_crops/
│       ├── processed/
│       └── snip_manifest.csv
│
├── computed_features/
│   └── {experiment_id}/
│       ├── mask_geometry_metrics.csv
│       ├── pose_kinematics_metrics.csv
│       ├── stage_predictions.csv
│       └── consolidated_snip_features.csv
│
├── quality_control/
│   └── {experiment_id}/
│       ├── segmentation_quality_qc.csv
│       ├── auxiliary_mask_qc.csv
│       ├── embryo_death_qc.csv
│       ├── surface_area_outliers_qc.csv
│       └── consolidated_qc_flags.csv
│
├── latent_embeddings/
│   └── {model_name}/
│       ├── {experiment_id}_embedding_manifest.csv
│       └── {experiment_id}_latents.csv
│
└── analysis_ready/
    └── {experiment_id}/
        └── features_qc_embeddings.csv
```

---

## Contract Files (Pre-Segmentation)

### `stitched_image_index.csv`
Purpose:
- Reporter output from stitched image builders.
- Lists what the builder attempted and what was materialized.

Core columns:
- `experiment_id`
- `microscope_id`
- `well_id`
- `well_index`
- `channel_id`
- `time_int`
- `frame_index`
- `image_id`
- `stitched_image_path`
- `materialization_status`
- `source_artifact_path`
- `source_artifact_kind`

### `frame_manifest.csv`
Purpose:
- Canonical frame-level input to segmentation and downstream joins.

Core columns:
- `experiment_id`
- `microscope_id`
- `well_id`
- `well_index`
- `channel_id`
- `channel_name_raw`
- `time_int`
- `frame_index`
- `image_id`
- `stitched_image_path`
- `micrometers_per_pixel`
- `frame_interval_s`
- `absolute_start_time`
- `experiment_time_s`
- `image_width_px`
- `image_height_px`
- `objective_magnification`
- `genotype`
- `treatment`
- `medium`
- `temperature`
- `start_age_hpf`
- `embryos_per_well`

Uniqueness key for both contracts:
- `(experiment_id, well_id, channel_id, time_int)`

---

## ID and Naming Conventions

```
experiment_id
└── well_index (A01, B12, ...)
    └── well_id = {experiment_id}_{well_index}
        └── channel_id (BF, GFP, ...)
            └── frame_index (0-based contiguous order)
                └── image_id = {well_id}_{channel_id}_t{frame_index}
                    └── embryo_id (starts at segmentation stage)
                        └── snip_id
```

Channel naming:
- `channel_id`: normalized, controlled naming used in joins and IDs.
- `channel_name_raw`: microscope-native label preserved for provenance.

---

## Validation Markers

Files marked as validated enforce schema and non-null rules:

- Pre-segmentation:
  - `plate_metadata.csv`
  - `scope_metadata_raw.csv`
  - `scope_metadata_mapped.csv`
  - `series_well_mapping.csv`
  - `stitched_image_index.csv`
  - `frame_manifest.csv`
- Post-segmentation:
  - `segmentation_tracking.csv`
  - `snip_manifest.csv`
  - `consolidated_snip_features.csv`
  - `consolidated_qc_flags.csv`
  - embedding outputs
  - `features_qc_embeddings.csv`

---

## Practical Flow for Scientists

When checking a new experiment:

1. Confirm `plate_metadata.csv` looks correct.
2. Confirm `scope_metadata_mapped.csv` has correct wells/channels/timing.
3. Confirm `stitched_image_index.csv` has expected materialization outcomes.
4. Confirm `frame_manifest.csv` has complete calibration and metadata rows.
5. Run segmentation from `frame_manifest.csv`.

---

## Deprecated Output

Removed/deprecated:
- `experiment_image_manifest.json`

Do not build new tooling that depends on this file.
