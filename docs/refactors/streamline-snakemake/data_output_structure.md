# MorphSeq Pipeline: Data Output Structure

**Status:** Phase 1-4 Implemented; Phase 5+ Planned (refactor)
**Audience:** Scientists and developers
**Last Updated:** 2026-02-28

**Note:** Phase 1-4 outputs and paths in this doc reflect the current implementation (`segmentation_and_tracking/` and `processed_snips/`). Downstream phases may still contain legacy outputs/paths until wired.

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
   - `frame_contract.csv` (canonical frame metadata table)
4. Canonical frame-level naming remains:
   - `channel_id`
   - `channel_name_raw`
   - `temperature`
   - required `micrometers_per_pixel`

## TL;DR
Pre-segmentation now has two key CSV contracts:

1. `plate_metadata.csv` is the user-provided experiment annotation.
2. `scope_series_metadata_raw.csv` is the microscope-side series output.
3. `scope_series_metadata_mapped.csv` is the scope-to-well mapping result.
4. `stitched_image_index.csv` is the materialization truth.
5. `frame_contract.csv` is the canonical frame-level table for segmentation.

Plate metadata re-enters later at the feature stage, not before segmentation.

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
│       ├── scope_series_metadata_raw.csv
│       ├── series_well_mapping.csv
│       ├── series_well_mapping_provenance.json
│       ├── scope_series_metadata_mapped.csv
│       ├── stitched_image_index.csv
│       └── frame_contract.csv
│
├── built_image_data/
│   └── {experiment_id}/
│       └── stitched_ff_images/
│           └── {well_index}/{channel_id}/{image_id}.tif
│
├── segmentation_and_tracking/                 # PHASE 3 OUTPUTS (implemented)
│   └── {experiment_id}/
│       ├── per_well/
│       │   └── {experiment_id}_{well_slug}/   # per-well shard (REAL files)
│       │       ├── contracts/
│       │       │   ├── frame_detections.parquet
│       │       │   ├── seed_selection.parquet
│       │       │   ├── embryo_track_instances.parquet
│       │       │   ├── embryo_mask_rle.parquet
│       │       │   ├── segmentation_tracking.csv
│       │       │   └── .segment_and_track.validated
│       │       ├── masks/
│       │       │   └── embryo_mask/{snip_id}_mask.png
│       │       └── artifacts/                 # optional/heavy (REAL or symlinks)
│       │           ├── raw_frames/{image_id}.jpg
│       │           ├── sam2_frames/00000.jpg
│       │           └── overlays/embryo_mask/{well_slug}_embryo_mask_overlay.mp4
│       ├── contracts/                          # merged experiment contracts (REAL)
│       │   ├── frame_detections.parquet
│       │   ├── seed_selection.parquet
│       │   ├── embryo_track_instances.parquet
│       │   ├── embryo_mask_rle.parquet
│       │   ├── segmentation_tracking.csv
│       │   └── .segmentation_tracking.validated
│       └── views/                              # symlink-only browse view (DISPOSABLE)
│           ├── wells/{well_slug} -> ../per_well/{experiment_id}_{well_slug}
│           ├── masks/embryo_mask/{well_slug} -> ../../per_well/.../masks/embryo_mask
│           └── videos/overlays/embryo_mask/{well_slug}_embryo_mask_overlay.mp4 -> ../../per_well/.../artifacts/overlays/embryo_mask/...
│
├── processed_snips/
│   └── {experiment_id}/
│       ├── per_well/
│       │   └── {well_id}/
│       │       ├── contracts/
│       │       │   ├── snip_manifest.parquet
│       │       │   ├── snip_manifest.csv
│       │       │   └── .snip_processing.validated
│       │       ├── processed/{snip_id}.jpg
│       │       ├── raw_crops/{snip_id}.tif            # optional (config: snip_processing.save_raw_crops)
│       │       └── artifacts/background_stats.json     # optional (for debugging/provenance)
│       ├── contracts/
│       │   ├── snip_manifest.parquet
│       │   ├── snip_manifest.csv
│       │   └── .snip_manifest.validated
│       └── views/                                     # symlink-only browse view (DISPOSABLE)
│           ├── wells/{well_slug} -> ../per_well/{well_id}
│           ├── processed/{well_slug} -> ../per_well/{well_id}/processed
│           └── raw_crops/{well_slug} -> ../per_well/{well_id}/raw_crops
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

### `frame_contract.csv`
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
  - `scope_series_metadata_raw.csv`
  - `scope_series_metadata_mapped.csv`
  - `series_well_mapping.csv`
  - `stitched_image_index.csv`
  - `frame_contract.csv`
- Post-segmentation:
  - `segmentation_and_tracking/{exp}/contracts/segmentation_tracking.csv`
  - `segmentation_and_tracking/{exp}/contracts/.segmentation_tracking.validated`
  - `processed_snips/{exp}/contracts/snip_manifest.csv`
  - `processed_snips/{exp}/contracts/.snip_manifest.validated`
  - `consolidated_snip_features.csv`
  - `consolidated_qc_flags.csv`
  - embedding outputs
  - `features_qc_embeddings.csv`

---

## Practical Flow for Scientists

When checking a new experiment:

1. Confirm `plate_metadata.csv` looks correct.
2. Confirm `scope_series_metadata_mapped.csv` has correct wells/channels/timing.
3. Confirm `stitched_image_index.csv` has expected materialization outcomes.
4. Confirm `frame_contract.csv` has complete calibration and metadata rows.
5. Run segmentation from `frame_contract.csv`.

---

## Deprecated Output

Removed/deprecated:
- `frame_contract.csv`

Do not build new tooling that depends on this file.
