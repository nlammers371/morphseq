# Streamline-Snakemake Refactor Documentation

**Organization Date:** 2025-11-06
**Status:** Refactor docs describe the target `stitched_image_index.csv` + `frame_manifest.csv` architecture (implementation may still be in progress)

---

## 2026-02-10 - Addendum, highlighting what we need to change in the original doc
- Keep all downstream (segmentation, snip processing, features, QC, embeddings, analysis-ready) logic as-is.
- Limit conceptual updates to ingest and pre-segmentation handoff only.
- Treat ingest as scope-first (YX1 and Keyence stay separate through extraction and mapping).
- Keep `materialize_stitched_images_*` as the stage name; scope builders implement the behavior.
- Keep frame-level handoff on `stitched_image_index.csv` + `frame_manifest.csv`.
- Use `channel_id`, preserve `channel_name_raw`, use `temperature`, and require `micrometers_per_pixel` in frame-level metadata.

---

## Quick TL;DR (For Scientists)
If you only remember one thing:

1. `stitched_image_index.csv` tells you what stitched images were materialized.
2. `frame_manifest.csv` is the canonical frame table that segmentation should trust.
3. `embryo_id` starts at segmentation, not during metadata ingest.

---

## Core Documentation (Target)

### 1. `processing_files_pipeline_structure_and_plan.md`
**Architecture spec**
- Why this design changed
- Full module structure
- Contract definitions and validation checks
- Scientist-friendly debugging flow

### 2. `snakemake_rules_data_flow.md`
**Rule-by-rule implementation spec**
- Exact stage flow through `frame_manifest.csv`
- Rule purposes and I/O expectations
- Naming conventions (`channel_id`, `channel_name_raw`, `temperature`)

### 3. `data_output_structure.md`
**Output file and directory spec**
- Canonical output tree
- Contract files and required columns
- Practical checklist for experiment validation

### 4. `DATA_INGESTION_AND_TESTING_STRATEGY.md`
**Data setup and testing guidance**
- Symlink strategy
- Test dataset guidance
- Stepwise validation approach

---

## Current vs Target Code Map (Where to Copy Patterns From)

This table is meant to answer: “where do I find an example of this *today*?” vs “what should it become in the refactor?”

| Stage | Current repo codepaths + artifacts | Target refactor codepaths + artifacts |
|---|---|---|
| Phase 1: plate metadata | `src/data_pipeline/metadata_ingest/plate/plate_processing.py` (invoked by `src/data_pipeline/pipeline_orchestrator/Snakefile`) → `data_pipeline_output/experiment_metadata/{exp}/plate_metadata.csv` | Same logical output, but treated as an input to the pre-segmentation handoff join (`frame_manifest.csv`). |
| Phase 1: scope metadata extraction | `src/data_pipeline/metadata_ingest/scope/yx1_scope_metadata.py` / `src/data_pipeline/metadata_ingest/scope/keyence_scope_metadata.py` → `data_pipeline_output/experiment_metadata/{exp}/scope_metadata.csv` | Split into `scope_metadata_raw.csv` + `scope_metadata_mapped.csv` per `processing_files_pipeline_structure_and_plan.md` and `snakemake_rules_data_flow.md`. |
| Phase 1: series→well mapping + alignment | Mapping helpers: `src/data_pipeline/metadata_ingest/mapping/series_well_mapper_yx1.py`, `src/data_pipeline/metadata_ingest/mapping/series_well_mapper_keyence.py`; join/validate: `src/data_pipeline/metadata_ingest/mapping/align_scope_plate.py` → `scope_and_plate_metadata.csv` | Target is explicit `series_well_mapping.csv` + provenance, then `apply_series_mapping` to produce `scope_metadata_mapped.csv`. |
| Phase 2: stitched image materialization | Builders: `src/data_pipeline/image_building/yx1/stitched_ff_builder.py`, `src/data_pipeline/image_building/keyence/stitched_ff_builder.py` → `data_pipeline_output/built_image_data/{exp}/stitched_ff_images/{well}/{channel}/...tif` | Keep scope-specific builders, but add reporter output `stitched_image_index.csv` during materialization (no crawler parsing). |
| Phase 2 handoff contract | Crawler + JSON manifest: `src/data_pipeline/metadata_ingest/manifests/generate_image_manifest.py` (+ `src/data_pipeline/schemas/image_manifest.py`) → `experiment_image_manifest.json` | Replace with two CSV contracts: `stitched_image_index.csv` + `frame_manifest.csv` (schemas + validators planned). |
| Phase 3+: segmentation and downstream | Code exists under `src/data_pipeline/segmentation/`, `src/data_pipeline/snip_processing/`, `src/data_pipeline/feature_extraction/`, `src/data_pipeline/quality_control/`, `src/data_pipeline/analysis_ready/` (not yet wired into `Snakefile`) | Same downstream logic, but segmentation consumes `frame_manifest.csv` as the canonical frame table (per the three core docs). |

Notes:
- The current `Snakefile` uses heredocs with `python`; for refactor wiring, prefer calling modules via `"$PYTHON" -m data_pipeline...` (see `DATA_INGESTION_AND_TESTING_STRATEGY.md`).
- Current stitched image filenames are `{well}_{channel}_t{time_int:04d}.tif`; the refactor docs assume `image_id` includes `well_id` (and therefore `experiment_id`) for globally unique IDs.

---

## Recommended Reading Order

1. `processing_files_pipeline_structure_and_plan.md`
2. `snakemake_rules_data_flow.md`
3. `data_output_structure.md`
4. `DATA_INGESTION_AND_TESTING_STRATEGY.md`

---

## Current Pre-Segmentation Flow

1. Normalize plate metadata.
2. Extract scope metadata.
3. Run scope-specific series mapping.
4. Apply mapping to produce `scope_metadata_mapped.csv`.
5. Materialize stitched images (scope-specific).
6. Emit and validate `stitched_image_index.csv`.
7. Build and validate `frame_manifest.csv`.
8. Start segmentation using `frame_manifest.csv`.

---

## Notes on Legacy Documents

- `logs/` and `_Archive/` retain historical planning and review context.
- Historical references to `experiment_image_manifest.json` are deprecated.
- Use the four core docs above for current implementation decisions.
