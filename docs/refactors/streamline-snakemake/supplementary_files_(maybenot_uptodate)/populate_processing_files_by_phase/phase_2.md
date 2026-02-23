# Phase 2 – Stitched FF Image Build

Goal: consume Phase 1 aligned metadata plus raw microscope data to
produce normalized stitched FF images and companion diagnostics. This
stage establishes the canonical `image_id`, `well_id`, and channel
naming (via `identifiers/parsing.py`) that all downstream rules rely on.

---

## Required Outputs

- `built_image_data/{exp}/stitched_ff_images/{well_id}/{channel}/{image_id}.tif`
  – canonical stitched FF images per microscope.
- `build_diagnostics/{exp}/stitching_{microscope}.csv` – per-frame QA
  metrics (focus scores, warnings, timing).

Both outputs must strictly follow the layout described in
`processing_files_pipeline_structure_and_plan.md` (Built Image Data
section). Diagnostics CSVs should include basic provenance columns
(experiment_id, well_id, channel, frame_index) plus any
microscope-specific QA metrics. No formal schema validation required -
diagnostics are for human review and retrospective analysis.

**Note:** If stitching quality becomes a production issue requiring
automated monitoring, we can add `schemas/build_diagnostics.py` later to
enforce strict column contracts. For initial implementation, lightweight
validation in each builder is sufficient.

---

## Entry Modules

### `image_building/shared/layout.py`

- Responsibilities: wrap `identifiers/parsing.py` so every builder uses
  the same logic for `well_id`, `image_id`, and stitched FF paths.
- Key helpers:
  - `make_well_id(experiment_id, well_index) -> str` (delegates to parsing)
  - `make_image_path(experiment_id, well_index, channel, frame_index, root) -> Path`
- All per-microscope builders must route path/ID creation through this
  module to guarantee consistent directory structure and naming.

### `image_building/keyence/stitched_ff_builder.py`

- Responsibilities: orchestrate Keyence z-stack collapse, tile stitching,
  flat-field correction, ID generation, and diagnostics writing.
- Inputs:
  - Raw tiles under `raw_image_data/Keyence/{exp}/`.
  - Aligned metadata:
    `input_metadata_alignment/{exp}/aligned_metadata/scope_and_plate.csv`.
- Must call `image_building.shared.layout.make_image_path` (or related
  helpers) when writing outputs.
- Outputs: Keyence slice of `stitched_ff_images/` tree + diagnostics CSV.
- Key requirements:
  - Use `identifiers/parsing.py` helpers for `well_id`, `image_id`.
  - Preserve normalized channel names from aligned metadata (e.g., `BF`,
    `GFP`).
  - Record focus metrics and warnings per frame.

### `image_building/keyence/z_stacking.py`

- Responsibilities: provide focus selection utilities for Keyence tile
  stacks and expose metrics to the builder.
- Must offer deterministic tile ordering, configurable focus methods, and
  structured metrics (per tile/per frame) for diagnostics.

### `image_building/yx1/stitched_ff_builder.py`

- Responsibilities: run YX1-specific preprocessing (channel ordering,
  bit depth, flat-field) while emitting the same stitched FF layout as
  the Keyence path.
- Inputs mirror the Keyence builder, with the raw source under
  `raw_image_data/YX1/{exp}/`.
- Key requirements:
  - Shared ID generation path via `identifiers.parsing` via
    `image_building.shared.layout`.
  - Include provenance columns in diagnostics (experiment_id, well_id,
    channel, frame_index) plus YX1-specific metrics.

---

## Supporting Configuration

- `data_pipeline/config/microscopes.py` – per-microscope channel
  normalization and tile geometry.
- `data_pipeline/config/runtime.py` – GPU/CPU device resolution for stitch
  workloads.
- `identifiers/parsing.py` – single source for experiment/well/image ID
  parsing and formatting.

---

## Validation & Handoff

- Builders should validate provenance columns exist (experiment_id,
  well_id, channel, frame_index) but can emit arbitrary
  microscope-specific metrics without schema enforcement.
- Smoke test each microscope path with representative data to ensure:
  - Output tree matches `built_image_data/{exp}/stitched_ff_images/...`.
  - Diagnosed frames align with metadata (matching `well_id`,
    `channel`, `frame_index`).
  - Channel normalization matches Phase 1 outputs (no fallback to raw
    `ch00` labels).
- Phase 2 completion unlocks Phase 3 (segmentation), which consumes the
  stitched images and image manifest (NOT diagnostics).

**Note:** Manifest generation (`metadata_ingest/manifests/generate_image_manifest.py`)
is Phase 2b and runs after image building completes. The manifest inventories the
stitched images and provides the authoritative frame list for segmentation.

---

## Design Decision: No Diagnostics Schema

Diagnostics CSVs are NOT schema-validated because:
1. No downstream Snakemake rules consume diagnostics as inputs
2. Diagnostics are for human QA review, not automated pipeline logic
3. Per-microscope metrics naturally vary (e.g., Keyence focus vs YX1 bit depth)
4. Enforcing artificial commonality would constrain useful microscope-specific metrics

**When to add schema later:**
- If automated stitching QA becomes production-critical (e.g., auto-reject experiments)
- If cross-experiment diagnostics aggregation requires strict column contracts
- If diagnostics feed into restful hooks or monitoring dashboards

**Until then:** Keep validation lightweight (provenance columns only) and let builders
emit whatever metrics are useful for debugging their microscope type.
