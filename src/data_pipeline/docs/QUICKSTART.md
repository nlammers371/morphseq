# Data Pipeline — Quick Start & Current State

## Overview

The morphseq data pipeline is a Snakemake workflow that takes raw YX1 microscope images
and produces a per-embryo feature table (`analysis_ready.csv`) with QC flags.

Entrypoint: `src/data_pipeline/pipeline_orchestrator/Snakefile`

---

## Prerequisites

- Conda environment: `segmentation_grounded_sam`
- Raw YX1 images at: `{data_root}/inputs/raw_image_data/YX1/{experiment}/`
- Plate metadata CSV at: `{data_root}/inputs/plate_metadata/{experiment}_well_metadata.xlsx` (optional)

---

## Running the Pipeline

### Full experiment (all wells)

```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  snakemake \
  --snakefile src/data_pipeline/pipeline_orchestrator/Snakefile \
  --config data_root=/path/to/data_pipeline_output \
  --cores 4
```

### Single well (smoke test / incremental)

```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  snakemake \
  --snakefile src/data_pipeline/pipeline_orchestrator/Snakefile \
  --config target_wells=B01 data_root=/path/to/data_pipeline_output \
  --cores 1
```

### Multiple specific wells

```bash
--config target_wells=B01,C03,D05 data_root=...
```

### Dry run (preview DAG without executing)

```bash
... --dry-run
```

### Force rerun a specific rule

```bash
... --forcerun compute_motion_qc
```

### Unlock after crash

```bash
... --unlock
```

---

## Key Config Options (`config.yaml`)

| Key | Default | Notes |
|-----|---------|-------|
| `data_root` | `{project_root}/data_pipeline_output` | Override to point at morphseq-docs or another location |
| `experiments` | `[20250912]` | List of experiment IDs to process |
| `microscope` | `YX1` | Microscope type |
| `target_wells` | (all) | Comma-separated well filter, e.g. `B01,C03` |
| `image_building.image_format` | `jpg` | Output format for stitched images (`jpg` or `tif`) |
| `image_building.device` | `cuda` | GPU device for flatfield correction |
| `segmentation_and_tracking.device` | `cuda` | GPU device for GDINO + SAM2 |

Defaults live in `src/data_pipeline/pipeline_orchestrator/config.yaml`.
Any key can be overridden with `--config key=value` on the command line.

---

## Two-Layer Artifact Design

The pipeline distinguishes two kinds of artifacts:

### Experiment-level, additive artifacts

These accumulate as wells are built. They are never filtered by `target_wells`.

| Artifact | Path | Semantics |
|----------|------|-----------|
| `stitched_inventory.csv` | `built_image_data/{exp}/` | All wells successfully built this experiment |
| `frame_contract.csv` | `experiment_metadata/{exp}/` | All built frames with physical metadata (source of truth) |

When you add a new well (`target_wells=C03`), `stitched_inventory.csv` and
`frame_contract.csv` grow to include C03 alongside any previously built wells like B01.

### Run-scoped selection artifacts

These describe what the current invocation chose to process.

| Artifact | Path | Semantics |
|----------|------|-----------|
| `selected_wells.txt` | `experiment_metadata/{exp}/` | Requested wells this run (written by `materialize_selected_wells`) |
| `wells.txt` | `experiment_metadata/{exp}/` | Final well IDs passed to downstream (written by `discover_wells` checkpoint) |

Changing `target_wells` changes `selected_wells.txt`, which triggers `discover_wells`
and all downstream per-well rules — but does **not** invalidate `frame_contract.csv`.

### Cascade invariant

A selected well that has not yet been built upstream is an error at `discover_wells` time.
The correct sequence: build the well first (image building → inventory → frame contract),
then select it for segmentation.

---

## Pipeline DAG (high level)

```
raw_images + plate_metadata
    ↓
extract_scope_metadata_yx1       scope_metadata__yx1.csv
map_series_to_wells_yx1          series_well_mapping.csv
apply_series_mapping_yx1         scope_metadata_mapped.csv
    ↓
materialize_selected_wells       selected_wells.txt   ← from target_wells config
    ↓
build_stitched_images_yx1        .well_{well_index}.done  (per well)
build_stitched_images_yx1_all    stitched_inventory.csv   (aggregate)
    ↓
build_frame_contract             frame_contract.csv       (experiment-level)
    ↓
discover_wells [checkpoint]      wells.txt                (run-scoped)
    ↓ (expands per well_id)
segment_and_track_per_well       segmentation_tracking.csv
run_snip_processing_per_well     snip_manifest.parquet
    ↓ (merge)
merge_segmentation_tracking      contracts/segmentation_tracking.csv
merge_snip_manifests             contracts/snip_manifest.parquet
    ↓
generate_auxiliary_masks         auxiliary_masks.csv

Feature extraction (parallel):
  compute_mask_geometry
  compute_pose_kinematics
  compute_fraction_alive
  compute_stage_predictions
    ↓
consolidate_features             consolidated_snip_features.csv

QC (parallel):
  compute_motion_qc              motion_qc_flags.csv
  compute_focus_qc               focus_qc_flags.csv
  compute_segmentation_qc        segmentation_qc_flags.csv
  compute_viability_qc           viability_qc_flags.csv
  compute_death_detection        death_detection_flags.csv
  compute_surface_area_qc        surface_area_qc_flags.csv
  compute_auxiliary_mask_qc      auxiliary_mask_qc_flags.csv
    ↓
consolidate_qc                   consolidated_qc_flags.csv
    ↓
assemble_analysis_ready          analysis_ready.csv
```

---

## Output Directory Layout

```
{data_root}/
  inputs/
    raw_image_data/YX1/{experiment}/       raw .nd2 files
    plate_metadata/                        well layout spreadsheets

  experiment_metadata/{experiment}/
    scope_metadata__yx1.csv
    series_well_mapping.csv
    scope_metadata_mapped.csv
    selected_wells.txt                     run-scoped selection
    wells.txt                              resolved well IDs (checkpoint output)
    frame_contract.csv                     experiment-level frame manifest
    plate_metadata.csv
    analysis_ready.csv                     FINAL OUTPUT

  built_image_data/{experiment}/
    stitched_ff_images/{well_index}/BF/    stitched jpg images
    .well_{well_index}.done               per-well build sentinel
    stitched_inventory.csv                 all built wells

  segmentation_and_tracking/{experiment}/
    per_well/{well_id}/contracts/          per-well segmentation CSVs
    contracts/segmentation_tracking.csv   merged experiment-level

  processed_snips/{experiment}/
    per_well/{well_id}/contracts/
    contracts/snip_manifest.parquet

  auxiliary_masks/{experiment}/
    contracts/auxiliary_masks.csv

  computed_features/{experiment}/
    mask_geometry/
    pose_kinematics/
    fraction_alive/
    stage_predictions/
    consolidated/consolidated_snip_features.csv

  quality_control/{experiment}/
    motion_qc/
    focus_qc/
    segmentation_qc/
    viability_qc/
    death_detection/
    surface_area_qc/
    auxiliary_mask_qc/
    consolidated_qc_flags.csv
```

---

## Current State (as of 2026-04-24)

### What is working end-to-end

- Metadata ingest: scope metadata extraction, series-to-well mapping, frame contract
- Image building: per-well stitched FF images (jpg), stitched_inventory.csv
- Two-layer artifact design: experiment-level frame_contract + run-scoped wells.txt
- Motion QC: `compute_motion_qc` produces `motion_qc_flags.csv` (threshold: ncc_p05 + min_tile_coverage=0.25)
- Focus QC stub: wired in DAG, produces flags CSV (full rel_entropy signal under development in `results/mcolon/20260423_focus_artifact_detection/`)

### Smoke test status

Experiment `20250912`, well `B01` has been run through:
- [x] scope metadata extraction
- [x] series-to-well mapping
- [x] stitched image building (jpg, `built_image_data/20250912/stitched_ff_images/B01/`)
- [x] stitched_inventory.csv
- [x] frame_contract.csv
- [x] motion QC (`motion_qc_flags.csv` produced, all frames pass)
- [ ] discover_wells → segmentation → features → analysis_ready (not yet run to completion)

Data root for the smoke test: `/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output`

To resume the B01 smoke test:
```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  snakemake \
  --snakefile src/data_pipeline/pipeline_orchestrator/Snakefile \
  --config target_wells=B01 data_root=/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output \
  --cores 1
```

### Known gaps / next work

- **focus_qc**: The rule is a stub; the real rel_entropy-based signal is being developed in `results/mcolon/20260423_focus_artifact_detection/`. Needs to be promoted to a proper module in `quality_control/`.
- **segmentation**: `segment_and_track_per_well` requires GDINO + SAM2 model weights to be configured in `config.yaml` under `segmentation_and_tracking.groundingdino` and `segmentation_and_tracking.sam2`. Models expected at `{data_root}/models/segmentation/`.
- **snip processing**: Wired in DAG but not smoke-tested.
- **`analysis_ready.csv`**: Final assembly rule (`assemble_analysis_ready`) is wired; depends on all QC + features. Not yet validated end-to-end.
- **`data_pipeline_output/` in project root**: There is a stale `data_pipeline_output/` directory at project root from early runs against wrong data_root. It is untracked and can be removed once confirmed safe.

---

## Source Layout

```
src/data_pipeline/
  pipeline_orchestrator/
    Snakefile                   main workflow
    config.yaml                 default configuration
  metadata_ingest/
    scope/yx1/                  YX1 scope metadata extraction + series mapping
    microscope_data_ingest/
      frame_contract/           build_frame_contract.py — experiment-level manifest
  image_building/yx1/           stitched flatfield builder
  auxiliary_masks/              UNet-based mask inference
  segmentation/grounded_sam2/   GDINO + SAM2 tracking
  snip_processing/              embryo crop extraction
  feature_extraction/           mask geometry, kinematics, stage, fraction alive
  quality_control/
    zstack_motion_qc/           NCC-based motion detection
    focus_qc/ (stub)            rel_entropy focus signal (WIP)
    segmentation_qc/
    consolidation/              merge all QC flags
  schemas/                      dataframe schema definitions
  io/                           validators
  docs/                         this file
```
