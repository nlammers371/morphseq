# Streamlined Snakemake User Guide

## What This Pipeline Produces
- `plate_metadata.csv`: user-provided experiment annotation
- `scope_series_metadata_raw.csv`: microscope series metadata extracted from raw files
- `scope_series_metadata_mapped.csv`: series-to-well mapping output
- `stitched_image_index.csv`: materialized frame paths and provenance
- `frame_contract.csv`: segmentation contract built from scope-derived and stitched-image metadata
- Phase 3 segmentation + tracking (per experiment):
  - `segmentation_and_tracking/<experiment>/contracts/segmentation_tracking.csv` (merged contract)
  - `segmentation_and_tracking/<experiment>/views/` (symlink-only browse view: videos/frames/masks)

Outputs live under:
- `data_pipeline_output/experiment_metadata/<experiment>/...`
- `data_pipeline_output/built_image_data/<experiment>/stitched_ff_images/...`
- `data_pipeline_output/segmentation_and_tracking/<experiment>/...`
- `data_pipeline_output/processed_snips/<experiment>/...`

## Quickstart
Run full smoke pipeline:

```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  /net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output/experiment_metadata/20240509_24hpf/.frame_contract.validated \
  --cores 4
```

Run Phase 3 segmentation_and_tracking (after frame contracts):

```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  data_pipeline_output/segmentation_and_tracking/20240418/contracts/.segmentation_tracking.validated \
  --cores 4
```

Where to look:
- Merged experiment contracts: `data_pipeline_output/segmentation_and_tracking/<experiment>/contracts/`
- Per-well shards: `data_pipeline_output/segmentation_and_tracking/<experiment>/per_well/<experiment>_<well>/`
- Browse overlay videos: `data_pipeline_output/segmentation_and_tracking/<experiment>/views/videos/overlays/embryo_mask/`

Run Phase 4 snip_processing (after Phase 3 segmentation_and_tracking):

```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  data_pipeline_output/processed_snips/20240418/contracts/.snip_manifest.validated \
  --cores 4
```

Where to look:
- Per-well processed snips: `data_pipeline_output/processed_snips/<experiment>/per_well/<well_id>/processed/`
- Merged snip manifest: `data_pipeline_output/processed_snips/<experiment>/contracts/snip_manifest.parquet`
- Browse view (symlinks): `data_pipeline_output/processed_snips/<experiment>/views/processed/`

Regenerate stitched frames only:

```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  /net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output/experiment_metadata/20240509_24hpf/.materialize_stitched_images.done \
  --cores 2 --forcerun materialize_stitched_images \
  --config overwrite_materialize_stitched_images=true
```

## Overwrite Behavior
- Global overwrite:
  - `overwrite_all: true` in `src/data_pipeline/pipeline_orchestrator/config.yaml`
- Step overwrite:
  - `overwrite_materialize_stitched_images: true`
  - or `overwrite_steps.materialize_stitched_images: true`
- When materialize overwrite is enabled, stale image files for selected wells/channels are pruned.

## Keyence-Specific Notes
- Tile/time semantics:
  - `T####` directory (or `_T####_` in filename) defines frame order (`time_int`) and legacy alias `time_int`.
  - XY layouts without `T*` are treated as single-timepoint (`time_int=0`, `time_int=0`); filename position token is used as tile ID.
- Stitched framing:
  - Uses shared tiler utility: `src/data_pipeline/image_building/utils/frame_tiler.py`
  - Legacy-compatible canvas scaling is applied from native tile width.

## Frame Index Migration
- Canonical frame key is now `time_int`.
- `time_int` remains a compatibility alias during migration.
- When both columns exist, validators enforce `time_int == time_int`.
- `scope_series_metadata_mapped.csv`, `stitched_image_index.csv`, and `frame_contract.csv` all emit:
  - `experiment_time_s`
  - `frame_interval_s`
  - `frame_interval_min`
  - `frame_interval_hr`
  - `elapsed_time_s`
  - `elapsed_time_min`
  - `elapsed_time_hr`

## New Stitched Index Diagnostics
`stitched_image_index.csv` now includes per-frame tiler diagnostics:
- `tiler_fallback_used`
- `tiler_qc_passed`
- `tiler_qc_reasons`
- `tiler_tile_count`
- `tiler_canvas_height_px`
- `tiler_canvas_width_px`
- `tiler_max_abs_shift_px`

For non-Keyence rows these are marked `not_applicable`/empty.

## Common Checks
Check one frame:

```bash
ls -la data_pipeline_output/built_image_data/20240509_24hpf/stitched_ff_images/B04/BF
```

Check stitched index rows:

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
"$PYTHON" - <<'PY'
import pandas as pd
df = pd.read_csv("data_pipeline_output/experiment_metadata/20240509_24hpf/stitched_image_index.csv")
print(df[(df.well_index=="B04") & (df.channel_id=="BF")][[
    "image_id", "time_int", "time_int", "image_height_px", "image_width_px",
    "tiler_fallback_used", "tiler_qc_passed", "tiler_qc_reasons"
]].to_string(index=False))
PY
```

## Overlay Video Configuration (Phase 3)
Overlay rendering is configured under `segmentation_and_tracking.qc_overlay` in
`src/data_pipeline/pipeline_orchestrator/config.yaml`.

Useful knobs:
- `qc_overlay.render.scale`: scale output video/frames (e.g. `3.0`)
- `qc_overlay.render.label_font_scale`, `qc_overlay.render.label_thickness`: embryo label text
- `qc_overlay.render.banner_font_scale`, `qc_overlay.render.banner_thickness`: image_id banner text

## Snip Processing Configuration (Phase 4)
Snip processing is configured under `snip_processing` in `src/data_pipeline/pipeline_orchestrator/config.yaml`.

Common knobs:
- `snip_processing.mask_type`: which mask head to process (default: `embryo`)
- `snip_processing.target_pixel_size_um`, `snip_processing.output_shape_hw`
- `snip_processing.save_raw_crops`: whether to write `raw_crops/{snip_id}.tif`
- `snip_processing.yolk_mask.enabled`: optionally look for yolk masks for rotation pivot
- `snip_processing.background_stats`: `fixed` or deterministic `estimate`
