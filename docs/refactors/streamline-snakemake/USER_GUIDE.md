# Streamlined Snakemake User Guide

## What This Pipeline Produces
- `scope_metadata_raw.csv`: microscope metadata extracted from raw files
- `scope_metadata_mapped.csv`: scope metadata mapped to plate well IDs
- `scope_and_plate_metadata.csv`: merged scope + plate metadata
- `stitched_image_index.csv`: materialized frame paths and provenance
- `frame_manifest.csv`: analysis-ready frame table

Outputs live under:
- `data_pipeline_output/experiment_metadata/<experiment>/...`
- `data_pipeline_output/built_image_data/<experiment>/stitched_ff_images/...`

## Quickstart
Run full smoke pipeline:

```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  /net/trapnell/vol1/home/mdcolon/proj/morphseq-docs/data_pipeline_output/experiment_metadata/20240509_24hpf/.frame_manifest.validated \
  --cores 4
```

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
  - `T####` directory (or `_T####_` in filename) defines `time_int`.
  - XY layouts without `T*` are treated as single-timepoint (`time_int=0`); filename position token is used as tile ID.
- Stitched framing:
  - Uses shared tiler utility: `src/data_pipeline/image_building/utils/frame_tiler.py`
  - Legacy-compatible canvas scaling is applied from native tile width.

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
    "image_id", "time_int", "image_height_px", "image_width_px",
    "tiler_fallback_used", "tiler_qc_passed", "tiler_qc_reasons"
]].to_string(index=False))
PY
```
