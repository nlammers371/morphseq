# Runbook: Metadata + Frame-Contracts Smoke Pipeline

## Scope
This runbook executes the Snakemake smoke workflow for metadata alignment and frame contracts on fixed wells:

- YX1 `20240418`: `A01`, `C01`
- Keyence `20240509_24hpf`: `A04`, `B04`

YX1 mapping is configured to use real stage coordinates (`xy_reference`).

## Prerequisites
1. Default config already points to:
   - raw images: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data`
   - plate metadata: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/well_metadata`
2. If you need alternate paths, edit `src/data_pipeline/pipeline_orchestrator/config.yaml`.
3. Use pinned interpreter:

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
"$PYTHON" -c 'import sys; print(sys.executable)'
```

## Dry run
From repo root:

```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile -n -j 4 all
```

## Execute metadata alignment only
```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile metadata_alignment_complete -j 4
```

## Execute full smoke pipeline
```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile -j 4 all
```

## Expected outputs
Per experiment under `data_pipeline_output/experiment_metadata/{experiment}/`:

- `plate_metadata.csv`
- `.plate_metadata.validated`
- `scope_metadata_raw.csv`
- `series_well_mapping.csv`
- `.physical_well_mapping.validated`
- `physical_well_mapping_diagnostics.json`
- `scope_metadata_mapped.csv`
- `scope_and_plate_metadata.csv`
- `stitched_image_index.csv`
- `.stitched_image_index.validated`
- `frame_manifest.csv`
- `.frame_manifest.validated`

Note: `frame_manifest.csv` is a plate-free physical frame inventory (scope + stitched paths). Plate annotations are joined later on demand.

Stitched images are written to:

- `data_pipeline_output/built_image_data/{experiment}/stitched_ff_images/{well}/{channel}/{image_id}.{ext}`

Default `ext` is `jpg` (configured in `frame_contracts.output_image_extension`).

Keyence materialization defaults to LoG focus-fused tile generation before stitching
(`frame_contracts.keyence.projection_method: "log"`), with filter scale
`frame_contracts.keyence.ff_filter_res_um` and device preference
`frame_contracts.keyence.device_preference`.

Overwrite behavior for stitched-image materialization:
- Default: `overwrite_all: false` and `overwrite_steps.materialize_stitched_images: false` (reuse existing image files).
- Per-step overwrite: set `overwrite_steps.materialize_stitched_images: true`.
- CLI-friendly per-step override: `--config overwrite_materialize_stitched_images=true`.
- Global overwrite: set `overwrite_all: true`.

## Smoke checks
```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
PYTHONPATH=src "$PYTHON" -m pytest tests/test_phase1_phase2_smoke_scope.py -q
```

## Troubleshooting
1. Missing plate metadata file:
   - confirm `{experiment}_well_metadata.xlsx` exists in `data_pipeline_output/inputs/plate_metadata/`.
2. YX1 mapping failures:
   - verify `metadata_alignment.yx1_mapping.ref_xy_csv` points to an existing coordinate reference CSV.
   - if you need to proceed without canonical A01-style mapping (not recommended), set `metadata_alignment.allow_unmapped_wells: true` (will produce S00-style well IDs).
3. Missing stitched files in validators:
   - re-run `materialize_stitched_images` and inspect `stitched_image_index.csv` paths.
