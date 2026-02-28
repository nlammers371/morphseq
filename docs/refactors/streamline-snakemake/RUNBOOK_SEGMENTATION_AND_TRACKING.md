# Runbook: Phase 3 Segmentation + Tracking (segmentation_and_tracking)

## Scope
This runbook executes Phase 3 segmentation + tracking from `frame_manifest.csv` and produces:
- per-well shards under `data_pipeline_output/segmentation_and_tracking/<exp>/per_well/...`
- merged experiment contracts under `data_pipeline_output/segmentation_and_tracking/<exp>/contracts/...`
- a symlink-only browse view under `data_pipeline_output/segmentation_and_tracking/<exp>/views/...`

Phase 3 uses GroundingDINO for detections + seed selection and SAM2 for native bidirectional propagation.

## Prerequisites
Pinned interpreter:

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
"$PYTHON" -c 'import sys; print(sys.executable)'
```

You must have Phase 2 frame contracts built:
- `data_pipeline_output/experiment_metadata/<exp>/frame_manifest.csv`
- `data_pipeline_output/experiment_metadata/<exp>/.frame_manifest.validated`

## Dry run
```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile -n -j 4 segmentation_and_tracking_complete
```

## Run Phase 3 for one experiment
```bash
snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile -j 4 \
  data_pipeline_output/segmentation_and_tracking/20240418/contracts/.segmentation_tracking.validated
```

## CPU vs CUDA
Phase 3 will fail fast if you request CUDA but CUDA driver initialization is blocked.

Common symptom: `nvidia-smi` works, but `cuInit` fails due to device-node permissions
(` /dev/nvidia-caps/nvidia-cap1 `).

To force CPU for now, set in `src/data_pipeline/pipeline_orchestrator/config.yaml`:
```yaml
segmentation_and_tracking:
  device: "cpu"
```

## Expected outputs
Per-well shard (example `20240418_A01`):
- `.../per_well/20240418_A01/contracts/segmentation_tracking.csv`
- `.../per_well/20240418_A01/contracts/.segment_and_track.validated`
- `.../per_well/20240418_A01/masks/embryo_mask/{snip_id}_mask.png`
- `.../per_well/20240418_A01/artifacts/overlays/embryo_mask/A01_embryo_mask_overlay.mp4`

Merged:
- `.../contracts/segmentation_tracking.csv`
- `.../contracts/.segmentation_tracking.validated`

Browse view (symlinks only):
- `.../views/videos/overlays/embryo_mask/A01_embryo_mask_overlay.mp4`

## Overlay tuning
Overlay render options live in `src/data_pipeline/pipeline_orchestrator/config.yaml`:
```yaml
segmentation_and_tracking:
  qc_overlay:
    render:
      scale: 3.0
      label_font_scale: 1.2
      label_thickness: 3
      banner_font_scale: 1.2
      banner_thickness: 3
```

