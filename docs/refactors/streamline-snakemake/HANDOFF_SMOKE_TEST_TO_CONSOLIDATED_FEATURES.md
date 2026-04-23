# Handoff: Smoke Test — Segmentation Through Consolidated Features

**Date:** 2026-04-23
**Branch:** `mdcolon/20260222_docs_snakemake_remake`
**Repo:** `/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs`
**Goal:** Run the full pipeline from `segment_and_track_per_well` through `consolidate_features`
on the smoke subset (experiment `20240418`, wells `A01` and `C01`) and verify every output
artifact exists and passes contract.

---

## Current state

### What is already done (do not re-run)
- `frame_contract.csv` — rebuilt from stitched inventory, 92 rows, wells A01/C01 only ✓
- `wells.txt` — A01, C01 only ✓
- `stitched_image_index.csv` — 92 rows ✓
- Stitched JPEG images — built for A01 (46 frames) and C01 in `built_image_data/20240418/stitched_ff_images/`
- UNet model weights — present in `data_pipeline_output/models/segmentation/`

### What is pending (the dry-run showed these 14 jobs)
```
segment_and_track_per_well       × 2  (A01, C01)
generate_auxiliary_masks_well    × 2  (A01, C01)
merge_segmentation_tracking      × 1
validate_segmentation_tracking   × 1
compute_mask_geometry            × 1
compute_curvature_metrics        × 1
compute_pose_kinematics          × 1
compute_fraction_alive           × 1
compute_stage_predictions        × 1
consolidate_features             × 1
aggregate_auxiliary_masks        × 1
```

### Config
`src/data_pipeline/pipeline_orchestrator/config.yaml`:
- `experiments: [20250912]` (default) — **pass `--config experiments='["20240418"]'` on every run**
- `experiment_wells.20240418: [A01, C01]` — already set correctly
- GDINO/SAM2 paths are currently empty strings in config — see known issues below

---

## Known issues to resolve before running

### 1. GDINO and SAM2 model paths are empty
`config.yaml` has:
```yaml
segmentation_and_tracking:
  groundingdino:
    config_path: ""
    weights_path: ""
  sam2:
    config_path: ""
    checkpoint_path: ""
```
The `segment_and_track_per_well` rule passes these as params. You need to locate the
actual model files and fill them in, or confirm whether the rule handles empty paths
gracefully (falls back to defaults, downloads, etc.).

Check: `src/data_pipeline/segmentation/grounded_sam2/run_per_well.py` — look for how it
handles empty config/weight paths.

### 2. `snip_processing` is not in the dry-run DAG
The dry-run shows no `run_snip_processing_per_well` or `merge_snip_manifests`. Either
these are gated behind something, or `consolidate_features` does not depend on snip
manifests. Verify whether snip processing is needed for `consolidate_features` to succeed,
or whether it runs on a separate target.

### 3. `compute_stage_predictions` needs `plate_metadata.csv`
This rule requires `plate_metadata.csv`. Check whether it exists:
```
data_pipeline_output/experiment_metadata/20240418/plate_metadata.csv
```
If not, either run `normalize_plate_metadata` first, or check whether the rule can run
without it (some feature rules have optional plate inputs).

---

## How to run

### Step 1 — dry-run first, verify DAG
```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  --config experiments='["20240418"]' \
  --until consolidate_features \
  --dry-run
```
Expected: 14 jobs (or fewer if some are already done). If the count is wrong, investigate
before proceeding.

### Step 2 — run segmentation + auxiliary masks (GPU-dependent, slow)
```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  --config experiments='["20240418"]' \
  --until validate_segmentation_tracking aggregate_auxiliary_masks \
  --cores 1
```
These are the long-running GPU steps. Run them first and verify outputs before continuing.

### Step 3 — run feature extraction through consolidate_features
```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  --config experiments='["20240418"]' \
  --until consolidate_features \
  --cores 4
```

---

## Verification checklist

Run these checks after Step 3 completes:

### Segmentation
```
data_pipeline_output/segmentation_and_tracking/20240418/
  per_well/A01/contracts/segmentation_tracking.csv     ← exists, non-empty
  per_well/C01/contracts/segmentation_tracking.csv     ← exists, non-empty
  contracts/segmentation_tracking.csv                  ← merged, non-empty
  contracts/.segmentation_tracking_merged.validated    ← exists
```

### Auxiliary masks
```
data_pipeline_output/auxiliary_masks/20240418/
  contracts/auxiliary_masks.csv                        ← exists, non-empty
  contracts/.auxiliary_masks.validated                 ← exists
```

### Features
```
data_pipeline_output/computed_features/20240418/
  mask_geometry/mask_geometry_metrics.csv              ← exists
  curvature_metrics/curvature_metrics.csv              ← exists
  pose_kinematics/pose_kinematics_metrics.csv          ← exists
  fraction_alive/fraction_alive.csv                    ← exists
  stage_predictions/stage_predictions.csv              ← exists
  consolidated/consolidated_snip_features.csv          ← exists
  consolidated/consolidated_snip_features.csv.validated ← exists
```

### Contract checks (run after pipeline)
```bash
conda run -n segmentation_grounded_sam --no-capture-output python - <<'EOF'
import pandas as pd

features = pd.read_csv(
    "data_pipeline_output/computed_features/20240418/consolidated/consolidated_snip_features.csv"
)
seg = pd.read_csv(
    "data_pipeline_output/segmentation_and_tracking/20240418/contracts/segmentation_tracking.csv"
)

print(f"consolidated_snip_features rows: {len(features)}")
print(f"segmentation_tracking rows: {len(seg)}")
print(f"unique wells in features: {sorted(features['well_id'].unique())}")
print(f"snip_id unique: {features['snip_id'].nunique() == len(features)}")
print(f"columns: {list(features.columns)}")
EOF
```

Expected:
- wells = `['A01', 'C01']` only
- `snip_id` unique
- row count plausible (46 frames × 2 wells × N embryos per well)

---

## If something fails

### Snakemake rule errors
Check `.snakemake/log/` for the most recent log file — it shows the exact shell command
and Python traceback.

### Module import errors
All Python runs must use:
```bash
PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src"
```
This is already set in every Snakefile shell block. If running manually, set it explicitly.

### Missing output from a rule
Check whether the rule's Python module exists and is importable:
```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  python -m data_pipeline.segmentation.grounded_sam2.run_per_well --help
```

### `consolidate_features` fails on missing columns
If schema validation fails, check `src/data_pipeline/schemas/features.py` for required
columns and compare against what the upstream feature rules actually write.

---

## Context for next agent

After this smoke test passes, the next steps are:

- **Agent A:** Wire QC integration (Chunk A) — see `AGENT_A_QC_BRIEFING.md`
- **Agent B:** Wire analysis-ready assembly (Chunk B) — see `AGENT_B_ANALYSIS_READY_BRIEFING.md`
- Both agents start from `consolidated_snip_features.csv` as their upstream input

Key files to read before Agent A or B work:
- `docs/refactors/streamline-snakemake/QC_AND_ANALYSIS_READY_SPEC.md`
- `docs/refactors/streamline-snakemake/AGENT_A_QC_BRIEFING.md`
- `docs/refactors/streamline-snakemake/AGENT_B_ANALYSIS_READY_BRIEFING.md`
- `src/data_pipeline/pipeline_orchestrator/Snakefile`
- `src/data_pipeline/pipeline_orchestrator/config.yaml`
