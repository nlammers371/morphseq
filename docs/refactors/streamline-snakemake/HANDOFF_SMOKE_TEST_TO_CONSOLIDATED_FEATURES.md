# Handoff: Smoke Test — Segmentation Through Consolidated Features, Then Analysis-Ready

**Date:** 2026-04-23
**Branch:** `mdcolon/20260222_docs_snakemake_remake`
**Repo:** `/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs`
**Goal:** Run the full pipeline from `segment_and_track_per_well` through
`consolidate_features` on the smoke subset (experiment `20240418`, wells `A01` and `C01`)
and verify every output artifact exists and passes contract.

Analysis-ready wiring is tracked separately in:
- `docs/refactors/streamline-snakemake/QC_AND_ANALYSIS_READY_SPEC.md`
- `docs/refactors/streamline-snakemake/AGENT_A_QC_BRIEFING.md`
- `docs/refactors/streamline-snakemake/AGENT_B_ANALYSIS_READY_BRIEFING.md`

Do not fold analysis-ready changes back into this handoff.

---

## Execution boundary 1 — YX1 smoke through `consolidate_features`

### Current state

#### What is already done (do not re-run)
- `frame_contract.csv` — rebuilt from stitched inventory, 92 rows, wells A01/C01 only ✓
- `wells.txt` — A01, C01 only ✓
- `stitched_image_index.csv` — 92 rows ✓
- Stitched JPEG images — built for A01 (46 frames) and C01 in `built_image_data/20240418/stitched_ff_images/`
- UNet model weights — present in `data_pipeline_output/models/segmentation/`

#### What is pending (the dry-run showed these 14 jobs)
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

#### Config
`src/data_pipeline/pipeline_orchestrator/config.yaml`:
- `experiments: [20250912]` (default) — **pass `--config experiments='["20240418"]'` on every run**
- `experiment_wells.20240418: [A01, C01]` — already set correctly
- GDINO/SAM2 paths are currently empty strings in config — see known issues below

---

### Blockers to resolve before running

#### 1. GDINO and SAM2 model paths are empty
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
Locate the actual model files and fill these in, or confirm that
`src/data_pipeline/segmentation/grounded_sam2/run_per_well.py` handles empty paths
gracefully (falls back to defaults or raises a clear error early).

#### 2. `snip_processing` is absent from the dry-run DAG
`consolidate_features` does not depend on snip manifests in the current Snakefile.
Snip processing runs independently via the `snip_processing_complete` target and is not
a blocker here. No action required.

#### 3. `compute_stage_predictions` needs `plate_metadata.csv`
This rule declares `plate_metadata.csv` as an input. Check whether it exists:
```
data_pipeline_output/experiment_metadata/20240418/plate_metadata.csv
```
If absent, run `normalize_plate_metadata` first, or confirm the rule accepts a missing
plate file without hard-failing.

---

### How to run

#### Commit first
Before running, commit the current working-tree pipeline fixes:
- Wildcard disambiguation in the top-level Snakefile
- Segmentation contract declaration (`segmentation_tracking.csv` as explicit output)
- Stage prediction sentinel separation (`.computed` vs `.validated`)
- Plate metadata schema relaxation (`treatment` no longer required)
- Stage prediction merge robustness (`well_index` KeyError fix)

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq-docs
git add src/data_pipeline/pipeline_orchestrator/ \
        src/data_pipeline/metadata_ingest/frame_contract/ \
        src/data_pipeline/feature_extraction/entrypoints/compute_stage_predictions.py \
        src/data_pipeline/segmentation_and_tracking/pipelines/validate_segmentation_and_tracking.py
git commit -m "Fix pipeline: wildcard constraints, segmentation contract, sentinel separation, schema relaxation, merge robustness"
```

#### Step 1 — dry-run, verify DAG
```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  --config experiments='["20240418"]' \
  --until consolidate_features \
  --dry-run
```
Expected: 14 jobs (or fewer if some are already done). If the count is wrong, investigate
before proceeding.

#### Step 2 — segmentation + auxiliary masks (GPU-dependent, slow)
```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  --config experiments='["20240418"]' \
  --until validate_segmentation_tracking aggregate_auxiliary_masks \
  --cores 1
```
Verify segmentation outputs exist and are non-empty before continuing.

#### Step 3 — feature extraction through `consolidate_features`
```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  snakemake -s src/data_pipeline/pipeline_orchestrator/Snakefile \
  --config experiments='["20240418"]' \
  --until consolidate_features \
  --cores 4
```

---

### Verification checklist

#### Segmentation
```
data_pipeline_output/segmentation_and_tracking/20240418/
  per_well/A01/contracts/segmentation_tracking.csv     ← exists, non-empty
  per_well/C01/contracts/segmentation_tracking.csv     ← exists, non-empty
  contracts/segmentation_tracking.csv                  ← merged, non-empty
  contracts/.segmentation_tracking_merged.validated    ← exists
```

#### Auxiliary masks
```
data_pipeline_output/auxiliary_masks/20240418/
  contracts/auxiliary_masks.csv                        ← exists, non-empty
  contracts/.auxiliary_masks.validated                 ← exists
```

#### Features
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

#### Contract checks (run after Step 3)
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

### If something fails

#### Snakemake rule errors
Check `.snakemake/log/` for the most recent log file — it shows the exact shell command
and Python traceback.

#### Module import errors
All Python runs must use:
```
PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src"
```
This is already set in every Snakefile shell block. If running manually, set it explicitly.

#### Missing output from a rule
Check whether the rule's Python module exists and is importable:
```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  python -m data_pipeline.segmentation.grounded_sam2.run_per_well --help
```

#### `consolidate_features` fails on missing columns
Check `src/data_pipeline/schemas/features.py` for required columns and compare against
what the upstream feature rules actually write.

---

## Follow-on work

Analysis-ready wiring is now tracked separately in the QC/analysis-ready spec and the
Agent A / Agent B briefings listed above. Keep this handoff focused on the smoke path
through `consolidate_features`.

## Key files

| File | Role |
|---|---|
| `src/data_pipeline/pipeline_orchestrator/Snakefile` | Main orchestrator — add rule + update `all` |
| `src/data_pipeline/pipeline_orchestrator/config.yaml` | Fill GDINO/SAM2 paths before smoke run |
| `docs/refactors/streamline-snakemake/QC_AND_ANALYSIS_READY_SPEC.md` | Column contract reference for follow-on work |
| `docs/refactors/streamline-snakemake/AGENT_A_QC_BRIEFING.md` | QC contract spec for the upstream stage |
| `docs/refactors/streamline-snakemake/AGENT_B_ANALYSIS_READY_BRIEFING.md` | Follow-on analysis-ready implementation brief |
