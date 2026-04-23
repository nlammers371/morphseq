# Handoff: Smoke Test — Segmentation Through Consolidated Features, Then Analysis-Ready

**Date:** 2026-04-23
**Branch:** `mdcolon/20260222_docs_snakemake_remake`
**Repo:** `/net/trapnell/vol1/home/mdcolon/proj/morphseq-docs`
**Goal:** Two sequential execution boundaries:
1. Run the full pipeline from `segment_and_track_per_well` through `consolidate_features`
   on the smoke subset (experiment `20240418`, wells `A01` and `C01`) and verify every output
   artifact exists and passes contract.
2. Once the YX1 smoke path is green, wire `analysis_ready/` as a first-class pipeline stage
   and commit it separately.

Do not start analysis-ready wiring until Step 1 is green.

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

## Execution boundary 2 — analysis-ready wiring (after YX1 smoke is green)

Do not start this until `consolidated_snip_features.csv` and its `.validated` sentinel
both exist and pass the contract checks above.

### Goal
Add `analysis_ready/` as a first-class pipeline stage. One new Snakemake rule
(`assemble_analysis_ready`) joins `consolidated_snip_features.csv` + `qc_flags.csv`
into `analysis_ready.csv`. Update the `all` target to require this output.

### Target module structure

```
src/data_pipeline/analysis_ready/
  core/
    assemble.py          ← pure join logic; no I/O, no CLI
  entrypoints/
    assemble_analysis_ready.py   ← CLI: parse args → load → call core → validators → writers
  io/
    loaders.py           ← load_consolidated_features(path), load_qc_flags(path) → DataFrames
    writers.py           ← write CSV + schema.json + .validated sentinel
  validators.py          ← assert_unique_snip_id, assert_1to1_join, assert_no_column_collisions,
                            validate_analysis_ready
```

**Existing stub to promote:**
`src/data_pipeline/analysis_ready/assemble_features_qc_embeddings.py` — move join logic
into `core/assemble.py`, push I/O to the entrypoint, keep computation pure.

**Schema file to update:**
`src/data_pipeline/schemas/analysis_ready.py`:
- rename `use_embryo` → `use_snip`
- remove `mask_quality_flag`
- add `viability_flag`, `motion_flag`

### Validator hard-failure requirements

`validators.py` must raise (not warn) on:
- Duplicate `snip_id` on either input side before the join
- Any `snip_id` present in one input but absent from the other
- Row count of output ≠ row count of each input
- Non-key column name collision between the two inputs

Implement as `assert_1to1_join(left_df, right_df, key="snip_id")` called before the
merge, not after. The join itself must then be deterministic.

### Output contract

```
data_pipeline_output/analysis_ready/{experiment}/
  analysis_ready.csv
  .analysis_ready.validated
  analysis_ready.schema.json
```

`analysis_ready.schema.json` must be generated from the canonical schema definition in
`src/data_pipeline/schemas/analysis_ready.py` at write time by `io/writers.py`. It must
not be stubbed or hand-authored.

`embedding_calculated` column must be present, non-null boolean, all `False` for MVP.

Sentinel naming: `.analysis_ready.validated` — matches the repo convention for
`.{artifact_name}.validated` (dot-prefixed, no path prefix).

### Snakemake changes

In `src/data_pipeline/pipeline_orchestrator/Snakefile`:

1. Add directory constants near the other path constants:
   ```python
   QUALITY_CONTROL_DIR = DATA_ROOT / "quality_control"
   ANALYSIS_READY_DIR = DATA_ROOT / "analysis_ready"
   ```

2. Add rule after `consolidate_features`:
   ```python
   rule assemble_analysis_ready:
       input:
           features = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv",
           features_validated = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv.validated",
           qc_flags = QUALITY_CONTROL_DIR / "{experiment}" / "consolidated" / "qc_flags.csv",
           qc_validated = QUALITY_CONTROL_DIR / "{experiment}" / "consolidated" / ".qc_flags.validated",
       output:
           table = ANALYSIS_READY_DIR / "{experiment}" / "analysis_ready.csv",
           validated = ANALYSIS_READY_DIR / "{experiment}" / ".analysis_ready.validated",
           schema_json = ANALYSIS_READY_DIR / "{experiment}" / "analysis_ready.schema.json",
       shell:
           """
           PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" \
             -m data_pipeline.analysis_ready.entrypoints.assemble_analysis_ready \
             --features-csv "{input.features}" \
             --qc-flags-csv "{input.qc_flags}" \
             --output-csv "{output.table}" \
             --output-schema-json "{output.schema_json}"
           """
   ```

3. Update `rule all` to target `.analysis_ready.validated` per experiment:
   ```python
   rule all:
       default_target: True
       input:
           expand(
               str(ANALYSIS_READY_DIR / "{experiment}" / ".analysis_ready.validated"),
               experiment=EXPERIMENTS,
           )
   ```

### QC dependency note

Agent A (QC wiring) may not be complete. For local development and dry-run verification
only, a mock `qc_flags.csv` may be used — it must have the correct column contract
(`snip_id`, `use_snip`, `viability_flag`, `motion_flag`) and match the `snip_id` set
in `consolidated_snip_features.csv`. The mock is strictly a development aid. It must not
be committed as a real artifact, must not be treated as final verification, and does not
satisfy the done criteria below. Final verification requires real QC output from Agent A.

### Done criteria for analysis-ready

1. `snakemake --config experiments='["20240418"]' --dry-run` shows `assemble_analysis_ready` in DAG
2. Running the smoke subset produces:
   - `analysis_ready/20240418/analysis_ready.csv`
   - `analysis_ready/20240418/.analysis_ready.validated`
   - `analysis_ready/20240418/analysis_ready.schema.json`
3. `analysis_ready.csv` has unique `snip_id`
4. Row count equals `consolidated_snip_features.csv` row count and `qc_flags.csv` row count
5. `embedding_calculated` is uniformly `False`
6. `analysis_ready.schema.json` exists and is valid JSON generated from the schema definition
7. Default `snakemake` (no target, no `--config`) resolves the DAG without error

Commit the analysis-ready wiring as a separate commit after the YX1 smoke commit.

---

## Key files

| File | Role |
|---|---|
| `src/data_pipeline/pipeline_orchestrator/Snakefile` | Main orchestrator — add rule + update `all` |
| `src/data_pipeline/pipeline_orchestrator/config.yaml` | Fill GDINO/SAM2 paths before smoke run |
| `src/data_pipeline/analysis_ready/assemble_features_qc_embeddings.py` | Existing stub — promote to `core/assemble.py` |
| `src/data_pipeline/schemas/analysis_ready.py` | Update QC column names |
| `docs/refactors/streamline-snakemake/QC_AND_ANALYSIS_READY_SPEC.md` | Column contract reference |
| `docs/refactors/streamline-snakemake/AGENT_B_ANALYSIS_READY_BRIEFING.md` | Detailed implementation brief |
| `docs/refactors/streamline-snakemake/AGENT_A_QC_BRIEFING.md` | QC contract spec (upstream of analysis-ready) |
