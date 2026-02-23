# Refactor‑013 Addendum (2025‑09‑14): Experiment Manager Update (Per‑Experiment Pipeline)

Status: Plan for implementation and audit  
Scope: Integrate Build01 → Build02/SAM2 → Build03 → Build04 → Build06 per‑experiment flow into ExperimentManager with clear inputs/outputs, status checks, and orchestration.

---

## Goals

- Treat each experiment as a first‑class unit with well‑defined inputs/outputs per stage.
- Add ExperimentManager properties and `needs_*` methods that reflect per‑experiment freshness.
- Provide minimal, reliable orchestration to run or resume any stage.
- Make `snip_id` traceable end‑to‑end with a single normalization strategy.

---

## End‑to‑End: Files, Inputs → Outputs (Per Experiment)

Below uses `{root}` for data root and `{exp}` for experiment name (e.g. `20250529_36hpf_ctrl_atf6`).

1) Build01 – Raw → Stitched FF images + metadata
- Inputs
  - Raw images (YX1 `.nd2` or Keyence tiles) and well metadata (`metadata/plate_metadata/{exp}_well_metadata.xlsx`).
- Outputs
  - Stitched FF images: `{root}/built_image_data/stitched_FF_images/{exp}/...`
  - Built metadata CSV: `{root}/metadata/built_metadata_files/{exp}_metadata.csv`

2) Build02 (legacy masks; optional if SAM2 used, but still useful for QC)
- Inputs
  - Stitched FF images from Build01.
  - Model weights under `{root}/segmentation/segmentation_models/`.
- Outputs
  - One directory per model: `{root}/segmentation/{model_name}_predictions/{exp}/...` (JPG masks)
  - Models of interest: embryo, yolk, focus, bubble, viability.

3) SAM2 pipeline (segmentation_sandbox; produces embryo masks + CSV)
- Inputs
  - Stitched FF images (Build01).
- Key Outputs
  - Exported embryo masks (PNG): `{root}/sam2_pipeline_files/exported_masks/{exp}/masks/*.png`
  - Per‑experiment CSV bridge (consumed by Build03):
    - Preferred: `{root}/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv`
    - Fallback (legacy): repository root `sam2_metadata_{exp}.csv`
  - Optional per‑experiment JSONs if configured (Phase 1 plan):
    - GDINO detections: `{root}/sam2_pipeline_files/detections/gdino_detections_{exp}.json`
    - SAM2 segmentations: `{root}/sam2_pipeline_files/segmentation/grounded_sam_segmentations_{exp}.json`

4) Build03 – Embryo processing + snip extraction (per experiment)
- Entry points
  - CLI: `src/run_morphseq_pipeline/steps/run_build03.py` (now exports snips by default)
  - Core: `src/build/build03A_process_images.py`
- Inputs
  - SAM2 CSV (above), stitched FF images (Build01), optional Build02 QC masks.
- Outputs
  - df01 per‑experiment: `{root}/metadata/build03_output/expr_embryo_metadata_{exp}.csv`
  - Snips (images): `{root}/training_data/bf_embryo_snips/{exp}/{snip_id}.jpg` (+ uncropped variant)
  - Snip masks: `{root}/training_data/bf_embryo_masks/emb_{snip_id}.jpg`, `yolk_{snip_id}.jpg`
- `snip_id` creation
  - Derived from image/time context; must be consistent across stages (use the normalizer below when joining).

5) Build04 – QC + stage inference (per experiment)
- Entry point
  - `src/build/build04_perform_embryo_qc.py::build04_stage_per_experiment`
- Inputs
  - df01 per‑exp: `{root}/metadata/build03_output/expr_embryo_metadata_{exp}.csv`
  - Stage reference: `{root}/metadata/stage_ref_df.csv` (plumbed via `stage_ref` argument)
- Outputs
  - df02 per‑exp: `{root}/metadata/build04_output/qc_staged_{exp}.csv`
  - Notes
    - SA outliers use internal controls first, stage_ref fallback; death lead‑time uses predicted_stage_hpf.

6) Build06 – Embeddings + df03 merge (per experiment)
- Entry points
  - CLI: `src/run_morphseq_pipeline/steps/run_build06_per_exp.py` (per‑exp, MVP)
  - Services: `src/run_morphseq_pipeline/services/gen_embeddings.py` (ensure latents, normalize, merge)
- Inputs
  - df02 per‑exp: `{root}/metadata/build04_output/qc_staged_{exp}.csv`
  - Latents per‑exp: `{root}/analysis/latent_embeddings/legacy/{model_name}/morph_latents_{exp}.csv`
    - Generator (Py3.9 subprocess): `src/run_morphseq_pipeline/services/generate_embeddings_py39.py`
- Outputs
  - df03 per‑exp: `{root}/metadata/build06_output/df03_final_ouput_with_latents_{exp}.csv`

Optional combine utilities
- df02 combine: concat all `qc_staged_{exp}.csv` → `metadata/combined_metadata_files/embryo_metadata_df02.csv`.
- df03 combine: concat all per‑exp df03 → `metadata/combined_metadata_files/embryo_metadata_df03.csv`.

---

## Global / Non‑Generated Inputs (Track for Visibility)

- Stage reference CSV: `{root}/metadata/stage_ref_df.csv`
- Perturbation name key CSV: `{root}/metadata/perturbation_name_key.csv`
- Well metadata Excel (per experiment): `{root}/metadata/plate_metadata/{exp}_well_metadata.xlsx`

ExperimentManager should expose these as properties so UIs/status views can quickly show their presence and timestamps. Build04 can bootstrap the perturbation key if missing, but managers should still surface it.

---

## `snip_id` Normalization (Single Source of Truth)

- Use the normalizer from `gen_embeddings.py` (and reuse in Build06 merge) to make joins deterministic.
- Contract: latents must contain `snip_id` equal to `Path(image_path).stem` (dataset enforces this), and df02’s `snip_id` is normalized before join. Avoid guessing from labels.

---

## ExperimentManager: Properties to Add (Per Experiment)

For class `Experiment` (or equivalent), add properties that map directly to per‑exp paths. Examples:

- Build01
  - `ff_dir`: `{root}/built_image_data/stitched_FF_images/{exp}`
  - `built_meta_csv`: `{root}/metadata/built_metadata_files/{exp}_metadata.csv`
- Build02 (optional QC masks)
  - `mask_dir(model_name)`: `{root}/segmentation/{model_name}_predictions/{exp}`
- SAM2
  - `sam2_csv_path`: `{root}/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv`
  - `sam2_masks_dir`: `{root}/sam2_pipeline_files/exported_masks/{exp}/masks`
  - `gdino_detections_path` (if emitted): `{root}/sam2_pipeline_files/detections/gdino_detections_{exp}.json`
  - `sam2_segmentations_path` (if emitted): `{root}/sam2_pipeline_files/segmentation/grounded_sam_segmentations_{exp}.json`
  - `sam2_mask_export_manifest_path` (if emitted): `{root}/sam2_pipeline_files/exported_masks/{exp}/mask_export_manifest_{exp}.json`
- Build03
  - `build03_path`: `{root}/metadata/build03_output/expr_embryo_metadata_{exp}.csv`
  - `snips_dir`: `{root}/training_data/bf_embryo_snips/{exp}`
- Build04
  - `build04_path`: `{root}/metadata/build04_output/qc_staged_{exp}.csv`
  - `stage_ref_csv`: `{root}/metadata/stage_ref_df.csv`
  - `perturbation_key_csv`: `{root}/metadata/perturbation_name_key.csv`
- Build01 (inputs)
  - `well_metadata_xlsx`: `{root}/metadata/plate_metadata/{exp}_well_metadata.xlsx`
- Build06
  - `latents_path(model_name)`: `{root}/analysis/latent_embeddings/legacy/{model}/morph_latents_{exp}.csv`
  - `build06_path`: `{root}/metadata/build06_output/df03_final_ouput_with_latents_{exp}.csv`

---

## ExperimentManager: `needs_*` Methods (Freshness)

Simple timestamp‑based freshness (True = needs work):

- `needs_build01()`: not `ff_dir` or not `built_meta_csv` exists.
- `needs_build02(model)`: not `mask_dir(model)` exists (optional in SAM2 flow).
- `needs_sam2()`:
  - Return True when the primary SAM2 CSV is missing.
  - If your flow requires masks for downstream steps, also return True when `sam2_masks_dir` is missing or empty.
  - Optionally include checks for per‑experiment SAM2 artifacts if those are part of your configured pipeline:
    - `gdino_detections_path`, `sam2_segmentations_path`, and `sam2_mask_export_manifest_path` (manifest that enumerates exported masks).
  - Provide reason strings (e.g., `missing_csv`, `masks_empty`, `missing_mask_manifest`).
- `needs_build03()`: not `build03_path` exists OR any upstream (SAM2 CSV, stitched FF) newer than `build03_path`.
- `needs_build04()`: not `build04_path` exists OR `build03_path` newer than `build04_path`.
- `needs_build06(model)`: not `build06_path` exists OR `build04_path` newer than `build06_path` OR `latents_path(model)` newer than `build06_path`.

Return also a string reason for UI (e.g., "missing", "upstream newer").

---

## Orchestration Flow (Per Experiment)

1) If `needs_build01` → run Build01.
2) If `needs_sam2` (or `needs_build02` in legacy flow) → run SAM2 (and/or Build02).
3) If `needs_build03` → run Build03 wrapper (will export snips by default).
4) If `needs_build04` → run `build04_stage_per_experiment`.
5) If `needs_build06(model)` → run Build06 per‑exp:
   - Default: generate missing latents; `--overwrite` regenerates latents too.
   - Merge df02 + latents; write df03 per‑exp.

All steps: log concise summaries (counts/coverage) and write atomically.

---

## Minimal UI / Status Reporting

For each experiment, show a single row (JSON or table) with:
- Exists: ff_dir, sam2_csv, build03, build04, latents(model), build06
- Freshness flags: needs_build03/04/06
- Counts: df02 rows (quality), embedding dims, join coverage (last run)

Persist a small sidecar `*.status.json` next to df03 per‑exp is optional (not MVP).

---

## Proposed Changes (Code Touch List)

- Add ExperimentManager properties + `needs_*` methods as above.
- Ensure CLI wrappers already present are callable from Manager:
  - Build03: `run_build03_pipeline` or CLI with `--data-root --exp`.
  - Build04: call `build04_stage_per_experiment(...)` directly.
  - Build06: call `run_build06_per_exp.py` or the underlying service.
- Normalize `snip_id` consistently in Build06 merge (already implemented in services).
- Keep logging concise by default; use `--verbose`/env flags to expand.

---

## snip_id Trace (Walkthrough)

- Creation (Build03): From SAM2 CSV rows and stitched image names → `snip_id` (e.g., `{exp}_{well}_t####`).
- Snip export: Writes JPEGs `{snip_id}.jpg` under `training_data/bf_embryo_snips/{exp}`.
- Latent generation: Dataset derives names from file paths; `snip_id = Path(image_path).stem` and writes `morph_latents_{exp}.csv` keyed by `snip_id`.
- Build04 df02: `snip_id` carried through (normalized as needed).
- Build06 merge: Normalizes `snip_id` on both sides, left‑joins df02 with latents, reports coverage.

---

## MVP vs Future

- MVP (implement now): properties, `needs_*`, per‑exp orchestration, atomic writes, concise logs, normalization, default latents generation.
- Future: parallel processing; YAML config; validation reports; cross‑experiment QA; rollbacks; manifests and/or WebDataset for large‑scale snip handling.

---

## Acceptance Checklist

- For a target experiment:
  - `needs_*` correctly reflect file presence/freshness.
  - Manager runs Build03 → Build04 → Build06 per‑exp and writes outputs in the listed locations.
  - df03 per‑exp joins with ≥90% coverage (warn if below) and persists atomically.

---

## Implementation Complete (2025‑09‑14)

### Completed Tasks ✅

1. **Centralized Path Integration**
   - ExperimentManager successfully imports and uses path functions from `src/run_morphseq_pipeline/paths.py`
   - All per-experiment paths now use centralized helpers: `get_build03_output()`, `get_build04_output()`, `get_build06_output()`
   - No inline path formatting - single source of truth maintained

2. **Per-Experiment `needs_*` Methods**
   - Added `exp.needs_build04()` for per-experiment Build04 logic
   - Updated `exp.needs_build06_per_experiment()` for per-experiment Build06 logic
   - Both use proper timestamp-based freshness checks with per-experiment file paths

3. **Deprecated Global Pipeline Checks**
   - `manager.needs_build04` marked deprecated with warning
   - `manager.needs_build06` marked deprecated with warning
   - Clear migration path provided to per-experiment methods

4. **Updated Pipeline Orchestration Display**
   - CLI now shows correct per-experiment flow:
     ```
     4️⃣ 🔄 Embryo processing (Build03) → per-exp df01 - would run
     5️⃣ 🔄 Per-exp QC & staging (Build04) → per-exp df02 - would run
     6️⃣ 🔄 Latent embeddings - would generate
     7️⃣ 🔄 Final merge (Build06) → per-exp df03 - would run
     ```
   - Removed confusing global df01/df02 status displays
   - Added clear dry-run completion messaging

5. **Pipeline Testing**
   - Tested with experiment `20250529_24hpf_ctrl_atf6`
   - Confirmed per-experiment architecture working correctly
   - Pipeline shows proper step-by-step per-experiment flow

### Current Status

**Primary Interface (Working):**
```bash
# Preview per-experiment pipeline
python -m src.run_morphseq_pipeline.cli pipeline e2e \
  --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
  --experiments 20250529_24hpf_ctrl_atf6 \
  --dry-run \
  --model-name 20241107_ds_sweep01_optimum

# Execute per-experiment pipeline
python -m src.run_morphseq_pipeline.cli pipeline e2e \
  --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
  --experiments 20250529_24hpf_ctrl_atf6 \
  --model-name 20241107_ds_sweep01_optimum
```

**Pipeline Flow (Per-Experiment):**
- Build01 → FF images + metadata
- Build02/SAM2 → QC masks / SAM2 segmentation
- Build03 → `{root}/metadata/build03_output/expr_embryo_metadata_{exp}.csv`
- Build04 → `{root}/metadata/build04_output/qc_staged_{exp}.csv`
- Build06 → `{root}/metadata/build06_output/df03_final_ouput_with_latents_{exp}.csv`

### File Status Verification

**Check exact file paths and status:**
```bash
# Status view (read-only per-experiment files)
python -m src.run_morphseq_pipeline.cli status \
  --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
  --model-name 20241107_ds_sweep01_optimum

# Detailed path verification (per-experiment file checker)
python -m scripts.check_paths_experiment \
  --data-root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
  --exp 20250529_24hpf_ctrl_atf6
```

## Pipeline Data Flow Accomplishments (2025‑09‑15)

### Major Bug Fixes & Data Success ✅

**1. YX1 Timestamp Extraction Fixed**
- **Issue:** YX1 `_get_imputed_time_vector()` was replacing ALL timestamps with incorrect ~20-second intervals
- **Fix:** Reverted to working legacy `_fix_nd2_timestamp()` function that preserves actual ND2 timestamps
- **Result:** YX1 experiments now show proper developmental time intervals (20+ minutes vs 20 seconds)
- **Impact:** `predicted_stage_hpf` values are now biologically meaningful
- **Example:** 20250519 experiment duration: 0.29h → 21.13h (proper developmental timeline)

**2. SAM2 Metadata Compatibility Fixed**
- **Issue:** SAM2 CSV missing `well`, `time_int`, `time_string` columns required by Build03
- **Fix:** Updated `export_sam2_metadata_to_csv.py` to extract these from SAM2 JSON structure:
  - `well` = `video_data.get('well_id')` (e.g., "E09")
  - `time_int` = `frame_index + 1` (0-based → 1-based)
  - `time_string` = `f"T{time_int:04d}"` (e.g., "T0001")
- **Result:** SAM2 CSV now fully compatible with Build03 pipeline

**3. Build06 --force Flag Fixed**
- **Issue:** CLI `--force` flag wasn't being passed to Build06 step causing "Output exists" errors
- **Fix:** Updated CLI to pass `overwrite=args.force` to `run_build06_per_experiment()`
- **Result:** Pipeline can now successfully overwrite existing outputs during reprocessing

**4. End-to-End Data Flow Verified**
- **Pipeline Status:** Successfully processed YX1 experiments through complete pipeline
- **Experiments Tested:** 20250519, 20250711, 20250515_part2, etc.
- **Data Quality:** Proper timestamp extraction → correct developmental staging → valid downstream analysis
- **Output Generation:** Per-experiment df03 files with correct latent embeddings

### Next Steps (TODO)

1. **Implement per-experiment Build04 runner** (currently shows warning)
2. **Add combine utilities** for creating global df01/df02/df03 from per-experiment files
3. **Remove legacy helper script** `scripts/test_manager_run_upto.py` → moved to `services/experiment_planner.py`

- Centralized paths (paths.py)
  - Single source of truth for all path templates, e.g., `get_sam2_csv(root, exp)`, `get_build04_output(root, exp)`, etc.
  - Code should call helpers rather than formatting path strings inline.

- Cascading orchestration (make‑like)
  - `run_to_stage('build06')` ensures dependencies are up‑to‑date in order: Build03 → Build04 → Build06.
  - Uses `needs_*` checks to decide which stages to run; resumes cleanly on partial runs.

- Robustness by default
  - Atomic writes for per‑exp outputs (write to tmp, then rename).
  - Status sidecar JSON written next to outputs with: status, counts, coverage, started/finished timestamps, and reason.
  - `needs_*` treats missing, zero‑byte, or schema‑invalid files as “needs work”.

- SAM2 deterministic inputs
  - Single authoritative SAM2 CSV path: `{root}/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv` (no fallback).

- Coverage policy
  - Build06 warns (does not fail) when join coverage < 0.90 and records coverage in the sidecar for later triage.

---

## Testing Strategy (using a fully processed real experiment)

1) Read‑only verification
  - Point `paths.py` helpers at the real data root and `{exp}`. Assert `os.path.exists()` for each generated path.
  - Run ExperimentManager `needs_*` on the completed experiment — all should return False.
  - `run_to_stage('build06', dry_run=True)` should report “up‑to‑date” with no actions.

2) Active testing with backups
  - Backup per‑exp outputs: `build03_output/`, `build04_output/`, `build06_output/`, and `bf_embryo_snips/{exp}/`.
  - Scenario A: Remove `build06_output` file + sidecar → `run_to_stage('build06')` should run only Build06 and regenerate df03.
  - Scenario B: Restore; then remove `build04_output` file → `run_to_stage('build06')` should run Build04 then Build06.
  - After each scenario, verify atomic writes, status sidecar contents (counts, coverage), and restore backups.
