# Experiment Manager: Orchestration and Path Logic

Status: current implementation notes and usage guide

This document explains how the Experiment Manager orchestrates the MorphSeq pipeline per experiment, how path resolution works (via a single paths.py), and how to dry‑run or execute individual steps and end‑to‑end flows.

---

## High‑Level Overview

- Experiments are first‑class. Each stage writes per‑experiment artifacts in predictable locations under the data root.
- A single source of truth for paths (`src/run_morphseq_pipeline/paths.py`) provides helpers to build all expected file/directory paths. No I/O in the helpers; callers check existence/freshness.
- The manager exposes simple “needs_*” checks to decide whether to run a step. The checks rely on file existence and basic timestamp freshness.
- You can:
  - Inspect status (read‑only)
  - Plan what would run up to a target step (dry‑run)
  - Run end‑to‑end or individual steps, only when needed

---

## Basic Usage

Assume a data root like `/path/to/morphseq_playground` and experiments such as `20250529_36hpf_ctrl_atf6` and `20250612_24hpf_wfs1_ctcf`.

- Read‑only status (concise table):
  - `python -m src.run_morphseq_pipeline.cli status --data-root /path/to/morphseq_playground --model-name 20241107_ds_sweep01_optimum`
  - Fields shown per experiment:
    - `FF` stitched FF images present (`built_image_data/stitched_FF_images/{exp}`)
    - `Z_STITCH` stitched Z stack present (Keyence only; best‑effort indicator)
    - `MASKS` legacy Build02 mask dirs present (✅ or x/5 with `--verbose`)
    - `SAM2` per‑experiment SAM2 CSV present
    - `B03` df01 per‑experiment CSV present
    - `B04` df02 per‑experiment CSV present
    - `LAT` per‑experiment latents CSV (for the selected model) present
    - `B06` df03 per‑experiment CSV present

- Dry‑run plan per experiment (no execution; plans to build06):
  - `python -m scripts.test_manager_run_upto --data-root /path/to/morphseq_playground --exp 20250529_36hpf_ctrl_atf6 20250612_24hpf_wfs1_ctcf --model-name 20241107_ds_sweep01_optimum`
  - Prints three sections per experiment:
    - `ImageGen      : FF | STITCH`
    - `Detect & Seg  : ExpMeta | GDINO | SEG | Masks | Manifest | CSV`
    - `QC and Latents: Build03 | Build04 | Build06`
    - A plan line showing which steps would run, in order (arrows)

- End‑to‑end dry‑run (full orchestration preview):
  - `python -m src.run_morphseq_pipeline.cli pipeline --data-root /path/to/morphseq_playground --experiments 20250529_36hpf_ctrl_atf6,20250612_24hpf_wfs1_ctcf --action e2e --dry-run --model-name 20241107_ds_sweep01_optimum`

When ready, re‑run the same `pipeline` command without `--dry-run` to execute. Use individual actions (sam2, build03, build04, build06) if you want to stop after a specific step.

---

## Paths: Single Source of Truth

All path construction lives in `src/run_morphseq_pipeline/paths.py`. Key helpers:

- Non‑generated inputs:
  - `get_stage_ref_csv(root)` → `{root}/metadata/stage_ref_df.csv`
  - `get_perturbation_key_csv(root)` → `{root}/metadata/perturbation_name_key.csv`
  - `get_well_metadata_xlsx(root, exp)` → `{root}/metadata/well_metadata/{exp}_well_metadata.xlsx`

- Build01:
  - `get_stitched_ff_dir(root, exp)`
  - `get_built_metadata_csv(root, exp)`

- SAM2 (Detect & Seg, authoritative per‑experiment locations):
  - `get_experiment_metadata_json(root, exp)` → `{root}/sam2_pipeline_files/raw_data_organized/experiment_metadata_{exp}.json`
  - `get_gdino_detections_json(root, exp)` → `{root}/sam2_pipeline_files/detections/gdino_detections_{exp}.json`
  - `get_sam2_segmentations_json(root, exp)` → `{root}/sam2_pipeline_files/segmentation/grounded_sam_segmentations_{exp}.json`
  - `get_sam2_masks_dir(root, exp)` → `{root}/sam2_pipeline_files/exported_masks/{exp}/masks`
  - `get_sam2_mask_export_manifest(root, exp)` → `{root}/sam2_pipeline_files/exported_masks/{exp}/mask_export_manifest_{exp}.json`
  - `get_sam2_csv(root, exp)` → `{root}/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv`

- Build03 / Build04 / Build06:
  - `get_build03_output(root, exp)` → df01 per‑exp
  - `get_build04_output(root, exp)` → df02 per‑exp
  - `get_latents_csv(root, model, exp)` → per‑exp latents
  - `get_build06_output(root, exp)` → df03 per‑exp

The `Experiment` and `ExperimentManager` classes use these helpers internally, keeping logic consistent across tools.

---

## Decision Logic (Summary)

- Image generation (Build01): run if FF/stitched outputs missing.
- Detect & Seg (SAM2):
  - Plan to run if ANY required SAM2 intermediary is missing, OR
  - If the SAM2 CSV is older than the stitched FF input (stale output).
  - Required intermediates per experiment:
    - ExpMeta JSON, GDINO JSON, SAM2 segmentations JSON, Masks dir (non‑empty), Mask export manifest JSON, SAM2 metadata CSV

- QC and Latents (per‑experiment):
  - Build03 “needs” if not yet contributed to df01 per‑exp, or inputs newer.
  - Build04 “needs” if missing or older than Build03 per‑exp.
  - Latents “needs” if per‑exp latents CSV missing for the selected model.
  - Build06 “needs” if missing or older than Build04/latents per‑exp.

- Planning order toward build06 target:
  - `build01 → sam2 → build03 → build04 → latents → build06`
  - The planner prints only prerequisites that are actually needed, with arrows.

---

## Running Individual Steps

You can execute steps individually via the manager/experiment methods or the CLI.

- SAM2 (per‑experiment):
  - Python: `exp.run_sam2(workers=8, ...)`
  - CLI (auto‑selected when using pipeline action `sam2`):
    - `python -m src.run_morphseq_pipeline.cli pipeline --data-root /path --experiments EXP --action sam2 --dry-run`

- Build03 (per‑experiment):
  - Python: `exp.run_build03(by_embryo=None, frames_per_embryo=None)`
  - CLI: `python -m src.run_morphseq_pipeline.cli pipeline --action build03 --dry-run`

- Build04 (per‑experiment df02 writer):
  - Python (via manager orchestration): `manager.run_build04()` for the cohort, or per‑exp wrapper when available.

- Latents (per‑experiment):
  - Python: `exp.generate_latents(model_name="20241107_ds_sweep01_optimum")`

- Build06 (per‑experiment df03 writer):
  - Python: `exp.run_build06_per_experiment(model_name=...)`
  - Manager can batch this with `manager.build06_per_experiments(model_name=...)`

In all cases you can use the dry‑run planner to preview the steps first.

---

## Testing and Validation

- Existence snapshot for key paths (per experiment):
  - `python -m scripts.check_paths_experiment --data-root /path/to/morphseq_playground --exp EXP1 EXP2 --model-name 20241107_ds_sweep01_optimum`

- Up‑to‑step planner (strict SAM2 intermediates):
  - `python -m scripts.test_manager_run_upto --data-root /path --exp EXP1 EXP2 --target build06 --model-name 20241107_ds_sweep01_optimum`
  - Output example:
    - `ImageGen      : FF ✅ | STITCH ✅`
    - `Detect & Seg  : ExpMeta ✅ | GDINO ✅ | SEG ✅ | Masks ✅ | Manifest ✅ | CSV ✅`
    - `QC and Latents: Build03 ✅ | Build04 ✅ | Build06 ✅`
    - `Plan          : build06: -> build03 -> build04 -> latents -> build06`

- End‑to‑end orchestration (dry‑run first):
  - `python -m src.run_morphseq_pipeline.cli pipeline --data-root /path --experiments EXP1,EXP2 --action e2e --dry-run --model-name 20241107_ds_sweep01_optimum`

---

## Notes & Tips

- Keep paths.py authoritative; add new helpers there and consume them in the manager/experiment code.
- The SAM2 rerun rule is intentionally conservative for reliability: missing any key intermediary (or stale CSV) will plan a rerun.
- If you maintain multiple models for latents, pass `--model-name` consistently to status/plan/merge tools to align checks.
- Prefer dry‑runs to sanity‑check work plans before executing on large datasets.
