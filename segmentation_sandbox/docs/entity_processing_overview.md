# Entity Processing Overview (segmentation_sandbox)


NOTE:this need to be pdate overwrite needs to be explicity to match this   
       45 +  6) Build06 (MorphSeq Pipeline - Generate df03 with 
          + embeddings)
       46 +  - Selectors: `--experiments`
       47 +  - Default: If no selector is provided, processes 
          + ALL experiments found in df02 (incremental: missing 
          + from df03 only).
       48 +  - Entity types: Experiments only.
       49 +  - Incremental: Process only experiments missing 
          + from df03 by default.
       50 +  - Overwrite semantics: **REQUIRES explicit 
          + experiment specification**:
       51 +    - `--overwrite --experiments "exp1,exp2"` → 
          + reprocess specific experiments
       52 +    - `--overwrite --experiments "all"` → reprocess 
          + ALL experiments (explicit)
       53 +    - `--overwrite` alone → **ERROR** (ambiguous, 
          + dangerous)
       54 +  - Quality filtering: Only processes embryos where 
          + `use_embryo_flag=True` (bypasses Build05 manual 
          + curation)
   
This note documents how each pipeline step selects what to process today: whether it defaults to processing everything, supports targeting specific entities (experiments/videos/images/snips), and whether it has an explicit `--process-all` switch vs. incremental behavior.

The goal is to make it easy to run the full pipeline or a subset consistently and prepare for a unified interface across steps.

## Terminology

- Entity: experiment ID, video ID, image ID, or snip ID depending on context.
- Selectors: CLI flags that limit which entities to process.
- Incremental: processes only new/missing items unless `--overwrite`/`--process-all` is used.

## Summary by Step

1) 01_prepare_videos.py (Video preparation and metadata)
- Selectors: `--entities_to_process` (new) or `--experiments_to_process` (legacy)
- Default: If no selector is provided, processes ALL experiments found in `--directory_with_experiments`.
- Entity types: Experiments only (treats `entities_to_process` as experiments for this step).
- Incremental: Supports dry-run; code inspects existing metadata and can detect experiments needing updates. No explicit `--process-all` switch; default is all unless filtered by a selector.

2) 03_gdino_detection.py (GroundingDINO detection + HQ filtering)
- Selectors: `--entities_to_process`
- Default: If no selector is provided, processes ALL images discovered via metadata.
- Entity types: Experiments, videos, or images. The script resolves the type automatically against metadata.
- Incremental: Skips work if no missing annotations for the target prompt/weights; performs HQ filtering unless `--skip-filtering` is set. No explicit `--process-all` flag.

3) 04_sam2_video_processing.py (SAM2 segmentation over videos)
- Selectors: `--entities_to_process`
- Default: If no selector is provided, processes ALL available videos derived from high‑quality DINO annotations.
- Entity types: Experiment IDs (filters videos by experiment prefix). No direct video/image lists supported via CLI for now.
- Incremental: Runs `process_missing_annotations` by video; supports `--overwrite` if you intend to redo processed videos. No explicit `--process-all` flag.

4) 05_sam2_qc_analysis.py (Quality control and flagging)
- Selectors: `--experiments`, `--videos`, `--images`, `--snips`
- Default: If no selectors are provided, runs INCREMENTAL QC (new entities only).
- Entity types: Experiments, videos, images, snips.
- Incremental: Yes by default. Explicit full reprocess via `--process-all`. Also supports `--dry-run` and `--output` to write to a separate file.

5) 06_export_masks.py (Export labeled masks)
- Selectors: `--entities-to-process` (note hyphen, not underscore)
- Default: If no selector is provided, considers ALL available experiments in the SAM2 annotations.
- Entity types: Experiments only (filters by experiment ID).
- Incremental: Exports missing only by default; `--overwrite` forces re‑export of all.

6) Build06 (MorphSeq Pipeline - Generate df03 with embeddings)
- Selectors: `--experiments`
- Default: If no selector is provided, processes ALL experiments found in df02 (incremental: missing from df03 only).
- Entity types: Experiments only.
- Incremental: Process only experiments missing from df03 by default.
- Overwrite semantics: **REQUIRES explicit experiment specification**:
  - `--overwrite --experiments "exp1,exp2"` → reprocess specific experiments
  - `--overwrite --experiments "all"` → reprocess ALL experiments (explicit)
  - `--overwrite` alone → **ERROR** (ambiguous, dangerous)
- Quality filtering: Only processes embryos where `use_embryo_flag=True` (bypasses Build05 manual curation)

## Current Flag Differences

- Underscore vs. hyphen:
  - Steps 01/03/04 use `--entities_to_process` (underscore).
  - Step 06 uses `--entities-to-process` (hyphen).

- Global full‑processing switch:
  - Step 05 supports `--process-all` (explicit).
  - Steps 01/03/04/06 default to all when no selectors are provided, but rely on incremental/"missing only" behavior internally; they do not expose a uniform `--process-all` flag.

- Selector breadth:
  - Step 03 accepts experiments/videos/images and auto‑infers types.
  - Step 04 expects experiment IDs and filters videos by prefix.
  - Step 05 accepts experiments/videos/images/snips explicitly.
  - Steps 01/06 focus on experiments.

## Practical Recipes

- Process everything end‑to‑end (typical):
  - 01: omit selector (all experiments)
  - 03: omit selector (all images)
  - 04: omit selector (all videos)
  - 05: use `--process-all` to audit all entities
  - 06: omit selector (all experiments; exports missing only)

- Process one or a few experiments:
  - Set `EXAMPLE_EXPS="expA,expB"` in the pipeline or pass per step:
    - 01: `--entities_to_process "expA,expB"`
    - 03: `--entities_to_process "expA,expB"`
    - 04: `--entities_to_process "expA,expB"`
    - 05: `--experiments "expA,expB"`
    - 06: `--entities-to-process "expA,expB"`

- Incremental vs full QC:
  - Default incremental: `05_sam2_qc_analysis.py --input ...`
  - Full re‑audit: `05_sam2_qc_analysis.py --input ... --process-all`

- Export re‑run for a subset:
  - Missing only (default): `06_export_masks.py --sam2-annotations ... --output ... --entities-to-process "expA"`
  - Force re‑export: add `--overwrite`.

## Proposed Unification (future improvement)

- Standardize selector naming to `--entities_to_process` everywhere (underscore), with consistent parsing of a comma‑separated list. Support both for backward compatibility where already shipped.
- Support `--process-all` uniformly across steps to force full processing regardless of incremental state, while keeping current defaults:
  - 01/03/04/06: default still processes all when no selectors provided, but add `--process-all` to explicitly force re‑work where incremental logic would otherwise skip.
  - 05: already implemented; keep as the canonical switch.
- Clarify accepted entity types per step:
  - 01/06: experiments only (fail fast if non‑experiment identifiers are passed via `--entities_to_process`).
  - 03: experiments/videos/images (auto‑detect via metadata).
  - 04: experiments (with optional future extension to `--videos`).
- Add `--dry-run` consistently where missing to preview work.

## File and Code References

- Pipeline driver: `segmentation_sandbox/scripts/pipelines/run_pipeline.sh`
- Step 01: `segmentation_sandbox/scripts/pipelines/01_prepare_videos.py`
- Step 03: `segmentation_sandbox/scripts/pipelines/03_gdino_detection.py`
- Step 04: `segmentation_sandbox/scripts/pipelines/04_sam2_video_processing.py`
- Step 05: `segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py`
- Step 06: `segmentation_sandbox/scripts/pipelines/06_export_masks.py`


## Implementation Plan (Detailed, no code changes yet)

Goal: Make “process-missing” the explicit, named default everywhere; use `--overwrite` only when truly recomputing. Clarify which entity types each step accepts. Keep a consistent selector scheme and startup banner that states exactly what will run.

Global conventions to adopt:
- Primary selectors (per step): `--experiments`, `--videos`, `--images`, `--snips` as applicable.
- Convenience alias (all steps): `--entities_to_process` (underscore) maps to the step’s valid entity type(s).
- Back-compat aliasing: keep previous names as hidden aliases and log a one-line deprecation notice when used (e.g., `--entities-to-process` in step 06; `--experiments_to_process` in step 01).
- Modes:
  - `--process-missing` (default True): explicit name for the normal behavior that processes only missing work.
  - `--overwrite`: recompute/regenerate even if outputs exist. Do not call this "process-all".
  - **IMPORTANT**: `--overwrite` without explicit selectors is DANGEROUS and ambiguous. Always require explicit entity specification:
    - `--overwrite --experiments "exp1,exp2"` → overwrite specific experiments
    - `--overwrite --experiments "all"` → overwrite ALL experiments (explicit intent)
    - `--overwrite` alone → should ERROR or warn about ambiguity
  - Only step 05 retains `--process-all` because it truly means a full re-audit of all entities.
- Startup banner: every step prints a clear banner at start: mode, selectors, counts, and what “no selectors” means. Example:
  - "Mode=process-missing. No selectors given → processing ALL available [images/videos/experiments] from metadata; limited to missing items. Use --experiments to target a subset; use --overwrite to force recompute."


### Step 01 — 01_prepare_videos.py

Current behavior highlights:
- Accepts `--entities_to_process` and legacy `--experiments_to_process`; treats entities as experiments only.
- Default is “all experiments under input dir”.
- Has `--dry-run` and verbose diagnostics.

Planned changes (argparse and flow):
- Add explicit `--experiments` as the primary selector (comma-separated list).
- Keep `--entities_to_process` and `--experiments_to_process` as aliases with `dest="experiments"`.
- Add `--process-missing` (action=store_true, default=True). This is mostly for explicitness; behavior already matches.
- Optional (future): add `--overwrite` if we want to force rebuild videos/metadata, but this may be more invasive; skip for now.

Concrete edits:
- Around the parser definitions (near the top, after `parser = argparse.ArgumentParser(...)` and existing adds):
  - Add:
    - `parser.add_argument("--experiments", help="Comma-separated experiment IDs to process (default: all)")`
    - `parser.add_argument("--process-missing", action="store_true", default=True, help="Process only missing items (default)")`
  - Change existing flags to aliases:
    - `parser.add_argument("--entities_to_process", dest="experiments", help="[Alias] Comma-separated experiment IDs")`
    - `parser.add_argument("--experiments_to_process", dest="experiments", help="[Deprecated alias] Comma-separated experiment IDs")`

- Post-parse mapping (replace current experiment_names parsing):
  - New logic:
    - If `args.experiments`: `selected_experiments = [e.strip() for e in args.experiments.split(',') if e.strip()]`
    - Else: `selected_experiments = None` (means “all”).
  - Print startup banner:
    - If `selected_experiments is None`: print all-mode message with counts of candidate experiments found, explicitly saying mode=process-missing.
    - Else: print targeted experiments list and count.

Why: Makes the “process-missing” default explicit and favors a clear, primary `--experiments` flag.

TODO (Step 01):
- [ ] Add `--experiments` and `--process-missing` args.
- [ ] Alias existing `--entities_to_process` and `--experiments_to_process` to `dest="experiments"`.
- [ ] Replace experiment list parsing with `args.experiments` only.
- [ ] Add startup banner printing mode and counts.


### Step 03 — 03_gdino_detection.py

Current behavior highlights:
- Has `--entities_to_process` (experiments/videos/images auto-resolved) and processes missing items by default.
- HQ filtering runs unless `--skip-filtering` is set.

Planned changes (argparse and flow):
- Add primary selectors: `--experiments`, `--videos`, `--images`.
- Keep `--entities_to_process` as a convenience alias. If provided, parse and auto-resolve types as today.
- Add `--process-missing` (default True) for explicitness.
- Keep `--overwrite` (new flag) to force re-detection on selected or all images. When `--overwrite` is used:
  - Re-run detection for targeted images regardless of existing annotations.
  - HQ filtering runs after detection as usual (unless `--skip-filtering` is set).
  - Optionally add `--recompute-hq` (default True when `--overwrite` is set; otherwise False) if we want finer control — can be deferred.

Concrete edits:
- Argparse block (after current arguments):
  - Add:
    - `parser.add_argument("--experiments", help="Comma-separated experiments to process")`
    - `parser.add_argument("--videos", help="Comma-separated videos to process")`
    - `parser.add_argument("--images", help="Comma-separated images to process")`
    - `parser.add_argument("--process-missing", action="store_true", default=True, help="Process missing items only (default)")`
    - `parser.add_argument("--overwrite", action="store_true", help="Force re-detection for targeted images")`
  - Alias:
    - `parser.add_argument("--entities_to_process", dest="entities_to_process", help="[Alias] Comma-separated entities (exp/video/image)")`

- Target resolution logic (early in main, after parsing):
  - If any of `--experiments`, `--videos`, `--images` specified, build explicit lists.
  - Else if `--entities_to_process` provided, resolve as current implementation.
  - Else: target = ALL images from metadata.

- Missing vs overwrite:
  - If `args.overwrite`:
    - Skip the early “nothing to do” exit.
    - Call `annotations.process_missing_annotations(..., overwrite=True)` and pass explicit target lists.
  - Else (process-missing): keep current missing-only flow.

- Startup banner:
  - Print mode, targets (counts), and whether HQ filtering is enabled.
  - Example: "Mode=process-missing; Prompt='individual embryo'; Targets: 2 experiments, 0 videos, 0 images; HQ filtering=ON".

Why: Avoids "process-all" overload, keeps overwrite semantics clear, and documents accepted entity types.

TODO (Step 03):
- [ ] Add `--experiments|--videos|--images|--process-missing|--overwrite` args.
- [ ] Implement target resolution precedence (explicit lists > entities_to_process > all).
- [ ] Adjust early-exit and processing branches to honor `--overwrite`.
- [ ] Add startup banner with mode, targets, and HQ status.


### Step 04 — 04_sam2_video_processing.py (run_sam2)

Current behavior highlights:
- Accepts `--entities_to_process` and filters videos by experiment prefix.
- Default is to process all available videos, missing-only.

Planned changes (argparse and flow):
- Add primary `--experiments` selector. Keep `--entities_to_process` alias to `dest="experiments"`.
- Add `--process-missing` (default True) and keep existing `--overwrite` flag semantics.
- Optional future: add `--videos` selector to directly target videos; for now, we’ll keep experiments only to avoid scope creep.

Concrete edits:
- Argparse block:
  - Add:
    - `parser.add_argument("--experiments", help="Comma-separated experiment IDs to process (default: all)")`
    - `parser.add_argument("--process-missing", action="store_true", default=True, help="Process only missing videos (default)")`
  - Alias:
    - `parser.add_argument("--entities_to_process", dest="experiments", help="[Alias] Comma-separated experiment IDs")`

- Selection logic (after grouping videos):
  - If `args.experiments`: filter videos by prefix match as today.
  - Else: process all available videos.

- Missing vs overwrite handling:
  - If `args.overwrite`: pass `overwrite=True` to `gsam.process_missing_annotations(...)` and include already processed videos in the target list.
  - Else: keep missing-only behavior.

- Startup banner:
  - Print mode, number of videos targeted, number of experiments represented, and clarify default when no selectors provided.

Why: Explicit default mode, clear targeting, and consistent with other steps.

TODO (Step 04):
- [ ] Add `--experiments` and `--process-missing` args with alias.
- [ ] Implement banner printing with counts and default explanation.
- [ ] Ensure overwrite path includes previously processed videos.


### Step 05 — 05_sam2_qc_analysis.py

Current behavior highlights:
- Has `--process-all` to do full re-audit; otherwise incremental (new entities only).
- Accepts `--experiments|--videos|--images|--snips`.

Planned changes (argparse and flow):
- Keep `--process-all` as is (accurate semantics).
- Add `--process-missing` (default True) as an explicit named default mode (no behavior change; only documentation clarity).
- Add convenience alias `--entities_to_process` that maps to `--experiments` (string → list) for parity with other steps.
- Startup banner clarifies mode and counts.

Concrete edits:
- Argparse block:
  - Add:
    - `parser.add_argument("--process-missing", action="store_true", default=True, help="Process only new entities (default)")`
    - `parser.add_argument("--entities_to_process", dest="experiments", help="[Alias] Comma-separated experiment IDs")`

- Run logic (in `main()` where QC is invoked):
  - If any target selectors provided, pass `target_entities` as today.
  - Mode: if `--process-all` true → full; else → process-missing.
  - Banner prints the mode and target counts.

Why: Surface the default mode explicitly without changing behavior.

TODO (Step 05):
- [ ] Add `--process-missing` and `--entities_to_process` alias.
- [ ] Add startup banner for mode and counts.


### Step 06 — 06_export_masks.py

Current behavior highlights:
- Uses `--entities-to-process` (hyphen) as the selector; defaults to exporting missing masks; `--overwrite` re-exports all.

Planned changes (argparse and flow):
- Introduce primary `--experiments` selector.
- Keep both `--entities_to_process` (underscore) and `--entities-to-process` (hyphen) as aliases with `dest="experiments"`. Print a one-line notice if the hyphen form is used (deprecated alias).
- Add `--process-missing` (default True), purely for explicitness.

Concrete edits:
- Argparse block:
  - Add:
    - `parser.add_argument("--experiments", help="Comma-separated experiment IDs to export (default: all)")`
    - `parser.add_argument("--process-missing", action="store_true", default=True, help="Export missing masks only (default)")`
  - Alias:
    - `parser.add_argument("--entities_to_process", dest="experiments", help="[Alias] Comma-separated experiments")`
    - `parser.add_argument("--entities-to-process", dest="experiments", help="[Deprecated alias] Comma-separated experiments")`

- Selection logic:
  - Prefer `args.experiments` if provided; otherwise all.
  - Continue to use exporter’s missing-only status unless `--overwrite` is set.

- Startup banner:
  - Print mode, targeted experiment count, and whether overwrite is active.
  - If deprecated flag used, print a single-line deprecation notice.

Why: Aligns selector naming and makes default mode explicit.

TODO (Step 06):
- [ ] Add `--experiments` and `--process-missing` args; add aliases for both underscore and hyphen forms.
- [ ] Implement banner printing and deprecation notice for hyphen option.


### Shared Utilities (optional, nice-to-have)

We may add a tiny shared helper for banners, but to avoid imports churn, it’s fine to inline a small banner function in each script:

Example inline helper:
```
def print_start_banner(step_name, mode, targets_desc, extra=None):
    print("=" * 60)
    print(f"{step_name}")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Targets: {targets_desc}")
    if extra:
        for k, v in extra.items():
            print(f"{k}: {v}")
    print()
```

Use mode values: `process-missing` or `overwrite`.


## Updated Usage Examples (to apply after changes)

- Full pipeline, default process-missing everywhere:
  - 01: `... 01_prepare_videos.py --directory_with_experiments ... --output_parent_dir ... --process-missing`
  - 03: `... 03_gdino_detection.py --config ... --metadata ... --annotations ... --process-missing`
  - 04: `... 04_sam2_video_processing.py --config ... --metadata ... --annotations ... --output ... --process-missing`
  - 05: `... 05_sam2_qc_analysis.py --input ... --process-missing`
  - 06: `... 06_export_masks.py --sam2-annotations ... --output ... --process-missing`

- Target selected experiments only (process-missing):
  - 01: `... 01_prepare_videos.py --experiments "20240418,20250612_30hpf_ctrl_atf6"`
  - 03: `... 03_gdino_detection.py --experiments "20240418"`
  - 04: `... 04_sam2_video_processing.py --experiments "20240418"`
  - 05: `... 05_sam2_qc_analysis.py --experiments "20240418"`
  - 06: `... 06_export_masks.py --experiments "20240418"`

- Force recompute on subset:
  - Add `--overwrite` to steps 03/04/06 as needed.


## Per-File TODO Checklist (one file at a time)

1) 01_prepare_videos.py
- [ ] Add `--experiments` and `--process-missing`
- [ ] Map `--entities_to_process` and `--experiments_to_process` → `dest="experiments"`
- [ ] Replace experiment parsing to use `args.experiments`
- [ ] Add startup banner with mode and counts

2) 03_gdino_detection.py
- [ ] Add `--experiments|--videos|--images|--process-missing|--overwrite`
- [ ] Implement target resolution precedence and banner
- [ ] Adjust detection/HQ branches to honor `--overwrite`

3) 04_sam2_video_processing.py
- [ ] Add `--experiments` and `--process-missing` (alias `--entities_to_process`)
- [ ] Implement banner; ensure overwrite includes processed videos

4) 05_sam2_qc_analysis.py
- [ ] Add `--process-missing` and alias `--entities_to_process` → `experiments`
- [ ] Add banner clarifying mode and counts

5) 06_export_masks.py
- [ ] Add `--experiments` and `--process-missing`
- [ ] Accept both `--entities_to_process` and `--entities-to-process` → `dest="experiments"`
- [ ] Add banner and one-line deprecation notice for hyphen alias
