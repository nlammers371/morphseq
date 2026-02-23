# Consolidated Action Items: Code Review Follow-Up
**Date:** 2025-11-06
**Reviews Analyzed:**
- `2025-11-06_codex-review.md` - Critical infrastructure gaps
- `20251106_claude_haiku_code_review.md` - Code quality & completeness

---

## Executive Summary

Two independent reviews of commit 0dd2857 agree on several **critical blocking issues** that must be fixed before the pipeline can run end-to-end:

- **Codex Review:** Identifies missing orchestration and broken metadata flows (more critical)
- **Claude Haiku:** Gives code A- grade but highlights schema/import inconsistencies

**Bottom Line:** The code is architecturally sound but has incomplete implementations and schema violations that prevent execution.

---

## 🔴 CRITICAL BLOCKING ISSUES (Fix First - Pipeline Won't Run)

### Issue 1: Invalid Schema Requires embryo_id in Phase 1
**Both reviews agree this breaks metadata alignment**

**Files:**
- `src/data_pipeline/schemas/scope_and_plate_metadata.py:8-30`
- `src/data_pipeline/metadata_ingest/mapping/align_scope_plate.py:45-75`

**Problem:** Schema requires `embryo_id` but it's not generated until Phase 3 segmentation. Phase 1 metadata alignment will fail validation immediately.

**Fix:** Remove `embryo_id` from REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA

**Blocker:** YES - Pipeline cannot proceed past Phase 1

---

### Issue 2: Missing Snakefile & Experiment Registry
**Codex review - critical architectural gap**

**Files:**
- `src/data_pipeline/pipeline_orchestrator/__init__.py` (empty)
- `src/data_pipeline/pipeline_orchestrator/config/__init__.py` (empty)
- No Snakefile anywhere in repo

**Problem:** No way to execute the modules in dependency order. Plan calls for `pipeline_orchestrator/Snakefile` and `config.registry` for experiment discovery.

**Fix:** Create Snakefile with Phase 1-8 rules (per snakemake_rules_data_flow.md)

**Blocker:** YES - Cannot run pipeline at all without orchestration

---

### Issue 3: Missing Embeddings Stage
**Codex review - documented output doesn't exist**

**Files:**
- `src/data_pipeline/embeddings/__init__.py` (empty stub only)
- Missing: `prepare_manifest.py`, `inference.py`, `subprocess_wrapper.py`, `file_validation.py`

**Problem:** Output structure spec (data_ouput_strcutre.md lines 91-95) promises embeddings CSVs that can't be produced.

**Fix:** Implement embeddings stage OR remove from Phase 8 spec

**Blocker:** YES for full pipeline, but could be stubbed like UNet

---

### Issue 4: Missing Snip Manifest Writer
**Codex review - required output never written**

**Files:**
- `src/data_pipeline/snip_processing/` (missing manifest writer)
- Schema: `src/data_pipeline/schemas/snip_processing.py`
- Output location: `processed_snips/{experiment_id}/snip_manifest.csv`

**Problem:** Phase 4 produces snips but never writes validated manifest. QC expects it as input.

**Fix:** Implement snip manifest writer in snip_processing module

**Blocker:** YES - QC cannot run without snip manifest

---

### Issue 5: YX1 Series Mapping Broken
**Codex review - metadata alignment fails for YX1**

**Files:**
- `src/data_pipeline/metadata_ingest/mapping/series_well_mapper_yx1.py:69,133-139`
- `src/data_pipeline/metadata_ingest/scope/yx1_scope_metadata.py:196-227`

**Problem:**
1. Mapper reads `plate_df['well']` (not schema-backed `well_index`)
2. Mapper output missing required columns (experiment_id, well_id, provenance)
3. No validation on mapper output
4. YX1 scope metadata fabricates numeric well_ids (`exp_00`)
5. Mapper output never joined back into scope metadata

**Fix:**
1. Add REQUIRED_COLUMNS_SERIES_MAPPING to schemas
2. Fix mapper to emit schema-compliant rows
3. Add validation call in mapper
4. Create join step that applies mapping before align_scope_plate

**Blocker:** YES - YX1 experiments cannot reach Phase 1b metadata alignment

---

### Issue 6: Runtime Error - Missing skimage Import
**Codex review - code will crash on execution**

**Files:**
- `src/data_pipeline/snip_processing/extraction.py:8-45`

**Problem:** Calls `skimage.exposure.rescale_intensity()` but never imports skimage module.

**Fix:** Add `import skimage.exposure` at top of file

**Blocker:** YES - Phase 4 crashes immediately on non-uint8 images

---

### Issue 7: CSV Formatter Missing Required Columns
**Both reviews agree - validation will fail**

**Files:**
- `src/data_pipeline/segmentation/grounded_sam2/csv_formatter.py:65-198`
- Schema: `src/data_pipeline/schemas/segmentation.py:8-39`

**Problem:**
1. Defines local schema instead of importing from centralized location
2. Local schema missing 5+ required columns: `experiment_id`, `video_id`, `well_index`, `time_int`, `centroid_x_px`, `centroid_y_px`
3. Creates two sources of truth → inevitable drift

**Fix:** Import REQUIRED_COLUMNS_SEGMENTATION_TRACKING from schemas module, add missing columns to output

**Blocker:** YES - Phase 3 output fails validation

---

### Issue 8: Segmentation QC Expects dict, Gets str
**Codex review - QC crashes on validation**

**Files:**
- `src/data_pipeline/segmentation/grounded_sam2/csv_formatter.py:170-191` (encodes to JSON string)
- `src/data_pipeline/quality_control/segmentation_qc/segmentation_quality_qc.py:32-47` (expects dict)

**Problem:** Formatter serializes RLE mask as JSON string, but QC module expects dict to call `pycocotools.mask.decode()`.

**Fix:** Either store RLE as structured JSON (parse before decoding) OR change QC to json.loads() first

**Blocker:** YES - Phase 6 crashes when processing masks

---

### Issue 9: QC Consolidation Missing Metadata
**Codex review - schema validation fails**

**Files:**
- `src/data_pipeline/quality_control/consolidation/consolidate_qc.py:107-125`
- Schema: `src/data_pipeline/schemas/quality_control.py:5-36`

**Problem:** Schema requires `experiment_id`, `time_int`, `fraction_alive` per row, but consolidation:
1. Starts from segmentation_qc which only has `snip_id`, `embryo_id`, `image_id`
2. Never merges tracking table to recover experiment/time context


**Fix:** Merge in segmentation_tracking or scope metadata to populate experiment/time columns

**Blocker:** YES - Phase 6 QC consolidation fails validation

---

## ⚠️ HIGH PRIORITY ISSUES (Fix Before Next Wave)

### Issue 10: Manifest Enrichment Not Channel-Aware
**Codex review - silent data corruption**

**Files:**
- `src/data_pipeline/metadata_ingest/manifests/generate_image_manifest.py:236-258`

**Problem:** When enriching frames with metadata, only filters on `(well_id, time_int)`. For multi-channel experiments, randomly reuses first row, contaminating BF with GFP timestamps/calibrations.

**Fix:** Filter on `(well_id, channel, time_int)` or `image_id`

**Blocker:** NO - but corrupts downstream data silently

---

### Issue 11: Feature Consolidation Join Misalignment
**Codex review - timestamp/calibration errors**

**Files:**
- `src/data_pipeline/feature_extraction/consolidate_features.py:74-91`

**Problem:** Joins metadata on `well_id` only, but scope_and_plate is keyed by `(well_id, time_int)`. Multiplies snips by frame count, attaches wrong calibrations.

**Fix:** Join on `['well_id', 'time_int']` with validation of one-to-one merge

**Blocker:** NO - but produces incorrect feature metadata

---

### Issue 12: Identifiers Module Never Implemented
**Both reviews - Week 1 task incomplete**

**Files:**
- `src/data_pipeline/identifiers/__init__.py` (empty)
- Plan: Should contain moved `parsing_utils.py` with ID parsing functions

**Problem:** No canonical ID parsing/validation. Modules invent their own concatenation logic.

**Fix:** Move `segmentation_sandbox/scripts/utils/parsing_utils.py` to `src/data_pipeline/identifiers/parsing.py`

**Blocker:** NO - code works without it, but ID consistency not enforced

---

## 🟡 MEDIUM PRIORITY ISSUES (Code Quality)

### Issue 13: Duplicate Columns in analysis_ready Schema
**Claude Haiku - schema maintenance**

**Files:**
- `src/data_pipeline/schemas/analysis_ready.py:26`

**Problem:** Direct list concatenation creates duplicates (experiment_id appears in base + features)

**Fix:** Use set deduplication before converting to list

**Blocker:** NO - inefficient but works

---

### Issue 14: Import Style Inconsistency
**Claude Haiku - 6 files use 3 different styles**

**Files:**
- `src/data_pipeline/metadata_ingest/plate/plate_processing.py:11` (absolute)
- `src/data_pipeline/feature_extraction/consolidate_features.py:13` (relative)
- `src/data_pipeline/quality_control/consolidation/consolidate_qc.py:31` (src. prefix)

**Problem:** Three import styles make codebase hard to navigate

**Fix:** Standardize on relative imports within data_pipeline package

**Blocker:** NO - works but inconsistent

---

### Issue 15: Missing skimage.exposure Import (Details)
**Also in Issue 6 above**

---

### Issue 16: Magic Numbers Without Constants
**Claude Haiku - maintenance issue**

**Files:**
- `src/data_pipeline/metadata_ingest/scope/keyence_scope_metadata.py:76-79` (time conversion)
- `src/data_pipeline/quality_control/consolidation/consolidate_qc.py:115` (death threshold 0.9)
- `src/data_pipeline/metadata_ingest/scope/yx1_scope_metadata.py:66` (1800.0 seconds)

**Fix:** Move to `src/data_pipeline/config/constants.py` with clear documentation

**Blocker:** NO - works but hard to maintain

---

## 📋 ACTION PLAN (Priority Order)

### Phase A: Critical Blocking Issues (Must Do Before Testing)

**1. Fix Schema Violations** (1-2 hours)
   - [ ] Remove `embryo_id` from scope_and_plate_metadata.py schema
   - [ ] Fix csv_formatter to emit all required segmentation_tracking columns
   - [ ] Add missing columns to QC consolidation output

**2. Fix YX1 Metadata Alignment** (2-3 hours)
   - [ ] Add REQUIRED_COLUMNS_SERIES_MAPPING to schemas
   - [ ] Fix series_well_mapper_yx1 output to be schema-compliant
   - [ ] Add validation call to series mapper
   - [ ] Create join step to apply series mapping before align_scope_plate

**3. Fix Runtime Errors** (30 minutes)
   - [ ] Add `import skimage.exposure` to snip_processing/extraction.py
   - [ ] Fix mask_rle serialization (either JSON structure or json.loads in QC) (do it however the current code handles it as we know this works, i think we serialize it into a the json)

**4. Implement Missing Outputs** (3-4 hours)
   - [ ] Write snip_manifest.csv in snip processing module
   - [ ] Stub or implement embeddings stage

**5. Create Snakefile & Orchestration** (2-3 hours)
   - [ ] Implement pipeline_orchestrator/Snakefile
   - [ ] Implement config.registry for experiment discovery
   - [ ] Wire all Phase 1-8 rules

**Total for Phase A: ~10-15 hours of focused work**

### Phase B: High Priority (Before Next Implementation Wave)

**6. Fix Metadata Joins** (1-2 hours)
   - [ ] Make manifest enrichment channel-aware
   - [ ] Fix feature consolidation join to use `(well_id, time_int)`

**7. Complete Identifiers Module** (1-2 hours)
   - [ ] Migrate parsing_utils.py to identifiers/parsing.py
   - [ ] Update all imports

**Total for Phase B: ~2-4 hours**

### Phase C: Quality Improvements (Nice to Have)

**8. Code Quality** (2-3 hours)
   - [ ] Standardize imports to relative style
   - [ ] Move magic numbers to constants module
   - [ ] Deduplicate analysis_ready schema

**Total for Phase C: ~2-3 hours**

---

## Summary by Reviewer

### Codex Review Findings
**Severity:** Critical - Infrastructure gaps
- Missing orchestration (Snakefile, registry)
- Broken YX1 metadata flow
- Missing implementations (embeddings, snip manifest)
- Silent data corruption (channel mixing)

### Claude Haiku Review Findings
**Severity:** Medium - Code quality
- Code is functionally sound (A- grade)
- Mostly consistency/completeness issues
- 3 critical schema issues + 8 recommendations
- Architecture is solid, execution incomplete

---

## Recommended Starting Point

**For fastest path to working pipeline:**

1. Start with Phase A critical blocking issues
2. Focus on issues 1, 2, 3, 6 first (they prevent ANY execution)
3. Then tackle issue 5 (YX1 specific) and issues 7-9 (validation)
4. Create Snakefile once blockers are fixed
5. Test Phase 1-2 before moving to Phase B

**Estimated time to working MVP: 2-3 days of focused work**

---

**Review compiled:** 2025-11-06
**Status:** Ready for prioritization and work assignment
