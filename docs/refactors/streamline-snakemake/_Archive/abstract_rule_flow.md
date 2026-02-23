# Abstract Pipeline Rule Flow

**Date:** 2025-10-09
**Purpose:** High-level view of Snakemake rule dependencies

---

## Rule Dependency DAG

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: RAW → IMAGES                                            │
├─────────────────────────────────────────────────────────────────┤
│  preprocess_keyence  OR  preprocess_yx1                         │
│      ↓                                                           │
│  stitched_FF/{experiment_id}/*.jpg                              │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: IMAGES → MASKS                                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─ SAM2 Pipeline ─────────────────────────────────────────┐   │
│  │  gdino_detect                                            │   │
│  │      ↓                                                   │   │
│  │  sam2_segment_and_track                                 │   │
│  │      ↓                                                   │   │
│  │  sam2_format_csv  ← NEW (creates tracking_table.csv)    │   │
│  │      ↓                                                   │   │
│  │  sam2_export_masks                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─ UNet Pipeline ──────────────────────────────────────────┐   │
│  │  unet_segment (viability, yolk, focus, bubble)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────┴───────────────┐
          ↓                               ↓
┌────────────────────────┐  ┌─────────────────────────────────────┐
│ STEP 3A: MASKS → SNIPS │  │ STEP 3B: MASKS → FEATURES           │
├────────────────────────┤  ├─────────────────────────────────────┤
│  extract_snips         │  │  compute_spatial_features           │
│                        │  │      ↓                               │
│  (INDEPENDENT)         │  │  compute_shape_features             │
│                        │  │      ↓                               │
└────────────────────────┘  │  infer_embryo_stage                 │
                            └─────────────────────────────────────┘
                                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: FEATURES + MASKS → QC FLAGS (ALL PARALLEL)              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─ Auxiliary Mask QC ────────────────────────────────────┐     │
│  │  qc_imaging        (UNet: yolk, focus, bubble)         │     │
│  │  qc_viability      (UNet: viability + SAM2 + stage)    │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌─ Segmentation QC ──────────────────────────────────────┐     │
│  │  qc_segmentation   (SAM2: mask quality)                │     │
│  │  qc_tracking       (SAM2: speed, trajectory)           │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌─ Feature QC ───────────────────────────────────────────┐     │
│  │  qc_size           (features: SA outlier)              │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: QC CONSOLIDATION                                        │
├─────────────────────────────────────────────────────────────────┤
│  consolidate_qc                                                 │
│      ↓                                                           │
│  compute_use_embryo                                             │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: EMBEDDINGS (QC-GATED)                                   │
├─────────────────────────────────────────────────────────────────┤
│  generate_embeddings  (uses use_embryo_flags.csv)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Rule Count Summary

**Total: 17 rules**

| Step | Rules | Parallel? |
|------|-------|-----------|
| 1. Preprocess | 2 (keyence OR yx1) | No (mutually exclusive) |
| 2. Segmentation | 5 (4 SAM2 + 1 UNet) | UNet parallel with SAM2 |
| 3A. Snips | 1 | Yes (independent) |
| 3B. Features | 3 | Sequential within branch |
| 4. QC Flags | 5 | Yes (all parallel) |
| 5. Consolidation | 2 | Sequential |
| 6. Embeddings | 1 | No |

---

## Key Parallelization Opportunities

### Parallel Group 1: After `preprocess`
```
unet_segment  ||  gdino_detect
```

### Parallel Group 2: After `sam2_export_masks` + `infer_embryo_stage`
```
extract_snips  ||  qc_imaging  ||  qc_viability  ||  qc_segmentation  ||  qc_tracking  ||  qc_size
```
*All 6 can run simultaneously!*

---

## Critical Path (Longest Dependency Chain)

```
preprocess
    ↓
gdino_detect
    ↓
sam2_segment_and_track
    ↓
sam2_format_csv
    ↓
sam2_export_masks
    ↓
compute_spatial_features
    ↓
compute_shape_features
    ↓
infer_embryo_stage
    ↓
qc_size  (or any QC module)
    ↓
consolidate_qc
    ↓
compute_use_embryo
    ↓
generate_embeddings
```

**Depth: 12 steps** (SAM2-heavy, features required for QC)

---

## Rule Dependencies Summary

### Rules with NO dependencies (entry points):
- `preprocess_keyence` OR `preprocess_yx1`

### Rules with SINGLE dependency:
- `gdino_detect` ← preprocess
- `unet_segment` ← preprocess
- `sam2_segment_and_track` ← gdino_detect
- `sam2_format_csv` ← sam2_segment_and_track
- `sam2_export_masks` ← sam2_format_csv + sam2_segment_and_track
- `compute_spatial_features` ← sam2_export_masks
- `compute_shape_features` ← sam2_export_masks
- `infer_embryo_stage` ← compute_shape_features
- `consolidate_qc` ← all 5 QC rules
- `compute_use_embryo` ← consolidate_qc
- `generate_embeddings` ← compute_use_embryo + extract_snips

### Rules with MULTIPLE dependencies:
- `extract_snips` ← sam2_export_masks + sam2_format_csv
- `qc_imaging` ← unet_segment + sam2_format_csv
- `qc_viability` ← unet_segment + sam2_export_masks + infer_embryo_stage
- `qc_segmentation` ← sam2_segment_and_track + sam2_format_csv
- `qc_tracking` ← sam2_format_csv + compute_spatial_features
- `qc_size` ← compute_shape_features + infer_embryo_stage

---

## Data Flow Validation

### ✅ All QC modules get what they need:
- **qc_imaging**: UNet masks (yolk, focus, bubble) + SAM2 tracking
- **qc_viability**: UNet viability + SAM2 masks + stage predictions
- **qc_segmentation**: SAM2 JSON + tracking table
- **qc_tracking**: SAM2 tracking + spatial features
- **qc_size**: Shape features + stage predictions

### ✅ No circular dependencies

### ✅ Consolidation happens AFTER all QC

### ✅ Embeddings use final filtered list

---

## Questions to Validate

1. **Is `sam2_format_csv` in the right place?**
   - Creates `tracking_table.csv` from `propagated_masks.json`
   - Required by: extract_snips, qc_imaging, qc_viability, qc_tracking
   - ✅ Yes, must happen before those rules

2. **Can UNet run in parallel with SAM2?**
   - UNet only needs stitched images
   - SAM2 only needs stitched images
   - ✅ Yes, fully independent

3. **Can snips run in parallel with features?**
   - Snips need: sam2_export_masks + tracking_table.csv
   - Features need: sam2_export_masks + tracking_table.csv
   - ✅ Yes, same inputs, independent computation

4. **Can all 5 QC rules run in parallel?**
   - Each has different inputs but all available after Step 3
   - ✅ Yes, no shared outputs

5. **Does consolidation need to wait for ALL QC?**
   - Yes, merges all 5 QC CSVs
   - ✅ Correct sequential dependency

---

## Design Decisions (Confirmed)

### ✅ UNet failures WILL fail the pipeline
- **qc_viability** requires UNet viability masks for death detection
- Death detection is critical, so UNet must succeed
- No optional QC for now (keep it simple)

### ✅ Stage inference is REQUIRED
- Both **qc_viability** and **qc_size** need `predicted_stage_hpf`
- Pipeline will fail if stage reference is missing
- Explicit dependency, no fallbacks

### ✅ Consolidation will fail if required QC is missing
- No graceful skipping (for now)
- If a QC module fails, the whole pipeline stops
- This ensures data quality (no partial QC results)

### ✅ `generate_embeddings` uses ONLY `use_embryo_flags.csv`
- Does NOT directly touch individual QC files
- `use_embryo_flags.csv` contains: snip_id, use_embryo (bool)
- Simple filter: just list of valid snip_ids per experiment
- Clean separation: QC → consolidation → use_embryo → embeddings

---

## Rule Flow Validated ✓

**All dependencies confirmed correct**
**No circular dependencies**
**Parallelization opportunities maximized**
**Critical path identified: 12 steps**
