# Plan: Two-Stage Canonical Alignment + Reference-Centric Registration

**Date implemented:** 2026-02-19
**Branch:** `marazzano-main`
**Status:** Implemented and verified

---

## Motivation / Bug Context

`CanonicalAligner.align()` conflated two logically distinct operations:
1. **Stage 1**: Single-embryo canonical alignment (PCA + scale + 0°/180°/flip + anchor shift)
2. **Stage 2**: Src→tgt registration (rotate tgt to match src via IoU sweep)

The bug: when `reference_mask` was passed to `align()`, the coarse 0°/180° candidate IoU
scoring compared center-anchored warped candidates against an anchor-shifted reference,
causing wrong orientation selection for sample_001 (`20251205_F11_e01`) and
sample_015 (`20251017_combined_H07_e01`).

Fix: strict separation of concerns via a locked two-stage architecture.

---

## Hard Constraints (Locked)

1. **UOT never calls registration functions.** UOT consumes masks as provided.
2. Stage 1 and Stage 2 are **never conflated** in production pipelines.
3. Stage 2 operates only on **already-canonical** masks.
4. Pipeline variations are **wrappers / dataset builders**, not modifications to core engines.

---

## Primary File Changed

**`src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py`**

All Stage 1 and Stage 2 logic lives here. No other core files were modified in this refactor.

---

## Stage 1: Canonical Alignment

### `CanonicalAligner.generic_canonical_alignment(mask, original_um_per_px, use_pca=True, return_debug=False) → (canonical_mask, meta)`

**Location:** `uot_grid.py`, class `CanonicalAligner`, ~line 535

- PCA rotation + scale + geometric 0°/180°/flip (no yolk, uses mask COM score)
- Anchor shift: mask COM → `self.anchor_point_xy`
- Candidate scoring: prefer COM in upper-left (score = `-(com_y + com_x)`)
- Calls `_apply_anchor_shift()` and `_validate_output_mask()` internally
- Returns `(canonical_mask, meta)` (two-tuple, no yolk)
- `meta["yolk_used"] = False`

### `CanonicalAligner.embryo_canonical_alignment(mask, original_um_per_px, yolk=None, use_pca=True, return_debug=False) → (canonical_mask, canonical_yolk, meta)`

**Location:** `uot_grid.py`, class `CanonicalAligner`, ~line 641

- If yolk is None/empty: **warns loudly** → delegates to `generic_canonical_alignment()`; `canonical_yolk = None`; `meta["yolk_com_yx"] = None`
- If yolk present: yolk-based anchor (`_coarse_candidate_select()`) + yolk COM pivot + anchor shift
- **`meta["yolk_com_yx"]`** — always present; contains post-alignment yolk COM in canonical coords (y, x) when yolk exists, or `None` when absent. This is the key output consumed by Stage 2.
- Returns `(canonical_mask, canonical_yolk, meta)` (three-tuple)
- `meta["yolk_used"] = True`

### `CanonicalAligner.align()` — deprecated wrapper

**Location:** `uot_grid.py`, ~line 776

- Emits `DeprecationWarning` on every call
- `reference_mask` parameter is a **no-op** — emits additional `DeprecationWarning`
- Delegates to `embryo_canonical_alignment()` when `use_yolk=True`
- Delegates to `generic_canonical_alignment()` when `use_yolk=False`
- Existing callers continue to work unchanged; the yolk-pivot sweep that `align()` previously triggered via `reference_mask` no longer occurs (Stage 2 is now separate)

### New internal helpers on `CanonicalAligner`

| Helper | Location (approx line) | Purpose |
|--------|------------------------|---------|
| `_coarse_candidate_select(mask, yolk, rotation_needed, scale, cx, cy, use_yolk)` | ~363 | Evaluate 0°/180°×flip geometric scoring with yolk; returns `(final_rotation, best_flip, best_yolk_yx, best_back_yx)` |
| `_apply_anchor_shift(aligned_mask, aligned_yolk, use_yolk)` | ~417 | Shift so feature COM → anchor_point_xy; returns `(mask, yolk, shift_x, shift_y, clamped, fit_impossible)` |
| `_validate_output_mask(final_mask, original_mask, scale, rotation, flip, shift_x, shift_y, fit_impossible, retained_ratio)` | ~489 | Raise RuntimeError if mask is empty or touches grid edges |

### `yolk_pivot_angle_range_deg` default change

`CanonicalAligner.__init__()` parameter `yolk_pivot_angle_range_deg` changed from **45.0 → 180.0**.

This affects `_yolk_pivot_rotate()`, which is now only called directly (not via `align()`).
In Stage 2, `generic_src_tgt_register()` does its own ±180° sweep independently.

---

## Stage 2: Src→Tgt Registration

All Stage 2 functions are **module-level** (not methods on `CanonicalAligner`).

### `_apply_pivot_rotation(mask, pivot_yx, angle_deg) → mask`

**Location:** `uot_grid.py`, ~line 815

3-line helper. Rotates `mask` about `pivot_yx` by `angle_deg` using OpenCV affine warp (clockwise-positive convention, consistent with `cv2.getRotationMatrix2D`).

```python
cy, cx = float(pivot_yx[0]), float(pivot_yx[1])
theta = np.radians(float(angle_deg))
cos_t, sin_t = np.cos(theta), np.sin(theta)
M = np.float32([
    [cos_t, -sin_t, cx*(1-cos_t) + cy*sin_t],
    [sin_t,  cos_t, cy*(1-cos_t) - cx*sin_t],
])
return cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST)
```

### `_apply_translation(mask, dyx) → mask`

**Location:** `uot_grid.py`, ~line 828

3-line helper. Translates `mask` by `(dy, dx)`.

### `_mask_com(mask) → (cy, cx)`

**Location:** `uot_grid.py`, ~line 836

Returns center of mass. Falls back to grid center if mask is empty.

### `_iou(a, b) → float`

**Location:** `uot_grid.py`, ~line 845

Standard IoU with `+1e-6` denominator guard.

---

### `generic_src_tgt_register(src_mask, tgt_mask, *, tgt_pivot_yx=None, src_pivot_yx=None, mode="rotate_only", angle_step_deg=1.0, min_iou_absolute=0.25, min_iou_improvement=0.02) → (refined_tgt_mask, meta)`

**Location:** `uot_grid.py`, ~line 853

Pure geometry engine. No biological assumptions.

**Algorithm:**
1. Resolve `tgt_pivot_yx`: use provided value or fall back to `_mask_com(tgt_mask)`
2. Resolve `src_pivot_yx`: use provided value or fall back to `_mask_com(src_mask)`
3. Compute `iou_before = _iou(tgt_mask, src_mask)`
4. Sweep angles `np.arange(-180.0, 180.0 + angle_step_deg, angle_step_deg)` (361 evals at default 1°)
   - For each angle: rotate tgt about `tgt_pivot_yx`, compute IoU with src
   - Track `iou_at_0` (angle closest to 0°), `best_iou`, `best_angle`
5. **Gating**: `apply = (best_iou >= min_iou_absolute) AND (best_iou >= iou_at_0 + min_iou_improvement)`
6. If `apply`: produce `refined = _apply_pivot_rotation(tgt_mask, tgt_pivot_yx, best_angle)`
7. If `mode == "rotate_then_pivot_translate"` and `apply`: additionally translate by `src_pivot_yx - tgt_pivot_yx`
8. Compute `iou_after = _iou(refined, src_mask)`

**Meta schema:**
```python
{
  "applied": bool,
  "mode": "rotate_only" | "rotate_then_pivot_translate",
  "best_angle_deg": float,       # winner of sweep
  "best_iou": float,             # IoU of sweep winner (before gating)
  "hit_boundary": bool,          # abs(best_angle_deg) >= 179.5°
  "angle_deg": float,            # = best_angle_deg if applied else 0.0
  "iou_before": float,           # IoU of tgt vs src before any transform
  "iou_after": float,            # IoU of returned mask vs src
  "tgt_pivot_yx": (cy, cx),      # pivot used
  "tgt_pivot_source": "provided" | "tgt_mask_com",
  "src_pivot_yx": (cy, cx),
  "src_pivot_source": "provided" | "src_mask_com",
  # Only present when mode == "rotate_then_pivot_translate":
  "translate_dyx": (dy, dx),     # = src_pivot_yx - tgt_pivot_yx (if applied) else (0,0)
}
```

**Invariants:**
- `applied=False` → `angle_deg=0.0`, `iou_after == iou_before`
- `applied=True` → `iou_after >= min_iou_absolute` and `iou_after >= iou_before + min_iou_improvement`
- `rotate_then_pivot_translate` + `applied` → `tgt_pivot_yx + translate_dyx ≈ src_pivot_yx` (within ~1px)

```python
# TODO: Add unimodality / peak-sharpness check — skip pivot on flat/multi-modal IoU curves
# (degenerate early-stage embryos with no clear A-P axis).
```

---

### `embryo_src_tgt_register(src_canonical, tgt_canonical, *, src_yolk_com_yx=None, tgt_yolk_com_yx=None, mode="rotate_only", angle_step_deg=1.0, min_iou_absolute=0.25, min_iou_improvement=0.02) → (refined_tgt_canonical, meta)`

**Location:** `uot_grid.py`, ~line 970

Thin biological dispatch wrapper over `generic_src_tgt_register()`.

**Pivot resolution:**
- `tgt_pivot_yx`: prefer `tgt_yolk_com_yx`; else **warn loudly** → pass `None` (generic resolves to mask COM)
- `src_pivot_yx`: prefer `src_yolk_com_yx`; else pass `None` (generic resolves to mask COM)

**Mode downgrade:**
- If `src_yolk_com_yx` is absent and `mode == "rotate_then_pivot_translate"`: silently downgrade to `"rotate_only"` (translation without both yolks is unreliable)

**Relabeled pivot sources in meta:**
- `"tgt_pivot_source"`: `"tgt_yolk_com"` (if `tgt_yolk_com_yx` provided) or `"tgt_mask_com"`
- `"src_pivot_source"`: `"src_yolk_com"` (if `src_yolk_com_yx` provided) or `"src_mask_com"`

These overwrite the `"provided"` / `"*_mask_com"` labels from `generic_src_tgt_register()`.

---

## Production Pipeline Architecture (not yet implemented as builders)

The two stage builders below are the intended production wrappers. **Neither has been implemented as a batch script in this PR** — the plan documents the intended architecture for future work.

### Stage 1 Builder: `build_canonical_dataset(...)`
- Calls `embryo_canonical_alignment()` once per embryo
- Outputs `canonical_*` dataset: canonical masks + yolk sidecars + meta including `yolk_com_yx`

### Stage 2 Builder: `build_ref_registered_dataset(ref_id, canonical_dataset, ...)`
- For each tgt:
  1. Load `src=ref_id` and `tgt` canonical artifacts (including `yolk_com_yx` from Stage 1 meta)
  2. Call `embryo_src_tgt_register(src_can, tgt_can, src_yolk_com_yx=..., tgt_yolk_com_yx=..., mode="rotate_only")`
  3. Propagate transform to tgt yolk sidecar using `reg_meta` only (no re-registration):
     ```python
     if reg_meta["applied"] and tgt_yolk_can is not None:
         tgt_yolk_registered = _apply_pivot_rotation(
             tgt_yolk_can, reg_meta["tgt_pivot_yx"], reg_meta["angle_deg"]
         )
     ```
  4. Write outputs to `ref_<REF_ID>_registered_*` dataset
- UOT consumes either `canonical_*` or `ref_<REF_ID>_registered_*` masks as-is

---

## Debug Script

**`results/mcolon/20260216_canonical_aligner_debug/debug_coarse_scoring_fix.py`**

Validates the Stage 1 + Stage 2 split on:
- Reference: `20250512_B09_e01` f113
- sample_001: `20251205_F11_e01` f50
- sample_015: `20251017_combined_H07_e01` f31

For each target:
1. Stage 1 via `embryo_canonical_alignment()` → `tgt_can_pre` + `tgt_yolk_com_yx`
2. Stage 2 via `embryo_src_tgt_register()` → `tgt_can_post` + `reg_meta`
3. IoU curve plot (full ±180° sweep) with gating thresholds marked
4. 2-panel overlay: pre vs post, src contour (red), tgt yolk overlay (blue), yolk COM marker (cyan)
5. Prints full `reg_meta` fields
6. Assertions: if `applied` → `iou_after >= iou_before + 0.02`
7. If `rotate_then_pivot_translate` + `applied` → assert `tgt_pivot_yx + translate_dyx ≈ src_pivot_yx` (within 1px)

Output: `debug_results/coarse_scoring_fix/`

**Verification result (2026-02-19):**
```
sample_001: applied=False, iou_before=0.8117, best_iou=0.8306, best_angle=-1.0°
sample_015: applied=False, iou_before=0.4521, best_iou=0.4617, best_angle=-2.0°
All assertions PASSED.
```

Both targets had `applied=False` because the improvement (0.019 and 0.010 respectively) fell below the `min_iou_improvement=0.02` gate — they were already well-aligned by Stage 1.

---

## Regression Check

**`results/mcolon/20260216_canonical_aligner_debug/debug_yolk_pivot_rotation.py`**

Calls `aligner.align()` (now a deprecated wrapper) and `_yolk_pivot_rotate()` directly.
All 3 embryos pass (post-pivot IoU ≥ pre-pivot IoU).

```
sample_012 (20251106_E09_e01 f20):  IoU 0.5054 → 0.5150  (+3.0°)  PASS
sample_013 (20251113_C04_e01 f14):  IoU 0.3992 → 0.4674  (+15.0°) PASS
sample_019 (20251205_F06_e01 f71):  IoU 0.5649 → 0.6140  (+7.0°)  PASS
```

---

## Verification Commands

```bash
# Debug script (Stage 1 + Stage 2)
PYTHONPATH=src /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260216_canonical_aligner_debug/debug_coarse_scoring_fix.py

# Regression check (existing yolk-pivot rotation)
PYTHONPATH=src /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
  results/mcolon/20260216_canonical_aligner_debug/debug_yolk_pivot_rotation.py
```

---

## What Was NOT Changed

- `preprocess.py` (`preprocess_pair_canonical`) — remains as legacy path, deprecated separately
- `run_transport.py` — UOT never calls registration (constraint 1)
- Any existing callers of `align()` — deprecated wrapper preserves backward compatibility
- `_yolk_pivot_rotate()` — still present, still callable directly; just no longer called by `align()`

---

## File Map

| File | Role |
|------|------|
| `src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py` | All Stage 1 + Stage 2 implementation |
| `results/mcolon/20260216_canonical_aligner_debug/debug_coarse_scoring_fix.py` | New debug/validation script |
| `results/mcolon/20260216_canonical_aligner_debug/debug_yolk_pivot_rotation.py` | Regression check (unchanged) |
| `results/mcolon/20260216_canonical_aligner_debug/PLAN_two_stage_canonical_alignment.md` | This document |
