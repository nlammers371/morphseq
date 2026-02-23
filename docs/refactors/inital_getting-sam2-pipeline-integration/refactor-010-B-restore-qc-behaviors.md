# Refactor-010-B: Restore QC Features in Build03A (SAM2 Path)

Created: 2025-09-03
Status: Complete ✅
Completed: 2025-09-03
Depends On: Refactor-010 (Standardize Embeddings), Refactor-009 (SAM2 Pipeline Validation)

## Executive Summary
- Problem: During the SAM2 integration, several QC signals computed in the legacy Build03A were simplified or disabled. Most critically, `fraction_alive` was hard-coded to 1.0 (forcing `dead_flag=False`) and yolk/focus/bubble flags were always False when those masks were absent.
- Goal: Restore legacy QC behavior in the SAM2 path while keeping SAM2 as the source of embryo identity/geometry. Specifically: compute `fraction_alive` when viability (via) masks exist; set `fraction_alive = NaN` when via is absent; re-enable yolk/focus/bubble and frame flags; recover speed and geometry metrics.
- Strategy: Implement test-first helper functions for mask-based QC and integrate them into `get_embryo_stats()` with graceful fallbacks. Update snip export to use Build02 masks correctly. Keep SAM2 integer masks as ground truth for embryo selection.

## Gap Analysis (Current vs Legacy)
- Fraction alive / dead_flag
  - Current: `segment_wells_sam2_csv()` seeds `fraction_alive = 1.0` (placeholder); `get_embryo_stats()` uses this to set `dead_flag`, so it never triggers.
  - Legacy: Computed `fraction_alive` per row using a via (viability) mask; set `dead_flag = (fraction_alive < ld_rat_thresh)`.
  - Required: Compute `fraction_alive` from via∧embryo overlap when via mask exists; otherwise set `fraction_alive = NaN` and skip `dead_flag` (False or NaN-based logic preserved downstream).

- Yolk presence (`no_yolk_flag`)
  - Current: Always False (placeholder).
  - Legacy: `no_yolk_flag = not any(yolk ∧ embryo_mask)`.
  - Required: Load yolk mask from Build02 outputs and compute overlap with selected embryo mask label; set accordingly.

- Focus and bubble flags
  - Current: Always False (placeholders).
  - Legacy: Compute distance transform on the inverse embryo mask; flag if nearest focus/bubble pixel is within `2 * qc_scale_px`.
  - Required: Restore proximity-based logic using Build02 focus and bubble masks if present.

- Frame flag and geometry
  - Current: Frame boundary check exists; surface area and PCA-based length/width reintroduced from mask pixels.
  - Legacy: Same behaviors established and tuned.
  - Required: Keep as-is; ensure consistency when mixing SAM2 mask geometry with Build02 masks via nearest-neighbor resize.

- Speed
  - Current: `speed = NaN` placeholder.
  - Legacy: Compute from delta(x,y)/delta(t) when consecutive frames available.
  - Required: Restore when `Time Rel (s)` and previous row available; otherwise keep NaN.

- Mask sourcing and resizing
  - Current: Snip export tries a hardcoded external yolk path; QC disabled in `get_embryo_stats()` due to missing Build02 masks.
  - Legacy: All masks loaded from `<root>/segmentation/<mask_type>_<model>/<date>/*{well}_t####*` and resized (order=0) to match working geometry.
  - Required: Load Build02 masks from `<root>/segmentation/...` (no hardcoded absolute paths). Resize to SAM2 embryo mask shape or FF geometry as needed.

## Scope
- Files: `src/build/build03A_process_images.py` (primary), with non-breaking changes; no schema changes to df01.
- Behaviors:
  - Compute `fraction_alive` when via mask available; set `NaN` when not.
  - Re-enable yolk/focus/bubble QC flags with mask-based proximity checks and fallbacks.
  - Restore `speed` when timing info present.
  - Fix snip export to source yolk from Build02 outputs; resize masks to match SAM2 geometry.

## Design (Test-First Helpers)
- `load_build02_masks(root, row) -> dict`
  - Loads via/yolk/focus/bubble masks from `<root>/segmentation/*_<model>/<date>/*{well}_t####*`.
  - Returns dict with missing keys omitted; caller handles fallbacks.

- `compute_fraction_alive(emb_mask: np.ndarray, via_mask: np.ndarray | None) -> float | NaN`
  - If `via_mask is None`: return NaN.
  - Else: dead_pixels = sum(emb_mask & via_mask); total_embryo = sum(embryo); return 1.0 - dead_pixels/total_embryo (guard zero-denominator).

- `compute_qc_flags(emb_mask: np.ndarray, yolk_mask: np.ndarray | None, focus_mask: np.ndarray | None, bubble_mask: np.ndarray | None, qc_scale_um: int, px_dim: float) -> dict`
  - `frame_flag`: truncation test with `qc_scale_px = ceil(qc_scale_um/px_dim)`.
  - `no_yolk_flag`: True if yolk missing or no overlap.
  - `focus_flag`/`bubble_flag`: if mask present → distance transform proximity check; else False.

- `compute_speed(prev_row, curr_row, px_dim) -> float | NaN`
  - Requires `xpos,ypos` and `Time Rel (s)` continuity.

All helpers tested on synthetic arrays before integration.

## Implementation Plan (Step-by-Step)
1) Unit tests for helpers (new tests)
- Create synthetic binary masks; test:
  - `compute_fraction_alive` returns correct value and NaN when via missing.
  - `compute_qc_flags` frame truncation, yolk overlap logic, and distance-based focus/bubble thresholds.
  - `compute_speed` correctness and NaN cases.

2) Add helper functions (local to module or dedicated utils)
- Implement functions in `build03A_process_images.py` or a small `qc_utils.py` under `src/build/` to avoid circular deps.
- Use nearest-neighbor `resize(..., order=0, preserve_range=True)` to align Build02 masks to SAM2 mask geometry.

3) Integrate into `get_embryo_stats()`
- Load SAM2 integer mask; derive selected embryo binary mask (by region_label).
- Call `load_build02_masks()`; compute `fraction_alive`, QC flags, and `speed`.
- Set `fraction_alive = NaN` if via absent; do not force `1.0` defaults.
- Maintain existing PCA geometry and frame checks; update `use_embryo_flag` logic remains in `compile_embryo_stats`.

4) Fix snip export yolk sourcing
- In `export_embryo_snips()`, load yolk from `<root>/segmentation/yolk_*` rather than hardcoded absolute paths; resize to SAM2 mask shape.
- If yolk missing, continue with zeros + warning (as today).

5) Fallbacks and warnings
- If Build02 masks absent: warn once per type and proceed; `fraction_alive=NaN`, flags default to safe values.

6) Integration tests
- Use a small experiment with and without Build02 masks:
  - Assert df01 contains finite geometry columns, `fraction_alive` finite when via present, NaN otherwise; flags behave as expected.

7) Documentation and UX
- Update docs and CLI guidance to explain that Build03 uses Build02 masks when available and degrades gracefully when absent.

## Data Flow & Fallbacks
- SAM2 remains the source for embryo identity and pixel selection (integer mask → per-label binary mask).
- Build02 masks, when present, enrich QC and `fraction_alive`:
  - via → `fraction_alive` and `dead_flag`.
  - yolk → `no_yolk_flag`.
  - focus/bubble → proximity flags.
- Absent masks do not break the pipeline:
  - `fraction_alive=NaN`, flags False except `frame_flag` from mask geometry.

## Validation Criteria
- When via is present, `fraction_alive` ∈ [0,1] (finite) and `dead_flag` matches threshold rule.
- When via is absent, `fraction_alive` is NaN (not 1.0), `dead_flag` computed accordingly (False or ignored downstream).
- `no_yolk_flag` correctly reflects overlap; focus/bubble flags respect proximity threshold.
- Snip export uses Build02 yolk when present; masks correctly resized to SAM2 geometry.

## Risks & Mitigations
- Mismatched geometries: Always resize legacy masks to the SAM2 mask or FF shape with order=0.
- Missing timing columns: Keep `speed=NaN` when `Time Rel (s)` not available.
- Performance: Distance transforms per row can be heavy; compute only when focus/bubble masks are present.

## Timeline (Completed ✅)
- ✅ **Implemented**: Helper functions in `src/build/qc_utils.py`
  - `compute_fraction_alive()` - handles NaN when via mask missing
  - `compute_qc_flags()` - frame/yolk/focus/bubble flags with fallbacks
  - `compute_speed()` - restored speed calculation
- ✅ **Integrated**: Modified `get_embryo_stats()` in `src/build/build03A_process_images.py`
  - Uses `_load_build02_masks_for_row()` for Build02 mask loading
  - Proper mask resizing to SAM2 geometry
  - Graceful fallbacks when Build02 masks absent
- ✅ **Validated**: Tests pass in `tests/test_build03A_integration_qc.py` and `tests/test_qc_utils.py`

## Implementation Status
**All acceptance criteria met:**
- [x] `fraction_alive` computed from SAM2 embryo + Build02 viability masks (NaN when via absent)
- [x] QC flags restored: yolk/focus/bubble/frame with Build02 mask dependencies
- [x] Speed calculation restored using position/time data
- [x] No hardcoded external paths - uses `<root>/segmentation/<model>_*/<date>/`
- [x] Test-first helper functions implemented and passing
- [x] Integration tests validate end-to-end functionality



