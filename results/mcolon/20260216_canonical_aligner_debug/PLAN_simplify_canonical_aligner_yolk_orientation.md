# Plan: Simplify Canonical Aligner — Yolk-Based Orientation

## Context

The canonical aligner (`CanonicalAligner` in `uot_grid.py`) computes two landmarks to choose among 4 rotation/flip candidates. The current back-vector computation (`_compute_back_direction_robust()`) is unstable for embryos O13, sample_013, and O19 due to:

1. **Over-engineered cascade with a logic bug**: Method 0 (yolk-proximal) computes pixels *near* the yolk and tries to use them as "back" — but the separation check rejects them (too close to head), falling through to ring expansion (Method 1) which overshoots at 3-10× yolk radius.

2. **Reference yolk never loads** in `s02_compute_ot_features.py`: `load_embryo_frame()` is missing `experiment_date`, `well`, `time_int` in usecols.

3. **Confusing naming**: Code calls yolk COM "head" and opposite extremity "back". We don't actually measure the head — we only have the yolk. Naming should reflect this.

---

## Plan

### Step 1: Rename to yolk-centric terminology

Throughout `CanonicalAligner` in `uot_grid.py`, rename all head/back references to reflect that we're using yolk, not head:

| Old name | New name | Meaning |
|----------|----------|---------|
| `head_yx` (in align loop) | `yolk_yx` | Yolk COM — anchor placed top-left |
| `back_yx` (in align loop) | `back_yx` | Back direction point — keep "back" since yolk IS the back |
| `head_weight` | `yolk_weight` | Scoring weight for yolk anchor |
| `back_weight` | `back_weight` | Keep — this is the actual back |
| `head_cost` | `yolk_cost` | Scoring penalty |
| `back_score` | `back_score` | Keep |
| `head_mask` | `yolk_feature_mask` | The mask used to compute yolk position |
| `head_yx_final` / `head_yx_pre_shift` | `yolk_yx_final` / `yolk_yx_pre_shift` | In meta dict |
| `back_yx_final` / `back_yx_pre_shift` | `back_yx_final` / `back_yx_pre_shift` | Keep |
| `_compute_back_com` | `_compute_back_direction` | Wrapper |
| `_compute_back_direction_robust` | `_compute_back_direction` | Merge into one method |
| `back_quantile` | `back_quantile` | Keep |
| `min_head_back_separation_ratio` | `min_yolk_back_separation_ratio` | Config param |

Key principle: **"yolk" = the yolk COM anchor. "back" = the direction defined by mass around the yolk.** No "head" references — we don't measure head.

The scoring logic in `align()` (lines 436-439) that picks the best rotation/flip should also be renamed:
```python
# OLD:
head_cost = head_yx[1] + head_yx[0]      # penalizes yolk being bottom-right
back_score = back_yx[1] + back_yx[0]      # rewards back being bottom-right
score = (self.back_weight * back_score) - (self.head_weight * head_cost)

# NEW:
yolk_cost = yolk_yx[1] + yolk_yx[0]       # penalizes yolk being far from top-left
back_score = back_yx[1] + back_yx[0]       # rewards back being bottom-right
score = (self.back_weight * back_score) - (self.yolk_weight * yolk_cost)
```
The scoring wants yolk near top-left (low y + low x) and back near bottom-right (high y + high x). This is the convention: **yolk/posterior → top-left, anterior → bottom-right** on the canonical grid. The naming should make this clear.

### Step 2: Rewrite `_compute_back_direction()` — yolk-surrounding centroid

Replace the multi-method cascade with a single clear approach:

**Geometry**: The back of the embryo is near the yolk. To find the back *direction*, sample embryo-mask pixels within a radius around the **yolk COM** and compute their centroid. This centroid IS the back point. All distance computations use the yolk COM as origin — **not** the embryo COM, which shifts with curvature and is unreliable.

```
1. Compute yolk COM (yolk_com_y, yolk_com_x) — this is the origin for everything
2. Compute yolk radius: r_yolk = sqrt(yolk_area / π)
3. Find all embryo-mask pixels within k × r_yolk of yolk COM
   (k = configurable, e.g. 1.5 — larger than the yolk itself
   to capture surrounding embryo tissue, not just yolk pixels)
4. If few pixels found (< some threshold), emit a WARNING (log) but continue —
   do NOT silently fall back to a different method. No minimum-pixel gate.
5. Compute centroid of those pixels = back_yx
6. If back_yx is off-mask, project to nearest embryo-mask pixel
   WITHIN the sampling disk (not globally — avoid projecting to the far end)
7. No fallback cascade. If yolk is None, raise or warn — the caller
   should ensure yolk is present for yolk-based alignment.
```

**No minimum-pixel threshold**: Don't gate on pixel count. If the disk has very few pixels, warn but use whatever is there. We can't standardize a threshold across different resolutions and grid sizes.

**Projection stays local**: If the centroid lands off-mask, project to the nearest mask pixel *within the sampling disk*, not globally. This prevents the artifact of projecting to a distant part of the embryo.

**No fallback methods**: Remove the quantile fallback and ring expansion entirely. If yolk is available, use yolk-surrounding centroid. Period. The no-yolk path should just warn/raise — callers using `align_mode="yolk"` must provide yolk.

**Important**: The `separation_origin_yx` parameter currently uses embryo mask COM. This must change to yolk COM when yolk is available. The scoring in `align()` already uses yolk COM for `yolk_yx` — the back direction must be measured from the same origin.

Remove from `__init__()`:
- `yolk_proximal_radius_k`
- `yolk_proximal_min_pixels`
- `yolk_ring_inner_k`, `yolk_ring_outer_k`, `yolk_ring_min_pixels`
- `back_quantile` (no fallback method needs it)
- `min_head_back_separation_ratio` (no separation gate — just use whatever centroid we get)

Add:
- `back_sample_radius_k: float = 1.5` — multiplier on yolk radius for the sampling disk. Named to clearly indicate: "how far around the yolk do we sample to define the back direction"

All variable names should track their origin. Example naming within `_compute_back_direction()`:
- `yolk_com_y, yolk_com_x` — yolk center of mass
- `r_yolk` — yolk equivalent radius
- `r_sample` — sampling disk radius (= back_sample_radius_k × r_yolk)
- `embryo_pixels_in_disk` — mask pixels within the disk
- `back_centroid_y, back_centroid_x` — centroid of those pixels
- `back_yx` — final back point (possibly projected to mask within disk)

In `align()` (line ~424), change `separation_origin_yx` from mask COM to yolk COM:
```python
# OLD: separation_origin_yx = self._center_of_mass(mask_w)
# NEW: separation_origin_yx = yolk_yx  (= yolk COM, already computed)
```
This ensures back-direction distances are measured from the yolk, not the embryo body centroid which shifts with curvature.

### Step 3: Fix reference yolk loading in `s02_compute_ot_features.py`

In `load_embryo_frame()` (line 69–75), add `experiment_date`, `well`, `time_int` to `usecols`.

### Step 4: Update debug script meta key references

In `debug_alignment_by_embryo_id.py`:
- `meta["head_yx_final"]` → `meta["yolk_yx_final"]`
- `meta["back_yx_final"]` → `meta["back_yx_final"]` (unchanged)
- `meta["debug"]["back_direction"]` — key stays same, internal structure simplified
- Update marker labels: "H" → "Y" (yolk), "B" stays

### Step 5: Update other meta consumers

Search for all files referencing `head_yx_final` and update:
- `debug_phase0_samples_012_013.py`
- `p0_ot_maps.py` (if it stores debug keys)
- Any other scripts in the debug directory

---

## Critical Files

| File | Changes |
|------|---------|
| `src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py` | Steps 1 & 2: rename head→yolk, rewrite back-direction to yolk-surrounding centroid, remove ring expansion params |
| `results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/s02_compute_ot_features.py` | Step 3: fix usecols in `load_embryo_frame()` |
| `results/mcolon/20260216_canonical_aligner_debug/debug_alignment_by_embryo_id.py` | Step 4: update meta key references |
| `results/mcolon/20260216_canonical_aligner_debug/debug_phase0_samples_012_013.py` | Step 5: update meta key references |
| `results/mcolon/20260215_roi_discovery_via_ot_feature_maps/p0_ot_maps.py` | Step 5: check and update if needed |

---

## Verification

1. Run the standalone debug script:
   ```bash
   /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
     /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260216_canonical_aligner_debug/debug_alignment_by_embryo_id.py
   ```

2. Visually inspect the 3 output PNGs in `debug_results/alignment_by_embryo_id/`:
   - Yolk marker (red "Y") should be at top-left, overlapping the blue yolk overlay
   - Back marker (cyan "B") should be near the yolk region — the centroid of embryo tissue surrounding the yolk. It will be close to but not identical to the yolk COM.
   - The yolk-to-back vector should point roughly in the posterior direction
   - The scoring should reliably pick the correct rotation/flip because both yolk_yx and back_yx are in the same region (back/posterior), giving a strong, consistent signal

3. Run the Phase 0 pipeline end-to-end:
   ```bash
   /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
     results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/s02_compute_ot_features.py \
     --reference-embryo-id 20250512_B09_e01 --reference-frame-index 113 \
     --n-wt 10 --n-mut 10 --stage-window 47-49 --seed 42 \
     --output-dir results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/output/phase0_rerun_simplified
   ```

---

## Future Work (out of scope)

**Yolk-pivot fine rotation**: After orientation is locked, do a fine rotation sweep around the yolk COM as pivot to maximize target-vs-reference IoU. This is a target-only refinement step that would come after the aligner produces a correctly oriented mask. Separate plan.

---

## Additional Note

`preprocess_pair_canonical()` in `preprocess.py` should be renamed to `source_target_joint_canonical_alignment()` in a future refactor to better reflect what it does: independently align src and tgt to canonical grid, then jointly crop/pad into a shared bounding box for OT. A TODO has been added to the function docstring.
