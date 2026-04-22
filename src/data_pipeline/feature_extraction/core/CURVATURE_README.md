# Curvature Metrics

This module computes curvature-based embryo features from the cleaned embryo mask.

The main split is:
- `mask_processing.py` owns generic embryo-mask cleanup.
- `curvature_skeletonization.py` owns skeletonization and centerline extraction.
- `curvature_metrics.py` owns the final curvature summaries.

## Why skeletonization instead of PCA

PCA is useful for a coarse long-axis estimate, but it does not give a usable centerline for curved or strongly bent embryos.

Skeletonization is the right choice here because:
- it follows the actual mask topology
- it can represent curved embryos without forcing a straight-line approximation
- it gives a centerline that can be differentiated into curvature
- it keeps the calculation tied to the real embryo outline rather than a global variance axis

PCA is still useful as a fallback or coarse geometry proxy, but not as the primary curvature method.

## Current outputs

The current curvature feature table writes:
- `mean_curvature_per_um`
- `median_curvature_per_um`
- `max_curvature_per_um`
- `centerline_length_um`
- `centerline_point_count`

## Inputs

Curvature should be computed from:
- `segmentation_and_tracking/{experiment}/per_well/{well_id}/masks/embryo_mask/`
- `segmentation_and_tracking/{experiment}/contracts/segmentation_tracking.csv`
- `experiment_metadata/{experiment}/frame_contract.csv`

The embryo mask is the primary input. The frame contract is used for calibration and row-level joins.

## Tuning knobs

### In `mask_processing.py`

These are the generic cleanup controls:
- `min_component_size`
- `fill_holes_first`
- `keep_largest_component`

Use these when the mask has:
- small islands
- holes in the embryo body
- stray fragments from segmentation

### In `curvature_skeletonization.py`

These controls affect the skeleton path:
- `min_component_size`

This is the place to touch if skeletonization is too fragmented or if the mask cleanup is too aggressive.

### In `curvature_metrics.py`

These controls affect the final numerical stability:
- smoothing window in `_smooth_series(...)`

This is the first thing to tweak if the curvature values are noisy, unstable, or overly sensitive to pixel-level jaggedness.

## What to tweak first

If the curvature output looks wrong, change things in this order:

1. Fix mask cleanup in `mask_processing.py`
2. Check skeletonization in `curvature_skeletonization.py`
3. Adjust smoothing in `curvature_metrics.py`
4. Only then consider changing the underlying geometry method

## What not to change casually

Do not replace skeletonization with PCA unless you are intentionally changing the meaning of the feature.

PCA answers:
- "what is the major axis of the embryo mask?"

Skeletonization answers:
- "what is the embryo centerline, and how curved is it along its true shape?"

Those are related, but they are not the same feature.

## Practical note

Curvature is intentionally isolated as its own module because skeletonization and centerline fitting can be slow. Keep it as a separate Snakemake job and a separate file so it does not block the faster geometry features.

