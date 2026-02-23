# 2026-01-21 Status Update

## Summary
- Implemented canonical alignment via PCA long-axis alignment + flip search + yolk/COM anchoring.
- Fixed PCA rotation sign (OpenCV clockwise) so aligned PCA axis should match grid long axis.
- Added fit-aware anchor shift clamping and clipping warnings (configurable to error).
- Added alignment debug script to inspect raw/pre-shift/final masks + metadata.
- Set canonical target resolution to **10.0 µm/px** (source um/px still respected).

## Key Files Touched
- src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py
- src/analyze/optimal_transport_morphometrics/uot_masks/preprocess.py
- src/analyze/utils/optimal_transport/config.py
- results/mcolon/20260121_uot-mvp/debug_uot_params.py
- results/mcolon/20260121_uot-mvp/debug_canonical_alignment.py

## Notes
- Alignment metadata now includes `pca_angle_deg`, `rotation_needed_deg`, `rotation_deg`,
  and `aligned_pca_angle_deg` for debugging.
- Anchor default moved to center (0.5, 0.5) to reduce clipping.

## Next Steps
- Pipeline yolk masks into preprocessing (yolk-loaded alignment and anchoring).
- Validate cross-embryo alignment visually after yolk integration.
