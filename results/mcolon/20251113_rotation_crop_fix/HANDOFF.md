# Embryo Extraction Scale Update: 6.5 → 7.8 μm/px

## Summary
Updated embryo extraction pipeline to use **7.8 μm/px** instead of 6.5 μm/px to reduce embryo clipping for large embryos during rotation/cropping.

## Key Changes

### 1. **Scale Change (Single Parameter)**
- **File**: `src/build/build03A_process_images.py`
- **Function**: `extract_embryo_snips()` (line 1336)
- **Parameter**: `outscale: float = 7.8` (default, was 6.5)
- **Effect**: Larger canvas per pixel → embryos fit with more margin during rotation

### 2. **Out-of-Frame Flag Now Available**
The extraction function now computes and returns an **`out_of_frame_flag`** for each embryo:
- **Location**: Returned by `export_embryo_snips()` (line 434)
- **Meaning**: `True` if > 2% of embryo mask was clipped during cropping
- **Threshold**: < 98% mask area retained (line 171 in image_utils.py)
- **Stored in**: `stats_df["out_of_frame_flag"]` (line 1477)

## What Happens in Build03

```
1. Load SAM2 mask → Clean mask
2. Load FF image → Determine px_dim_raw from metadata
3. Rescale to target resolution: rescale_factor = px_dim_raw / outscale
4. Get embryo orientation angle
5. Rotate image and mask
6. Crop to 576×256 canvas centered on embryo
7. Check mask area retention:
   - area_retained = mask_after / mask_before
   - out_of_frame = (area_retained < 0.98)  ← FLAG SET HERE
8. Post-process: CLAHE + Gaussian smoothing + noise blending
9. Save extracted snip
```

## Next Steps: Build04 Integration

### The `out_of_frame_flag` needs to be integrated into Build04 QC pipeline:

**Similar to existing flags:**
- `well_qc_flag` - well-level QC
- `sam2_qc_flags` - SAM2 detection confidence QC
- **NEW**: `out_of_frame_flag` - embryo clipping QC

**Action Required:**
1. Pass `out_of_frame_flag` from Build03 output to Build04 input
2. Add it to quality control thresholds in Build04
3. Filter or flag embryos where `out_of_frame_flag == True`
4. Document in Build04 QC logic

## Comparison: 6.5 vs 7.8 μm/px

| Metric | 6.5 μm/px | 7.8 μm/px | Benefit |
|--------|-----------|-----------|---------|
| Canvas Size (mm) | 3.74 × 1.66 | 4.50 × 2.00 | +20% larger |
| Large Embryos (3.5mm) | ~94% fill | ~78% fill | More margin |
| Clipping Risk | HIGH | LOW | Reduces OOF |

## Testing

Run direct comparison:
```bash
python test_build03_direct.py
```

This tests both scales on problematic embryos and reports `out_of_frame` status.

## Files Modified/Created

- ✅ `src/build/build03A_process_images.py` - Default outscale already 7.8
- ✅ `results/mcolon/20251113_rotation_crop_fix/test_build03_direct.py` - Direct test script
- ⏳ **TODO**: Build04 integration for `out_of_frame_flag` handling

## Notes

- **Yolk mask** is still used for head/tail orientation (critical for consistency)
- Future optimization: Could calculate orientation from curvature analysis instead
- The 0.98 threshold (98% retention) is hardcoded in `crop_embryo_image()` (line 171 image_utils.py)
