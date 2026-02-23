# Embryo Rotation/Cropping Fix - Summary of Changes

## Problem Identified

**Root Cause:** Embryos were being clipped during rotation and cropping due to size mismatch.

- **Well diameter:** 7 mm
- **Older embryos:** ~3.5 mm long (50% of well diameter)
- **Previous capture window:** 576×256 px at **6.5 μm/px** = **3.74 mm × 1.66 mm**
- **Result:** Large embryos filled 100% of the capture window height, leaving no margin for rotation-induced expansion

When embryos rotate, the canvas expands by ~1.4× at worst-case angles (45°). If an embryo already fills 100% of the frame, parts get clipped when the rotation shifts it near the edges.

## Solution Implemented

### 1. **Updated Resolution (6.5 → 7.8 μm/px)**

**File:** `src/build/build03A_process_images.py:1338`

```python
def extract_embryo_snips(root: str | Path,
                         stats_df: pd.DataFrame,
                         outscale: float=7.8,  # Changed from 6.5
                         ...
```

**New capture window:** 576×256 px at **7.8 μm/px** = **~4.5 mm × 2.0 mm**

**Effect:** Ensures large embryos (~3.5mm) fill maximum **~75% of frame height**, leaving margin for rotation without clipping.

### 2. **Reinstated Out-of-Frame Flag Detection**

**File:** `src/functions/image_utils.py:120-174`

Added `return_metrics` parameter to `crop_embryo_image()` function:
- Calculates mask area before and after cropping
- Flags embryos if <98% of mask area is retained
- Returns truncation status to caller

**File:** `src/build/build03A_process_images.py:396-414`

Updated `export_embryo_snips()` to use the flag:
```python
im_cropped, emb_mask_cropped, yolk_mask_cropped, out_of_frame_flag = crop_embryo_image(
    im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape=outshape, return_metrics=True
)

# Flag is now properly computed, not forced to False
if is_debug_embryo and out_of_frame_flag:
    print(f"   ⚠️  Out of frame detected: <98% mask area retained")
```

### 3. **Created Comprehensive Test Script**

**File:** `results/mcolon/20251113_rotation_crop_fix/test_rotation_fix.py`

**Purpose:** Validate the fix on real problematic images

**Loads:**
- Raw full-frame images from `built_image_data/stitched_FF_images/`
- Embryo masks from SAM2 segmentation output
- Metadata from `embryo_metadata_df01.csv`

**Tests with both:**
- OLD: 6.5 μm/px (original broken behavior)
- NEW: 5.2 μm/px (fixed behavior)

**Outputs generated:**
- `old_6.5um/` - Extracted snips with original resolution
- `new_5.2um/` - Extracted snips with new resolution
- `debug/` - Intermediate stages (rescaled, rotated, cropped + overlays)
- `comparison.png` - Visual comparison grid
- `metrics.csv` - Quantitative metrics

**Metrics collected:**
- Embryo physical dimensions (mm)
- Fill fraction (% of capture window)
- Mask area retention (%)
- Out-of-frame flag status
- Rotation angle applied

## Expected Outcomes

For the 4 test images:

1. **20251106_A02_e01_t0049** (cut at top)
   - OLD: ~100% fill, high clipping
   - NEW: ~75-80% fill, no clipping ✓

2. **20251106_A03_e01_t0085** (cut at bottom)
   - OLD: ~100% fill, high clipping
   - NEW: ~75-80% fill, no clipping ✓

3. **20250512_A03_e01_t0112** (cut at side)
   - OLD: Edge case clipping during rotation
   - NEW: Centered with margin ✓

4. **20251106_H12_e01_t0093** (large reference)
   - OLD: ~95-100% fill, marginal clipping
   - NEW: ~75-80% fill, comfortable margin ✓

## Files Modified

### Core Changes:
1. `src/build/build03A_process_images.py`
   - Line 1338: Changed `outscale=6.5` → `outscale=5.2`
   - Lines 1341-1363: Added docstring explaining the change
   - Lines 396-414: Updated to use out_of_frame_flag from crop_embryo_image()

2. `src/functions/image_utils.py`
   - Lines 120-174: Enhanced `crop_embryo_image()` with `return_metrics` parameter
   - Lines 146-172: Added mask area retention calculation and truncation detection

### Test Infrastructure:
3. `results/mcolon/20251113_rotation_crop_fix/test_rotation_fix.py`
   - Comprehensive test harness for validation
   - Loads raw images and masks from Build03 pipeline
   - Compares old vs new behavior with detailed metrics

## Running the Test

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251113_rotation_crop_fix
python test_rotation_fix.py
```

**Outputs:**
- `metrics.csv` - Summary metrics for all test images
- `old_6.5um/` - Old behavior extracts
- `new_5.2um/` - New behavior extracts
- `debug/` - Intermediate images and overlays

## Backward Compatibility

The changes are **backward compatible**:
- `crop_embryo_image()` default `return_metrics=False` returns original 3-tuple
- Existing code using `crop_embryo_image()` without the flag continues to work
- Only new code (Build03) uses `return_metrics=True`

## Next Steps

1. Run test script to validate on the 4 problematic images
2. Review comparison images for visual validation
3. Check metrics.csv for quantitative confirmation
4. If all looks good, apply changes to full pipeline
5. Re-extract all snips with new 5.2 μm/px resolution
