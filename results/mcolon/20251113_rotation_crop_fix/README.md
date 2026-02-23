# Embryo Rotation/Cropping Fix - Test Results Directory

This directory contains the implementation and testing infrastructure for fixing the embryo clipping bug in the rotation/cropping pipeline.

## Quick Start

### Run the Test
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251113_rotation_crop_fix
python test_rotation_fix.py
```

### View Results
After running the test, check:
- **`metrics.csv`** - Quantitative comparison (fill fraction, mask retention, flags)
- **`old_6.5um/`** - Extracted snips using old parameters (showing clipping)
- **`new_7.8um/`** - Extracted snips using new parameters (should show no clipping)
- **`debug/`** - Intermediate processing stages:
  - `*_rescaled.jpg` - After rescaling to target μm/px
  - `*_rotated.jpg` - After rotation with expanded canvas
  - `*_cropped.jpg` - Final crop output
  - `*_overlay.jpg` - Mask overlay on image

## What Was Fixed

### The Problem
- Large embryos (~3.5 mm) were being cut off during rotation/cropping
- Root cause: Embryos filled 100% of the capture window (3.74 mm at 6.5 μm/px)
- When rotated, expansion pushed them outside the crop bounds → pixels got clipped

### The Solution
1. **New resolution:** 7.8 μm/px (instead of 6.5 μm/px)
   - Capture window: 4.5 mm × 2.0 mm
   - Ensures embryos fill max ~75% of frame, leaving rotation margin
   - Proportionality maintained: same μm/px → same physical scale

2. **Restored truncation detection:** `out_of_frame_flag`
   - Flags embryos where <98% of mask area retained after cropping
   - Helps identify any remaining edge cases

## Test Images

The script tests on 4 real problematic images:

| Image | Issue | Expected Fix |
|-------|-------|--------------|
| `20251106_A02_e01_t0049` | Cut at top | Should be fully visible at 5.2 μm/px |
| `20251106_A03_e01_t0085` | Cut at bottom | Should be fully visible at 5.2 μm/px |
| `20250512_A03_e01_t0112` | Cut at side | Should be fully visible at 5.2 μm/px |
| `20251106_H12_e01_t0093` | Large embryo | Reference case - should look good at both resolutions |

## Key Metrics to Check

In `metrics.csv`:

- **`embryo_length_mm`** - Physical embryo length (should match expectation ~3-3.5mm)
- **`fill_fraction_height`** - Percentage of capture window filled:
  - OLD (6.5 μm/px): Should be ~0.95-1.0 (100% fill = problem!)
  - NEW (5.2 μm/px): Should be ~0.70-0.80 (safe margin)
- **`area_retained`** - Fraction of mask area after cropping:
  - Should be >0.98 for both old and new
  - <0.98 indicates truncation
- **`out_of_frame`** - Boolean flag:
  - NEW should have fewer or no True values compared to OLD

## Implementation Files

Modified files (all backward compatible):

1. **`src/build/build03A_process_images.py`**
   - Default `outscale=5.2` (was 6.5)
   - Added docstring documenting the change

2. **`src/functions/image_utils.py`**
   - Enhanced `crop_embryo_image()` with optional metrics return
   - Calculates mask area retention and truncation flag

## Next Steps

1. ✅ Review the test outputs
2. ✅ Verify metrics show improvement (fill fraction ~80%, no truncation)
3. ⏳ If metrics look good, apply pipeline to full dataset
4. ⏳ Re-extract all snips with new 5.2 μm/px resolution
5. ⏳ Retrain VAE models with new snips (may need retraining due to different image scale)

## Notes

- **Proportionality preserved:** 1.4× expansion during rotation still maintains μm/px ratio
- **Backward compatible:** Existing code continues to work with old behavior if needed
- **Black padding:** Crop boundaries shown as black for easy visual inspection
- **SAM2 masks used:** Loads masks from SAM2 segmentation pipeline (not legacy masks)

## Questions?

See `CHANGES_SUMMARY.md` for detailed technical documentation.
