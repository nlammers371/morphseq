# Analysis: Why Extracted Crops Seem Too Big

## Observed Metrics
```
20251106_A02_e01_t0049: Fill=6.9%, Size=0.26×0.10 mm
20251106_A03_e01_t0085: Fill=9.4%, Size=0.26×0.09 mm
20250512_A03_e01_t0112: Fill=14.1%, Size=0.25×0.13 mm
20251106_H12_e01_t0093: Fill=19.1%, Size=0.71×0.13 mm
```

## Root Cause Analysis

### 1. **Output Canvas Size: 576×256 pixels**
   - This is the hardcoded default in build03 (line 1369)
   - Designed for typical embryos at later developmental stages
   - Current test embryos are **extremely small** relative to this canvas

### 2. **The Fill Fraction Math is CORRECT**
   - Fill fraction = (embryo_bounding_box_height / canvas_height)
   - For 20251106_A02_e01_t0049: ~40 px embryo height / 576 px canvas = **6.9%** ✓
   - The algorithm is working properly - the embryos are just small

### 3. **Why This Happens**

The cropping algorithm (crop_embryo_image in image_utils.py):
```
1. Find embryo center of mass: y_mean, x_mean
2. Create a 576×256 canvas
3. Crop a 576×256 region centered on embryo from rotated image
4. Place in output canvas
5. Result: Large white space around small embryo
```

**This is exactly what build03 does** - the extraction logic is identical.

## Comparison: Test Script vs Build03

| Aspect | Test Script | Build03 |
|--------|-------------|---------|
| Outshape | [576, 256] | [576, 256] (default) |
| Cropping Logic | `crop_embryo_image_with_metrics()` | `crop_embryo_image()` |
| Logic Difference | None - identical | Same algorithm |
| Fill Threshold Enforcement | None | Build03 may have QC filtering |

## Actual Issue: The Embryos Are Small

Looking at size metrics:
- **H12_e01_t0093**: 0.71 × 0.13 mm (largest, ~19% fill)
- **A02_e01_t0049**: 0.26 × 0.10 mm (smallest, ~7% fill)

These are **genuinely small embryos**. The 576×256 canvas is appropriate for normal-sized embryos, but these are below typical size.

## Solutions

### Option 1: Use Adaptive Outshape
```python
# Scale canvas size based on embryo size
if embryo_size_mm < 0.5:
    outshape = [288, 128]  # Half size for small embryos
else:
    outshape = [576, 256]  # Standard size
```

### Option 2: Apply QC Filtering
Build03 likely filters embryos with fill_fraction < threshold:
```python
if fill_fraction_height < 0.10:
    flag_as_low_quality = True
```

### Option 3: Accept Current Output
The extraction is working correctly - small embryos just have low fill fractions.
This matches what build03 would produce.

## Verification

✅ **Cropping logic**: Identical to build03's `crop_embryo_image()`
✅ **Canvas size**: Uses build03 default [576, 256]
✅ **Rotation**: Applied correctly (angles look reasonable)
✅ **Masks**: Loading and extracting correctly (verified with diagnostics)
✅ **Math**: Fill fractions calculated correctly for the canvas size

## Conclusion

**The extraction is NOT broken - it's working exactly like build03.**

The perceived issue is that:
1. These test embryos are small (0.26-0.71 mm)
2. The 576×256 canvas is large relative to them
3. Small embryos → low fill fractions (6-19%)
4. This is expected and correct behavior

To get higher fill fractions, you would need either:
- Larger embryos, OR
- Adaptive/smaller output canvas, OR
- Different QC filtering strategy
