# Handoff: UOT Output Upsampling to Canonical Grid

**Date**: 2026-01-21
**Commit**: 56c74023
**Status**: Critical issue identified - needs resolution before parameter sweep

---

## Problem Statement

The UOT parameter debugging script (`debug_uot_params.py`) currently has a **visualization mismatch**:

- ✅ **Input masks**: Created on canonical grid (256×576)
- ❌ **UOT outputs**: Downsampled to (81×81) for computational efficiency
- ❌ **Visualizations**: Show 81×81 data in top-left corner of 256×576 plot

**Expected behavior**: All outputs should be upsampled back to canonical grid for visualization.

---

## Current Workflow

```
Input mask (256×576)
    ↓
Downsampling (for efficiency)
    ↓
UOT computation on support points
    ↓
Output: velocity_field_yx_hw2 (81×81×2)  ← Problem: stays downsampled
Output: mass_created_hw (81×81)           ← Problem: stays downsampled
Output: mass_destroyed_hw (81×81)         ← Problem: stays downsampled
    ↓
Visualization on canonical grid (256×576) ← Mismatch!
```

---

## Root Cause Analysis

### Where does downsampling happen?

1. **In UOT pipeline preprocessing**:
   - `downsample_factor` parameter controls downsampling
   - Default: `downsample_factor=4` → reduces resolution by 4×
   - Location: `src/analyze/optimal_transport_morphometrics/uot_masks/preprocess.py`

2. **In support point sampling**:
   - `max_support_points=5000` limits number of points
   - Sparse sampling from mask for computational efficiency
   - Location: `src/analyze/utils/optimal_transport/multiscale_sampling.py`

3. **Transport maps reconstruction**:
   - Maps are created on downsampled grid
   - Location: `src/analyze/utils/optimal_transport/transport_maps.py`

### Where should upsampling happen?

**Option 1: In `transport_maps.py` (recommended)**
- Upsample velocity/creation/destruction maps back to canonical grid
- Happens automatically in pipeline
- All downstream code gets canonical-resolution outputs

**Option 2: In visualization code (not recommended)**
- Upsample only for plotting
- Pipeline outputs stay downsampled
- Inconsistent outputs across different uses

---

## Investigation Plan

### Step 1: Trace the data flow

**Files to examine**:

1. `src/analyze/optimal_transport_morphometrics/uot_masks/run_transport.py`
   - `run_uot_pair()` function
   - Where does `result.velocity_field_yx_hw2` get created?
   - What is `work_shape_hw` vs `transform_meta`?

2. `src/analyze/utils/optimal_transport/transport_maps.py`
   - `compute_transport_maps()` function
   - How are velocity/creation/destruction computed?
   - What grid are they on?

3. `src/analyze/optimal_transport_morphometrics/uot_masks/preprocess.py`
   - `preprocess_pair()` function
   - How does `downsample_factor` affect output shape?
   - What's the relationship between input shape and output shape?

### Step 2: Understand canonical grid integration

**Key questions**:

1. When `use_canonical_grid=True`, what happens to:
   - Input mask transformations?
   - Output shape expectations?
   - Transport map resolution?

2. Is there an existing upsampling mechanism?
   - Check `src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py`
   - Look for inverse transforms

3. What is `transform_meta` used for?
   - Does it contain scale factors?
   - Can we use it to upsample?

### Step 3: Design solution

**Two approaches to consider**:

#### Approach A: Subset by mask (not bbox)
```python
# Current (suspected): Uses bounding box → downsamples entire bbox
bbox = get_bounding_box(mask)
work_region = mask[bbox]

# Proposed: Use mask directly → only compute on actual embryo
support_coords = get_mask_coordinates(mask)  # Only (y,x) where mask=1
# This naturally gives sparse representation, no unnecessary bbox padding
```

**Pros**:
- More efficient (only compute on actual embryo pixels)
- Natural sparsity
- No bbox-induced downsampling

**Cons**:
- May need to map back to full grid for visualization
- Irregular grid (not a rectangle)

#### Approach B: Upsample outputs to canonical grid
```python
# After computing transport maps on downsampled grid (81×81)
# Upsample back to canonical grid (256×576)

from scipy.ndimage import zoom

canonical_h, canonical_w = (256, 576)
downsample_h, downsample_w = velocity_field.shape[:2]

scale_h = canonical_h / downsample_h
scale_w = canonical_w / downsample_w

velocity_field_canonical = zoom(velocity_field, (scale_h, scale_w, 1), order=1)
mass_created_canonical = zoom(mass_created, (scale_h, scale_w), order=1)
mass_destroyed_canonical = zoom(mass_destroyed, (scale_h, scale_w), order=1)
```

**Pros**:
- Simple to implement
- Maintains rectangular grid
- Works with existing pipeline

**Cons**:
- Interpolation may introduce artifacts
- Not true high-resolution computation

#### Approach C: Compute at canonical resolution (no downsampling)
```python
config = UOTConfig(
    downsample_factor=1,      # No downsampling!
    downsample_divisor=1,     # No divisor adjustment
    max_support_points=None,  # Use all mask points (or very high limit)
    # ... rest of config
)
```

**Pros**:
- True canonical resolution throughout
- No upsampling artifacts
- Simplest conceptually

**Cons**:
- Much slower (5000+ points vs current ~1000)
- May hit memory/time limits
- Defeats purpose of efficient downsampling

---

## Recommended Next Steps

### Step 1: Investigate current behavior
```bash
# Create a test script to examine intermediate outputs
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq
python -c "
from src.analyze.optimal_transport_morphometrics.uot_masks import run_uot_pair, build_problem
from src.analyze.utils.optimal_transport import UOTConfig, UOTFramePair, UOTFrame, MassMode
import numpy as np

# Create canonical grid mask
mask = np.zeros((256, 576), dtype=np.uint8)
mask[100:150, 250:300] = 1

config = UOTConfig(downsample_factor=4, max_support_points=5000)
pair = UOTFramePair(src=UOTFrame(embryo_mask=mask), tgt=UOTFrame(embryo_mask=mask))

# Build problem to see intermediate shapes
problem = build_problem(pair, config)
print(f'Input mask shape: {mask.shape}')
print(f'Work shape: {problem.work_shape_hw}')
print(f'Support src shape: {problem.src.coords_yx.shape}')
print(f'Support tgt shape: {problem.tgt.coords_yx.shape}')
print(f'Transform meta: {problem.transform_meta}')

# Run full pipeline
result = run_uot_pair(pair, config)
print(f'Velocity field shape: {result.velocity_field_yx_hw2.shape}')
print(f'Mass created shape: {result.mass_created_hw.shape}')
"
```

### Step 2: Check for existing upsampling

Search the codebase:
```bash
grep -r "zoom\|resize\|upsample\|interpolate" src/analyze/optimal_transport_morphometrics/
grep -r "work_shape_hw\|transform_meta" src/analyze/optimal_transport_morphometrics/
```

### Step 3: Implement solution

Based on findings, implement one of:

1. **If bbox subsetting is the issue**: Modify to use mask-based subsetting
2. **If no upsampling exists**: Add upsampling in `compute_transport_maps()`
3. **If performance allows**: Test with `downsample_factor=1` on canonical grid

### Step 4: Update debug script

Once outputs are canonical-resolution:
```python
# In debug_uot_params.py
# These should now all be (256, 576) or close to it
assert result.velocity_field_yx_hw2.shape[:2] == CANONICAL_GRID_SHAPE, \
    f"Velocity field {result.velocity_field_yx_hw2.shape} != canonical {CANONICAL_GRID_SHAPE}"
```

---

## Files to Modify

### Primary targets:
1. `src/analyze/utils/optimal_transport/transport_maps.py`
   - Add upsampling to canonical grid in `compute_transport_maps()`

2. `src/analyze/optimal_transport_morphometrics/uot_masks/preprocess.py`
   - Check if `transform_meta` stores canonical grid info
   - Add inverse transform if needed

3. `results/mcolon/20260121_uot-mvp/debug_uot_params.py`
   - Update once pipeline fixed
   - Add assertions for output shapes

### Secondary targets:
4. `src/analyze/optimal_transport_morphometrics/uot_masks/run_transport.py`
   - May need to pass canonical grid info to transport_maps

5. `src/analyze/utils/optimal_transport/config.py`
   - Consider adding `output_resolution` parameter?

---

## Testing Strategy

### Test 1: Verify upsampling works
```python
# Create mask on canonical grid
mask_canonical = make_circle(CANONICAL_GRID_SHAPE, (128, 288), 40)

# Run UOT
result = run_uot_pair(...)

# Check output shapes
assert result.velocity_field_yx_hw2.shape == (*CANONICAL_GRID_SHAPE, 2)
assert result.mass_created_hw.shape == CANONICAL_GRID_SHAPE
```

### Test 2: Verify no distortion
```python
# For identity test (circle -> same circle):
# - Velocity should be near-zero everywhere
# - Creation/destruction should be near-zero everywhere
# - No edge artifacts from upsampling
```

### Test 3: Verify physical units preserved
```python
# Surface areas should match regardless of downsampling
# created_area_um2 = created_mass * (um_per_px)^2
# Should be consistent before/after upsampling
```

---

## Context for Next Session

### What works:
- ✅ Test mask generation on canonical grid (256×576)
- ✅ Visualization axis limits set to canonical grid
- ✅ Parameter sweep framework
- ✅ Diagnostic metrics collection

### What needs fixing:
- ❌ UOT outputs (81×81) need upsampling to canonical grid (256×576)
- ❌ Velocity/creation/destruction maps shown in top-left corner only

### Priority:
**HIGH** - Cannot run meaningful parameter sweep until outputs are on canonical grid

### Next actions:
1. Trace data flow through `run_uot_pair()` → `compute_transport_maps()`
2. Identify where shape reduction happens (bbox? downsampling? support sampling?)
3. Implement upsampling solution
4. Test with debug script
5. Run full parameter sweep

---

## References

- Canonical grid spec: `canonical_grid_implementation.md`
- UOT config: `src/analyze/utils/optimal_transport/config.py:76-100`
- Transport maps: `src/analyze/utils/optimal_transport/transport_maps.py`
- Preprocessing: `src/analyze/optimal_transport_morphometrics/uot_masks/preprocess.py`

---

## Questions to Answer

1. **What is `work_shape_hw`?**
   - Is this the downsampled shape?
   - Is this what transport maps are computed on?

2. **What is in `transform_meta`?**
   - Does it store scale factors?
   - Can we use it for inverse transform?

3. **Is there a canonical grid transform?**
   - Check `uot_grid.py` for transform logic
   - Are there forward/inverse transform functions?

4. **Where does 81 come from?**
   - 256 / 4 = 64 (if downsample_factor=4)
   - 81 doesn't match clean division
   - Padding? Rounding? Different mechanism?

---

**Status**: Ready for investigation. Start with Step 1 of investigation plan.
