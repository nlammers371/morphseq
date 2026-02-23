# Canonical Grid Implementation - Status Report

**Date**: 2026-01-21
**Status**: Phase 1 Complete, Phase 2 Partially Complete, Testing Reveals Issues

## Summary

The canonical grid infrastructure has been successfully implemented with the following components:

1. **Core canonical grid module** (`uot_grid.py`) - ✅ Complete
2. **Physical unit extraction** from CSV metadata - ✅ Complete
3. **Canonical grid preprocessing** - ✅ Complete (with caveats)
4. **Integration with UOT pipeline** - ✅ Complete
5. **Basic validation tests** - ✅ Passing
6. **End-to-end UOT tests** - ⚠️ Issues identified

## Implemented Components

### 1. Core Canonical Grid Module (`uot_masks/uot_grid.py`)

Created with the following functionality:

- `CanonicalGridConfig`: Configuration for 576×256 grid @ 7.8 μm/px
- `GridTransform`: Dataclass tracking transformation metadata
- `compute_grid_transform()`: Computes transformation parameters
- `apply_grid_transform()`: Resamples masks to canonical grid
- `rescale_velocity_to_um()`: Converts pixel velocities to micrometers
- `rescale_distance_to_um()`: Converts pixel distances to micrometers

**Key features**:
- Integrated with `src/data_pipeline/snip_processing/rotation.py` for yolk-based alignment
- Supports fallback to centroid-based alignment when yolk unavailable
- Handles arbitrary source resolutions
- Tracks effective um/px including downsampling

### 2. Metadata Extraction (`uot_masks/frame_mask_io.py`)

Updated all mask loading functions to:
- Extract `um_per_pixel` from CSV columns `Height (um)` / `Height (px)`
- Store in `UOTFrame.meta` dictionary
- Updated `DEFAULT_USECOLS` to include physical dimension columns

### 3. Canonical Grid Preprocessing (`uot_masks/preprocess.py`)

Added `preprocess_pair_canonical()` function:
- QC cleans both masks
- Computes **shared transform** based on union of masks (preserves spatial relationships)
- Applies rotation (yolk-based or centroid-based)
- Rescales to target resolution (7.8 μm/px)
- Crops/pads to fixed 256×576 grid centered on union centroid
- Returns canonical masks + detailed transform metadata

**Critical design decision**: Uses shared transform for mask pairs to preserve spatial relationships between source and target.

### 4. UOT Pipeline Integration (`uot_masks/run_transport.py`)

Modified `run_uot_pair()` to support two modes:

**Legacy mode** (`use_canonical_grid=False`):
- Original preprocessing (bbox cropping, optional centroid alignment)
- Results in arbitrary pixel units

**Canonical grid mode** (`use_canonical_grid=True`):
- Calls `preprocess_pair_canonical()`
- Transforms both masks to 256×576 @ 7.8 μm/px
- Automatically rescales velocity fields to micrometers
- Stores transform metadata for reconstruction

### 5. Configuration (`utils/optimal_transport/config.py`)

Added to `UOTConfig`:
```python
use_canonical_grid: bool = False
canonical_grid_um_per_pixel: float = 7.8
canonical_grid_shape_hw: tuple[int, int] = (256, 576)
canonical_grid_align_mode: str = "yolk"  # "yolk" | "centroid" | "none"
```

##Validation Tests

### ✅ Basic Grid Tests (PASSING)

File: `test_canonical_grid_basic.py`

**Results**:
- Grid shape test: ✓ All masks correctly transformed to 256×576
- Scale factor computation: ✓ Correct ratios (0.5, 1.0, 2.0)
- Velocity rescaling: ✓ 10 px → 78 μm (at 7.8 μm/px)
- Distance rescaling: ✓ Handles downsampling correctly
- Resolution invariance: ⚠️ Some discretization effects (expected)

**Key finding**: The grid transformation mathematics are correct.

### ⚠️ End-to-End UOT Tests (ISSUES IDENTIFIED)

File: `test_canonical_uot_synthetic.py`

**Test setup**: Circle → Circle shifted by known amount (78 μm diagonal)

**Results**:
- High res (5.0 μm/px): Mean velocity 1079 μm (**1284% error**)
- Canonical res (7.8 μm/px): Mean velocity NaN (**no transport**)
- Low res (10.0 μm/px): Mean velocity 2 μm (**97% error**)

**Analysis**:
1. Velocities are wildly incorrect
2. No resolution invariance
3. Canonical resolution case shows zero transport (perfect overlap?)
4. Other resolutions show massive errors

**Potential causes**:
1. **Sampling artifacts**: After downsampling to canonical grid, the shift may be too small relative to the support point sampling (max 5000 points from 147k pixel grid)
2. **Centering issues**: The union-based centering may be inadvertently aligning the circles more than intended
3. **Coordinate scaling**: The `coord_scale` parameter may need adjustment for the canonical grid
4. **Marginal relaxation**: Current value (10.0) may be too permissive, allowing excessive mass creation/destruction instead of transport

## Known Issues & Limitations

### 1. Yolk Masks Not Available
- Yolk masks generated in `src/build/build02B_segment_bf_main.py`
- Stored in `morphseq_playground/segmentation/yolk_v1_0050_predictions`
- **Not exported with embryo mask CSV**
- Current workaround: Use `align_mode="centroid"` fallback
- **TODO**: Future pipeline should export yolk mask paths with embryo data

### 2. Velocity/Distance Rescaling May Be Incomplete
- Velocity field rescaled to μm ✓
- Mean/max transport distances in metrics **not yet rescaled**
- Cost values still in pixel²·mass units (may need rescaling for interpretability)

### 3. Test Failures Indicate Deeper Issues
The synthetic UOT tests reveal that something is fundamentally wrong with the end-to-end pipeline:
- Either the canonical grid transform is breaking spatial relationships
- Or the UOT solver parameters need adjustment for the canonical grid scale
- Or the sampling strategy doesn't work well on the canonical grid

### 4. Resolution Invariance Not Achieved
The core goal of the canonical grid (resolution-independent comparisons) is not yet validated because the UOT results vary dramatically across resolutions.

## Next Steps (Prioritized)

### Immediate (Debug Failing Tests)
1. **Visualize transformed masks**: Check if circles are actually overlapping after canonical grid transform
2. **Check support point distribution**: Verify sampling isn't too sparse
3. **Test without sampling**: Set `max_support_points` very high to rule out sampling issues
4. **Adjust UOT parameters**: Try lower `marginal_relaxation`, different `epsilon`, adjusted `coord_scale`

### Short Term (Validation)
1. **Identity test**: Same mask → same mask should give zero cost/velocity
2. **Pure translation with overlap**: Smaller shift to ensure overlap after transform
3. **Real embryo consecutive frames**: Test on actual data with small time gaps

### Medium Term (Refinement)
1. **Integrate yolk masks**: Add yolk mask path to CSV export pipeline
2. **Rescale all metrics**: Convert costs and distances to physical units
3. **Optimize sampling**: May need stratified sampling tuned for canonical grid
4. **Benchmarking**: Compare canonical vs legacy preprocessing on real data

### Long Term (Production)
1. **Cross-embryo comparisons**: Validate consistent orientation across embryos
2. **Timeseries analysis**: Run on full developmental trajectories
3. **Phenotype extraction**: Use interpretable velocities for morphological dynamics

## Files Modified/Created

### Created:
- `src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py`
- `results/mcolon/20260121_uot-mvp/canonical_grid_implementation.md` (plan document)
- `results/mcolon/20260121_uot-mvp/test_canonical_grid_basic.py` (basic tests - passing)
- `results/mcolon/20260121_uot-mvp/test_canonical_uot_synthetic.py` (UOT tests - failing)
- `results/mcolon/20260121_uot-mvp/test_canonical_grid_cep290.py` (CEP290 test - not yet working)

### Modified:
- `src/analyze/optimal_transport_morphometrics/uot_masks/frame_mask_io.py`
  - Added `_compute_um_per_pixel()` helper
  - Updated all loading functions to extract um/px

- `src/analyze/optimal_transport_morphometrics/uot_masks/preprocess.py`
  - Added `preprocess_pair_canonical()` function
  - Implements shared transform for mask pairs

- `src/analyze/optimal_transport_morphometrics/uot_masks/run_transport.py`
  - Updated `run_uot_pair()` to support canonical grid mode
  - Automatic velocity rescaling when `use_canonical_grid=True`

- `src/analyze/utils/optimal_transport/config.py`
  - Added canonical grid configuration parameters to `UOTConfig`

## Test Commands

```bash
# Basic grid tests (PASSING)
python results/mcolon/20260121_uot-mvp/test_canonical_grid_basic.py

# UOT synthetic tests (FAILING - needs debug)
python results/mcolon/20260121_uot-mvp/test_canonical_uot_synthetic.py

# CEP290 real data test (not yet working - frame indices issue)
python results/mcolon/20260121_uot-mvp/test_canonical_grid_cep290.py
```

## Conclusion

The canonical grid infrastructure is **architecturally complete** but **not yet functionally validated**. The basic transformations work correctly, but integration with the UOT solver reveals issues that need debugging. The most likely culprit is interaction between the canonical grid parameters and the UOT solver configuration (sampling, marginal relaxation, coordinate scaling).

**Recommendation**: Focus debugging efforts on the synthetic translation test with visualization to understand where the pipeline breaks down.
