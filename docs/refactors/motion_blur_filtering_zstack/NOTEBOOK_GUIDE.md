# Validation Notebook Guide

**Notebook**: [results/mcolon/20251019/zstack_filtering_motion.ipynb](../../../results/mcolon/20251019/zstack_filtering_motion.ipynb)

## Overview

This notebook validates the z-stack filtering implementation with two complementary approaches:

1. **Simplified notebook implementations** - For rapid prototyping and understanding
2. **Production function testing** - Verifies the actual `export_utils.py` functions work correctly

## Notebook Structure

### Section 1: Setup & Imports (Cells 1-6)

**Purpose**: Load dependencies, define helper functions, load labeled examples

**Key cells:**
- Cell 1: Imports and IO helpers (`read_nd2_frame`, `laplacian_variance`, `log_focus_response`, etc.)
- Cell 2: Load lookup CSV with labeled examples
- Cell 3-5: View individual frame examples and compute basic metrics

**What to check:**
- ✅ All imports successful
- ✅ Lookup table loads correctly (10 rows: 4 bad, 3 okay, 3 great)
- ✅ Can load frames from ND2 files

---

### Section 2: Notebook Validation Functions (Cells 7-12)

**Purpose**: Test filtering logic with simplified implementations

**Key functions defined:**
- `filter_by_peak_relative()` - Sharpness-based filtering
- `compute_frame_correlations()` - Motion detection
- `filter_by_correlation_bilateral()` - Bilateral motion flagging
- `hybrid_filter()` - Combined filtering
- `naive_focus_stack()` - Simplified focus stacking
- `validate_filtering_on_example()` - Test filtering on one example
- `plot_validation_results()` - Visualize before/after
- `create_comparison_table()` - Generate comparison DataFrame

**Example usage cells:**
- **Quick test on bad image** - See immediate improvement
- **Quick test on great image** - Verify no degradation
- **Full validation loop** - Test all 10 labeled examples
- **Comparison table** - Quantitative summary
- **Before/after grid** - Visual comparison

**What to check:**
- ✅ Bad images show visible improvement after filtering
- ✅ Great images remain high quality (>80% frames kept)
- ✅ Rejection rates are reasonable:
  - Bad images: 40-60% rejection expected
  - Okay images: 20-40% rejection
  - Great images: <20% rejection

**Expected findings:**
- Peak-relative filtering (α=0.7) is a good starting point
- Correlation filtering may be too aggressive for some stacks
- Hybrid method provides good balance

---

### Section 3: Production Function Testing (Cells 13+)

**Purpose**: Verify the actual `export_utils.py` implementations work correctly

#### Cell Block 1: Import Production Functions
```python
from src.build.export_utils import (
    compute_slice_quality_metrics,
    filter_bad_slices,
    LoG_focus_stacker_with_filtering,
    LoG_focus_stacker
)
```

**What to check:**
- ✅ Imports succeed without errors
- ✅ Functions are callable

#### Cell Block 2: Test Baseline LoG Focus Stacker
Loads a bad image example and runs the standard `LoG_focus_stacker` (no filtering).

**What to check:**
- ✅ Function runs without errors
- ✅ Output shape is correct (Y, X) for single image
- ✅ LoG scores shape is (Z, Y, X)
- ✅ Works on both CPU and GPU

#### Cell Block 3: Test QA Metrics Computation
Tests `compute_slice_quality_metrics()` function.

**What to check:**
- ✅ Returns expected metrics:
  - `peak_log`: Maximum sharpness
  - `mean_log`: Average sharpness
  - `min_corr`: Minimum correlation (should be low for bad images)
  - `median_corr`: Median correlation
- ✅ Visualizations show:
  - Sharpness curve (should vary across slices)
  - Correlation plot (should show drops for motion artifacts)

**Example output for bad image:**
```
Peak LoG: 0.0451
Mean LoG: 0.0234
Min correlation: 0.42  ← Low indicates motion!
Median correlation: 0.89
```

#### Cell Block 4: Test Filtering Methods
Tests all 4 filtering modes:
1. No filtering (baseline)
2. Peak-relative (α=0.7)
3. Correlation (≥0.9)
4. Hybrid (α=0.7, corr≥0.9)

**What to check:**
- ✅ All methods run without errors
- ✅ Filtering reduces frame count appropriately
- ✅ Output images differ from original
- ✅ Metrics are populated correctly

**Expected behavior:**
- **No filtering**: All frames kept (baseline)
- **Peak-relative**: Moderate rejection (20-40%)
- **Correlation**: More aggressive rejection (30-50%)
- **Hybrid**: Most aggressive (40-60% for bad images)

#### Cell Block 5: Category Comparison
Tests filtering across all 3 categories (bad/okay/great).

**What to check:**
- ✅ Bad images:
  - High rejection rate (40-60%)
  - Low min_corr (<0.85)
  - Visible improvement in filtered image
- ✅ Okay images:
  - Moderate rejection (20-40%)
  - Moderate min_corr (0.85-0.95)
  - Some improvement
- ✅ Great images:
  - Low rejection (<20%)
  - High min_corr (>0.95)
  - Minimal change (quality preserved)

**Sample output:**
```
Category         File                      Frames       Rejection    Min Corr
Bad Images       20250912_B10_ch00_t0092   5/11         54.5%        0.420
Okay Images      20250912_C04_ch00_t0111   8/11         27.3%        0.870
Great Images     20250912_C04_ch00_t0028   10/11        9.1%         0.970
```

---

## Validation Checklist

### ✅ Section 1: Basic Setup
- [ ] All imports load successfully
- [ ] Lookup CSV contains 10 labeled examples
- [ ] Can read frames from ND2 files
- [ ] Basic metrics (laplacian, LoG) compute correctly

### ✅ Section 2: Notebook Validation
- [ ] Filtering functions run without errors
- [ ] Bad images show improvement with filtering
- [ ] Great images remain unchanged (>80% retention)
- [ ] Comparison table shows expected rejection rates
- [ ] Visual comparisons look reasonable

### ✅ Section 3: Production Testing
- [ ] Export_utils imports succeed
- [ ] Baseline LoG_focus_stacker works
- [ ] QA metrics function returns correct values
- [ ] QA visualizations show expected patterns
- [ ] All 4 filtering methods run successfully
- [ ] Category comparison shows expected behavior:
  - Bad images: high rejection, low correlation
  - Great images: low rejection, high correlation

### ✅ Final Validation
- [ ] Determined optimal alpha value (recommended: 0.7)
- [ ] Decided on filtering method (peak_relative vs hybrid)
- [ ] Documented findings in notebook or separate file
- [ ] Ready to integrate into build pipeline (optional)

---

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src.build.export_utils'`

**Solution**:
```python
import sys
from pathlib import Path
REPO_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(REPO_ROOT))
```

### GPU/CUDA Issues
**Problem**: CUDA out of memory or not available

**Solution**:
```python
device = "cpu"  # Force CPU
# Or free GPU memory:
torch.cuda.empty_cache()
```

### ND2 Loading Errors
**Problem**: `IndexError` when loading ND2 frames

**Solution**: Check that `time_int` and `series_num` are valid:
```python
with ND2File(str(nd2_path)) as f:
    print(f"Shape: {f.to_dask().shape}")  # Should be (T, P, Z, Y, X)
    print(f"Max time: {f.to_dask().shape[0] - 1}")
    print(f"Max series: {f.to_dask().shape[1] - 1}")
```

### No Improvement on Bad Images
**Possible causes:**
1. Alpha too high (try 0.5 or 0.6)
2. Z-stack window too small (increase to ±10 frames)
3. Wrong filtering method (try hybrid instead of peak_relative)

### Great Images Degraded
**Possible causes:**
1. Alpha too low (try 0.8 or 0.9)
2. Correlation threshold too high (try 0.85 instead of 0.95)
3. Using hybrid when peak_relative is sufficient

---

## Next Steps After Validation

1. **Document findings**:
   - What alpha value works best?
   - Which method (peak_relative, correlation, hybrid)?
   - Any special cases or edge cases discovered?

2. **Optional: Integrate into build pipeline**:
   - See [IMPLEMENTATION_PLAN.md Phase 3](./IMPLEMENTATION_PLAN.md#phase-3-build-pipeline-integration)
   - Add QA metrics collection to build scripts
   - Add optional filtering flags

3. **Future enhancements**:
   - Anchor correlation method (for drift detection)
   - Golden template approach (for highest robustness)
   - Automatic threshold selection (adaptive methods)

---

## References

- **Implementation Plan**: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
- **Quick Start**: [README.md](./README.md)
- **Labeled Examples**: [bad_image_examples.md](./bad_image_examples.md)
- **Production Code**: [export_utils.py:530-787](../../../src/build/export_utils.py#L530-L787)
