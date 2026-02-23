# Z-Stack Motion Blur Filtering - Quick Start Guide

## Overview

This refactor adds z-stack filtering to remove motion artifacts and out-of-focus frames **before** applying LoG focus stacking, improving final image quality.

## Files Modified/Created

### Documentation
- **[IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)** - Complete implementation plan with technical details
- **[bad_image_examples.md](./bad_image_examples.md)** - Labeled test cases (bad/okay/great images)
- **[frame_nd2_lookup.csv](./frame_nd2_lookup.csv)** - Mapping from JPEG paths to ND2 indices

### Code
- **[export_utils.py](../../../src/build/export_utils.py)** - Added 3 new functions:
  - `compute_slice_quality_metrics()` - Extract QA metrics from z-stacks
  - `filter_bad_slices()` - Filter out bad frames using peak-relative and/or correlation methods
  - `LoG_focus_stacker_with_filtering()` - Wrapper that optionally filters before stacking

### Validation Notebook
- **[results/mcolon/20251019/zstack_filtering_motion.ipynb](../../../results/mcolon/20251019/zstack_filtering_motion.ipynb)** - Enhanced with:
  - Core filtering functions (peak-relative, correlation, hybrid)
  - Visual validation helpers
  - Example usage cells for testing on labeled images

## Quick Start: Validate Filtering Methods

### Step 1: Run the Validation Notebook

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq
jupyter lab results/mcolon/20251019/zstack_filtering_motion.ipynb
```

**Key sections to run (in order):**

#### Section 1: Setup & Basic Metrics (Cells 1-6)
1. Load imports and helper functions
2. Load lookup table
3. View single frame examples

#### Section 2: Notebook Validation Functions (Cells 7-12)
4. Test filtering with simplified notebook implementations
5. Quick tests on bad/great images
6. Full validation loop on all examples
7. Comparison tables and plots

#### Section 3: Production Function Testing (NEW - Cells 13+)
8. **Import production functions** from `export_utils.py`
9. **Test baseline LoG focus stacker** (verify it works)
10. **Test QA metrics computation** (visualize sharpness & correlation)
11. **Test filtering methods** (peak-relative, correlation, hybrid)
12. **Compare across categories** (bad/okay/great images)

**Expected behavior:**
- Bad images: Should show improvement with filtering
- Great images: Should remain unchanged (high retention rate >80%)
- QA metrics should correctly identify problem frames

### Step 2: Determine Optimal Thresholds

Based on visual inspection, decide on:
- **`alpha`** (peak-relative threshold): Typical range 0.6-0.8
  - Lower α (0.6) = more aggressive filtering
  - Higher α (0.8) = more conservative
- **`corr_threshold`** (correlation threshold): Typical range 0.85-0.95
  - Lower (0.85) = more aggressive motion detection
  - Higher (0.95) = only flag severe motion

**Success criteria:**
- Bad images show visible improvement
- Great images retain quality (>80% frames kept)
- Rejection rate ~10-30% for good stacks, ~40-60% for bad stacks

### Step 3: Integrate into Build Pipeline (Optional)

Once validated, the filtering can be integrated into the build scripts. See [IMPLEMENTATION_PLAN.md Phase 3](./IMPLEMENTATION_PLAN.md#phase-3-build-pipeline-integration) for details.

## Current Status

✅ **Completed:**
- Implementation plan document
- Core filtering functions added to `export_utils.py`
- Validation notebook enhanced with:
  - Peak-relative sharpness filter
  - Frame-to-frame correlation (with bilateral flagging)
  - Hybrid filter (combines both)
  - Visual validation helpers

⏳ **Next Steps (User Action Required):**
1. **Run validation notebook** on labeled examples
2. **Visually inspect** before/after comparisons
3. **Document findings** - which threshold values work best?
4. **Decide** whether to proceed with build integration

## Filtering Methods

### Method 1: Peak-Relative Sharpness (Recommended Starting Point)
```python
# Filter out frames with LoG score < peak * alpha
filter_method = "peak_relative"
alpha = 0.7  # Keep frames with ≥70% of peak sharpness
```

**Pros:**
- Fast, simple, interpretable
- Self-normalizing per stack
- Good for removing out-of-focus frames

**Use when:** Mainly concerned with blur, not motion

### Method 2: Correlation (Motion Detection)
```python
# Flag frames with low correlation to neighbors
filter_method = "correlation"
corr_threshold = 0.9  # Flag if corr(i, i+1) < 0.9
```

**Pros:**
- Detects motion artifacts even in sharp frames
- Bilateral flagging (flags both frames in bad pairs)

**Use when:** Mainly concerned with motion/jitter

### Method 3: Hybrid (Recommended for Final Integration)
```python
# Combine both filters
filter_method = "hybrid"
alpha = 0.7
corr_threshold = 0.9
```

**Pros:**
- Most robust - catches both blur AND motion
- Handles complex failure modes

**Use when:** Want maximum quality assurance

## Example Usage

### In Notebook (Validation)
```python
# Load example
bad_example = lookup_df[lookup_df["category"] == "Bad Images"].iloc[0]

# Test filtering
results = validate_filtering_on_example(
    bad_example,
    alpha_values=[0.6, 0.7, 0.8],
    corr_values=[0.90],
    method="peak_relative"
)

# Visualize
plot_validation_results(results)
```

### In Build Pipeline (Future)
```python
# In export_utils.py - already implemented
from src.build.export_utils import LoG_focus_stacker_with_filtering

ff, log_scores, metrics = LoG_focus_stacker_with_filtering(
    stack_zyx,
    filter_size=5,
    device="cuda",
    enable_filtering=True,      # Turn on filtering
    filter_alpha=0.7,           # Validated threshold
    filter_method="peak_relative",
    return_metrics=True         # Get QA metrics
)

# Check metrics
print(f"Kept {metrics['n_kept']}/{metrics['n_total']} frames")
print(f"Rejection rate: {metrics['rejection_rate']*100:.1f}%")
```

## Troubleshooting

### "No improvement on bad images"
- Try **more aggressive filtering**: lower alpha (0.5-0.6)
- Try **hybrid method** instead of peak-relative alone
- Check if z-stack window is too small (increase `stack_window` parameter)

### "Great images degraded"
- Try **more conservative filtering**: higher alpha (0.8-0.9)
- Use **peak-relative only** instead of hybrid
- Check rejection rates - should be <20% for great images

### "All frames rejected"
- Safety check prevents this - will fallback to sharpness-only filtering
- If happening frequently, thresholds are too aggressive

## References

- **Technical Details**: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
- **Labeled Examples**: [bad_image_examples.md](./bad_image_examples.md)
- **Code**:
  - [export_utils.py:530-787](../../../src/build/export_utils.py#L530-L787)
  - [zstack_filtering_motion.ipynb](../../../results/mcolon/20251019/zstack_filtering_motion.ipynb)

## Questions?

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for:
- Detailed algorithm descriptions
- Future enhancements (anchor correlation, golden template, etc.)
- Build pipeline integration instructions
- Validation checklists
