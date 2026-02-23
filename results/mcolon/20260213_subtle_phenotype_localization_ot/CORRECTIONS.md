# Corrections Applied to Implementation Plan

**Date**: 2026-02-13

## User-Requested Clarifications

### 1. HPF Bin Selection
- **Original**: 26-28 hpf (early timepoint)
- **Corrected**: **48 hpf** (47-49 hpf window)
- **Rationale**: 48 hpf represents mature phenotype stage with better WT vs mutant differentiation

### 2. Centerline Extraction Method
- **Original**: `LocalPrincipalCurve` from `src/analyze/spline_fitting/lpc_model.py`
- **Corrected**: **Geodesic skeletonization** from `segmentation_sandbox/scripts/body_axis_analysis/centerline_extraction.py`
- **Key function**: `extract_centerline(mask, method='geodesic')`
- **Rationale**: Use the same skeletonization method that's used in the data pipeline for curvature calculation (consistency)

**Method details**:
- Gaussian blur preprocessing (sigma=15.0, threshold=0.7)
- Skeletonization via `skimage.morphology`
- Geodesic distance along skeleton
- B-spline smoothing (s=5.0) for curvature
- Automatic head-to-tail orientation via `orient_spline_head_to_tail()`

### 3. Smoothing Terminology
- **Original**: "heat kernel smoothing"
- **Corrected**: **"Gaussian kernel smoothing"**
- **Key clarification**: Gaussian kernel also provides a **density measure** (not just smoothing)
- **Implementation**: `scipy.ndimage.gaussian_filter` (2D) and `gaussian_filter1d` (1D)

---

## Files Updated

1. **`IMPLEMENTATION_PLAN.md`**:
   - Changed HPF bin from 26-28 to 48 hpf throughout
   - Updated Section 5 to use geodesic skeletonization instead of LPC
   - Replaced "heat kernel" with "Gaussian kernel"
   - Updated function signatures and dependencies
   - Updated config snippet

2. **`config.yaml`**:
   - `hpf_bin_start: 47.0`, `hpf_bin_end: 49.0`
   - Renamed `heat_kernel_sigma_*` to `gaussian_kernel_sigma_*`
   - Added comment about density measure

3. **`README.md`**:
   - Updated HPF bin to 48 hpf
   - Updated infrastructure section to reference geodesic skeletonization
   - Added density measure note for Gaussian kernel
   - Updated key design decisions

---

## Key Method Details (For Reference)

### Geodesic Centerline Extraction

**Location**: `segmentation_sandbox/scripts/body_axis_analysis/centerline_extraction.py`

**Usage**:
```python
from body_axis_analysis.centerline_extraction import extract_centerline

# Extract centerline with automatic orientation
spline_x, spline_y, curvature, arc_length = extract_centerline(
    mask,
    method='geodesic',              # default
    preprocess='gaussian_blur',     # default
    orient_head_to_tail=True,       # default
    um_per_pixel=10.0,
    bspline_smoothing=5.0,
    random_seed=42
)

# Or get full intermediate results
results = extract_centerline(mask, return_intermediate=True)
```

**Returns**:
- `spline_x, spline_y`: Smoothed centerline coordinates (N points)
- `curvature`: Curvature along centerline (rad/μm)
- `arc_length`: Arc length along centerline (μm)

**Advantages**:
- Robust to highly curved embryos
- Same method as data pipeline (consistent curvature calculations)
- Automatic preprocessing and orientation

### Gaussian Kernel Smoothing

**For 2D maps** (cost density, displacement fields):
```python
from scipy.ndimage import gaussian_filter

smoothed_field = gaussian_filter(field_2d, sigma=2.0)
# Also provides density measure (blurred values = local density)
```

**For 1D profiles** (along-S curves, AUROC profiles):
```python
from scipy.ndimage import gaussian_filter1d

smoothed_profile = gaussian_filter1d(profile_1d, sigma=1.0)
```

**Key point**: Gaussian kernel provides both smoothing and a density measure (blurred mask values represent local density of mass).

---

## No Changes to PLAN.md

The original `PLAN.md` specification document was **not modified** as requested. All corrections were applied only to the implementation plan and configuration files.

---

## Status

**Ready to proceed**: All corrections applied. Implementation can begin with Section 0 (data loading).
