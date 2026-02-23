# Body Axis Analysis Package

Consolidated suite for zebrafish embryo centerline extraction and curvature analysis.

## Quick Start

```python
from segmentation_sandbox.scripts.body_axis_analysis import extract_centerline

# Extract centerline from mask (uses Geodesic method by default)
spline_x, spline_y, curvature, arc_length = extract_centerline(mask)

# Tune B-spline smoothing if needed
spline_x, spline_y, curvature, arc_length = extract_centerline(mask, bspline_smoothing=3.0)

# Get full analysis results
results = extract_centerline(mask, return_intermediate=True)
print(f"Total length: {results['stats']['total_length']:.1f} pixels")
print(f"Mean curvature: {results['stats']['mean_curvature']:.6f}")
print(f"Method: {results['stats']['method']}")
```

## Package Structure

```
body_axis_analysis/
├── __init__.py                  # Package exports
├── README.md                    # This file
├── METHODS_DECISIONS.md         # Detailed decision rationale
├── centerline_extraction.py     # Main high-level API
├── geodesic_method.py           # Geodesic skeleton approach (primary)
├── pca_method.py                # PCA slicing approach (fallback)
├── mask_preprocessing.py        # Gaussian blur + alpha shape
└── spline_utils.py              # Head/tail identification utilities
```

## Methods

### Geodesic (Primary Method - Default)

**Best for:** Robust analysis, curved/complex embryos

- Extracts centerline using geodesic distance along skeleton
- Handles cases where head is near tail
- Applies B-spline smoothing (s=5.0) for curvature measurement
- **~1.3s per embryo** ⚡ (optimized Oct 2025)
- **Reproducible** with fixed random seed (default: 42)

**Use when:**
- Embryos are highly curved or have complex shapes
- Robustness is more important than speed
- Working with unknown or challenging data
- **Recommended as default**

### PCA (Fast Fallback)

**Best for:** Speed-critical applications, normal embryo shapes

- Extracts centerline via PCA-based slicing perpendicular to principal axis
- Much faster (~5.3s per embryo, 2.8x speedup)
- ~97.5% agreement with Geodesic on normal shapes

**Use when:**
- Speed is critical
- Embryos are known to be normally shaped
- Batch processing of large datasets
- Morphology metrics: extent > 0.35 AND solidity > 0.6 AND eccentricity < 0.98

### Method Comparison

```python
from body_axis_analysis import compare_methods

comparison = compare_methods(mask)
print(f"Hausdorff distance: {comparison['hausdorff_distance']:.1f} pixels")
print(f"Aligned distance: {comparison['mean_aligned_distance']:.1f} pixels")
# Use to validate method choice for your data
```

## Key Features

### Preprocessing
- **Gaussian blur** (default, <0.1s): Smooths boundaries, removes spines
- **Alpha shape** (experimental, not yet implemented): For future geodesic method variants
- Applied to all masks before spline fitting

### Orientation
- Automatic head→tail orientation based on width tapering
- Biological: head is wider than tail
- Optional: disable with `orient_head_to_tail=False`

### B-Spline Smoothing
- Fixed parameter s=5.0 (tested optimal)
- Adaptive: longer centerlines get more smoothing
- Analytical derivative computation for accurate curvature

### Curvature Computation
```
κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
```
- Computed from B-spline analytical derivatives
- More accurate than numerical differentiation
- Returned in 1/pixels or 1/microns (based on um_per_pixel)

## API Reference

### Main Function

```python
extract_centerline(mask, method='geodesic', preprocess='gaussian_blur',
                   orient_head_to_tail=True, um_per_pixel=1.0,
                   bspline_smoothing=5.0, random_seed=42,
                   return_intermediate=False)
```

**Key Parameters:**
- `method`: 'geodesic' (default) or 'pca'
- `bspline_smoothing`: B-spline smoothing (default=5.0), tunable for your data
- `random_seed`: For reproducible geodesic endpoint detection (default=42)

**Returns:**
- `(spline_x, spline_y, curvature, arc_length)` if `return_intermediate=False`
- Full results dict if `return_intermediate=True`

### Method Comparison

```python
from segmentation_sandbox.scripts.body_axis_analysis import compare_methods

comparison = compare_methods(mask)
print(f"Hausdorff distance: {comparison['hausdorff_distance']:.1f}")
```

### Individual Analyzers

```python
from segmentation_sandbox.scripts.body_axis_analysis import (
    GeodesicCenterlineAnalyzer,
    PCACenterlineAnalyzer
)

# Advanced usage with custom parameters
analyzer = GeodesicCenterlineAnalyzer(mask, um_per_pixel=1.0, bspline_smoothing=5.0)
results = analyzer.analyze()

```

## Documentation

See **METHODS_DECISIONS.md** for:
- Detailed reasoning for each decision
- Method comparison analysis results
- Parameter tuning guidelines
- When to use each method
- Troubleshooting guide

## Key Results from Analysis

### 1000-Embryo Comparison (PCA vs Geodesic)
- 97.5% agreement (Hausdorff < 114.78px)
- 2.5% disagreement on extremely curved embryos
- PCA fails when: extent < 0.35 OR solidity < 0.6 OR eccentricity > 0.98

### B-Spline Smoothing
- Tested: 0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0
- Optimal: 5.0 balances noise reduction and feature preservation
- Adaptive scaling: `s = 5.0 * len(centerline)`

### Mask Cleaning
- Conditional opening when solidity < 0.6
- Saves ~40% of computation (40% of embryos have solidity > 0.6)
- Effective at removing spines without harming thin structures

## Performance

| Operation | Time (per embryo) | Notes |
|-----------|------------------|-------|
| Mask preprocessing | <0.1s | Gaussian blur, always applied |
| Geodesic extraction | **~1.1s** | ⚡ Optimized (was ~14.4s) |
| PCA extraction | ~5.2s | Fast fallback |
| B-spline smoothing | <0.1s | Same for both methods |
| Curvature computation | <0.1s | Analytical derivatives |
| **Total (Geodesic)** | **~1.3s** | ⚡ 12.65x faster (was ~14.6s) |
| **Total (PCA)** | **~5.3s** | Fast path |

### Speed Optimization (Oct 2025)

✅ **Graph construction: O(N²) → O(N)**
- Replaced pairwise distance checks with hash-table based 8-neighbor lookup
- **Result: 12.65x median speedup** (3.6s → 0.285s per embryo)
- **Geometric accuracy: Perfect** (0.0px Hausdorff distance)
- All downstream analysis automatically benefits from speedup

See `results/mcolon/20251028/` for detailed benchmark results and comparison plots.

## Integration with Existing Code

### Replacing Old Methods

Old code:
```python
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
# ... various scripts with different centerline methods
```

New code:
```python
from segmentation_sandbox.scripts.body_axis_analysis import extract_centerline
x, y, curv, arc = extract_centerline(mask)  # Single unified API
```

### Migration Notes

- `bodyaxis_spline_utils.py` moved to `body_axis_analysis/spline_utils.py`
- `mask_cleaning.py` stays in `scripts/utils/` (imported by body_axis_analysis)
- Skeleton pruning removed (didn't work reliably)
- All experimental methods left in `results/mcolon/` for reference

## What's Included

✅ **Consolidated:**
- Geodesic centerline extraction (class + unified API)
- PCA centerline extraction (class + unified API)
- Mask preprocessing (Gaussian blur + alpha shape)
- Spline utilities (head/tail identification, alignment)
- Full documentation of decisions and rationale

⚠️ **Toggleable (in mask_preprocessing.py):**
- Gaussian blur (default)
- Alpha shape (alternative)

🚫 **Removed (didn't work):**
- Skeleton pruning (fragile, no universal parameters)
- Alternative mask refinement methods (kept in results/ for research)

## Testing

To test the package:

```python
import numpy as np
from segmentation_sandbox.scripts.body_axis_analysis import extract_centerline

# Create test mask
mask = np.zeros((100, 200), dtype=bool)
mask[40:60, 20:180] = True  # Simple rectangle

# Extract centerline
x, y, curv, arc = extract_centerline(mask, method='auto')
print(f"Length: {arc[-1]:.1f} px, Mean curvature: {curv.mean():.6f} 1/px")
```

## Benchmarking Speedups

Use `geodesic_speedup.py` to compare the baseline geodesic analyzer with
prototype optimizations. All variants operate on the same cleaned and
preprocessed mask to ensure a fair timing comparison.

```bash
python -m segmentation_sandbox.scripts.body_axis_analysis.geodesic_speedup \
    --mask /path/to/mask.npy \
    --clean \
    --preprocess gaussian_blur \
    --gaussian-sigma 10 \
    --gaussian-threshold 0.7 \
    --repeats 5
```

The script reports per-analyzer runtimes, speed-ups relative to the baseline,
and Hausdorff distances to confirm that centerline geometry remains equivalent.

## Questions & Issues

1. **Method disagreement**: Use `compare_methods()` to see both results
2. **Unexpected centerline**: Check morphology metrics (extent, solidity, eccentricity)
3. **Performance issues**: Use `method='pca'` for speed or check mask preprocessing
4. **Spline looks wrong**: Increase `bspline_smoothing` or check original mask quality

See METHODS_DECISIONS.md for more troubleshooting.

## Related Files

- **mask_cleaning.py**: 5-step mask cleaning pipeline (imported automatically)
- **results/mcolon/20251027/**: Comparison analysis scripts
- **results/mcolon/20251022/**: B-spline parameter testing
- **results/mcolon/20251020/**: Morphology analysis
