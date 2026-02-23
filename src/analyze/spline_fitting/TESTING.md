# Usage Examples for spline_fitting Module

This doc focuses on usage patterns and module layout. Refer to the examples below
to validate end-to-end workflows.

## Usage Examples

```python
# Basic LPC fitting
from src.analyze.spline_fitting import LocalPrincipalCurve
lpc = LocalPrincipalCurve(bandwidth=0.5)
lpc.fit(points, num_points=50)
curve = lpc.cubic_splines[0]

# Bootstrap with uncertainty
from src.analyze.spline_fitting import spline_fit_wrapper
spline_df = spline_fit_wrapper(df, pca_cols=['PC1', 'PC2', 'PC3'], n_bootstrap=100)

# Multi-group fitting
all_splines = spline_fit_wrapper(df, group_by='phenotype', pca_cols=['PC1', 'PC2', 'PC3'])

# Alignment
from src.analyze.spline_fitting import quaternion_alignment
R, t = quaternion_alignment(curve1, curve2)

# Metrics
from src.analyze.spline_fitting import rmse, segment_direction_consistency
error = rmse(curve1, curve2)
sim, cov = segment_direction_consistency(curve1, curve2, k=10)
```

## Module Organization

```
src/analyze/spline_fitting/
├── README.md             ✓ Comprehensive guide
├── PLAN.md               ✓ Implementation plan
├── __init__.py           ✓ Clean public API (20 exports)
├── lpc_model.py          ✓ Standalone core algorithm
├── bootstrap.py          ✓ With group_by support
├── curve_ops.py          ✓ Geometry operations
├── alignment.py          ✓ Quaternion + legacy procrustes
├── dynamics.py           ✓ Journeys + developmental shifts
├── viz.py                ✓ Augmentors + convenience functions
├── fitter.py             ✓ Placeholder for future class
├── _compat.py            ✗ Removed (no shims)
└── utils/
    └── spline_metrics.py ✓ Comparison metrics
```

## Backwards Compatibility

- ✅ `group_by=None` preserves original single-spline API
- ✅ All original function signatures unchanged
- ✅ Return types match original implementations

## Next Steps

1. Add deprecation warnings to old import paths in `src/functions/`
2. Update notebooks to use new import paths
3. Consider implementing SplineFitter class (see fitter.py placeholder)
