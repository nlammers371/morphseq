# Plan: Consolidate Spline/LPC Utilities into `src/analyze/spline_fitting/`

## Summary

Move scattered spline fitting and LocalPrincipalCurve code from `src/functions/` into a new consolidated module at `src/analyze/spline_fitting/`.

---

## Current State

| File | Size | Key Contents |
|------|------|--------------|
| `src/functions/spline_morph_spline_metrics.py` | 75KB | LPC class, segmentation, 3D projection, developmental |
| `src/functions/spline_fitting_v2.py` | 17KB | LPC class (duplicate), bootstrap fitting |
| `src/functions/improved_build_splines.py` | 16KB | Bootstrap wrapper |
| `src/functions/embryo_df_performance_metrics.py` | 125KB | LPC class (duplicate), comparison metrics |

**Problem:** `LocalPrincipalCurve` duplicated in 3 files.

---

## Target Structure

```
src/analyze/spline_fitting/
├── __init__.py           # Public API exports
├── lpc_model.py          # LocalPrincipalCurve class (STANDALONE - no deps)
├── bootstrap.py          # Bootstrap fitting with group_by support
├── fitter.py             # PLACEHOLDER for future SplineFitter class
├── curve_ops.py          # All curve geometry: discretize, segment, project
├── alignment.py          # quaternion_alignment (+ legacy procrustes with warning)
├── dynamics.py           # Journey simulation, developmental shifts
├── viz.py                # add_spline_to_fig() + plot_3d_with_spline()
├── utils/
│   ├── __init__.py
│   └── spline_metrics.py # rmse, rmsd, segment_direction_consistency, dispersion
└── _compat.py            # Removed (no shims)
```

**Key design decisions:**
- `lpc_model.py` is **standalone** - no dependencies on other modules (confirmed by code review)
- `curve_ops.py` - all curve geometry in one place (discretize → segment → project are tightly coupled)
- `dynamics.py` - standard term for systems changing over time
- `alignment.py` - quaternion is primary; procrustes kept with **strong warning** it's not validated
- Metrics in `utils/spline_metrics.py` - separate from alignment logic
- `viz.py` has both:
  - **Augmentors**: `add_spline_to_fig()` for composability
  - **Convenience**: `plot_3d_with_spline()` for quick one-liners (matches `plot_3d_scatter` API)

---

## Implementation Steps

1. **Create `lpc_model.py`** - Merge 3 LPC implementations into one canonical class
2. **Create `bootstrap.py`** - Move `spline_fit_wrapper()`, add `group_by` parameter
3. **Create `fitter.py`** - Placeholder file with docstring for future SplineFitter class
4. **Create `curve_ops.py`** - Move all curve geometry operations (segmentation + projection)
5. **Create `alignment.py`** - Move quaternion_alignment; keep procrustes with strong warning
6. **Create `utils/spline_metrics.py`** - Move metrics (rmse, direction consistency, dispersion)
7. **Create `dynamics.py`** - Move trajectory dynamics functions
8. **Create `viz.py`** - Both augmentors AND `plot_3d_with_spline()`
9. **Create `__init__.py`** - Export public API
10. **Create `_compat.py`** - Removed (no shims)
11. **Update old files** - Re-export from new location with warnings

---

## Verification

1. Import test: `from src.analyze.spline_fitting import LocalPrincipalCurve`
2. Run existing notebooks that use spline fitting
3. Verify deprecation warnings for old import paths
4. Test augmentor pattern: `add_spline_to_fig(fig, curve)` on existing figure
5. Test convenience function: `plot_3d_with_spline(df, coords, spline=curve)`
