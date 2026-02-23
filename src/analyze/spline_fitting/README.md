# Spline Fitting

Tools for fitting smooth curves through trajectory data using Local Principal Curves (LPC).

## Quick Start

```python
from src.analyze.spline_fitting import LocalPrincipalCurve

# Fit a curve through 3D trajectory points
lpc = LocalPrincipalCurve(bandwidth=0.5)
lpc.fit(points, start_points=start_point)
curve = lpc.cubic_splines[0]  # Fitted spline coordinates
```

## Module Overview

### `lpc_model.py` - Local Principal Curve (Standalone)
The core algorithm that iteratively fits a smooth curve through point clouds.
- `LocalPrincipalCurve` class - bandwidth-based curve fitting
- Cubic spline interpolation utilities
- **No dependencies on other spline_fitting modules** - fully self-contained

### `bootstrap.py` - Uncertainty Estimation
Fit splines with resampling to quantify uncertainty.
- `spline_fit_wrapper()` - bootstrap spline fitting with optional `group_by`
  - `group_by=None`: fits single spline (backwards compatible)
  - `group_by='phenotype'`: fits one spline per group, returns combined DataFrame
- Returns mean spline + confidence intervals per group

### `fitter.py` - Future SplineFitter Class (Placeholder)
Reserved for future object-oriented API:
```python
# Future API (not implemented yet)
fitter = SplineFitter(df, group_by='phenotype')
fitter.fit()
fitter.project(other_df)
fitter.splines  # dict of fitted splines
fitter.projections  # dict of projection results
```

### `curve_ops.py` - Curve Geometry Operations
All operations on the discretized curve: segmentation, projection, point mapping.
- `discretize_curve()` - split curve into ~200 micro-segments (internal)
- `split_spline()` - divide curve into k segments by arc-length
- `assign_points_to_segments()` - map data points to segments
- `create_spline_segments_for_df()` - end-to-end segmentation workflow
- `project_onto_plane()` - 2D plane projection
- `project_points_onto_reference_spline()` - 3D reference projection

### `alignment.py` - Curve Alignment
Align curves for comparison.
- `quaternion_alignment()` - align two curves using quaternion rotation (recommended)
- `procrustes_alignment()` - **LEGACY, NOT VALIDATED** - kept for historical reference only

### `dynamics.py` - Trajectory Dynamics
Analyze how systems change over time along splines.
- `run_bootstrap_journeys()` - simulated random walks
- `compute_developmental_shifts()` - stage-to-stage changes

### `viz.py` - Visualization
Spline-specific visualization functions.

**Augmentor functions** (add to existing figures):
- `add_spline_to_fig()` - add spline curve to a Plotly figure
- `add_uncertainty_tube()` - add bootstrap confidence tube

**Convenience function** (matches `plotting_3d.py` API):
- `plot_3d_with_spline()` - wraps `plot_3d_scatter` + adds spline overlay
  - Same API as `plot_3d_scatter`, plus `spline=` parameter
  - For quick one-liners when you just want scatter + spline

### `utils/spline_metrics.py` - Spline Metrics
Quantitative metrics for comparing splines.
- `segment_direction_consistency()` - direction similarity
- `calculate_dispersion_metrics()` - trajectory spread
- `rmse()`, `rmsd()` - error metrics

## Visualization Patterns

### Pattern 1: Augmentor (Most Flexible)
Build up a figure step by step:

```python
from src.analyze.trajectory_analysis.viz.plotting import plot_3d_scatter
from src.analyze.spline_fitting.viz import add_spline_to_fig

fig = plot_3d_scatter(df, coords=['PC1', 'PC2', 'PC3'], color_by='phenotype')
add_spline_to_fig(fig, fitted_curve, color='red', width=5)
fig.show()
```

### Pattern 2: Convenience Function (Quick One-Liner)
When you just want scatter + spline:

```python
from src.analyze.spline_fitting.viz import plot_3d_with_spline

fig = plot_3d_with_spline(
    df, coords=['PC1', 'PC2', 'PC3'],
    spline=fitted_curve,
    color_by='phenotype'
)
```

## Common Workflows

Quick examples for common tasks. For detailed function documentation, see module descriptions below.

### 1. Fit a spline through trajectory data
```python
from src.analyze.spline_fitting import LocalPrincipalCurve

lpc = LocalPrincipalCurve(bandwidth=0.5)
lpc.fit(data_points, start_points=anchor_point)
fitted_curve = lpc.cubic_splines[0]
```

### 2. Fit splines with bootstrap uncertainty
```python
from src.analyze.spline_fitting import spline_fit_wrapper

# Fit single spline (backwards compatible)
wt_spline = spline_fit_wrapper(wt_df, pca_cols=['PC1', 'PC2', 'PC3'], n_bootstrap=100)

# Fit multiple splines by group (NEW: group_by parameter)
all_splines = spline_fit_wrapper(
    df,
    group_by='phenotype',  # fits one spline per phenotype
    pca_cols=['PC1', 'PC2', 'PC3'],
    n_bootstrap=100
)
# Returns DataFrame with 'phenotype' column + spline coordinates + SE columns
```

### 3. Segment a trajectory
```python
from src.analyze.spline_fitting import create_spline_segments_for_df

segmented_df = create_spline_segments_for_df(
    df, n_segments=5,
    segment_labels=['early', 'mid1', 'mid2', 'mid3', 'late']
)
```

### 4. Project points onto a reference curve
```python
from src.analyze.spline_fitting import project_points_onto_reference_spline

projected_df = project_points_onto_reference_spline(
    embryo_df, reference_spline, pca_cols=['PC1', 'PC2', 'PC3']
)
```

### 5. Align and compare two curves
```python
from src.analyze.spline_fitting import quaternion_alignment
from src.analyze.spline_fitting.utils import segment_direction_consistency

aligned = quaternion_alignment(curve1, curve2)
consistency = segment_direction_consistency(curve1, curve2)
```

---

## Module Details

For detailed information about each module's functions and parameters, see sections below.
