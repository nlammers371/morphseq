# Optimal Transport Utilities

Reusable optimal transport infrastructure for mask-based temporal analyses.

## Purpose

This module provides **general-purpose optimal transport utilities** that can be used across different analyses, not just embryo morphometrics. The code here is agnostic to the specific biological application.

## What's Included

### Core Data Structures (`config.py`)
- `UOTConfig`: Configuration for UOT solvers
- `UOTFrame`, `UOTFramePair`: Frame containers
- `UOTSupport`: Point cloud representation (coords + weights)
- `UOTProblem`: Full problem specification
- `UOTResultWork`: Work-grid-only solver output (no mixed frames)
- `UOTResultCanonical`: Canonical-grid-only lifted output (no mixed frames)
- `SamplingMode`, `MassMode`: Configuration enums

### Working Grid Seam (`working_grid.py`)
- `WorkingGridPair`: canonical<->work mapping plus prepared work-grid densities
- `prepare_working_grid_pair()`: crop/pad/downsample into a solver-friendly work grid
- Lifting helpers: `WorkingGridPair.lift_*` + `lift_work_result_to_canonical()`

### Backends (`backends/`)
- **`base.py`**: Abstract backend interface
- **`pot_backend.py`**: CPU implementation using POT library
- Pluggable design allows swapping solvers (e.g., GPU via JAX/ott-jax)

### Density Transforms (`density_transforms.py`)
- `mask_to_density_uniform()`: Uniform mass (0A mode)
- `mask_to_density_boundary_band()`: Boundary-weighted mass (0B mode)
- `mask_to_density_distance_transform()`: Distance-transform mass (0C mode)
- `enforce_min_mass()`: Fallback for near-zero mass

### Multiscale & Sampling (`multiscale_sampling.py`)
- `downsample_density()`: Sum-pooling for area-preserving downsampling
- `pad_to_divisible()`: Pad arrays to match divisor requirements
- `build_support()`: Extract point cloud from density with optional sampling

### Transport Maps (`transport_maps.py`)
- `compute_transport_maps()`: Convert coupling matrix to:
  - Mass created/destroyed maps
  - Velocity field (barycentric projection)

### Metrics (`metrics.py`)
- `summarize_metrics()`: Compute summary statistics from UOT results

## Example Usage

```python
import numpy as np

from analyze.utils.coord.grids.canonical import CanonicalGridConfig, to_canonical_grid_mask
from analyze.utils.optimal_transport import (
    MassMode,
    UOTConfig,
    WorkingGridConfig,
    lift_work_result_to_canonical,
    prepare_working_grid_pair,
    run_uot_on_working_grid,
)

# 1) Canonicalize upstream (coord owns geometry)
canonical_cfg = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576), align_mode="yolk")
src_can = to_canonical_grid_mask(mask_src_raw, um_per_px=um_per_px_src, yolk_mask=yolk_src, cfg=canonical_cfg)
tgt_can = to_canonical_grid_mask(mask_tgt_raw, um_per_px=um_per_px_tgt, yolk_mask=yolk_tgt, cfg=canonical_cfg)

# 2) Prepare work grid (working_grid owns crop/pad/downsample)
working_cfg = WorkingGridConfig(downsample_factor=2, padding_px=16, mass_mode=MassMode.UNIFORM)
pair = prepare_working_grid_pair(src_can, tgt_can, working_cfg)

# 3) Solve on work grid (solver/backends are math only)
solver_cfg = UOTConfig(epsilon=1e-2, marginal_relaxation=10.0, max_support_points=5000)
res_work = run_uot_on_working_grid(pair, config=solver_cfg)

# 4) Lift to canonical (interpretation/mapping step)
res_canon = lift_work_result_to_canonical(res_work, pair)
```

## Design Principles

1. **Backend-agnostic preprocessing**: All density transforms, downsampling, and sampling work with NumPy/SciPy and don't depend on the solver backend.

2. **Pluggable solvers**: Only the `solve()` method differs between backends. This allows easy GPU upgrades (JAX/ott-jax) without rewriting preprocessing.

3. **Physical mass preservation**: Densities are not normalized per-frame. The solver normalizes internally, then rescales outputs back to source mass units.

4. **Explicit coordinates**: All coordinates use `(y, x)` convention internally. Visualization code handles conversion to `(x, y)` for plotting.

## Potential Use Cases Beyond Morphometrics

- **Cell tracking**: Track cell populations across frames
- **Melanocyte migration**: Analyze pigment cell movement patterns
- **Tissue dynamics**: Compare tissue masks over time
- **Any temporal mask analysis**: Where you want to quantify morphological change

## Related

For embryo-specific analysis code (I/O, preprocessing, visualization), see:
- `src/analyze/optimal_transport_morphometrics/uot_masks/`
