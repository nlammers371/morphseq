# Tutorial 04 Implementation Handoff

## Summary

Implemented Tutorial 04 demonstrating cluster projection methodology for CEP290 experiments, along with a robust projection API in the main codebase. The tutorial shows how to project new trajectories onto reference cluster definitions while avoiding extrapolation bias.

## What Was Accomplished

### 1. Created New Projection Module (`src/analyze/trajectory_analysis/projection.py`)

**Location**: `src/analyze/trajectory_analysis/projection.py`

**Key Functions**:
- `project_onto_reference_clusters()` - High-level projection API (recommended)
- `compute_cross_dtw_distance_matrix()` - Cross-dataset DTW computation
- `assign_clusters_nearest_neighbor()` - NN cluster assignment
- `assign_clusters_knn_posterior()` - KNN with uncertainty quantification

**Critical Feature - Automatic Time Window Detection**:
```python
# Automatically finds temporal intersection to avoid extrapolation
window_start = max(source_min, ref_min)
window_end = min(source_max, ref_max)
```

This solves the major issue where comparing trajectories with different temporal coverage would create extrapolation artifacts (NaN values) that biased DTW distances.

### 2. Tutorial 04: Cluster Projection Script

**File**: `results/mcolon/20260127_create_src_analyze_tutorial/04_cluster_projection.py`

**What it demonstrates**:
- Projects CEP290 experiments (20260122, 20260124) onto reference clusters from 7 older experiments
- Shows batch effect from temporal coverage differences
- Uses new `project_onto_reference_clusters()` API

**Key Results**:
- **20260122** (12-47 hpf): 63.7% Not Penetrant, 25.7% Low_to_High (reasonable for early window)
- **20260124** (27-77 hpf): 54.1% Low_to_High (captures full penetrance trajectory)
- **Chi-square**: χ² = 20.739, p = 0.0001 (significant batch effect from temporal coverage)

### 3. Fixed Import Issues

**Problem**: After recent refactoring (commit `5e57a782`), `PHENOTYPE_COLORS` was removed but still being imported in multiple places.

**Solution**:
- Removed `PHENOTYPE_COLORS` and `PHENOTYPE_ORDER` from `src/analyze/viz/styling/__init__.py`
- These are B9D2-specific and should be imported directly when needed
- Updated `src/analyze/trajectory_analysis/config.py` to import directly

### 4. Updated Plotting to Use `plot_proportions`

Replaced custom matplotlib proportion plotting with the new unified `plot_proportions()` API from `src/analyze/viz/plotting/proportions.py`. This provides:
- Faceted layouts (row/col by variables)
- Consistent styling
- Automatic count annotations
- Cleaner, more maintainable code

## The Critical Issue We Solved

### Problem: Extrapolation Bias in DTW

**Original approach**:
- Used a single global time window (11-80 hpf) for all comparisons
- When 20260122 (only goes to 47 hpf) was interpolated to the full 11-80 hpf grid, timepoints 47-80 hpf became NaN
- DTW was comparing these NaN-padded trajectories against full-length references
- Result: 87.6% misclassified as "Not Penetrant" (totally wrong!)

**Solution**:
- Use experiment-specific time windows (intersection of source and reference)
- Only compare real data (no extrapolation, no NaN padding)
- Result: 63.7% Not Penetrant (much more reasonable!)

### Why Automatic Time Window Detection Matters

For projection to work correctly:
1. **Source and reference must use the same time grid** (for DTW to work)
2. **But the grid must only span where BOTH have real data** (avoid extrapolation)
3. **This is different for each source experiment** (different temporal coverage)

The new `project_onto_reference_clusters()` handles this automatically.

## API Usage Example

### Old Way (Manual, Error-Prone)
```python
# Manually filter to time window
TIME_WINDOW = (25, 50)  # Hardcoded, might not be right for all experiments!
df_source_filtered = df_source[
    (df_source['hpf'] >= TIME_WINDOW[0]) &
    (df_source['hpf'] <= TIME_WINDOW[1])
]
df_ref_filtered = df_ref[...]

# Prepare arrays manually
X_source, source_ids, time_grid = prepare_multivariate_array(df_source_filtered, ...)
X_ref, ref_ids, _ = prepare_multivariate_array(df_ref_filtered, time_grid=time_grid, ...)

# Compute DTW manually
D_cross = compute_cross_dtw_distance_matrix(X_source, X_ref, ...)

# Assign clusters manually
df_proj = assign_clusters_nearest_neighbor(D_cross, source_ids, ref_ids, ...)
```

### New Way (Automatic, Robust)
```python
from src.analyze.trajectory_analysis import project_onto_reference_clusters

# Single function call, automatic time window detection
assignments, time_grid = project_onto_reference_clusters(
    source_df=df_new_experiment,
    reference_df=df_reference,
    reference_cluster_map=cluster_map,
    reference_category_map=category_map,
    metrics=['baseline_deviation_normalized'],
    sakoe_chiba_radius=20,
)
```

## Files Modified/Created

### Created
- `src/analyze/trajectory_analysis/projection.py` - New projection module
- `results/mcolon/20260127_create_src_analyze_tutorial/04_cluster_projection.py` - Tutorial script
- `results/mcolon/20260127_create_src_analyze_tutorial/README_tutorial_04.md` - Tutorial documentation

### Modified
- `src/analyze/trajectory_analysis/__init__.py` - Added projection exports
- `src/analyze/viz/styling/__init__.py` - Removed B9D2-specific imports
- `src/analyze/trajectory_analysis/config.py` - Fixed phenotype color imports

### Supporting Files (Already Existed)
- `results/mcolon/20260127_create_src_analyze_tutorial/projection_utils.py` - Local copy for `compare_cluster_frequencies()`

## Outputs Generated

```
output/
├── figures/04/
│   ├── projection_results/
│   │   ├── 20260122_projection_nn.csv          # 113 embryo assignments
│   │   ├── 20260124_projection_nn.csv          # 98 embryo assignments
│   ├── cluster_projection_trajectories.png     # Faceted trajectories
│   ├── proportion_by_experiment.png            # Using new plot_proportions()
│   └── batch_effect_analysis.png               # Distance distributions
└── results/
    └── cluster_frequency_comparison.csv        # Chi-square test results
```

## Future Work / TODOs

### 1. NaN Handling in DTW (Not Implemented Yet)
Currently DTW doesn't explicitly handle NaN values. Options:
- **Option A**: Treat NaN timepoints as infinite cost (skip comparison)
- **Option B**: Mask NaN timepoints and only compute DTW over valid regions

**Recommendation**: Don't implement yet. The automatic time window detection already prevents NaN values from extrapolation, which was the main issue. Real missing observations (sparse timepoints) are rare and should be handled case-by-case.

### 2. Remove `fill_edges` Parameter
The `fill_edges` parameter in `interpolate_to_common_grid_multi_df()` is a footgun that enables extrapolation. Should be removed to prevent users from shooting themselves in the foot.

**Location**: `src/analyze/trajectory_analysis/utilities/trajectory_utils.py`

### 3. Move `compare_cluster_frequencies` to Main Codebase
Currently in `projection_utils.py` (local copy). Should be moved to:
- `src/analyze/trajectory_analysis/projection.py` OR
- `src/analyze/utils/statistics.py` (if more general)

### 4. Add Projection Validation
Add checks to warn users if:
- Time window intersection is very small (< 10 hpf)
- Distance distributions are unusual (very high mean/std)
- Many embryos have high nearest_distance (poor projection quality)

## Testing

**Validation**:
- ✅ Script runs without errors
- ✅ Automatic time window detection works correctly
- ✅ Results are biologically reasonable (not 87% misclassification!)
- ✅ Chi-square test shows expected batch effect
- ✅ Proportion plots generated correctly with new API

**Quick Test**:
```bash
python results/mcolon/20260127_create_src_analyze_tutorial/04_cluster_projection.py
```

## Key Takeaways

1. **Never extrapolate when doing DTW projection** - only compare real data
2. **Time windows must be experiment-specific** - different experiments have different coverage
3. **Automatic detection is critical** - manual windows are error-prone and don't scale
4. **The new projection API handles all of this** - use it instead of manual steps

## Questions/Decisions Needed

None - implementation is complete and working. Future work items above are optional improvements, not blockers.

---

**Date**: 2026-02-02
**Implemented by**: Claude (with user guidance)
**Ready for**: Production use in analyses
