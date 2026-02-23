# Faceted Plotting Refactoring Summary

## Changes Made

Simplified the faceted plotting API by merging `overlay` and `color_by` parameters into a single `color_by_grouping` parameter.

### Motivation

In practice, `overlay` and `color_by` were almost always set to the same value (e.g., `overlay='genotype', color_by='genotype'`). This redundancy added API complexity without providing real flexibility.

### Before

```python
plot_trajectories_faceted(
    df,
    col_by='cluster',      # What makes separate subplots
    overlay='genotype',    # What makes separate lines within a subplot
    color_by='genotype',   # What determines line color (usually same as overlay!)
)
```

### After

```python
plot_trajectories_faceted(
    df,
    col_by='cluster',            # What makes separate subplots
    color_by_grouping='genotype', # Groups AND colors lines within each subplot
)
```

---

## Files Modified

### 1. Core Plotting Module
**File**: `src/analyze/trajectory_analysis/facetted_plotting_refactored.py`

**Changes**:
- Renamed `overlay` → `color_by_grouping` in all function signatures
- Removed `color_by` parameter completely
- Simplified `build_color_state()`: removed `color_by` param, reduced to 3 dict entries
- Simplified color resolution in `_build_traces_for_cell()`: removed 4-level cascade
- Updated internal variable names: `overlay_col` → `color_by_grouping`, `overlay_val` → `grouping_val`

**Impact**:
- Removed ~15-20 lines of cascading color logic
- Function signatures reduced by 1 parameter
- Clearer mental model: `color_by_grouping` = grouping AND coloring within subplots

### 2. Pair Analysis Plotting
**File**: `src/analyze/trajectory_analysis/pair_analysis/plotting.py`

**Changes**:
- Updated 3 function calls: `overlay=` → `color_by_grouping=`
- Removed `color_by=` arguments (redundant with new API)

### 3. Usage Files
**File**: `results/mcolon/20251218_MD-DTW-morphseq_analysis/run_analysis.py`
- Updated `plot_multimetric_trajectories()` call: `overlay=` → `color_by_grouping=`

**File**: `results/mcolon/20251215_test_updated_pair_plotting/test_row_labels.py`
- Updated test: `overlay='genotype', color_by='genotype'` → `color_by_grouping='genotype'`

**File**: `results/mcolon/20251215_test_updated_pair_plotting/test_refactored_plotting.py`
- Updated 2 test cases with new parameter name

---

## API Changes Summary

| Function | Old Parameters | New Parameters |
|----------|---------------|----------------|
| `plot_trajectories_faceted()` | `overlay=..., color_by=...` | `color_by_grouping=...` |
| `plot_multimetric_trajectories()` | `overlay=..., color_by=...` | `color_by_grouping=...` |

**Migration Guide**:
```python
# If you had:
plot_trajectories_faceted(df, col_by='X', overlay='Y', color_by='Y')

# Change to:
plot_trajectories_faceted(df, col_by='X', color_by_grouping='Y')

# If you had:
plot_trajectories_faceted(df, col_by='X', overlay=None, color_by=None)

# Change to:
plot_trajectories_faceted(df, col_by='X', color_by_grouping=None)
```

---

## Testing

A comprehensive test script (`test_color_by_grouping_refactor.py`) was created and run successfully:
- ✓ Basic plotting without grouping
- ✓ Plotting with color_by_grouping
- ✓ Multimetric plotting with color_by_grouping
- ✓ Verified old parameters (`overlay`, `color_by`) are removed

All tests pass.

---

## What's Preserved

- IR pattern (TraceData/SubplotData/FigureData dataclasses)
- Homogeneity check for mean line coloring
- Backend dispatch (Plotly/Matplotlib/both)
- All plotting functionality remains identical
- Grid layouts (row_by, col_by) unchanged

---

## Benefits

1. **Simpler API**: One parameter instead of two
2. **Clearer intent**: `color_by_grouping` explicitly states it groups AND colors
3. **Less code**: ~15-20 lines of color resolution logic removed
4. **Same functionality**: No loss of plotting capabilities
