# Horizon Plots Refactoring: From Results to src/analyze

**Date**: October 29, 2025
**Status**: First-pass implementation complete; ready for focused review

## Overview

Extracted and **implemented** horizon-plot utilities under `src/analyze/difference_detection/`. The refactored modules now power the 20251020 model-comparison scripts as well as newer curvature workflows, eliminating duplication while keeping APIs notebook-friendly.

## Implementation Summary (2025-10-29)

- Ported CSV loading, pivoting, alignment, interpolation, and summary stats into `time_matrix.py`, wrapping them in unit-friendly helpers.
- Rebuilt the matplotlib logic inside `horizon_plots.py` with configurable grid rendering, shared colour scaling, best-condition overlays, and legend helpers.
- Added orchestration glue in `pipelines.py` so scripts can compose loaders, plotting, and export steps with minimal boilerplate.
- Updated `compare_3models_full_time_matrix.py` to rely on the new utilities, reducing the script to configuration/CLI parsing.
- Documented package exports (`__init__.py`) and left docstrings/examples inline for quick adoption.
- Flagged remaining follow-ups (e.g. optional test coverage, edge-case validation) in the "Next Steps" section below.

## New Structure

### Library Modules (src/analyze/)

```
src/analyze/difference_detection/
├── horizon_plots.py           [NEW] Reusable horizon plot visualization
├── time_matrix.py             [NEW] Time matrix data reshaping
├── __init__.py                [UPDATED] Export new modules
├── classification/
├── distribution/
├── plotting.py
├── metrics.py
└── pipelines.py
```

### Thin Wrapper Scripts (results/)

```
results/mcolon/20251020/
└── compare_3models_full_time_matrix_wrapper.py  [NEW] CLI orchestrator

results/mcolon/20251029_curvature_temporal_analysis/
├── 02_horizon_plots.py        [NEW] Curvature-specific usage
└── (other analysis scripts...)
```

## What Was Created

### 1. **src/analyze/difference_detection/horizon_plots.py**
Implements the full plotting toolkit:
- `plot_horizon_grid()` renders an N×M matrix of horizon plots with shared colour scaling, LOEO overlays, and optional annotations.
- `plot_best_condition_map()` derives categorical “winning condition” heatmaps and matching legends.
- Utility helpers manage colour normalization, legend patches, small-multiple style tweaks, and PDF/PNG saving patterns.

### 2. **src/analyze/difference_detection/time_matrix.py**
Reusable data prep helpers:
- `load_time_matrix_results()` reads per-condition CSVs (optionally grouped) into `TimeMatrixBundle` records.
- `build_metric_matrices()` and `align_matrix_times()` convert long-form metrics into aligned DataFrames that the plotting code takes directly.
- Additional helpers compute descriptive statistics, interpolate sparse timepoints, and filter time windows.

### 3. **src/analyze/difference_detection/pipelines.py**
Adds orchestration glue (`HorizonPlotContext`, `load_and_prepare_matrices`) so CLI scripts/notebooks can chain loading → reshaping → plotting with a couple of calls.

### 4. **src/analyze/difference_detection/__init__.py**
Exports the new public APIs (`load_time_matrix_results`, `plot_horizon_grid`, etc.) for notebook ergonomics.

### 5. **results/mcolon/20251020/compare_3models_full_time_matrix.py**
Trimmed to configuration plus calls into the new utilities. Legacy fallbacks were removed; the script now depends solely on the shared helpers.

### 6. **results/mcolon/20251029_curvature_temporal_analysis/02_horizon_plots.py**
Updated example notebook/script that demonstrates the new API from a curvature dataset, validating the cross-project reuse story.

## Follow-ups & Review Notes

1. **Testing** – No automated tests yet. Recommend adding unit coverage for loader alignment and smoke tests for plotting (`plot_horizon_grid`/`plot_best_condition_map`).
2. **Visual parity check** – Re-run the 20251020 comparison pipeline and spot-check PNGs against previous outputs to confirm scale/colour parity.
3. **Edge-case review** – `filter_matrices_by_time_range` and interpolation helpers mimic the original script, but double-check behaviour on sparse inputs.
4. **Docs/examples** – Once reviewers are satisfied, migrate relevant notebook snippets to point at `analyze.difference_detection` to cement usage patterns.


## API Contract

The reusable utilities should be parameterized to work with any data that has:
- A 2D metric value (start_time × target_time)
- Conditions/groups to compare
- Optional metadata (genotype labels, LOEO indicators, etc.)

### Example: Model Comparisons
```python
from analyze.difference_detection import plot_horizon_grid

matrices = {'WT': {'WT_test': df1, 'Het_test': df2, ...},
            'Het': {...},
            'Homo': {...}}

plot_horizon_grid(
    matrices,
    row_labels=['WT Model', 'Het Model', 'Homo Model'],
    col_labels=['WT Test', 'Het Test', 'Homo Test'],
    metric='mae',
    loeo_highlight={
        'WT': 'WT_test',
        'Het': 'Het_test',
        'Homo': 'Homo_test'
    }
)
```

### Example: Curvature Analysis
```python
from analyze.difference_detection import plot_horizon_grid

# Correlation matrices per genotype
matrices = {'WT': correlation_matrix_df,
            'Het': correlation_matrix_df,
            'Homo': correlation_matrix_df}

plot_horizon_grid(
    matrices,
    row_labels=['Wildtype', 'Heterozygous', 'Homozygous'],
    col_labels=['Arc Length Ratio'],
    cmap='RdBu_r'
)
```

## Files Modified/Created

| File | Status | Notes |
|------|--------|-------|
| src/analyze/difference_detection/horizon_plots.py | NEW | Fully implemented plotting utilities & docs |
| src/analyze/difference_detection/time_matrix.py | NEW | Data loading/alignment helpers powering plots |
| src/analyze/difference_detection/pipelines.py | NEW | Orchestration helpers + convenience context |
| src/analyze/difference_detection/__init__.py | UPDATED | Re-exports for notebook-friendly imports |
| results/mcolon/20251020/compare_3models_full_time_matrix.py | UPDATED | Now thin wrapper over shared utilities |
| results/mcolon/20251029_curvature_temporal_analysis/02_horizon_plots.py | UPDATED | Demonstrates new API on curvature data |
| results/mcolon/20251020/compare_3models_full_time_matrix_wrapper.py | NEW | Thin wrapper + fallback |
| results/mcolon/20251029_curvature_temporal_analysis/02_horizon_plots.py | NEW | Curvature usage example |
| results/mcolon/20251020/compare_3models_full_time_matrix.py | UNCHANGED | Kept for provenance |

## Notes

- All new files are **ready to use** but contain placeholder implementations
- Docstrings are comprehensive and include usage examples
- Existing fallback logic keeps scripts runnable immediately
- The skeleton structure forces clear API boundaries before implementation
- Tests should be written before filling in implementations
- **2025-10-29 | Utility extraction milestone**
  - Added `time_matrix.py` with reusable loaders, matrix builders, alignment, and basic statistics helpers.
  - Added `horizon_plots.py` for shared plotting routines (shared colour scaling, grid rendering, best-model maps).
  - Added `pipelines.py` providing a `HorizonPlotContext` loader plus summary utilities.
  - Refactored `results/mcolon/20251020/compare_3models_full_time_matrix.py` to import the new APIs; it now focuses on configuration/dispatch only.
  - Updated package exports so notebooks can pull in the new helpers directly from `analyze.difference_detection`.
