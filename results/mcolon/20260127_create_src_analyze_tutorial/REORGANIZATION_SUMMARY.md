# Tutorial Reorganization Summary

**Date:** 2026-02-05
**Status:** ✅ Complete

## What Was Done

### 1. Updated Scripts 01-02 for CEP290 Data

**Modified:**
- `01_feature_over_time.py` - Now uses CEP290 experiments (20260122, 20260124)
- `02_3d_pca_scatter.py` - Now uses CEP290 experiments (20260122, 20260124)

**Already Correct:**
- `03_dtw_clustering.py` - Already used CEP290 data
- `04_cluster_projection.py` - Already CEP290 projection analysis
- `05_projection_splines.py` - Already works on projection outputs
- `06_projection_classification_test.py` - Already works on projection outputs

### 2. Archived B9D2 Tutorial Scripts

**Location:** `_archive_b9d2_tutorial/`

**Moved 6 scripts:**
- 04_cluster_labeling.py
- 05_faceted_feature_plots.py
- 06_proportions.py
- 07_spline_per_cluster.py
- 08_difference_detection.py
- 09_plot_results.py

**Created:** `_archive_b9d2_tutorial/README.md` - Explains original B9D2 exploratory work

### 3. Archived Investigation Scripts

**Location:** `_investigation_dtw_methods/`

**Moved 7 scripts:**
- 04b_compare_clustering_methods.py
- 04c_distance_matrix_test.py
- 04d_direct_distance_comparison.py
- 04e_normalization_alternatives.py
- 04f_clustering_normalization_test.py
- 04g_per_metric_dtw_combination.py
- 04h_cross_experiment_validation.py

**Moved 8 documentation files:**
- EXECUTION_COMPLETE.md
- FINAL_CONCLUSIONS.md
- HANDOFF_tutorial_04_projection.md
- IMPLEMENTATION_STATUS.md
- INDEX.md
- README_04g_04h.md
- RUN_04h.md

**Created:** `_investigation_dtw_methods/README.md` - Summarizes DTW method investigation findings

### 4. Rewrote Main README

**File:** `README.md`

**New Focus:**
- CEP290 cross-experiment analysis as canonical flow
- Key finding: Temporal coverage determines penetrance detection
- Clear tutorial progression: 01 → 02 → 03 → 04 → 05 → 06
- Comprehensive documentation of all 6 scripts
- API reference for key patterns
- Archive directory documentation

## Final Tutorial Structure

### Canonical Flow (6 Scripts)

```
01_feature_over_time.py              → CEP290 feature visualization
02_3d_pca_scatter.py                 → CEP290 PCA analysis
03_dtw_clustering.py                 → CEP290 within-experiment clustering
04_cluster_projection.py             → Cross-experiment projection (KEY)
05_projection_splines.py             → Splines for projected groups
06_projection_classification_test.py → Statistical validation
```

### Archive Directories

```
_archive_b9d2_tutorial/        → Original B9D2 exploratory work (6 scripts)
_investigation_dtw_methods/    → DTW method investigation (7 scripts + 8 docs)
```

### Documentation

```
README.md                  → Main tutorial documentation (CEP290-focused)
README_tutorial_04.md      → Detailed script 04 documentation
projection_utils.py        → Utilities for script 04
```

## Key Finding Documented

**Temporal Coverage is Critical:**
- 20260122 (11-47 hpf): Cannot detect "Low_to_High" trajectories reliably
- 20260124 (27-77 hpf): Extended window captures full penetrance dynamics
- Batch effect p = 0.0081 between experiments
- Implication: Design experiments with sufficient imaging windows

## Files Kept Unchanged

- `projection_utils.py` - Used by script 04
- `README_tutorial_04.md` - Documents script 04 methodology
- `test_cep290_quick.py` - Test script
- `01_debug_feature_overtime.py` - Debug script
- `output/` directory - Generated results

## Verification Checklist

- [x] Scripts 01-02 updated to use CEP290 data (build04_output)
- [x] Script 03 already uses CEP290 data (verified)
- [x] Scripts 04-06 already use CEP290 data (verified)
- [x] B9D2 scripts archived (6 scripts)
- [x] Investigation scripts archived (7 scripts)
- [x] Documentation files archived (8 files)
- [x] Archive READMEs created (2 files)
- [x] Main README rewritten (CEP290-focused)
- [x] Directory structure clean and organized

## Summary

The tutorial has been successfully reorganized to make CEP290 cross-experiment analysis the canonical flow. The original B9D2 exploratory work and DTW method investigation are preserved in well-documented archive directories. The main tutorial now provides a clean, linear progression (01→02→03→04→05→06) demonstrating CEP290 penetrance trajectory analysis with cross-experiment projection.

**Result:** Clean, focused tutorial documenting the key finding that temporal coverage determines ability to detect penetrance dynamics.
