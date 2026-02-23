# Surface Area Outlier Detection Analysis

**Date**: 2025-10-08
**Status**: In Progress
**Goal**: Fix one-sided SA outlier detection to catch embryos that are too small

---

## Problem Statement

The current SA outlier detection in `build04_perform_embryo_qc.py` only flags embryos with surface areas that are **too large**, missing embryos that are **too small**.

### Evidence:
- **20250711_F06_e01**: Consistently 29-48% of normal size (0.29x-0.48x vs p95 reference)
- **20250711_H07_e01**: Undersized at 48-88% of normal size (0.48x-0.88x vs p95 reference)
- **Current implementation**: Line 679 & 723 only check `SA > threshold`
- **Result**: 0 flags for F06 and H07 despite being obvious outliers

### Root Cause:
```python
# Line 617 comment:
"""
Note: This performs one-sided outlier detection, only flagging embryos that are
unusually large (surface_area > threshold). Small embryos are not flagged as they
may represent valid biological phenotypes rather than technical errors.
"""

# Line 679, 723:
df["sa_outlier_flag"] = df["surface_area_um"].to_numpy() > thresholds  # Only ">"
```

**Why this is wrong**: Small embryos can indicate:
- Incomplete segmentation (mask doesn't capture full embryo)
- Dead/dying embryos with arrested growth
- Technical artifacts (focus, tracking failures)

---

## Solution Plan

### Phase 1: Build Robust Reference Dataset

**Goal**: Aggregate SA distributions from all experiments to establish normal ranges

**Approach**:
1. Collect all `qc_staged_*.csv` files from `metadata/build04_output/`
2. Filter for **true wild-type controls only**:
   - `genotype in ['wik', 'ab', 'wik-ab']` (pure strain backgrounds)
   - `chem_perturbation == 'None'` or `isna()` (no chemical treatment)
   - `use_embryo_flag == True` (passed basic QC)
3. Calculate stage-binned statistics:
   - Bin by 0.5 hpf increments
   - Compute p5 (lower bound), p50 (median), p95 (upper bound) for each bin
   - Track sample size (n) for confidence
4. Smooth curves with Savitzky-Golay filter (window=5, poly=2)
5. **Output**: `sa_reference_curves.csv` with columns: `stage_hpf, p5, p50, p95, n`

**Script**: `build_sa_reference.py`

---

### Phase 2: Tune Thresholds

**Goal**: Find optimal upper/lower multipliers that catch true outliers without excessive false positives

**Test Cases**:
- **True positives**: 20250711_F06_e01, 20250711_H07_e01 (known bad embryos)
- **Target false positive rate**: <5% of embryos flagged

**Test Grid**:
- Upper threshold: `SA > k_upper * p95` where k_upper ∈ {1.2, 1.4, 1.6}
- Lower threshold: `SA < k_lower * p5` where k_lower ∈ {0.6, 0.7, 0.8}

**Visualizations**:
1. Plot SA vs stage with p5/p50/p95 reference bands
2. Overlay F06_e01 and H07_e01 trajectories
3. Show decision boundaries for different k values
4. Bar chart of flagging rates by threshold

**Script**: `tune_thresholds.py`

---

### Phase 3: Implement Two-Sided Detection

**Goal**: Add lower bound checking to catch small embryos

**Simple approach** (no over-engineering):
1. Load reference curves (p5, p95 vs stage)
2. For each embryo timepoint:
   - Interpolate expected p5 and p95 at current stage
   - Flag if `SA > k_upper * p95` **OR** `SA < k_lower * p5`
3. Keep analysis code in `tests/sa_outlier_analysis/` for now
4. Migration to production code happens later after validation

**Script**: `apply_two_sided_qc.py`

**Validation**:
- Run on 20250711 → verify F06 and H07 get flagged
- Run on all experiments → measure overall flag rate
- Manual spot-check ~20 flagged embryos

---

## Current Status

### Completed:
- ✅ Initial exploratory analysis (`analyze_sa_outliers.py`)
- ✅ Identified problem: one-sided detection missing small embryos
- ✅ Validated F06/H07 as true positives (too small)

### In Progress:
- 🔄 Documenting problem and solution plan (this file)

### To Do:
- ⏳ Build reference dataset from all build04 CSVs
- ⏳ Calculate and plot p5/p50/p95 curves
- ⏳ Tune thresholds using F06/H07 test cases
- ⏳ Implement two-sided detection

---

## Files in This Directory

```
tests/sa_outlier_analysis/
├── README.md                    # This file - problem statement and plan
├── analyze_sa_outliers.py       # Initial analysis (completed)
├── summary.txt                  # Output from initial analysis
├── build_sa_reference.py        # Phase 1: Build reference curves (to create)
├── tune_thresholds.py           # Phase 2: Optimize thresholds (to create)
└── apply_two_sided_qc.py        # Phase 3: Implement solution (to create)
```

### Outputs (to be generated):
```
tests/sa_outlier_analysis/outputs/
├── sa_reference_curves.csv      # p5/p50/p95 vs stage from all data
├── reference_plot.png           # Visualization of reference curves
├── threshold_tuning.png         # Decision boundaries for different k values
├── flagging_rates.png           # Bar chart of flag rates by threshold
└── validation_summary.txt       # Results on test datasets
```

---

## Migration Notes

**Important**: Keep all analysis in `tests/` folder until validated.

**Future work** (after validation):
1. Move validated logic to `src/data_pipeline/quality_control/surface_area_outlier_detection.py`
2. Deprecate `_sa_qc_with_fallback()` in `build04_perform_embryo_qc.py`
3. Update documentation in `docs/refactors/`

**Design principle**: Simple, straightforward data analysis. No over-engineering.

---

## References

- Current implementation: `src/build/build04_perform_embryo_qc.py:599-725`
- Death detection pattern: `src/data_pipeline/quality_control/death_detection.py`
- Build04 data location: `metadata/build04_output/qc_staged_*.csv`
