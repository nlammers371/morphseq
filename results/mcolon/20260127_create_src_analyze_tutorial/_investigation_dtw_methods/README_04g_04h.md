# Tutorial 04g/04h: Multivariate vs Per-Metric DTW Analysis

## Quick Summary

**Question:** Should we use per-metric DTW instead of multivariate DTW for better cross-experiment robustness?

**Answer:** **NO** - Multivariate DTW is the winner.

**Result:** Multivariate DTW achieves **11.6% disagreement** vs per-metric DTW's **20.4% disagreement** in cross-experiment tests.

**Decision:** **Keep multivariate DTW as default.** Document per-metric DTW as interpretability tool only.

---

## Results at a Glance

| Method | Cross-Experiment ARI | Disagreement | Verdict |
|--------|---------------------|--------------|---------|
| **Multivariate DTW** | **0.884** | **11.6%** | ✅ WINNER |
| Per-Metric DTW | 0.796 | 20.4% | ❌ 8.8% worse |

**Consistency:**
- Multivariate: Same 11.6% across both tests (consistent)
- Per-Metric: Varies 13.2% to 27.6% (inconsistent)

---

## Key Findings

### 1. Multivariate DTW Wins on Robustness
- Better cross-experiment performance
- More consistent across different experiment pairs
- Z-score normalization actually HELPS (counter-intuitive!)

### 2. Per-Metric DTW Wins on Interpretability
- Reveals curvature dominates clustering (r=0.95) vs length (r=0.69)
- Can quantify individual metric contributions
- Useful for publication figures and hypothesis generation

### 3. The Hypothesis Was Wrong
**Expected:** Per-metric would avoid Z-score artifacts
**Reality:** Z-scores create better cross-experiment consistency than raw values

### 4. Surprising Discovery
**Expected 25% disagreement** (from Tutorial 04d)
**Achieved 11.6% disagreement** (much better!)

---

## Recommendation

### What to Do

✅ **KEEP multivariate DTW as default** for:
- Cross-experiment projection
- Production pipelines
- Standard clustering workflows

✅ **USE per-metric DTW** (from tutorial code) for:
- Understanding metric contributions
- Publication figures
- Debugging clustering results
- Hypothesis generation

❌ **DO NOT refactor core library** to add per-metric option
- Complexity not justified (performs worse)
- Tutorial code available for specialized use

---

## Where to Find Results

### Main Documentation
- **`FINAL_CONCLUSIONS.md`** - Complete analysis and recommendations
- **`output/04g/TAKEAWAYS.md`** - Tutorial 04g within-experiment findings
- **`output/04h/FINAL_ANALYSIS.md`** - Tutorial 04h cross-experiment results

### Quick Results
- **`output/04h/results/disagreement_summary.csv`** - Quantified metrics
- **`output/04h/figures/disagreement_comparison.png`** - Main result plot
- **`output/04h/figures/confusion_matrices/`** - Per-experiment confusion matrices

### Tutorial Scripts
- **`04g_per_metric_dtw_combination.py`** - Per-metric DTW implementation (within-experiment)
- **`04h_cross_experiment_validation.py`** - Cross-experiment validation test

---

## Test Details

### Tutorial 04g: Within-Experiment Test
- **Dataset:** Experiment 20250512 (54 embryos)
- **Finding:** 87-89% cluster agreement between methods
- **Key insight:** Curvature (r=0.95) dominates over length (r=0.69)
- **Conclusion:** Per-metric works well within single experiment

### Tutorial 04h: Cross-Experiment Test
- **Reference:** Experiment 20250512 (54 embryos)
- **Test 1:** 20251017_combined (57 embryos) → multivariate 11.6%, per-metric 13.2%
- **Test 2:** 20251106 (70 embryos) → multivariate 11.6%, per-metric 27.6%
- **Conclusion:** Per-metric fails cross-experiment validation

---

## Why Multivariate DTW Wins

1. **Better performance:** 11.6% vs 20.4% disagreement
2. **Consistent:** Same result across different experiments
3. **Simpler:** Already in core library, well-tested
4. **Counter-intuitive benefit:** Z-scores help cross-experiment consistency

### Why Z-Score Normalization Helps

Even though Z-scores are "global," they create better cross-experiment consistency because:
- Standardize relative variability (σ), not just position (μ)
- Biological variability is more consistent than absolute ranges
- Single normalization (before DTW) is more stable than multiple (after DTW)

---

## Value of Per-Metric DTW

While it loses on cross-experiment robustness, per-metric DTW provides:

### Interpretability (Tutorial 04g Finding)
**Metric contributions to clustering:**
- Curvature (baseline_deviation): r = 0.95 - **DOMINANT**
- Body length (total_length): r = 0.69 - Secondary

**Biological insight:** Body shape trajectory matters much more than absolute size.

This insight is valuable and cannot be obtained from multivariate DTW after clustering.

### When to Use Per-Metric DTW
✅ Understanding which metrics drive results
✅ Creating interpretable figures
✅ Debugging unexpected patterns
✅ Within-experiment sensitivity analysis

❌ NOT for cross-experiment projection
❌ NOT for production pipelines

---

## Quick Start

### View Main Results
```bash
# Quantified metrics
cat output/04h/results/disagreement_summary.csv

# Full analysis
cat FINAL_CONCLUSIONS.md

# Visualizations
ls -lh output/04h/figures/
```

### Use Per-Metric DTW for Interpretability
See `04g_per_metric_dtw_combination.py` for complete working implementation.

Key function: `compute_per_metric_dtw()` computes DTW separately per metric and combines.

---

## File Organization

```
.
├── README_04g_04h.md                      # This file (quick summary)
├── FINAL_CONCLUSIONS.md                   # Complete analysis
├── EXECUTION_COMPLETE.md                  # Execution summary
├── IMPLEMENTATION_STATUS.md               # Implementation details
│
├── 04g_per_metric_dtw_combination.py     # Tutorial 04g script
├── 04h_cross_experiment_validation.py    # Tutorial 04h script
│
└── output/
    ├── README.md                          # Output directory guide
    ├── 04g/                               # Tutorial 04g results (organized)
    │   ├── TAKEAWAYS.md                   # Comprehensive analysis
    │   ├── IMPLEMENTATION_SUMMARY.md      # Technical details
    │   ├── figures/                       # 4 PNG visualizations
    │   ├── results/                       # Numerical summary
    │   └── logs/                          # Execution log
    │
    └── 04h/                               # Tutorial 04h results (generated)
        ├── FINAL_ANALYSIS.md              # Detailed analysis
        ├── figures/                       # Comparison plots
        │   ├── disagreement_comparison.png
        │   └── confusion_matrices/
        ├── results/                       # disagreement_summary.csv
        └── logs/                          # run.log
```

---

## Bottom Line

**Multivariate DTW is the winner** - it should remain the default method for trajectory clustering.

**Per-metric DTW has value** - as an interpretability tool for understanding which metrics drive clustering.

**The decision is data-driven** - based on rigorous testing across 235 embryos, 6 distance matrices, and multiple experiments.

**11.6% cross-experiment disagreement is acceptable** - much better than the 25% we were trying to solve.

---

**For full details, see:** `FINAL_CONCLUSIONS.md`

**Date:** February 3, 2026
