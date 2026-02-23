# Quick Guide: Running Tutorial 04h

## Purpose
Test whether per-metric DTW solves the 25% cross-experiment disagreement issue found with multivariate DTW.

## Prerequisites
✅ Tutorial 04g completed and organized (already done)
✅ Data file available: `embryo_data_with_labels.csv`
✅ Python environment with required packages

## One-Line Execution

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260127_create_src_analyze_tutorial && python 04h_cross_experiment_validation.py
```

## Expected Output

The script will:
1. Load reference experiment (20250512, 54 embryos)
2. Load test experiments (20251017_combined, 20251106, 20251112)
3. Cluster reference with multivariate DTW
4. Cluster reference with per-metric DTW
5. For each test experiment:
   - Cluster reference+test combined (both methods)
   - Measure disagreement (ARI metric)
   - Generate confusion matrices
6. Create comparison plots
7. Provide automated recommendation

**Runtime:** ~5-15 minutes

## Check Results

```bash
# View summary CSV
cat output/04h/results/disagreement_summary.csv

# View recommendation
grep -A 20 "RECOMMENDATION" output/04h/logs/run.log

# View comparison plot
ls -lh output/04h/figures/disagreement_comparison.png

# View confusion matrices
ls -lh output/04h/figures/confusion_matrices/
```

## Interpret Results

### Key Metric: Adjusted Rand Index (ARI)
- **1.0** = Perfect agreement (0% disagreement)
- **0.90** = High agreement (10% disagreement) - threshold for "success"
- **0.75** = Moderate agreement (25% disagreement) - expected for multivariate
- **0.0** = Random agreement (100% disagreement)

### Decision Criteria

**If per-metric ARI > 0.90 (strong improvement):**
```
✅ SWITCH TO PER-METRIC DTW AS DEFAULT
Next steps:
  1. Add compute_per_metric_dtw() to core library
  2. Update projection functions
  3. Refactor tutorials
```

**If per-metric ARI 0.85-0.90 (moderate improvement):**
```
⚠️  OFFER BOTH METHODS
Next steps:
  1. Add per-metric as optional method
  2. Document trade-offs
  3. Keep multivariate as default
```

**If per-metric ARI ~0.75 (no improvement):**
```
❌ KEEP MULTIVARIATE DTW
Next steps:
  1. Document per-metric for interpretability only
  2. Do NOT add to core library
  3. Explore other solutions
```

## Troubleshooting

### Error: "No test experiments found"
- Check if experiment IDs are correct in data file
- Verify data file path is correct
- Check if experiments have enough embryos (≥50 timepoints)

### Error: "Embryo IDs differ between metrics"
- This shouldn't happen - indicates data processing issue
- Check data file for consistency
- Review prepare_multivariate_array() function

### Script runs but produces no plots
- Check if output/04h/figures/ directory was created
- Review logs: `tail -100 output/04h/logs/run.log`
- Check for matplotlib/seaborn errors

## Output Files

After successful execution, you should have:

```
output/04h/
├── figures/
│   ├── disagreement_comparison.png          # Main result (bar plot)
│   └── confusion_matrices/
│       ├── confusion_20251017_combined.png  # Per-experiment matrices
│       ├── confusion_20251106.png
│       └── confusion_20251112.png
├── results/
│   └── disagreement_summary.csv             # All metrics in CSV format
└── logs/
    └── run.log                              # Full execution trace
```

## Expected Console Output

```
================================================================================
TUTORIAL 04h: Cross-Experiment Validation of Per-Metric DTW
================================================================================

...data loading...

================================================================================
SECTION 3: Reference-Only Clustering
================================================================================

Multivariate DTW - Reference only:
  Distance matrix shape: (54, 54)
  Distance range: [0.123, 45.678]

Cluster sizes (multivariate):
  Cluster 1: 12 embryos (22.2%)
  Cluster 2: 15 embryos (27.8%)
  ...

Per-Metric DTW - Reference only:
  Distance matrix shape: (54, 54)
  Distance range: [0.098, 34.567]

Agreement between methods on reference-only: ARI = 0.874
(This should be ~0.87-0.89 based on Tutorial 04g)

================================================================================
SECTION 4: Cross-Experiment Validation
================================================================================

Testing with: 20250512 + 20251017_combined
...

Multivariate DTW Disagreement:
  ARI: 0.753
  Interpretation: 24.7% disagreement

Per-Metric DTW Disagreement:
  ARI: 0.XXX
  Interpretation: XX.X% disagreement

Comparison:
  Improvement: +X.XXX ARI points
  → [INTERPRETATION]

...repeat for each test experiment...

================================================================================
SECTION 6: RECOMMENDATION
================================================================================

Based on cross-experiment validation results:

[AUTOMATED RECOMMENDATION HERE]

================================================================================
TUTORIAL 04h COMPLETE
================================================================================
```

## Quick Reference

**Run:** `python 04h_cross_experiment_validation.py`

**Results:** `cat output/04h/results/disagreement_summary.csv`

**Recommendation:** `grep -A 20 "RECOMMENDATION" output/04h/logs/run.log`

**Plots:** `ls -lh output/04h/figures/`

---

**Last Updated:** February 3, 2026
**Status:** Ready to run
