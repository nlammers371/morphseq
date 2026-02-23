# Tutorial Output Directory

This directory contains organized results from trajectory analysis tutorials investigating per-metric DTW vs multivariate DTW for cross-experiment robustness.

## Directory Structure

```
output/
├── 04g/                    # Per-Metric DTW Investigation (within-experiment)
│   ├── figures/            # 4 PNG visualizations
│   ├── results/            # Numerical results summary
│   ├── logs/               # Execution log
│   ├── TAKEAWAYS.md        # Comprehensive analysis document
│   └── IMPLEMENTATION_SUMMARY.md
│
└── 04h/                    # Cross-Experiment Validation (to be created)
    ├── figures/            # Comparison plots and confusion matrices
    ├── results/            # Disagreement metrics CSV
    └── logs/               # Execution log
```

## Quick Start

### View Tutorial 04g Results
```bash
# Read comprehensive analysis
cat output/04g/TAKEAWAYS.md

# View numerical results
cat output/04g/results/04g_results_summary.txt

# Check figures
ls -lh output/04g/figures/
```

### Run Tutorial 04h (Cross-Experiment Validation)
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260127_create_src_analyze_tutorial
python 04h_cross_experiment_validation.py

# View results
cat output/04h/results/disagreement_summary.csv
grep -A 20 "RECOMMENDATION" output/04h/logs/run.log
```

## Tutorial Summaries

### Tutorial 04g: Per-Metric DTW Investigation
**Question:** Does computing DTW separately for each metric produce more robust results than multivariate DTW?

**Dataset:** Experiment 20250512 (54 embryos, within-experiment only)

**Key Findings:**
- Per-metric DTW achieves **87-89% cluster agreement** with multivariate DTW
- Curvature dominates clustering (r=0.95) over body length (r=0.69)
- Normalization of distance matrices is critical
- Euclidean combination strategy performs best

**Conclusion:** Per-metric DTW is viable and interpretable, but **cross-experiment test needed** to determine if it solves the 25% disagreement issue.

### Tutorial 04h: Cross-Experiment Validation
**Question:** Does per-metric DTW reduce cross-experiment cluster assignment disagreement compared to multivariate DTW?

**Design:**
- Reference: Experiment 20250512 (54 embryos)
- Test: Multiple experiments with different time windows
- Compare: Reference-only vs reference+test combined clustering
- Metric: Adjusted Rand Index (ARI) - higher is better

**Hypothesis:**
- Multivariate DTW: ~25% disagreement (ARI ~0.75) - known from Tutorial 04d
- Per-metric DTW: <10% disagreement (ARI >0.90) - hypothesis

**Decision Criteria:**
- **If per-metric ARI >0.90**: Switch to per-metric as default (strong improvement)
- **If per-metric ARI 0.85-0.90**: Offer both methods (moderate improvement)
- **If per-metric ARI ~0.75**: Keep multivariate (no improvement)

**Status:** Script created, awaiting execution

## Key Visualizations

### Tutorial 04g Figures
1. **distance_comparison.png** - Multivariate vs per-metric distance matrices (r=0.96)
2. **cluster_comparison.png** - Confusion matrix showing 87-89% agreement
3. **metric_contributions.png** - Individual metric correlations (curvature dominates)
4. **strategy_comparison.png** - Comparison of combination methods (euclidean best)

### Tutorial 04h Figures (To Be Generated)
1. **disagreement_comparison.png** - Bar plot comparing methods across experiments
2. **confusion_matrices/** - Per-experiment cluster reassignment patterns

## Background: The 25% Disagreement Problem

**Issue:** Cross-experiment projection using multivariate DTW shows ~25% disagreement in cluster assignments.

**Root Cause:**
- Global Z-score normalization depends on time window coverage
- Different experiments → different time coverage → different means/stds
- Same raw value gets normalized to different Z-scores
- Cluster assignments change unpredictably

**Documented In:** `src/analyze/trajectory_analysis/KNOWN_ISSUES.md`

**Proposed Solution:** Per-metric DTW may avoid this by:
1. Using raw values per metric (no cross-metric normalization)
2. Normalizing distance matrices independently per metric
3. Being more robust to experiment-specific value ranges

**Test:** Tutorial 04h validates whether this solution works

## References

### Related Documentation
- **KNOWN_ISSUES.md** - Documents cross-experiment disagreement problem
- **Tutorial 04d** - Discovered 25% disagreement issue
- **Tutorial 04f** - Single-metric clustering baseline
- **Tutorial 04g** - Per-metric vs multivariate (within-experiment)
- **Tutorial 04h** - Cross-experiment validation (critical test)

### Core Functions
- `compute_trajectory_distances()` - Multivariate DTW (current)
- `compute_per_metric_dtw()` - Per-metric DTW (implemented in tutorials)
- `compute_md_dtw_distance_matrix()` - Core DTW algorithm
- `prepare_multivariate_array()` - Data prep and normalization

---

**Last Updated:** February 3, 2026
