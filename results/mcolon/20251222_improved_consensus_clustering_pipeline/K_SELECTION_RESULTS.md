# K Selection Results & Design Choices

**Date:** December 22-23, 2025  
**Analysis:** MD-DTW Consensus Clustering Pipeline

---

## Executive Summary

This document describes the design choices and results from implementing a two-phase clustering pipeline that **selects optimal k before filtering**, using IQR distance-based outlier removal and membership quality assessment.

**Key Finding:** IQR distance filtering (2.0×) removes ~8% of global outliers while preserving cluster structure, enabling robust k-selection across k=[2,3,4,5,6] with 100 bootstrap iterations per k.

---

## Design Philosophy

### Problem Statement

**Original Issue:** k-NN filtering (Stage 1) kept "shitty embryos" that formed stable local clusters, leading to:
- 0% removal rate in Stage 1 (all embryos passed)
- Problematic embryos persisting through pipeline
- Optimal k unknown before filtering decisions

**Root Cause:** k-NN uses **local** neighborhood distances → embryos in tight (but biologically meaningless) clusters have low k-NN distances → they pass filtering even if globally aberrant.

### Solution: IQR Distance Filtering

**Switch from Local to Global Outlier Detection:**

| Method | Distance Metric | Detection Scope | Removal Rate | Use Case |
|--------|----------------|-----------------|--------------|----------|
| **k-NN** | k-nearest neighbors | Local clusters | 0% (k=3,5,10) | Preserves stable clusters |
| **IQR** | Median distance to all | Global outliers | ~8% (2.0×) | Removes aberrant embryos |

**IQR Method:**
```
median_dist = median(D[i, :])  # Distance from embryo i to all others
Q1, Q3 = percentiles(median_distances, [25, 75])
IQR = Q3 - Q1
threshold = Q3 + multiplier × IQR
outliers = embryos where median_dist > threshold
```

**Multiplier Sensitivity:**
- 1.5× → 12.7% removed (aggressive)
- 2.0× → 8.9% removed (balanced) ← **chosen**
- 4.0× → 3.8% removed (lenient)

**Rationale:** 2.0× provides moderate filtering that removes global outliers without over-filtering, leaving sufficient data for robust k-selection.

---

## Two-Phase Pipeline Design

### Phase 1: Light Filtering + K Selection

```
Input: Full distance matrix D (N × N)
  ↓
Apply IQR 2.0× filtering → Remove ~8% global outliers
  ↓
Evaluate k = [2, 3, 4, 5, 6] with 100 bootstrap iterations each
  ↓
Compute metrics per k:
  - Silhouette score (cluster separation)
  - Mean max_p (assignment confidence)
  - Mean entropy (assignment ambiguity)
  - Membership % (core/uncertain/outlier)
  ↓
Select best k (highest % core members)
```

### Phase 2: Consensus Clustering at Optimal K

```
Use selected k from Phase 1
  ↓
Run 100 bootstrap iterations
  ↓
Build co-association matrix (evidence accumulation)
  ↓
Hierarchical clustering on consensus
  ↓
Apply Stage 2 filtering (remove uncertain/outliers if needed)
  ↓
Final cluster assignments
```

**Key Advantage:** Knowing optimal k **before** aggressive filtering allows:
1. Data-driven k selection on representative sample
2. Targeted removal of low-quality assignments
3. No circular dependency (filter → pick k → filter again)

---

## Membership Quality Classification

Bootstrap consensus clustering provides **posterior probabilities** for each embryo's cluster assignment. We classify membership quality using two metrics:

### Metrics

1. **max_p:** Maximum posterior probability across all clusters
   - High max_p → confident assignment
   - Low max_p → uncertain assignment

2. **log_odds_gap:** Log-odds difference between top 2 clusters
   ```
   log_odds_gap = log(p_1st / (1 - p_1st)) - log(p_2nd / (1 - p_2nd))
   ```
   - Large gap → clear winner
   - Small gap → ambiguous between clusters

### Classification Decision Boundaries

```python
if max_p >= 0.7:
    if log_odds_gap >= 1.0:
        category = 'core'        # High confidence, clear assignment
    else:
        category = 'uncertain'   # High confidence, but close 2nd choice
elif max_p >= 0.5:
    category = 'uncertain'       # Moderate confidence
else:
    category = 'outlier'         # Low confidence, doesn't fit well
```

**Biological Interpretation:**
- **Core:** Embryos with stereotyped trajectories, strong cluster membership
- **Uncertain:** Boundary embryos, transitional phenotypes, or noisy trajectories
- **Outlier:** Aberrant embryos that don't fit any cluster well

---

## K Selection Metrics

### 1. Membership Distribution (Primary)

**Goal:** Maximize core members, minimize outliers

Best k has:
- Highest % core members (confident assignments)
- Low % outliers (most embryos fit some cluster)
- Balanced uncertain % (some ambiguity is expected)

### 2. Mean Max Posterior

**Goal:** High average confidence

- Target: mean(max_p) > 0.7
- Low values → many uncertain assignments
- Monotonically increases with k (more clusters → easier to fit)

### 3. Mean Entropy

**Goal:** Low assignment ambiguity

```
entropy = -Σ p_k × log2(p_k)
```

- 0 bits → deterministic assignment (one cluster = 100%)
- High → probability spread across many clusters
- Generally decreases with k (more clusters → less ambiguity per cluster)

### 4. Silhouette Score

**Goal:** Well-separated clusters

- Range: [-1, 1]
- Positive → embryos closer to own cluster than others
- Negative → poor cluster separation
- Can decrease with k (more clusters → denser space)

**Trade-off:** Silhouette often peaks at low k (fewer, broader clusters), while membership quality improves with moderate k (more specialized clusters).

---

## Results Summary

### Test Configuration

- **Data:** 291 embryos across 4 experiments (20251119, 20251121, 20251104, 20251125)
- **Distance Matrix:** MD-DTW (multivariate dynamic time warping)
- **Stage 1 Filtering:** IQR 2.0× → 265 embryos retained (~8.9% removed)
- **K Range Tested:** [2, 3, 4, 5, 6]
- **Bootstrap Iterations:** 100 per k
- **Metrics:** 2 trajectories (curvature, body length)

### K Selection Output

Generated files in `k_selection_results/`:
- `k2_membership_trajectories.png` → Membership quality plots for k=2
- `k3_membership_trajectories.png` → k=3
- `k4_membership_trajectories.png` → k=4
- `k5_membership_trajectories.png` → k=5
- `k6_membership_trajectories.png` → k=6
- `k_selection_comparison.png` → Summary comparison across all k
- `k_selection_summary.csv` → Quantitative metrics table

### Interpretation Guidelines

**Look for:**
1. **Highest % core** → optimal granularity (k not too small/large)
2. **Biological interpretability** → clusters match known phenotypes
3. **Trajectory separation** → core members have distinct trends
4. **Stable membership** → uncertain/outlier embryos consistent across plots

**Red flags:**
- High % outliers → k too large (overfitting)
- Low % core → k too small (underfitting) or poor cluster structure
- All metrics flat → data may not have strong cluster structure

---

## Implementation Notes

### Code Organization

**Core modules:**
- `distance_filtering.py` → IQR and k-NN filtering methods
- `k_selection.py` → K evaluation pipeline, helper functions
- `consensus_pipeline.py` → Two-stage consensus clustering
- `cluster_classification.py` → Membership quality classification
- `facetted_plotting.py` → Multimetric trajectory visualization

**Key functions:**
- `identify_outliers(D, method='iqr', threshold=2.0)` → Stage 1 filtering
- `evaluate_k_range(D, embryo_ids, k_range)` → Test multiple k values
- `classify_membership_2d(max_p, log_odds_gap)` → Core/uncertain/outlier
- `add_membership_column(df, classification)` → Add category for plotting
- `plot_multimetric_trajectories(df, col_by='cluster', color_by_grouping='membership')` → Faceted plots

### Pipeline Defaults

**Updated in `consensus_pipeline.py`:**
```python
def run_consensus_clustering(
    ...
    stage1_method='iqr',        # Changed from 'knn'
    stage1_threshold=2.0,       # IQR multiplier
    ...
):
```

**Backward compatibility:** Both methods supported, user can override via `stage1_method` parameter.

---

## Future Enhancements

### Plotting Improvements

1. **Manual color mapping for membership:**
   - Currently uses automatic colors from `color_by_grouping`
   - Add explicit color dict: `{'core': 'green', 'uncertain': 'orange', 'outlier': 'red'}`

2. **Legend positioning:**
   - Matplotlib legends occasionally overlap with data
   - Consider moving legends outside plot area or adjusting placement

### Workflow Refinements

1. **Automated k recommendation:**
   - Current: manual inspection of plots + metrics
   - Future: scoring function combining all metrics → auto-select best k

2. **Stage 2 filtering options:**
   - Current: uncertain/outliers kept in final results
   - Add parameter to auto-remove based on membership category

3. **Cross-validation:**
   - Test stability of k-selection across different bootstrap samples
   - Report confidence intervals on membership %

---

## Conclusions

**Design Choices Summary:**

1. ✅ **IQR distance filtering** → Removes global outliers, preserves cluster structure
2. ✅ **Two-phase pipeline** → Select k before aggressive filtering
3. ✅ **Membership quality** → Core/uncertain/outlier classification enables targeted filtering
4. ✅ **Bootstrap consensus** → Robust to sampling variability, quantifies assignment confidence
5. ✅ **Multi-metric evaluation** → Silhouette, max_p, entropy, membership % provide complementary views

**Impact:** Pipeline now removes problematic embryos early (IQR filtering), evaluates k on clean data, and provides interpretable quality metrics for each cluster assignment.

**Next Steps:**
1. Review generated plots in `k_selection_results/`
2. Select optimal k based on membership distribution + biological interpretation
3. Run Phase 2 consensus clustering at selected k
4. Apply Stage 2 filtering if needed (remove uncertain/outliers)
5. Proceed with downstream trajectory analysis

---

**Contact:** mdcolon  
**Last Updated:** December 23, 2025
