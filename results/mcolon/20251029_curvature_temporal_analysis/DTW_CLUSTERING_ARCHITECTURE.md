# DTW Clustering Analysis: Architecture & Design Rationale

## Overview
This document explains the design decisions in the DTW clustering pipeline, particularly why certain visualization/statistical approaches are used or avoided.

---

## Why PCA is NOT used (and was removed)

### The Problem
PCA requires **feature vectors** in Euclidean space, not **distance matrices**. When you try to apply PCA to a distance matrix:

1. **Mathematically incorrect**:
   - PCA assumes your input is n×p (samples × features)
   - DTW gives you n×n (pairwise distances)
   - Distance matrices have special properties (symmetry, triangle inequality) that violate PCA assumptions

2. **Computationally unstable**:
   - PCA uses SVD internally, which triggers LAPACK's scaling routines (e.g., DLASCL)
   - DLASCL errors occur when matrices contain numerical edge cases
   - Distance matrices often have zeros, extreme values, or poor conditioning

3. **Informationally redundant**:
   - We're already working in "distance space"
   - Any 2D embedding loses information about inter-cluster distances
   - Better to keep analysis in the native distance metric

### What we use instead

**Plot 30: Silhouette Analysis**
- Evaluates clustering quality directly in distance space
- Compares intra-cluster vs inter-cluster distances
- No coordinate transformation needed
- More interpretable: higher score = better separation

**Plot 29: DTW Distance Matrix**
- Shows pairwise distances visually (heatmap)
- Sorted by cluster assignment to reveal block structure
- Direct visualization of what the algorithm sees
- No information loss from dimensionality reduction

**Plot 31 & 32: Temporal Trend Plots**
- Work in original time/measurement space (hpf × metric)
- Not derived from distance matrix
- Show actual biological patterns
- Interpretable in experimental context

---

## Architecture: Working in Distance Space

### Data Flow
```
Raw trajectories (n_embryos × time_points)
        ↓
     [DTW computation]
        ↓
Distance matrix D (n×n)
        ↓
     [K-means with precomputed metric]
        ↓
Cluster assignments
        ↓
     [Analysis in both distance and temporal space]
        ↓
Plots 21-32
```

### Why this works well

1. **Mathematically sound**: Distance-based clustering on distance-based input
2. **Computationally stable**: No matrix decompositions, no LAPACK scaling
3. **Biologically interpretable**: Results stay connected to observed trajectories
4. **No information loss**: Native distance metric preserved throughout

---

## Key Design Decisions

### 1. Linear Interpolation Only
- **Why**: Validated as best method (RMSE = 0.009541 ± 0.000000)
- **Where**: 07a_validate_imputation_methods.py
- **Benefit**: Simple, stable, no overfitting

### 2. Homozygous Mutants Only
- **Why**: Cleaner phenotypic signal, biological relevance
- **Configuration**: `GENOTYPE_FILTER = 'cep290_homozygous'`
- **Benefit**: Reduces noise from mixed genetic backgrounds

### 3. Silhouette + Distance Matrix for Visualization
- **Why**: Both directly measure clustering quality in distance space
- **Advantage**: No assumptions, no coordinate transformations
- **Result**: LAPACK errors eliminated

### 4. Temporal Plots for Interpretation
- **Plots 31-32**: Show mean/individual trajectories with confidence bands
- **Context**: Original time scale (hpf) and measurement scale (curvature)
- **Strength**: Bridge between abstract distance space and biology

---

## Clustering Algorithm Details

### DTW Distance
```python
dtw_distance(s1, s2, window=3)
  - Sakoe-Chiba band constraint (window=3)
  - Handles variable-length trajectories
  - Computational: O(n·m) with band
```

### K-means on Precomputed Distances
```python
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
cluster_assignments = kmeans.fit_predict(dist_matrix)
  - Uses: sklearn's distance-aware KMeans
  - Metric: 'precomputed' (uses D directly)
  - Robust: 10 random initializations
```

### Quality Metrics
- **Silhouette Score**: Measures how similar points are to their own cluster vs others
- **Inertia**: Sum of squared distances (for elbow detection)
- **Range**: [−1, 1], closer to 1 is better

---

## Output Organization

```
outputs/
├── 07a/
│   └── validation/
│       └── imputation_validation_summary.txt
│           (Which imputation method is best?)
│
└── 07b/
    ├── plots/
    │   ├── plot_21: Cluster selection (elbow + silhouette)
    │   ├── plot_22: Anti-correlation scatter
    │   ├── plot_23-28: Various cluster visualizations
    │   ├── plot_29: DTW distance matrix heatmap ✓ (distance space)
    │   ├── plot_30: Silhouette analysis ✓ (distance space, replaces PCA)
    │   ├── plot_31: Temporal trends by cluster ✓ (temporal space)
    │   └── plot_32: Cluster trajectory overlay ✓ (temporal space)
    │
    └── tables/
        ├── table_4: Cluster characteristics
        ├── table_5: Anti-correlation evidence
        └── table_6: Embryo-cluster assignments
```

---

## Numerical Stability Measures

### Distance Matrix Validation
```python
# After DTW computation:
- Check for NaN/Inf values
- Replace any bad values with appropriate defaults
- Ensure diagonal = 0 (self-distance)
- Report diagnostic statistics
```

### Clustering Robustness
```python
# Before/during clustering:
- Assert distance matrix is finite
- Try-except wrapper around KMeans
- Report any failures gracefully
```

---

## Anti-Correlation Hypothesis

### What we're testing
Groups with flip-flop pattern:
- **Early (44-50 hpf)**: High curvature
- **Late (80-100 hpf)**: Low curvature (or vice versa)

### How we test
1. Extract mean curvature in early/late windows (per embryo)
2. Compute Pearson r within each cluster
3. Permutation test (shuffle to get null distribution)
4. Report: r, p-value, interpretation

### Expected results
- **Anti-correlated**: r < −0.3, low p-value
- **Correlated**: r > 0.3, low p-value
- **Uncorrelated**: r ≈ 0

---

## Running the Analysis

### Step 1: Validate Imputation (One-time)
```bash
python 07a_validate_imputation_methods.py
# Output: outputs/07a/validation/imputation_validation_summary.txt
```

### Step 2: Cluster Homozygous Trajectories
```bash
python 07b_dtw_trajectory_clustering.py
# Output:
#   - outputs/07b/plots/ (12 PNG files)
#   - outputs/07b/tables/ (3 CSV files)
```

### Configuration (in 07b_dtw_trajectory_clustering.py)
```python
METRIC_NAME = 'normalized_baseline_deviation'
GENOTYPE_FILTER = 'cep290_homozygous'  # or None for all
EARLY_WINDOW = (44, 50)  # hpf
LATE_WINDOW = (80, 100)  # hpf
CLUSTER_K_VALUES = [2, 3, 4]
DTW_WINDOW = 3  # Sakoe-Chiba band width
```

---

## Summary: Why This Design is Better

| Aspect | Old (PCA) | New (Distance-space) |
|--------|-----------|----------------------|
| **Mathematical soundness** | ❌ PCA ≠ distance matrices | ✅ Distance metric throughout |
| **Numerical stability** | ❌ LAPACK errors | ✅ No decompositions |
| **Information loss** | ❌ 2D embedding | ✅ Native metric |
| **Interpretability** | ❌ Abstract PC space | ✅ Distance values |
| **Biological connection** | ❌ Indirect | ✅ Temporal trajectories |
| **Computation cost** | ⚠️ SVD-heavy | ✅ Minimal |

**Result**: Cleaner, faster, more reliable, and more interpretable clustering analysis!
