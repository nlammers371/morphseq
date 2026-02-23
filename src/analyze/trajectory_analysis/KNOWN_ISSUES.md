# Known Issues and Limitations

## 1. Normalization Dependency in Cross-Experiment Projection

**Status**: Fundamental limitation (not easily solvable)
**Affects**: `projection.py` - all projection functions
**Discovered**: 2026-02-03 (Tutorial 04d testing)

### Problem Description

When projecting experiments with different temporal coverage onto a reference dataset, **global Z-score normalization produces inconsistent results**. The same embryo-reference pair can produce different DTW distances depending on which time window is used for analysis.

### Root Cause

The normalization pipeline works as follows:

```python
# Current implementation in prepare_multivariate_array()
means = np.nanmean(X, axis=(0, 1))  # Mean across ALL embryos and timepoints
stds = np.nanstd(X, axis=(0, 1))    # Std across ALL embryos and timepoints
X_normalized = (X - means) / stds
```

**The issue**: Different time windows contain different data, producing different mean/std statistics:

- **Experiment A** (12-47 hpf, 71 timepoints):
  - Normalization uses data from 113 embryos × 71 timepoints
  - mean_A, std_A

- **Experiment B** (27-77 hpf, 101 timepoints):
  - Normalization uses data from 98 embryos × 101 timepoints
  - mean_B, std_B

- **Combined** (12-77 hpf, 131 timepoints):
  - Normalization uses data from 211 embryos × 131 timepoints (with NaNs)
  - mean_combined, std_combined

**Result**: `mean_A ≠ mean_B ≠ mean_combined`, so the **same raw value gets normalized to different values**.

### Impact

**Quantified in Tutorial 04d**:

- **Distance correlation for same embryo-reference pairs**:
  - Experiment 20260122: r = 0.65 (should be 1.0!)
  - Experiment 20260124: r = 0.94 (should be 1.0!)
  - Maximum difference: 297 (should be ~0)

- **Cluster assignment disagreement**:
  - 25% of embryos assigned to different categories when using separate vs combined projection
  - Not random - systematic due to normalization differences

### Why This Matters

1. **Reproducibility**: Same embryo can get different cluster assignments depending on which other experiments are included in the analysis

2. **Batch effects**: Experiments with different temporal coverage will have systematic differences in cluster assignments

3. **Comparisons**: Cannot directly compare cluster proportions across experiments with different time windows

### Why It's Hard to Solve

The fundamental problem: **You need a normalization reference, but often don't have one**.

**Potential solutions and their issues**:

1. ❌ **Reference-based normalization**
   - Requires a stable, comprehensive reference dataset
   - Not available in many use cases (e.g., first experiment with a new mutant)
   - Reference must cover the full time range of all experiments

2. ❌ **Per-embryo normalization**
   - Loses biological information about between-embryo variation
   - Can't distinguish "high curvature" from "low curvature" phenotypes

3. ❌ **Fixed normalization parameters**
   - Requires pre-computing from a large representative dataset
   - Brittle if experimental conditions change (different imaging setup, etc.)
   - Not generalizable across labs/conditions

4. ❌ **No normalization**
   - Only works if all metrics are on similar scales
   - Breaks with multi-metric DTW

### Workarounds

Since there's no perfect solution, here are **pragmatic strategies**:

#### For Projection (Recommended)

1. **Project experiments separately** when temporal coverage differs significantly (<50% overlap)
   - Accept that ~25% near boundaries will differ
   - Focus on high-confidence assignments

2. **Batch experiments with similar time windows together**
   - Process all 12-47 hpf experiments together
   - Process all 27-77 hpf experiments together

3. **Use K-NN with posteriors** to identify uncertain cases
   - Low max_posterior → embryo near cluster boundary
   - Flag these for manual inspection

4. **Document time windows clearly**
   - Report which time window was used for projection
   - Compare proportions only within same time window

#### For Clustering (Different from Projection)

When clustering a **single experiment** (not projection):
- This issue is less severe because all embryos use the same normalization
- Still affects comparisons across experiments, but within-experiment clustering is consistent

### Testing

**Tutorial 04d** (`04d_direct_distance_comparison.py`) provides a test that quantifies this issue:
- Compares distances for the same embryo-reference pairs in different time windows
- Correlation should be 1.0 if normalization were consistent
- Actually: r ≈ 0.65-0.94

**To verify this issue persists**:
```bash
cd results/mcolon/20260127_create_src_analyze_tutorial
python 04d_direct_distance_comparison.py
```

Expected output:
- Correlation < 0.95 for at least one experiment
- Max absolute difference > 100
- Conclusion: "TEST FAILED: Significant differences detected"

### Related Files

- `projection.py` - Contains warning in module docstring
- `utilities/dtw_utils.py` - `prepare_multivariate_array()` performs normalization
- Tutorial 04: Demonstrates 25% disagreement in cluster assignments
- Tutorial 04d: Direct test of distance consistency

### Future Work

Potential improvements (not currently planned):

1. **Add normalization options** to `prepare_multivariate_array()`:
   - `normalize='global'` (current default)
   - `normalize='per_embryo'`
   - `normalize='reference'` (requires ref_mean, ref_std parameters)
   - `normalize=False`

2. **Document best practices** for specific use cases
   - When to use which normalization
   - How to batch experiments appropriately

3. **Posterior-based confidence scores**
   - Implement K-NN projection by default
   - Automatically flag low-confidence assignments

### Important Exception: Single-Metric Clustering

**Status**: Not an issue for within-experiment clustering
**Tested**: Tutorial 04f
**Finding**: For single-metric analysis within a single experiment, normalization does NOT matter.

#### Evidence

Tutorial 04f tested clustering experiment 20260122 (113 embryos, 4 clusters):
- **Distance correlation**: 1.0 (perfect)
- **Adjusted Rand Index**: 1.0 (identical clusters)
- **Cluster distributions**: Exactly the same [2, 47, 63, 1]

Raw vs normalized features produce **identical clustering results**.

#### Why This Works

Within a single experiment, normalization is just a **linear rescaling**:
- All embryos use the same mean and std
- Relative distances are preserved
- Nearest neighbors don't change
- Clustering is identical

#### Recommendation for Single-Metric Clustering

**You can safely use `normalize=False` for single-metric clustering** within one experiment:

```python
from src.analyze.trajectory_analysis import compute_trajectory_distances

# For single metric - normalization doesn't matter
D, embryo_ids, time_grid = compute_trajectory_distances(
    df,
    metrics=['baseline_deviation_normalized'],
    normalize=False,  # ← Can use False for single metric!
)
```

**Benefits:**
- Simpler (one less parameter to worry about)
- More interpretable (distances in actual metric units)
- Avoids the cross-experiment normalization issue entirely

**When you still need normalization:**
- Multi-metric DTW (different metrics have different scales)
- Cross-experiment projection (though this has the issues described above)

#### Testing

Tutorial 04f demonstrates this:
```bash
cd results/mcolon/20260127_create_src_analyze_tutorial
python 04f_clustering_normalization_test.py
```

Expected: ARI = 1.0, identical cluster assignments

### When to Be Concerned

This is a **problem** when:
- Projecting experiments with very different temporal coverage (<50% overlap)
- Needing exact reproducibility of cluster assignments
- Comparing cluster proportions across experiments with different time windows
- Using multi-metric analysis across experiments

This is **less of a concern** when:
- Single-metric clustering within one experiment (normalization doesn't matter!)
- Experiments have similar temporal coverage (>75% overlap)
- Using projection for exploratory analysis, not final decisions
- Focusing on well-separated clusters (far from boundaries)
- Reporting proportions within experiments, not across

---

## 2. [Placeholder for Future Issues]

*Document additional known issues here as they are discovered.*

---

**Last Updated**: 2026-02-03
**Contributors**: Discovered during Tutorial 04/04d development
