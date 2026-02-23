# DTW Method Investigation

Investigation into whether per-metric DTW improves cross-experiment clustering robustness compared to multivariate DTW.

## Key Findings

### Winner: Multivariate DTW

- **Cross-experiment disagreement**: 11.6% (multivariate) vs 20.4% (per-metric)
- **Consistent performance** across experiment pairs (20260122-20260124, 20260122-REF, 20260124-REF)
- **Already in core library**, well-tested and battle-proven
- **Better suited for production** pipelines

### Value of Per-Metric DTW

- **Interpretability**: Reveals which metrics drive clustering decisions
- **Key insight**: Curvature (r=0.95) dominates over length (r=0.69) in determining cluster assignments
- **Use case**: Understanding results and debugging, not production pipelines
- **Diagnostic tool**: Helps identify which features are most informative for trajectory similarity

## Conclusion

**Keep multivariate DTW as default**. Use per-metric DTW for interpretability only when you need to understand which features are driving your clustering results.

## Investigation Scripts

These scripts systematically tested different DTW approaches:

- **04b_compare_clustering_methods.py** - Comparison of multivariate vs per-metric DTW
- **04c_distance_matrix_test.py** - Distance matrix validation
- **04d_direct_distance_comparison.py** - Direct distance comparison between methods
- **04e_normalization_alternatives.py** - Testing normalization strategies
- **04f_clustering_normalization_test.py** - Impact of normalization on clustering
- **04g_per_metric_dtw_combination.py** - Per-metric DTW implementation and testing
- **04h_cross_experiment_validation.py** - Cross-experiment validation (key test)

## Documentation Files

- **README_04g_04h.md** - Quick summary of scripts 04g and 04h
- **FINAL_CONCLUSIONS.md** - Complete analysis and recommendations
- **IMPLEMENTATION_STATUS.md** - Implementation details and status
- **RUN_04h.md** - Instructions for running the cross-experiment validation

## Experimental Design

The investigation compared clustering stability across three experiment pairs:
1. **20260122 vs 20260124**: Two new CEP290 experiments
2. **20260122 vs REF**: New experiment vs established reference
3. **20260124 vs REF**: New experiment vs established reference

**Reference clusters** were derived from 7 CEP290 experiments (20251229 analysis).

## Key Metrics

**Disagreement rates** (percentage of embryos with inconsistent cluster assignments):

| Method | 20260122-20260124 | 20260122-REF | 20260124-REF | **Mean** |
|--------|-------------------|--------------|--------------|----------|
| Multivariate | 12.0% | 10.5% | 12.4% | **11.6%** |
| Per-metric | 20.0% | 19.5% | 21.7% | **20.4%** |

**Winner**: Multivariate DTW with ~9% lower disagreement rate.

## Implementation Notes

Per-metric DTW combines distances across features:
```
d_combined = w_curv * d_curvature + w_len * d_length
```

Where weights were tested as:
- Equal weighting: w_curv = w_len = 0.5
- Variance-based: Proportional to feature variance
- Learned: Optimized for cluster separation

**Result**: None of the per-metric variants outperformed multivariate DTW.

## When to Use Each Method

**Multivariate DTW** (default):
- Production pipelines
- Cross-experiment projection
- When you need robust, consistent results
- When computational efficiency matters

**Per-metric DTW** (diagnostic):
- Understanding which features drive clustering
- Debugging unexpected cluster assignments
- Identifying which measurements are most informative
- Educational demonstrations of trajectory analysis

## Related Work

This investigation was motivated by concerns about batch effects in cross-experiment projection (see `README_tutorial_04.md` in parent directory). The goal was to determine if per-metric DTW could reduce sensitivity to temporal coverage differences.

**Finding**: Per-metric DTW did not improve cross-experiment robustness. The batch effects observed in Tutorial 04 are fundamental to temporal coverage limitations, not an artifact of the DTW method.
