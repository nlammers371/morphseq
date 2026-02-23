# Bootstrap Assignment Posteriors: Robust Clustering Quality Assessment

**Implementation Date:** 2025-11-06  
**Status:** Ready for validation

## Quick Start

```bash
# Run comparison pipeline
python compare_methods_v2.py --k_min 2 --k_max 5

# Generate plots
python plot_quality_comparison.py
python plot_posterior_heatmaps.py --k 3
```

## Overview

Implements bootstrap assignment posteriors for assessing cluster membership quality. Addresses failures of co-association-based methods by computing per-embryo cluster assignment probabilities with proper label alignment.

**Key Innovation:** Computes p_i(c) directly from aligned bootstrap iterations instead of using pairwise co-association as a proxy.

## Core Files

- `bootstrap_posteriors.py` - Label alignment + posterior computation
- `adaptive_classification.py` - 2D gating classifier
- `compare_methods_v2.py` - Main comparison script
- `plot_quality_comparison.py` - Visualization
- `plot_posterior_heatmaps.py` - Heatmap plots
- `plot_cluster_trajectories.py` - Trajectory plots

## Method

1. **Label Alignment:** Hungarian algorithm aligns cluster IDs across bootstrap iterations
2. **Posterior Calculation:** p_i(c) = (# times embryo i assigned to c) / (# times i sampled)
3. **Quality Metrics:** max_p (confidence), entropy (uncertainty), log_odds_gap (disambiguation)
4. **Classification:** 2D gating using max_p ≥ 0.8 AND log_odds_gap ≥ 0.7 for core membership

## Expected Runtime

<2 minutes total (loads pre-computed bootstrap results, no re-clustering)

## Dependencies

numpy, scipy, matplotlib, seaborn, pandas

See `cluster_assignment_quality.md` for detailed documentation.
