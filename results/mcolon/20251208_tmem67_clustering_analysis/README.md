# tmem67 Genotyping Analysis via Consensus Clustering

## Overview

This analysis identifies tmem67 mutant embryos when genotype labels are unreliable using bootstrap consensus clustering with posterior probability analysis.

**Key Approach:**
- Cluster ALL embryos together (ignore unreliable genotype labels)
- Test k=3,4,5,6 clusters using DTW-based trajectory similarity
- Use posterior probabilities to robustly identify optimal k
- Identify mutant clusters by average normalized baseline deviation > 0.05

**Author:** Generated via Claude Code
**Date:** 2025-12-08
**Experiment:** 20250711

## Quick Start

```bash
cd results/mcolon/20251208_tmem67_clustering_analysis
python run_tmem67_clustering.py
```

**Expected Runtime:** 20-30 minutes

## Analysis Pipeline

### Stage 1: Data Loading
- Loads curvature metrics and metadata
- Merges on `snip_id`
- Filters by `min_timepoints >= 3`
- **NO genotype filtering** (key difference from typical analysis)

### Stage 2: Trajectory Extraction & Alignment
- Extracts normalized baseline deviation trajectories
- Interpolates to common time grid (0.5 hpf steps)
- Converts to arrays for DTW computation

### Stage 3: DTW Distance Computation
- Computes pairwise DTW distances (Sakoe-Chiba window=3)
- Caches distance matrix for reuse

### Stage 4: Bootstrap Clustering (k=3,4,5,6)
For each k value:
- 100 bootstrap iterations, 80% sampling per iteration
- Hierarchical clustering with average linkage
- Generates consensus cluster assignments

### Stage 5: Posterior Probability Analysis
- Computes `p_matrix`: (n_embryos × k) posterior probabilities
- Calculates quality metrics:
  - `max_p`: Confidence per embryo
  - `entropy`: Uncertainty per embryo
  - `log_odds_gap`: Top-2 cluster separation

### Stage 6: Membership Quality Classification
- Classifies embryos as core/uncertain/outlier
- Core: max_p ≥ 0.8 AND log_odds_gap ≥ 0.7
- Outlier: max_p < 0.5

### Stage 7: Mutant Cluster Identification
For each cluster:
1. Compute cluster mean trajectory (binned mean across all embryos)
2. Calculate cluster average = mean(cluster_mean_trajectory)
3. **Flag as mutant if cluster_average > 0.05**

### Stage 8: Visualization
- Individual trajectories by cluster (PNG)
- **Cluster mean trajectories** (PNG + interactive Plotly HTML)
- Interactive plots focus on cluster-level averages with hover showing:
  - Cluster ID
  - Number of embryos
  - Cluster average
  - Mutant/WT status
  - Membership quality breakdown

### Cross-k Comparison
- Posterior metrics vs k (4-panel plot)
- Composite scoring for optimal k recommendation:
  - Weighted combination: max_p (30%), entropy (25%), core_fraction (25%), silhouette (20%)

## Output Structure

```
output/20250711/
├── data/
│   ├── dtw_distance_matrix.npy           # Cached DTW distances
│   ├── df_interpolated.pkl               # Aligned trajectories
│   ├── bootstrap_results_k3.pkl          # Bootstrap results per k
│   └── posteriors_k3.pkl                 # Posterior analysis per k
├── figures/
│   ├── k3/, k4/, k5/, k6/               # Per-k visualizations
│   │   ├── trajectories_by_cluster.png   # Individual + mean
│   │   ├── cluster_means.png             # Cluster means comparison
│   │   └── interactive_cluster_means.html # MAIN interactive plot
│   └── comparison/
│       ├── posterior_metrics_vs_k.png    # Quality metrics across k
│       └── optimal_k_recommendation.png  # Composite scores
└── tables/
    ├── cluster_characteristics_k3.csv    # Per-cluster stats per k
    ├── mutant_embryos_k3.csv             # Identified mutants per k
    └── optimal_k_analysis.csv            # Cross-k comparison
```

## Key Outputs

### Cluster Characteristics Table
Per k value, contains:
- `cluster_id`: Cluster number (0 to k-1)
- `n_embryos`: Number of embryos in cluster
- `cluster_average`: Mean of cluster mean trajectory (across all timepoints)
- `is_putative_mutant`: Boolean flag (cluster_average > 0.05)
- `n_core`, `n_uncertain`, `n_outlier`: Membership quality breakdown
- `core_fraction`: Fraction of core members
- `embryo_ids`: Semicolon-separated list of embryo IDs

### Mutant Embryos List
Per k value, contains:
- `embryo_id`: Individual embryo identifiers in putative mutant clusters

### Optimal K Analysis Table
Comparison across k values:
- `k`: Number of clusters
- `avg_max_p`: Average confidence
- `avg_entropy`: Average uncertainty
- `core_fraction`: Fraction of core members
- `silhouette`: Cluster separation score
- `n_mutant_embryos`: Total mutants identified

### Interactive Plots
**Focus:** Cluster mean trajectories (not individual embryos)

**Hover shows:**
- Cluster ID and mutant/WT status
- Time (hpf) and mean value
- Cluster average (across all time)
- Number of embryos
- Membership quality (core/uncertain/outlier counts)

## Configuration

Edit `config.py` to customize:
- `K_VALUES`: Cluster range to test (default: [3,4,5,6])
- `N_BOOTSTRAP`: Bootstrap iterations (default: 100)
- `MUTANT_THRESHOLD`: Cluster average threshold (default: 0.05)
- `DTW_WINDOW`: Sakoe-Chiba band width (default: 3)
- `THRESHOLD_MAX_P`: Core membership confidence (default: 0.8)
- `GRID_STEP`: Time interpolation step (default: 0.5 hpf)

## Interpretation Guide

### How to Identify Mutants
1. **Look at recommended k** from optimal_k_analysis
2. **Examine cluster means** in `figures/k{recommended}/cluster_means.png`
3. **Clusters ABOVE red threshold line (0.05)** are putative mutants
4. **Check cluster_average** in `cluster_characteristics_k{recommended}.csv`
5. **Extract embryo IDs** from `mutant_embryos_k{recommended}.csv`

### Quality Checks
- **Cluster sizes should be balanced** (not all embryos in one cluster)
- **Core fraction should be > 30%** (indicates stable clustering)
- **Average max_p should be > 0.6** (indicates confident assignments)
- **Silhouette score should be positive** (indicates good separation)

### Validation
- Compare identified mutants with reported tmem67 genotypes (if available)
- Manually inspect trajectories in interactive plots
- Check if mutant clusters show consistent elevated deviation over time
- Verify core_fraction is high in mutant clusters (indicates tight grouping)

## Files in This Directory

- `config.py`: Configuration parameters
- `run_tmem67_clustering.py`: Main analysis script
- `cluster_analysis_utils.py`: Helper functions for mutant identification
- `plotting_utils.py`: Visualization functions (PNG + Plotly)
- `README.md`: This file
- `output/`: Results directory (created on first run)

## Dependencies

Required packages (from src/analyze):
- `src.analyze.trajectory_analysis` (DTW, clustering, posterior analysis)
- `sklearn.metrics` (silhouette_score)
- `numpy`, `pandas` (data manipulation)
- `matplotlib` (static plots)
- `plotly` (interactive plots)

## Troubleshooting

**Issue:** "No module named 'src.analyze'"
- **Solution:** Run from project root or ensure `sys.path` includes morphseq root

**Issue:** "File not found" for curvature or metadata
- **Solution:** Check `CURV_DIR` and `META_DIR` paths in `config.py`

**Issue:** Very low core_fraction (< 20%)
- **Solution:** Try adjusting `THRESHOLD_MAX_P` or `THRESHOLD_LOG_ODDS_GAP` in `config.py`

**Issue:** All/none clusters flagged as mutant
- **Solution:** Adjust `MUTANT_THRESHOLD` or check data quality (plot raw trajectories)

## References

**Methods:**
- DTW distance: Dynamic Time Warping for trajectory similarity
- Bootstrap consensus clustering: Monti et al. (2003) Machine Learning
- Posterior probabilities: Ensemble-based cluster assignment confidence

**Related analyses:**
- `results/mcolon/20251104_preliminary_curvature_across_experiments/` - Original clustering approach
- `results/mcolon/20251205_b9d2_analysis_updated_plots/` - Plotting inspiration

## Contact

For questions about this analysis, see the Claude Code session that generated it.
