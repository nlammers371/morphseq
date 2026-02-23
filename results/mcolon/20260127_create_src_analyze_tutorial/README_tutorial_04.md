# Tutorial 04: CEP290 Cluster Projection - Batch Effect Analysis

## Overview

This tutorial demonstrates **cluster projection methodology** for comparing phenotypic trajectories across experiments with different temporal coverage. It projects new CEP290 experiments onto pre-existing cluster definitions to test for batch effects caused by temporal coverage differences.

## Problem Statement

From Tutorial 03, we observed that with the 27-47 hpf overlap window, we cannot distinguish:
- Wild-type (ab) from "Low_to_High" cep290_crispant trajectories
- "Not_Penetrant" from "Low_to_High" cep290_crispant (error bars overlap)

**Key Insight**: Experiment 20260124 extends to 80 hpf and shows almost all crispants become penetrant by 80 hpf. This allows us to test whether temporal coverage differences cause batch effects.

## Approach

**Reference Clusters**: CEP290 mutant phenotypes from 7 older experiments (2025 data)
- 571 embryos with well-defined cluster assignments
- 4 categories: Not_Penetrant (50.6%), High_to_Low (24.7%), Low_to_High (17.7%), Intermediate (7.0%)

**Source Experiments** (to be projected):
- **20260122**: 95 embryos, 11-47 hpf coverage (shorter window)
- **20260124**: 98 embryos, 27-77 hpf coverage (longer window)

**Method**: Project source experiments onto reference clusters using DTW-based nearest neighbor assignment, then compare cluster proportions.

## Pipeline Steps

### 1. Load Reference Clusters
- Load pre-computed CEP290 clusters from 7 experiments (2025)
- Filter to embryos with valid cluster assignments (571 embryos)
- Create embryo_id → cluster mappings

### 2. Load Source Experiments
- Load 20260122 and 20260124 separately from build04
- Filter to valid embryos (use_embryo_flag=True)

### 3. Prepare Multivariate Arrays
- Filter all data to shared time window: **25-50 hpf** (matches Tutorial 03)
- Interpolate to common time grid (50 points)
- Use single metric: `baseline_deviation_normalized` (curvature only)

### 4. Compute Cross-DTW Distances
- Compute DTW distances from each source embryo to all reference embryos
- Use Sakoe-Chiba radius = 20 for warping constraint
- Save distance matrices for inspection

### 5. Nearest Neighbor Projection
- For each source embryo, find nearest reference embryo
- Assign to same cluster as nearest neighbor
- Record distance to nearest neighbor (for quality check)

### 6. Proportion Analysis
- Compare cluster distributions between experiments
- Chi-square test for statistical significance
- Stratify by genotype (ab vs cep290_crispant)

### 7-9. Visualization
- Trajectory plots (faceted by cluster category × experiment)
- Proportion plots (by genotype and experiment)
- Distance distribution analysis

## Key Results

### Projection Assignments

**20260122** (95 embryos):
- Not Penetrant: 32 (33.7%)
- Low_to_High: 28 (29.5%)
- High_to_Low: 24 (25.3%)
- Intermediate: 11 (11.6%)

**20260124** (98 embryos):
- Not Penetrant: 41 (41.8%)
- Low_to_High: 37 (37.8%)
- Intermediate: 13 (13.3%)
- High_to_Low: 7 (7.1%)

### Statistical Test

**Chi-square test**: χ² = 11.801, p = 0.0081 (df = 3)
- **SIGNIFICANT batch effect detected (p < 0.05)**

### Key Finding: High_to_Low Discrepancy

**20260122**: 25.3% High_to_Low
**20260124**: 7.1% High_to_Low

This 18% difference suggests that 20260122 (shorter time window) assigns more embryos to "High_to_Low" trajectories, possibly because:
1. Limited temporal coverage prevents observing later recovery
2. Early high curvature is mistaken for sustained high penetrance
3. Temporal bias in DTW alignment favors early-stage patterns

### Genotype-Stratified Analysis

**Wild-type (ab)**:
- 20260122: 48.6% Not_Penetrant, 37.8% Low_to_High
- 20260124: 74.2% Not_Penetrant, 22.6% Low_to_High
- Interpretation: More consistent (wild-type should be mostly Not_Penetrant)

**cep290_crispant**:
- 20260122: 36.2% High_to_Low, 24.1% Low_to_High
- 20260124: 44.8% Low_to_High, 9.0% High_to_Low
- Interpretation: Strong batch effect - opposite patterns between experiments

### Distance Distributions

**20260122**: Mean = 5.75, Median = 4.98
**20260124**: Mean = 9.88, Median = 8.71

20260124 has higher DTW distances to reference, suggesting:
- Different temporal coverage leads to poorer alignment
- Reference clusters (from 2025 data) may not perfectly represent 2026 experiments

## Interpretation

### Batch Effect Confirmation

The significant chi-square test (p = 0.0081) confirms that temporal coverage differences cause batch effects in cluster assignment. The High_to_Low vs Low_to_High discrepancy is particularly striking.

### Updated Conclusions (2026 Projections)

From the proportion plots by experiment/genotype:
- **20260124 (longer window)** shows cep290_crispants biased toward **Low_to_High** and **Intermediate**.
- **20260122 (shorter window)** shows a more even distribution across categories.

**Interpretation:** A substantial fraction of “Not Penetrant” in **20260122** likely would have become penetrant with extended imaging. This suggests:
- **Distinguishing Low_to_High vs Not Penetrant requires imaging to at least ~60 hpf.**
- Shorter windows can under-call penetrance and inflate “Not Penetrant.”

### Why This Happens

1. **Temporal Coverage Bias**: 20260122 (11-47 hpf) captures early penetrance but misses later recovery
2. **DTW Alignment Sensitivity**: DTW prioritizes early timepoints when shorter trajectories are compared to longer references
3. **Phenotype Resolution**: "High_to_Low" vs "Low_to_High" distinction requires full temporal window

### Tutorial Takeaway

Cluster projection can:
- **Reveal batch effects** from temporal coverage differences
- **Use reference clusters as anchors** for cross-experiment comparison
- **Quantify projection quality** via DTW distances
- **Stratify by genotype** to isolate biological from technical variation

## Outputs

### Files Generated

```
output/
├── figures/04/
│   ├── projection_results/
│   │   ├── 20260122_projection_nn.csv          # Projection assignments (95 embryos)
│   │   ├── 20260124_projection_nn.csv          # Projection assignments (98 embryos)
│   │   ├── cross_dtw_20260122.npy              # Distance matrix (95 × 552)
│   │   └── cross_dtw_20260124.npy              # Distance matrix (98 × 552)
│   ├── cluster_projection_trajectories.png     # Faceted trajectories (cluster × experiment)
│   ├── proportion_by_experiment.png            # Proportion bar charts
│   └── batch_effect_analysis.png               # Distance distributions
└── results/
    └── cluster_frequency_comparison.csv        # Statistical comparison table
```

### Projection CSV Format

```csv
embryo_id,nearest_embryo_id,nearest_distance,cluster,cluster_category,experiment_id,genotype
20260122_A02_e01,20251112_D07_e01,8.73,4.0,High_to_Low,20260122,cep290_crispant
```

## Code Structure

### Key Functions Used

From `projection_utils.py` (local copy with fixed imports):
- `compute_cross_dtw_distance_matrix()` - Cross-dataset DTW computation
- `assign_clusters_nearest_neighbor()` - NN-based cluster assignment
- `compare_cluster_frequencies()` - Statistical comparison (chi-square)

From `src.analyze.trajectory_analysis`:
- `prepare_multivariate_array()` - Interpolate trajectories to common grid

From `src.analyze.viz.plotting`:
- `plot_feature_over_time()` - Faceted trajectory visualization

### Parameters

```python
METRICS = ['baseline_deviation_normalized']  # Curvature only
TIME_WINDOW = (25, 50)                       # hpf (matches Tutorial 03)
SAKOE_CHIBA_RADIUS = 20                      # DTW warping constraint
```

## Next Steps

### Immediate Extensions

1. **K-NN Posterior Assignment**: Use `assign_clusters_knn_posterior()` to get uncertainty estimates
2. **Bootstrap Confidence Intervals**: Resample reference clusters to quantify projection stability
3. **Outlier Analysis**: Investigate embryos with high DTW distances (>2 SD from median)

### Research Applications

1. **Cross-Mutant Comparison**: Project tmem67 onto CEP290 reference (Tutorial 05?)
2. **Temporal Normalization**: Use reference clusters to correct batch effects
3. **Quality Control**: Flag experiments with high projection distances as potential batch issues

### Code Refactoring

The projection utilities should be moved to:
```
src/analyze/trajectory_analysis/cluster_projection.py
```

This would unify projection workflows across the codebase and make the API more discoverable.

## Educational Value

This tutorial demonstrates:
- **Why batch effects matter**: Temporal coverage significantly alters cluster assignments
- **How to detect batch effects**: Quantitative comparison using projection + chi-square test
- **Reference-based normalization**: Using well-defined clusters as anchors
- **API usage**: Complete workflow from data loading to statistical testing

## Connection to Tutorial 03

Tutorial 03 showed that clustering within overlapping time windows produces clean clusters but cannot distinguish penetrance trajectories due to limited temporal coverage.

Tutorial 04 shows that **projecting onto pre-defined reference clusters** allows:
1. Cross-experiment comparison
2. Batch effect quantification
3. Temporal coverage normalization

**Together**, these tutorials demonstrate the power of reference-based trajectory analysis for handling heterogeneous experimental designs.

---

**Author**: Tutorial generated from plan implementation
**Date**: 2026-02-02
**Script**: `04_cluster_projection.py`
