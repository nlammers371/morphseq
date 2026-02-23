# 07b DTW Clustering Analysis - Specification

## Purpose
Cluster homozygous mutant embryo curvature trajectories using Dynamic Time Warping (DTW) distance and test for anti-correlated early/late curvature patterns.

---

## Input
- Full dataframe from `load_data.py`
- Metric: `normalized_baseline_deviation`
- Filter: `cep290_homozygous` genotype only

---

## Step-by-Step Process

### STEP 1: Data Extraction & Preparation
**Input**: Raw dataframe
**Action**:
- Filter for `cep290_homozygous` genotype
- Extract columns: `embryo_id`, `predicted_stage_hpf`, `normalized_baseline_deviation`
- Drop rows with NaN metric values
- Create long format: embryo_id, hpf, metric_value
- Extract per-embryo trajectories (minimum 3 timepoints each)

**Output**:
- List of trajectories (n_embryos × variable length)
- List of embryo IDs (n_embryos)
- Long-format dataframe (all data points)
- Print: Number of embryos, mean trajectory length

---

### STEP 2: Missing Data Handling
**Input**: Long-format trajectories_df
**Action**:
- Apply linear interpolation to handle any remaining missing values
- For each embryo, interpolate NaN values between valid timepoints
- Use scipy.interpolate.interp1d with linear extrapolation

**Output**: Imputed trajectories_df
**Print**: Confirmation that imputation complete

---

### STEP 3: Trajectory Interpolation to Common Timepoints
**Input**: List of trajectories (variable lengths, long-format dataframe)
**Action**:
- Find minimum and maximum hpf values across all trajectories
- Create common timepoint grid (e.g., 0.5 hpf intervals from min to max)
- For each embryo trajectory:
  - Use scipy.interpolate.interp1d with linear interpolation to map to common grid
  - NO edge/boundary padding - only interpolate within observed range
  - Trajectories shorter than common grid length will remain truncated
- Store both: interpolated_trajectories (at common timepoints) and original trajectory lengths

**Output**: Interpolated trajectories at common timepoints, original trajectory lengths array
**Print**: Min/max hpf range, common timepoint grid length, interpolated shape

---

### STEP 4: DTW Distance Matrix Computation
**Input**: Padded trajectories (n_embryos × max_length)
**Action**:
- Compute pairwise DTW distances using Sakoe-Chiba band (window=3)
- Fill n×n symmetric distance matrix
- Validate: check for NaN/Inf, ensure diagonal is 0

**Output**: Distance matrix D (n×n)
**Print**:
- Progress (every 10 embryos)
- Validation: NaN count, Inf count, shape, min/max/mean distance

---

### STEP 5: K-means Clustering
**Input**: Distance matrix D
**Action**:
- Test k = [2, 3, 4]
- For each k:
  - Run KMeans with metric='precomputed', n_init=10, random_state=42
  - Compute silhouette score
  - Store: assignments, silhouette, inertia

**Output**: clustering_results dict with k as keys
**Print**: k value, silhouette score for each k

---

### STEP 6: Select Best K
**Input**: clustering_results for all k
**Action**:
- Choose k with highest silhouette score

**Output**: best_k, best_assignments
**Print**: Best k, its silhouette score

---

### STEP 7: Extract Early/Late Means
**Input**: Original dataframe, embryo_ids, best_assignments
**Action**:
- For each embryo, extract timepoints in:
  - Early window: 44-50 hpf
  - Late window: 80-100 hpf
- Compute mean metric value for each window (per embryo)
- Create array: early_means_arr, late_means_arr (one value per embryo)

**Output**: Two arrays of means
**Print**: Window ranges

---

### STEP 8: Anti-Correlation Test
**Input**: cluster assignments, early_means_arr, late_means_arr
**Action**:
- For each cluster:
  - Get early and late mean values for embryos in that cluster
  - Filter out NaN values
  - If ≥3 valid pairs:
    - Compute Pearson r between early and late
    - Compute p-value
    - Run permutation test (shuffle late values, recompute r, get null distribution)
    - Compute permutation p-value
    - Classify: anti-correlated (r < -0.3), correlated (r > 0.3), uncorrelated
  - Store all stats

**Output**: anticorr_results dict
**Print**: For each cluster:
- n_embryos, early_mean, late_mean, pearson_r, p_value, permutation_p, interpretation

---

### STEP 9: Create Output Dataframe
**Input**: embryo_ids, best_assignments, early_means_arr, late_means_arr, original df
**Action**:
- Create DataFrame with columns:
  - embryo_id
  - cluster (assignment)
  - early_mean
  - late_mean
  - genotype (from original df)

**Output**: output_df
**Print**: shape and column names

---

### STEP 10: Generate Plots
**Input**: All results from previous steps
**Action**: See PLOTS section below
**Output**: 12 PNG files
**Print**: Filename as each plot is saved

---

### STEP 11: Generate Tables
**Input**: All results
**Action**: See TABLES section below
**Output**: 3 CSV files
**Print**: Filename as each table is saved

---

### STEP 12: Print Summary
**Input**: All results
**Action**:
- Print completion message
- Print output directory locations
- Print output_df shape and columns

**Output**: Console output
**Print**: Final summary

---

## Plots to Generate

### Plot 21: Cluster Selection Metrics
- **2×2 grid**
- Panel A: Elbow curve (k vs inertia)
- Panel B: Silhouette scores (k vs silhouette)
- Panel C: Placeholder (Gap Statistic)
- Panel D: Placeholder (Penetrance)
- Highlight best_k in red

### Plot 22: Anti-Correlation Scatter
- **Single plot**
- X-axis: early_mean (early window curvature)
- Y-axis: late_mean (late window curvature)
- Points colored by cluster
- Regression line per cluster
- Annotations: cluster ID, Pearson r, p-value, permutation p, interpretation
- Flip-flop pattern = anti-correlated (line with negative slope)

### Plots 23-28: Cluster Visualizations
- Various cluster characteristics (genotype distribution, etc.)

### Plot 29: DTW Distance Matrix
- **Heatmap** showing pairwise DTW distances
- Rows/columns sorted by cluster assignment (shows block structure)
- Colorbar showing distance values
- Cluster boundary lines

### Plot 30: Silhouette Analysis
- **Bar chart** showing silhouette scores for each k
- Best k highlighted in red

### Plot 31: Temporal Trends by Cluster
- **Subplots** (one per cluster, side-by-side)
- Per subplot:
  - Individual trajectories (light gray, α=0.15)
  - Mean trajectory (bold black line)
  - IQR band (25th-75th percentile, α=0.25)
  - ±1 SD band (α=0.15, blue)
  - Early window (cyan vertical band, α=0.15)
  - Late window (red vertical band, α=0.15)
  - Title: Cluster ID, n_embryos, genotype breakdown, Pearson r, p-value, interpretation

### Plot 32: Cluster Trajectories Overlay
- **Single plot** with all clusters overlaid
- Mean trajectory per cluster (colored line)
- ±1 SD confidence band per cluster (lighter shade)
- Early window (cyan vertical band, α=0.1)
- Late window (red vertical band, α=0.1)
- Legend showing cluster IDs
- Shows overall comparison of trajectories

---

## Tables to Generate

### Table 4: Cluster Characteristics
- Columns: Cluster, n_embryos, %_WT, %_Het, %_Homo, Early_mean, Late_mean, Penetrance_t100
- One row per cluster
- Save to: `table_4_cluster_characteristics.csv`

### Table 5: Anti-Correlation Evidence
- Columns: Cluster, Early_mean, Late_mean, Pearson_r, P_value, Permutation_p_value, Interpretation
- One row per cluster
- Save to: `table_5_anticorrelation_evidence.csv`

### Table 6: Embryo-Cluster Assignments
- Columns: embryo_id, genotype, cluster_assignment, early_curvature, late_curvature, penetrant_at_t100
- One row per embryo
- Save to: `table_6_embryo_cluster_assignments.csv`

---

## Output Directory Structure

```
results/mcolon/20251029_curvature_temporal_analysis/outputs/07b_dtw_clustering_analysis/
├── plots/
│   ├── plot_21_cluster_selection.png
│   ├── plot_22_anticorrelation_scatter.png
│   ├── plot_23-28_*.png
│   ├── plot_29_distance_matrix.png
│   ├── plot_30_silhouette_analysis.png
│   ├── plot_31_temporal_trends.png
│   └── plot_32_trajectory_overlay.png
│
└── tables/
    ├── table_4_cluster_characteristics.csv
    ├── table_5_anticorrelation_evidence.csv
    └── table_6_embryo_cluster_assignments.csv
```

---

## Configuration (At Top of File)

```python
METRIC_NAME = 'normalized_baseline_deviation'
GENOTYPE_FILTER = 'cep290_homozygous'
EARLY_WINDOW = (44, 50)  # hpf
LATE_WINDOW = (80, 100)  # hpf
DTW_WINDOW = 3
CLUSTER_K_VALUES = [2, 3, 4]
```

---

## Error Handling

- **Distance matrix validation**: Assert no NaN/Inf values after DTW computation
- **Clustering**: Try-except around KMeans operations, skip k if fails
- **Plotting**: Try-except around each major plot section, report errors clearly
- **Anti-correlation test**: Handle cases where clusters have <3 valid early/late pairs

---

## Success Criteria

✅ All 12 plots generated without errors
✅ All 3 tables generated
✅ No LAPACK errors
✅ Console output shows progress at each step
✅ Files saved to correct locations
✅ Anti-correlation hypothesis tested for all clusters
