# DTW Clustering Pipeline - Plotting Functions Implementation Summary

## Status: ✅ CORE PLOTS IMPLEMENTED

Six critical visualization functions have been implemented to evaluate clustering quality across different k values and visualize temporal patterns.

---

## Implemented Plotting Functions

### 1. **K-Selection Metrics Comparison** ✅
**Module**: `select-k-simple.py`
**Function**: `plot_metric_comparison(metrics, best_k=None, dpi=200)`

**Purpose**: Evaluate different k values to justify the selected k

**Visualization**:
- 2x2 subplot panel showing:
  - **Panel 1**: Silhouette coefficient (green line, higher is better)
  - **Panel 2**: Gap statistic with error bars (purple line)
  - **Panel 3**: Spectral eigengap (orange markers)
  - **Panel 4**: Bootstrap stability ARI (brown line, shows consistency)

**Key Features**:
- Red dashed line highlights selected best k
- Yellow annotation box at bottom shows final recommendation
- Grid and legend for clarity
- All metrics on same figure for easy comparison

**Example Output**:
```
output/3_select_k/plots/k_selection_metrics.png
```

**Question Answered**: "Is k=2 actually the best choice?"
- Look at silhouette scores: if k=2 is low compared to k=3,4, then k=2 may not be optimal
- Look at stability (ARI): if k=2 has low ARI, clusters are unstable
- Multi-metric comparison shows if consensus exists

---

### 2. **Temporal Trends by Cluster** ✅
**Module**: `fit-models-module.py`
**Function**: `plot_cluster_trajectories(trajectories, common_grid, cluster_id)`

**Purpose**: Show all individual trajectories within a cluster with statistical summaries

**Visualization**:
- Single cluster with:
  - Individual trajectories (gray, alpha=0.3) - see trajectory diversity
  - Mean trajectory (black, bold) - cluster center
  - ±1 SD band (blue, transparent) - trajectory variability
  - Linear fit (red dashed) - trend direction with R² value

**Key Features**:
- Shows individual embryo behavior (not just averages)
- High alpha (0.3) makes patterns visible despite overlap
- R² shows how linear the temporal trend is
- SD band indicates cluster coherence

**Example Output**:
```
output/5_fit_models/plots/cluster_trajectories_k2_c0.png
output/5_fit_models/plots/cluster_trajectories_k2_c1.png
```

**Question Answered**: "Do embryos in this cluster actually have similar trajectories?"
- If trajectories are completely overlapped: good clustering
- If trajectories are scattered: cluster may not be homogeneous
- If SD band is huge: poor cluster cohesion
- R² shows if trajectory is monotonic or complex

---

### 3. **Cluster Trajectory Overlay Comparison** ✅
**Module**: `fit-models-module.py`
**Function**: `plot_cluster_comparison(trajectories_by_cluster, common_grid)`

**Purpose**: Compare cluster means side-by-side to see cluster separation

**Visualization** (Two-panel design):

**LEFT PANEL**: Individual trajectories by cluster
- Each cluster's trajectories in its own color (tab10 palette)
- Alpha=0.3 for individual embryos
- Thick colored lines overlay cluster means
- Shared y-axis across all clusters for fair comparison

**RIGHT PANEL**: Mean trajectories with confidence intervals
- Cluster mean curves (thick, bold)
- ±1 SD confidence interval (same color, transparent)
- Linear fits for each cluster (dashed, bold)
- Shared y-axis for direct comparison

**Key Features**:
- Two-pass plotting for shared y-axis limits (fair comparison)
- Color consistency helps identify clusters
- Individual trajectories show variability
- Fits show temporal trends per cluster
- Side-by-side comparison clarifies separation

**Example Output**:
```
output/5_fit_models/plots/cluster_comparison_k2.png
output/5_fit_models/plots/cluster_comparison_k3.png
```

**Question Answered**: "Are the clusters really separated? Do they have different temporal patterns?"
- Clear separation of lines = good clustering
- Overlapped means = poor clustering
- Different linear fits = meaningful biological differences
- Similar trajectories across clusters = k may be too high

---

### 4. **Bootstrap Stability Heatmap** ✅
**Module**: `cluster-module.py`
**Function**: `plot_coassoc_matrix(C, labels=None, k=None, dpi=100)`

**Purpose**: Visualize clustering stability from bootstrap resamples

**Visualization**:
- Heatmap of co-association matrix
- Embryos sorted by cluster (creates block structure if stable)
- Colors: red=high co-association (stable pairs), blue=low
- Block diagonals show stable clusters
- Noisy pattern = unstable clustering

**Key Features**:
- Sorted by cluster to reveal block structure
- Value 1.0 = always clustered together, 0.0 = never
- Clear blocks indicate stable, well-separated clusters
- Noisy background = uncertain, overlapping clusters
- Color range: 0 (blue) to 1 (red)

**Example Output**:
```
output/2_select_k/plots/bootstrap_k2_coassoc.png
output/2_select_k/plots/bootstrap_k3_coassoc.png
```

**Question Answered**: "How stable are the clusters from bootstrap resampling?"
- Clear red blocks on diagonal = stable clusters
- Noisy/fuzzy blocks = unstable, uncertain clusters
- Off-diagonal red patches = spurious cluster merging
- Mostly blue = weak clustering signal

**THIS IS WHY K=2 SHOWS HIGH UNCERTAINTY**: If the co-association matrix is noisy (not clear blocks), then many embryos have uncertain membership because they don't cluster consistently across bootstrap resamples.

---

### 5. **Membership Distribution** ✅
**Module**: `membership-module.py`
**Function**: `plot_membership_distribution(classification, cluster_stats=None)`

**Purpose**: Show core/uncertain/outlier breakdown

**Visualization** (Two-panel):

**LEFT PANEL**: Overall distribution
- Bar chart with three categories
- Green=Core (stable, >70% co-association)
- Yellow=Uncertain (medium stability, 30-70%)
- Red=Outlier (unstable, <30%)
- Count labels on each bar

**RIGHT PANEL**: Per-cluster breakdown
- Stacked bars showing composition of each cluster
- Same color scheme (green/yellow/red)
- Shows if certain clusters are mostly uncertain

**Example Output**:
```
output/4_membership/plots/membership_distribution_k2.png
```

**Question Answered**: "Why is membership mostly uncertain?"
- High yellow = clusters aren't stable across bootstrap resamples
- Different clusters have different stability (visible in right panel)
- Suggests k may not be optimal or data is noisy
- Compare with stability heatmap for interpretation

**FOR YOUR K=2 CASE**: If most are uncertain (yellow), the co-association heatmap should show noisy patterns (not clear blocks), confirming the clusters aren't stable.

---

### 6. **Supporting Utility Functions** ✅
**Module**: `plot_utils.py`

**Key Helper Functions**:

```python
# Shared utilities
get_cluster_colors(n_clusters)           # Tab10 color palette for clusters
pad_trajectories_to_common_length()      # Handle variable-length data
compute_trajectory_stats()                # Mean and SD for trajectories
fit_linear_regression()                   # Linear trend with R²
get_shared_ylims()                        # Fair axis limits across plots
plot_distance_matrix()                    # DTW distance heatmap
plot_silhouette_scores()                  # Silhouette by k
```

**These ensure**:
- Consistent colors across all plots
- Proper NaN handling for variable-length trajectories
- Shared y-axis for fair comparison (critical for evaluation)
- Reproducible plotting style

---

## Why These Plots Answer Your Question

**Your observation**: "k=2 has high uncertainty - how is this good?"

**These plots help you evaluate**:

1. **K-Selection Metrics** → Shows if k=2 is statistically optimal
   - Low silhouette? k=2 may not be best
   - Low ARI stability? Clusters not stable

2. **Co-association Heatmap** → Shows why uncertainty is high
   - Noisy pattern = many embryos cluster inconsistently
   - Clear blocks = stable despite uncertainty ratings

3. **Temporal Trends** → Shows if clusters are biologically meaningful
   - Different temporal patterns per cluster = meaningful separation
   - Similar patterns = maybe wrong k

4. **Cluster Overlay** → Direct visual comparison
   - Overlapped means = poor separation, explains uncertainty
   - Well-separated = maybe uncertainty is just due to small sample size

5. **Membership Distribution** → Shows stability breakdown
   - Mostly uncertain = global instability problem
   - Mostly core = outliers are exception, clustering good

---

## Usage Example

```python
# After running explore.py, view plots:

# 1. Check k selection
# output/3_select_k/plots/k_selection_metrics.png
# Are silhouette, gap, eigengap, ARI all recommending k=2?

# 2. Check stability
# output/2_select_k/plots/bootstrap_k2_coassoc.png
# Are there clear red blocks? Or noisy?

# 3. Check trajectories
# output/5_fit_models/plots/cluster_comparison_k2.png
# Are cluster means well-separated? Do they have different trends?

# 4. Check membership
# output/4_membership/plots/membership_distribution_k2.png
# How many core vs uncertain per cluster?
```

---

## Next Steps to Evaluate K=2

**Run the complete pipeline with plotting**:
1. Execute `explore.py`
2. View all plots in `output/` directory
3. Compare k=2 vs k=3 vs k=4 side-by-side
4. Check if other k values have:
   - Better silhouette scores
   - More stable clusters (better blocks in heatmap)
   - Clearer cluster separation
   - Higher core membership percentage

**If k=2 is NOT optimal**:
- Try k=3 or k=4
- May need to adjust DTW window or bootstrap settings
- Or data may not naturally cluster into clear groups

**If k=2 IS optimal despite uncertainty**:
- Uncertainty may be due to small sample size
- Clusters may be meaningful even if boundaries are soft
- Consider using "uncertain" category in downstream analysis

---

## Plotting Integration Status

✅ **Completed**:
- K-selection metrics (4-panel comparison)
- Temporal trends per cluster
- Cluster overlays with 2-panel layout
- Bootstrap stability heatmaps
- Membership distribution
- Shared plotting utilities

⏳ **Not Yet Integrated into explore.py**:
- Still need to add calls to generate these plots automatically when running pipeline

**These plots are ready to use - just need to integrate into the main pipeline script.**

