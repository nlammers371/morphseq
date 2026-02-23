# Multi-Experiment Hierarchical Consensus Clustering Analysis

**Date Created**: 2025-01-05
**Purpose**: Streamlined DTW clustering across multiple experiments and genotypes with organized output structure

---

## Overview

This analysis runs hierarchical consensus clustering on curvature trajectory data across 4 experiments, automatically analyzing all genotypes within each experiment. Results are organized by experiment → genotype → plot type for easy navigation.

---

## Key Files

### Scripts
- **`run_hierarchical_consensus_clustering.py`** - Main analysis script
- **`config.py`** - Configuration (experiments, k values, thresholds, paths)

### Utilities Created
- **`src/analyze/utils/plotting.py`** - General plotting utility with `plot_embryos_metric_over_time()` function

---

## What This Script Does

### 1. Data Loading
- Loads experiments: `['20250305', '20250416', '20250711', '20251020']`
- Merges curvature metrics + metadata for each experiment
- Automatically detects unique genotypes per experiment

### 2. Clustering Analysis
- **Method**: Hierarchical consensus clustering only (streamlined)
- **K values**: 2-8 clusters
- **For each experiment × genotype combination**:
  - Runs DTW distance computation
  - Performs bootstrap consensus clustering (100 iterations, 80% sampling)
  - Computes membership classification (core/uncertain/outlier)
  - Generates plots organized by type

### 3. Output Structure
```
output/
├── {experiment_id}/          # e.g., "20250305"
│   ├── {genotype}/           # e.g., "cep290_homozygous"
│   │   ├── coassoc_matrices/
│   │   │   ├── coassoc_k2.png
│   │   │   ├── coassoc_k3.png
│   │   │   └── ...
│   │   ├── temporal_trends/
│   │   │   ├── temporal_trends_k2.png
│   │   │   └── ...
│   │   ├── cluster_overlays/
│   │   │   ├── cluster_overlay_k2.png
│   │   │   └── ...
│   │   └── membership_vs_k.png
│   ├── genotype_overlay.png  # NEW: All genotypes overlaid
│   └── data/
│       └── results_{genotype}.pkl
```

---

## Configuration Settings

### Key Parameters (in `config.py`)

**Experiments**:
```python
EXPERIMENTS = ['20250305', '20250416', '20250711', '20251020']
```

**Clustering**:
```python
K_VALUES = [2, 3, 4, 5, 6, 7, 8]
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.8
```

**Membership Thresholds** (relaxed for less stringent assignment):
```python
CORE_THRESHOLD = 0.70      # Was 0.80
OUTLIER_THRESHOLD = 0.4    # Was 0.3
```

**Transparency**:
```python
TRAJECTORY_ALPHA = 0.8  # All membership categories use same alpha
```

---

## Plot Types Generated

### 1. Co-association Matrices
- Heatmaps showing how often pairs of embryos cluster together
- One per method per k value
- Ordered by cluster assignment

### 2. Temporal Trends with Membership
- Individual trajectories colored by membership status (core/uncertain/outlier)
- Mean trajectory (black line)
- ±SD band (blue shading)
- Linear regression fit with R²
- **Alpha = 0.8 for all trajectories** (no longer varies by membership)

### 3. Cluster Trajectory Overlays
- Two-panel plot:
  - **Left**: Individual trajectories colored by cluster
  - **Right**: Mean trajectories with ±SD and linear fits overlaid
- Direct comparison of clusters

### 4. Membership vs K Plots
- Shows how core/uncertain/outlier percentages change across k values
- Two sub-panels: line plot + stacked area plot
- One per method (aggregates all k values)

### 5. Genotype Overlay Plots (NEW)
- All genotypes for an experiment overlaid on one plot
- Individual trajectories (light, α=0.2) + mean per genotype (bold, α=0.9)
- **Smoothing enabled by default** (rolling window = 5)
- Located at: `output/{experiment_id}/genotype_overlay.png`

---

## New Plotting Utility

### `src/analyze/utils/plotting.py`

**Function**: `plot_embryos_metric_over_time()`

**Purpose**: Flexible general-purpose plotting for embryo trajectories over time

**Key Features**:
- Color by any categorical column (`genotype`, `cluster`, `phenotype`, etc.)
- Toggle individual trajectories, means, ±SD bands
- **Built-in smoothing** (default `smooth_window=5`)
- Customizable transparency, line widths, colors
- Auto-saves to specified path

**Example Usage**:
```python
from src.analyze.utils.plotting import plot_embryos_metric_over_time

# Genotype overlay with smoothing
fig = plot_embryos_metric_over_time(
    df,
    color_by='genotype',
    smooth_window=5  # Default, can adjust or set to 1 for no smoothing
)

# Cluster overlay with SD bands
fig = plot_embryos_metric_over_time(
    df,
    color_by='cluster',
    show_individual=False,
    show_sd_band=True
)
```

---

## Edge Case Handling

### Small Sample Sizes
The script gracefully handles experiments/genotypes with few trajectories:
- **Skips k values** when k ≥ n_samples
- **Checks cluster validity** (skips if consensus finds only 1 cluster)
- **Wraps silhouette/membership** in try-except to handle failures

### Example:
- 9 samples with k=7 may result in only 1 consensus cluster → skipped
- 4 samples automatically skips k ≥ 4

---

## Running the Analysis

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251104_preliminary_curvature_across_experiments
python run_hierarchical_consensus_clustering.py
```

### Expected Runtime
- ~5-10 minutes per experiment/genotype combination
- Depends on number of trajectories and k values tested
- Progress printed to console with VERBOSE_OUTPUT=True

---

## Data Sources

### Curvature Metrics
- Location: `morphseq_playground/metadata/body_axis/summary/`
- Files: `curvature_metrics_{experiment_id}.csv`
- Contains: Body axis metrics (baseline_deviation_um, arc_length_ratio, etc.)

### Metadata
- Location: `morphseq_playground/metadata/build06_output/`
- Files: `df03_final_output_with_latents_{experiment_id}.csv`
- Contains: Genotype, embryo_id, predicted_stage_hpf, embeddings

### Merged Data
- Script automatically merges on `snip_id`
- Uses `baseline_deviation_normalized` from metadata
- Renames to `normalized_baseline_deviation` for consistency

---

## Key Differences from `compare_clustering_methods.py`

### Streamlined
- **One method only**: Hierarchical consensus (removed k-medoids, direct DTW methods)
- **Organized outputs**: By experiment → genotype → plot_type (not flat structure)
- **Multi-experiment**: Loops over experiments automatically
- **Dynamic genotypes**: Detects genotypes per experiment (not hardcoded)

### New Features
- **Genotype overlay plots**: Visual comparison across genotypes
- **Flexible plotting utility**: Reusable `plot_embryos_metric_over_time()` function
- **Smoothing**: Built-in trajectory smoothing (rolling window)

### Configuration
- **Relaxed thresholds**: Core=0.70, Outlier=0.4 (more inclusive)
- **Consistent alpha**: All trajectories α=0.8 (removed uncertainty-dependent transparency)

---

## Troubleshooting

### "No data for experiment X"
- Check that curvature and metadata files exist for that experiment
- Verify `source_experiment` column is populated correctly

### "Skipped (too few samples)"
- Normal for small genotype groups (e.g., unknown genotypes)
- K values automatically adjusted based on sample size

### Import errors
- Ensure working directory is morphseq root
- Check that `src/analyze/utils/plotting.py` exists
- Verify `plot_utils.py` and other modules from `20251103_DTW_analysis` are accessible

---

## Next Steps / TODOs

- [ ] Review genotype overlay plots for each experiment
- [ ] Compare clustering results across experiments
- [ ] Identify stable k values per experiment/genotype
- [ ] Analyze temporal trends in cluster means
- [ ] Export cluster assignments for downstream analysis

---

## Notes

- **Membership classification** is now less stringent (more core members, fewer outliers)
- **All plots use smoothing** by default (window=5) for cleaner visualization
- **Genotype overlays** provide quick visual comparison before diving into cluster details
- **Output organization** makes it easy to compare same plot type across genotypes/experiments
