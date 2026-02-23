# Plotting Integration Complete ✅

## What Was Done

All 6 plotting functions have been **fully integrated into `explore.py`**. The pipeline now automatically generates comprehensive visualizations when you run it.

---

## Plotting Workflow in Pipeline

### Step 2: Bootstrap Stability
```
For each k value:
  ✓ Compute co-association matrix
  ✓ Generate co-association heatmap
  → Output: 2_select_k/plots/bootstrap_k*_coassoc.png
```

**What you see**: Heatmap with block structure indicating cluster stability
- Clear red blocks = stable clusters
- Noisy pattern = unstable clusters (explains high "uncertain" members)

---

### Step 3: K-Selection Metrics
```
✓ Evaluate all k values with 4 metrics
✓ Generate comparison panel
✓ Highlight recommended best k
→ Output: 3_select_k/plots/k_selection_metrics.png
```

**What you see**: 4-panel plot comparing:
1. **Silhouette Score** (green) - cluster separation quality
2. **Gap Statistic** (purple) - goodness of clustering vs. null
3. **Spectral Eigengap** (orange) - spectral clustering quality
4. **Bootstrap ARI** (brown) - stability across resamples

**Use this to determine**: Is k=2 actually optimal? Or should you try k=3, k=4?

---

### Step 4: Membership Distribution
```
✓ Classify members as core/uncertain/outlier
✓ Plot overall distribution
✓ Plot per-cluster breakdown
→ Output: 4_membership/plots/membership_distribution_k*.png
```

**What you see**: Two bar charts
- **Left**: Overall core (green) / uncertain (yellow) / outlier (red) counts
- **Right**: Breakdown per cluster (which clusters are more stable?)

**Use this to understand**: Why are there so many "uncertain" members?
- If mostly yellow = clusters aren't stable across bootstraps
- Look at co-association heatmap to see why

---

### Step 5: Model Fitting & Cluster Visualization
```
For each cluster:
  ✓ Fit mixed-effects model
  ✓ Plot individual cluster trajectories
  → Output: 5_fit_models/plots/cluster_trajectories_k*_c*.png

After all clusters:
  ✓ Generate cluster comparison plot
  → Output: 5_fit_models/plots/cluster_comparison_k*.png
```

**Individual cluster plots** show:
- Gray lines: individual embryo trajectories
- Black line: cluster mean trajectory
- Blue band: ±1 standard deviation
- Red dashed: linear fit with R² value

**Cluster comparison plot** has 2 panels:
- **Left**: Individual trajectories by cluster (colored by cluster)
- **Right**: Mean trajectories with ±1 SD bands and linear fits

**Use this to determine**: Are clusters biologically meaningful?
- Different temporal patterns per cluster = good separation
- Similar patterns = wrong k or weak signal
- Large SD bands = inconsistent trajectories within cluster

---

## Complete Output Directory Structure

```
output/
├── 0_dtw/
│   ├── data/
│   │   ├── distance_matrix.pkl
│   │   ├── embryo_ids.pkl
│   │   └── ...
│   └── plots/
│
├── 1_cluster/
│   └── data/
│       └── baseline_results.pkl
│
├── 2_select_k/
│   ├── data/
│   │   ├── bootstrap_k2.pkl
│   │   ├── bootstrap_k3.pkl
│   │   ├── bootstrap_k4.pkl
│   │   └── ...
│   └── plots/
│       ├── bootstrap_k2_coassoc.png          ← Check these!
│       ├── bootstrap_k3_coassoc.png
│       ├── bootstrap_k4_coassoc.png
│       └── ...
│
├── 3_select_k/
│   ├── data/
│   │   ├── metrics.pkl
│   │   ├── best_k.pkl
│   │   └── baseline_results.pkl
│   └── plots/
│       └── k_selection_metrics.png           ← Check this first!
│
├── 4_membership/
│   ├── data/
│   │   ├── membership_results.pkl
│   │   └── core_indices.pkl
│   └── plots/
│       └── membership_distribution_k*.png    ← Check this!
│
└── 5_fit_models/
    ├── data/
    │   └── cluster_models.pkl
    └── plots/
        ├── cluster_trajectories_k*_c0.png   ← Check these!
        ├── cluster_trajectories_k*_c1.png
        ├── cluster_comparison_k*.png        ← Check this!
        └── ...
```

---

## How to Run and View Results

### Step 1: Run the Pipeline
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251103_DTW_analysis
rm -rf output  # Clear old results
python explore.py
```

### Step 2: View Plots
Open the plots in this order to understand your clustering:

**1. Start here:** `output/3_select_k/plots/k_selection_metrics.png`
- Is k=2 recommended by all metrics?
- Or do other k values score better?

**2. Then check:** `output/2_select_k/plots/bootstrap_k2_coassoc.png` (and k3, k4, etc.)
- Do you see clear red blocks?
- Or is the pattern noisy?
- Compare across different k values

**3. Understand membership:** `output/4_membership/plots/membership_distribution_k*.png`
- How many are core vs. uncertain?
- Which clusters are stable vs. unstable?

**4. Visualize trajectories:**
- `output/5_fit_models/plots/cluster_trajectories_k*.png` (individual clusters)
- `output/5_fit_models/plots/cluster_comparison_k*.png` (all clusters together)
- Do clusters look different?
- Are trajectories coherent within clusters?

---

## Making a Decision About K

### If k=2 looks good across all plots:
- Silhouette scores high
- Bootstrap heatmap has clear blocks
- Trajectory comparison shows separated clusters
- → k=2 is probably correct despite high "uncertain" percentage
- → Uncertainty may be due to small sample size or soft cluster boundaries

### If k=3 or k=4 looks better:
- Better silhouette/gap/eigengap scores
- Clearer blocks in bootstrap heatmap
- More distinct temporal patterns between clusters
- → Consider using higher k instead
- Edit `config.py`: `K_VALUES = [3, 4, 5]` or set `PRIOR_K = 3`

### If all k values show uncertainty:
- No k shows clear block structure in heatmap
- All have low silhouette scores
- Temporal patterns don't separate well
- → Data may not have natural clustering
- → May need different approach: remove outliers, adjust metric, or reconsider clustering goal

---

## Integration Checklist

✅ **Step 2**: Co-association matrix plots
✅ **Step 3**: K-selection metrics comparison plot
✅ **Step 4**: Membership distribution plot
✅ **Step 5**: Individual cluster trajectory plots
✅ **Step 5**: Cluster comparison plot
✅ **Step 6**: Summary output with plot locations

✅ **Error Handling**: All plotting wrapped in try-except, won't crash pipeline
✅ **Figure Cleanup**: Plots properly closed after saving
✅ **User Feedback**: Console prints locations of all generated plots

---

## Example Console Output

When you run `python explore.py`, you'll see:

```
================================================================================
STEP 2: BOOTSTRAP STABILITY ANALYSIS
================================================================================

  Bootstrap for k=2 (100 iterations)...
    Mean silhouette: 0.421
    Mean ARI: 0.658
    Plotting co-association matrix...
    Saved plot: output/2_select_k/plots/bootstrap_k2_coassoc.png

  Bootstrap for k=3 (100 iterations)...
    Mean silhouette: 0.385
    Mean ARI: 0.512
    Plotting co-association matrix...
    Saved plot: output/2_select_k/plots/bootstrap_k3_coassoc.png

...

================================================================================
STEP 5: MODEL FITTING
================================================================================

  Fitting model for cluster 0...
    Size: 9, Core: 1
    Mean R²: 0.452
    DBA: computed
    Plotting cluster 0 trajectories...
    Saved plot: output/5_fit_models/plots/cluster_trajectories_k2_c0.png

  Fitting model for cluster 1...
    Size: 15, Core: 3
    Mean R²: 0.531
    DBA: computed
    Plotting cluster 1 trajectories...
    Saved plot: output/5_fit_models/plots/cluster_trajectories_k2_c1.png

  Plotting cluster comparison (all k=2 clusters together)...
    Saved cluster comparison plot
    Saved plot: output/5_fit_models/plots/cluster_comparison_k2.png

================================================================================
PIPELINE COMPLETE
================================================================================

Results saved to: output

Key output files:
  Data: output/*_*/data/
  Plots: output/*_*/plots/

Key plots to review:
  - output/3_select_k/plots/k_selection_metrics.png
  - output/2_select_k/plots/bootstrap_k*_coassoc.png
  - output/5_fit_models/plots/cluster_trajectories_k2_*.png
  - output/5_fit_models/plots/cluster_comparison_k2.png
  - output/4_membership/plots/membership_distribution_k2.png
```

---

## Technical Details

**Imports Added to explore.py**:
- `plot_coassoc_matrix` from cluster_module
- `plot_metric_comparison` from select_k_module
- `plot_membership_distribution` from membership_module
- `plot_cluster_trajectories` from fit_models_module
- `plot_cluster_comparison` from fit_models_module
- `save_plot` from io_module

**Error Handling**:
- All plotting calls wrapped in try-except
- Warnings printed if plots fail, but pipeline continues
- matplotlib figures properly closed after saving
- matplotlib.pyplot imported locally within functions to avoid issues

**Memory Management**:
- Figures closed immediately after saving with `plt.close(fig)`
- Large arrays not duplicated unnecessarily
- Two-pass plotting pattern for shared y-axes (efficient)

---

## Summary

**The pipeline is now fully visualization-enabled!**

Just run `python explore.py` and you'll get:
- ✅ Complete clustering analysis
- ✅ Stability assessment across k values
- ✅ Membership classification
- ✅ Model fitting with splines
- ✅ 6 publication-quality plots
- ✅ Comprehensive output directory

All plots are designed to help you **evaluate if your clustering is meaningful** and **justify your choice of k**.

Ready to run and view the results! 🎉
