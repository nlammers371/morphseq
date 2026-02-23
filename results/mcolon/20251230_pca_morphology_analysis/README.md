# PCA Morphology Space Analysis - B9D2 Phenotypes

**Date:** 2025-12-30
**Status:** Phase A Complete ✅ | Phase B Ready 🚀
**Location:** `/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251230_pca_morphology_analysis/`

---

## 🎯 Project Goal

Visualize b9d2 phenotypes in **morphology space** (PCA of VAE embeddings) and apply DTW clustering to test if the 4 phenotypes (CE, HTA, BA-rescue, non-penetrant) recapitulate in this reduced dimensional space.

---

## ✅ Phase A: Visualization (COMPLETE)

### What Was Built

#### 1. **3D Plotting Module** (`src/analyze/trajectory_analysis/plotting_3d.py`)

Single unified function following `facetted_plotting.py` API patterns:

```python
plot_3d_scatter(
    df,
    coords=['PCA_1', 'PCA_2', 'PCA_3'],  # List of 3 column names
    color_by='phenotype',
    color_palette=PHENOTYPE_COLORS,
    # Optional features
    show_lines=True,         # Connect points per embryo over time
    x_col='predicted_stage_hpf',
    show_mean=True,          # Show mean trajectory per group
    output_path=Path('plot.html'),
)
```

**Key features:**
- Follows `facetted_plotting` naming conventions (`color_by`, `line_by`, `color_palette`)
- `coords` parameter (list of 3 strings) - cleaner than separate x/y/z
- Saves to HTML (interactive) and PNG (if kaleido installed)
- Optional trajectory lines and group means

#### 2. **PCA Embedding Module** (`src/analyze/trajectory_analysis/pca_embedding.py`)

Functions for PCA transformation with WT reference subtraction:

```python
# Step 1: FIT PCA (learn transformation from data)
pca, scaler, z_mu_cols = fit_pca_on_embeddings(
    df,
    z_mu_cols=None,  # Auto-detects z_mu_b* columns
    n_components=3
)

# Step 2: TRANSFORM (apply to any data)
df = transform_embeddings_to_pca(df, pca, scaler)
# Adds columns: PCA_1, PCA_2, PCA_3

# Step 3: Compute WT reference (time-binned average)
wt_ref = compute_wt_reference_by_time(
    df,
    pca_cols=['PCA_1', 'PCA_2', 'PCA_3'],
    wt_embryo_ids=wildtype_ids,
    bin_width=2.0
)

# Step 4: Subtract WT reference (get deviation)
df = subtract_wt_reference(df, wt_ref, ['PCA_1', 'PCA_2', 'PCA_3'])
# Adds columns: PCA_1_delta, PCA_2_delta, PCA_3_delta
```

**Why two functions (fit vs transform)?**
- `fit_pca_on_embeddings` - **LEARNS** the PCA axes from data (call ONCE)
- `transform_embeddings_to_pca` - **APPLIES** the same transformation (call on any data)
- Ensures all embryos are in the same coordinate system

#### 3. **Analysis Script** (`pca_phenotype_analysis.py`)

Complete workflow script that:
1. Loads data from experiments 20251121, 20251125
2. Assigns phenotype labels (CE, HTA, BA-rescue, non-penetrant)
3. Fits PCA on z_mu_b columns (80 biological VAE features)
4. Computes WT reference and deviation trajectories
5. Creates 6 interactive 3D visualizations with multiple coloring schemes

**Run it:**
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251230_pca_morphology_analysis
python pca_phenotype_analysis.py
```

### Results Generated

Output in `output/` directory:

| File | Description |
|------|-------------|
| `pca_raw_by_phenotype.html` | 3D scatter colored by **phenotype** (CE, HTA, BA-rescue, non-penetrant) |
| `pca_raw_by_stage.html` | 3D scatter colored by **developmental stage (hpf)** - continuous Viridis colorscale |
| `pca_raw_by_pair.html` | 3D scatter colored by **pair** (cross background) |
| `pca_raw_trajectories.html` | 3D scatter with **trajectory lines by phenotype** (shows phenotype-level movement patterns) |
| `pca_raw_individual_trajectories.html` | 3D scatter with **individual embryo trajectories** (each embryo is a unique color with traced path) |
| `pca_delta_3d.html` | 3D scatter of **PCA delta** (WT-subtracted, deviation space) colored by phenotype |

**Key Finding:**
- **3 PCA components capture 86% of variance**
  - PC1: 40.4% (likely reflects anterior-posterior or body size)
  - PC2: 25.0% (likely reflects curvature/body axis changes)
  - PC3: 20.6% (likely reflects fine morphological details)

**Data Summary:**
- 187 embryos total across 2 experiments (20251121, 20251125)
  - CE: 38 embryos
  - HTA: 24 embryos
  - BA-rescue: 7 embryos
  - non-penetrant: 118 embryos
- WT reference: 35 wildtype embryos, 58 time bins (11-125 hpf)
- Total timepoints: 23,074 measurements

**Visualization Features Added:**
- ✅ **Continuous coloring** - Color by predicted_stage_hpf to see temporal progression through PCA space
- ✅ **Categorical coloring** - Color by phenotype, pair, or embryo_id
- ✅ **Interactive trajectories** - Hover to see embryo_id, stage, phenotype, and PCA coordinates
- ✅ **Multiple trajectory views** - Both phenotype-level patterns and individual embryo trajectories

---

## 🚀 Phase B: DTW Clustering (READY TO START)

### Objective

Apply DTW clustering on PCA delta trajectories to test whether the 4 phenotypes (CE, HTA, BA-rescue, non-penetrant) recapitulate/separate as distinct clusters in morphology space.

### Workflow: 4 Steps

#### **Step 1: DTW Distance Computation**

Compute pairwise DTW distances between all embryo trajectories in PCA delta space:

```python
from src.analyze.trajectory_analysis import compute_trajectory_distances

# Load the processed dataframe (from pca_phenotype_analysis.py output)
# Or re-run steps 1-6 of pca_phenotype_analysis.py to regenerate it

# Compute pairwise DTW distances on PCA delta trajectories
D, embryo_ids, time_grid = compute_trajectory_distances(
    df,
    metrics=['PCA_1_delta', 'PCA_2_delta', 'PCA_3_delta'],
    time_col='predicted_stage_hpf',
    normalize=True,
    sakoe_chiba_radius=3
)

print(f"Distance matrix shape: {D.shape}")
print(f"Embryo count: {len(embryo_ids)}")
```

**What this does:**
- Computes DTW distance between every pair of embryos based on their PCA delta trajectories
- `normalize=True`: Standardizes each PC before DTW (recommended)
- `sakoe_chiba_radius=3`: Limits warping window (speedup, reasonable constraint)
- Returns: Distance matrix `D` (n_embryos × n_embryos), embryo IDs, and time grid

**Alternative: Use more PCA components**
```python
# For potentially better clustering, fit PCA with more components
# Then repeat steps 4-6 of pca_phenotype_analysis.py with n_components=5 or 10
```

#### **Step 2: Hierarchical Clustering with Bootstrap**

Cluster embryos and estimate robustness via bootstrapping:

```python
from src.analyze.trajectory_analysis import run_bootstrap_hierarchical

results = run_bootstrap_hierarchical(
    D,
    k=4,  # Number of clusters (match phenotype count)
    embryo_ids=embryo_ids,
    n_bootstrap=100
)

print(f"Modal cluster assignments: {results['modal_cluster']}")
print(f"Bootstrap support values: {results['bootstrap_support']}")
```

**What this does:**
- Runs hierarchical clustering k times with subsampled data (bootstrap)
- `modal_cluster`: Most frequent cluster assignment (consensus)
- `bootstrap_support`: How often each embryo gets its modal cluster (0-1, higher = more robust)
- Returns results dict with cluster assignments and support values

**Optional: Try different k values**
```python
# Test multiple cluster counts
for k in [3, 4, 5]:
    results_k = run_bootstrap_hierarchical(D, k=k, embryo_ids=embryo_ids, n_bootstrap=100)
    print(f"k={k}: cluster distribution = {np.bincount(results_k['modal_cluster'])}")
```

#### **Step 3: Assign Clusters Back to Data and Visualize**

Add cluster assignments to dataframe and create comparison visualizations:

```python
# Map cluster assignments back to main dataframe
cluster_map = {embryo_ids[i]: results['modal_cluster'][i] for i in range(len(embryo_ids))}
df['dtw_cluster'] = df[EMBRYO_ID_COL].map(cluster_map)

# Plot 1: DTW clusters in PCA delta space
fig_clusters = plot_3d_scatter(
    df,
    coords=['PCA_1_delta', 'PCA_2_delta', 'PCA_3_delta'],
    color_by='dtw_cluster',
    line_by='embryo_id',
    min_points_per_line=10,
    title='DTW Clusters in PCA Delta Space (Morphology-Based)'
)

# Plot 2: True phenotypes in same space (for comparison)
fig_phenotypes = plot_3d_scatter(
    df,
    coords=['PCA_1_delta', 'PCA_2_delta', 'PCA_3_delta'],
    color_by='phenotype',
    color_palette=PHENOTYPE_COLORS,
    color_order=PHENOTYPE_ORDER,
    line_by='embryo_id',
    min_points_per_line=10,
    title='True Phenotypes in PCA Delta Space (Ground Truth)'
)

# Plot 3: Side-by-side with trajectories
fig_clusters_traj = plot_3d_scatter(
    df,
    coords=['PCA_1_delta', 'PCA_2_delta', 'PCA_3_delta'],
    color_by='dtw_cluster',
    show_lines=True,
    x_col='predicted_stage_hpf',
    line_width=1.5,
    title='DTW Cluster Trajectories in PCA Delta Space'
)
```

**Save figures:**
```python
fig_clusters.write_html('output/dtw_clusters_pca_delta.html')
fig_phenotypes.write_html('output/phenotypes_pca_delta.html')
fig_clusters_traj.write_html('output/dtw_clusters_trajectories.html')
```

#### **Step 4: Quantify Cluster-Phenotype Agreement**

Measure how well DTW clusters match the true phenotypes:

```python
import pandas as pd
import numpy as np

# Get one row per embryo with cluster and phenotype
df_embryos = df.drop_duplicates('embryo_id')[
    ['embryo_id', 'dtw_cluster', 'phenotype']
].copy()

# === Confusion Matrix ===
confusion = pd.crosstab(
    df_embryos['phenotype'],
    df_embryos['dtw_cluster'],
    rownames=['True Phenotype'],
    colnames=['DTW Cluster'],
    margins=True
)
print("\n" + "="*50)
print("CONFUSION MATRIX: DTW Clusters vs True Phenotypes")
print("="*50)
print(confusion)

# === Per-Cluster Purity ===
print("\n" + "="*50)
print("CLUSTER PURITY ANALYSIS")
print("="*50)

cluster_purities = []
for cluster_id in sorted(df_embryos['dtw_cluster'].unique()):
    cluster_df = df_embryos[df_embryos['dtw_cluster'] == cluster_id]
    phenotype_counts = cluster_df['phenotype'].value_counts()
    dominant_pheno = phenotype_counts.index[0]
    dominant_count = phenotype_counts.iloc[0]
    total_in_cluster = len(cluster_df)
    purity = dominant_count / total_in_cluster

    cluster_purities.append(purity)
    print(f"\nCluster {cluster_id} (n={total_in_cluster}):")
    print(f"  Dominant phenotype: {dominant_pheno} ({dominant_count}/{total_in_cluster})")
    print(f"  Purity: {purity:.2%}")
    print(f"  Breakdown: {dict(phenotype_counts)}")

mean_purity = np.mean(cluster_purities)
print(f"\nMean cluster purity: {mean_purity:.2%}")

# === Phenotype-to-Cluster Mapping ===
print("\n" + "="*50)
print("PHENOTYPE-TO-CLUSTER MAPPING")
print("="*50)

for pheno in PHENOTYPE_ORDER:
    if pheno in df_embryos['phenotype'].values:
        pheno_df = df_embryos[df_embryos['phenotype'] == pheno]
        cluster_dist = pheno_df['dtw_cluster'].value_counts()
        main_cluster = cluster_dist.index[0]
        main_count = cluster_dist.iloc[0]
        purity = main_count / len(pheno_df)

        print(f"\n{pheno} (n={len(pheno_df)}):")
        print(f"  Main cluster: {main_cluster} ({main_count}/{len(pheno_df)})")
        print(f"  Cluster purity: {purity:.2%}")
        print(f"  Distribution: {dict(cluster_dist)}")

# === Summary Statistics ===
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Total embryos: {len(df_embryos)}")
print(f"Number of DTW clusters: {df_embryos['dtw_cluster'].nunique()}")
print(f"Number of phenotypes: {df_embryos['phenotype'].nunique()}")
print(f"Mean cluster purity: {mean_purity:.2%}")
```

**Expected outputs:**
- **High purity (>0.7):** Phenotypes separate well in morphology space
- **Low purity (<0.5):** Phenotypes don't differentiate in PCA space; may need different metrics
- **Mixed results:** Some phenotypes separate (e.g., CE distinct) while others mix

**What to look for:**
- Does CE (short) cluster separately from HTA/BA-rescue (curved)?
- Do HTA and BA-rescue cluster together or separately?
- Does non-penetrant scatter near the origin (near WT in delta space)?

---

## 📁 File Structure

```
/net/trapnell/vol1/home/mdcolon/proj/morphseq/
│
├── src/analyze/trajectory_analysis/
│   ├── plotting_3d.py              ← 3D Plotly plotting (supports continuous & categorical coloring)
│   ├── pca_embedding.py            ← PCA transformation utilities
│   ├── dtw_distance.py             ← DTW distance computation
│   ├── plot_config.py              ← PHENOTYPE_COLORS, PHENOTYPE_ORDER
│   └── __init__.py                 ← Exports all functions
│
└── results/mcolon/20251230_pca_morphology_analysis/
    ├── README.md                   ← THIS FILE (Handoff Document)
    ├── pca_phenotype_analysis.py   ← Phase A: Main analysis script (COMPLETE ✅)
    └── output/
        ├── pca_raw_by_phenotype.html          (colored by phenotype)
        ├── pca_raw_by_stage.html              (colored by developmental stage - continuous)
        ├── pca_raw_by_pair.html               (colored by pair)
        ├── pca_raw_trajectories.html          (phenotype-level trajectories)
        ├── pca_raw_individual_trajectories.html (individual embryo trajectories)
        ├── pca_delta_3d.html                  (WT-subtracted PCA)
        └── [Phase B outputs will be added here]
            ├── dtw_clusters_pca_delta.html
            ├── phenotypes_pca_delta.html
            ├── dtw_clusters_trajectories.html
            └── cluster_purity_analysis.txt
```

**Key file locations for Phase B:**
- Analysis script template: See "Phase B: DTW Clustering" section above
- Plotting function: `src/analyze/trajectory_analysis/plotting_3d.py` (line 90-397)
- DTW functions: `src/analyze/trajectory_analysis/dtw_distance.py`
- Clustering function: `src/analyze/trajectory_analysis/dtw_distance.py` (look for `run_bootstrap_hierarchical`)

---

## 🔑 Key Design Decisions

1. **PCA on z_mu_b only** (80 dims) - Consistent with b9d2_phenotype_comparison.py, excludes non-biological features
2. **WT reference = wildtype only** - Uses only b9d2_wildtype genotype (not non-penetrant hets)
3. **WT subtraction per time bin** - Captures deviation from normal development trajectory
4. **3 PCA components for visualization** - Always use 3 for 3D plots
5. **DTW on PCA delta** - Will cluster on deviation trajectories; can use more PCA dims if needed
6. **API consistency** - Follows `facetted_plotting.py` patterns (`color_by`, `line_by`, `coords` list)

---

## 🧪 Validation Checklist

### Phase A (Completed ✅)

- [x] PCA captures >80% variance (achieved 86%)
- [x] WT reference covers full time range (11-125 hpf, 58 bins)
- [x] All rows get WT reference (100% coverage)
- [x] 3D plots render correctly (6 HTML files saved)
- [x] Continuous coloring works (by predicted_stage_hpf)
- [x] Categorical coloring works (by phenotype, pair, embryo_id)
- [x] Trajectory lines render correctly (individual and phenotype-level)
- [x] Interactive hover shows coordinates and metadata
- [x] Default opacity matches original plotting functions (0.65)

### Phase B (Ready to Start 🚀)

- [ ] DTW distance matrix computed (Step 1)
- [ ] Hierarchical clustering with bootstrap completed (Step 2)
- [ ] Cluster-phenotype visualizations created (Step 3)
- [ ] Confusion matrix and purity analysis completed (Step 4)
- [ ] Results saved to output directory

---

## 🔗 Dependencies

**Existing modules used:**
- `src.analyze.trajectory_analysis.data_loading.load_experiment_dataframe`
- `src.analyze.trajectory_analysis.plot_config.PHENOTYPE_COLORS`
- `src.analyze.trajectory_analysis.facetted_plotting.plot_multimetric_trajectories`
- `results.mcolon.20251228_b9d2_phenotype_comparisons.b9d2_phenotype_comparison`

**New dependencies:**
- `sklearn.decomposition.PCA`
- `sklearn.preprocessing.StandardScaler`
- `plotly.graph_objects`

---

## 📊 Expected Outcomes (Phase B)

If phenotypes truly separate in morphology space:
- **CE (short)** should cluster separately (different PC than HTA/BA-rescue)
- **HTA/BA-rescue (curved)** may cluster together or separately depending on timing
- **non-penetrant** should scatter near WT reference (near origin in delta space)
- **Cluster purity >0.7** would indicate strong morphology-phenotype correspondence

If phenotypes DON'T separate:
- Clusters may mix phenotypes (low purity)
- Suggests phenotypes differ in ways not captured by VAE embeddings
- May need different metrics or time windows

---

## 🐛 Common Issues

### Issue: "No z_mu_b* columns found"
**Fix:** Check that DataFrame has VAE embedding columns. Load from `df03_final_output_with_latents_*.csv`

### Issue: "WT reference subtracted for X/Y rows (low %)"
**Fix:** Check `bin_width` matches between `compute_wt_reference_by_time` and `subtract_wt_reference`

### Issue: Plots show "No Data Available"
**Fix:** Check `min_points_per_line` - reduce if embryos have few timepoints

### Issue: PNG export fails
**Fix:** Install kaleido: `pip install kaleido` (HTML export always works)

---

## 📝 How to Pick Up Phase B (For Next Model)

### Quick Start (30 mins)

1. **Read this document** - You're doing this now! ✅
2. **Run Phase A (if needed):**
   ```bash
   cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251230_pca_morphology_analysis
   python pca_phenotype_analysis.py
   ```
   This generates `df` with all PCA columns and phenotype labels.

3. **Create Phase B script:**
   ```bash
   touch dtw_clustering_analysis.py  # Create new file
   ```
   Copy the code snippets from **"Phase B: DTW Clustering"** section above into this script.

4. **Run Phase B:**
   ```bash
   python dtw_clustering_analysis.py
   ```

### Code Template for Phase B Script

Save this as `dtw_clustering_analysis.py`:

```python
"""Phase B: DTW Clustering Analysis"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.analyze.trajectory_analysis import (
    compute_trajectory_distances,
    run_bootstrap_hierarchical,
    plot_3d_scatter,
)
from src.analyze.trajectory_analysis.plot_config import PHENOTYPE_COLORS, PHENOTYPE_ORDER

# Import Phase A script functions (or re-run steps 1-6)
from results.mcolon.20251228_b9d2_phenotype_comparisons.b9d2_phenotype_comparison import (
    load_all_phenotypes,
    load_experiment_data,
    extract_wildtype_embryos,
)
from results.mcolon.20251228_b9d2_phenotype_comparisons.b9d2_phenotype_distribution_by_pair import (
    prepare_phenotype_dataframe,
)
from src.analyze.trajectory_analysis import (
    fit_pca_on_embeddings,
    transform_embeddings_to_pca,
    compute_wt_reference_by_time,
    subtract_wt_reference,
)

# ============================================================================
# PHASE B: DTW CLUSTERING
# ============================================================================

def main():
    print("="*70)
    print("PHASE B: DTW CLUSTERING ON PCA DELTA TRAJECTORIES")
    print("="*70)

    # [Copy steps 1-6 from pca_phenotype_analysis.py OR load pre-computed df]
    # ... (data loading, PCA fitting, WT subtraction)

    # STEP 1: DTW Distance Computation
    print("\n[Step 1/4] Computing DTW distances...")
    D, embryo_ids, time_grid = compute_trajectory_distances(
        df,
        metrics=['PCA_1_delta', 'PCA_2_delta', 'PCA_3_delta'],
        time_col='predicted_stage_hpf',
        normalize=True,
        sakoe_chiba_radius=3
    )
    print(f"  Distance matrix shape: {D.shape}")

    # STEP 2: Hierarchical Clustering
    print("\n[Step 2/4] Running bootstrap hierarchical clustering...")
    results = run_bootstrap_hierarchical(
        D,
        k=4,
        embryo_ids=embryo_ids,
        n_bootstrap=100
    )
    print(f"  Clusters assigned: {len(np.unique(results['modal_cluster']))}")

    # STEP 3: Visualizations
    print("\n[Step 3/4] Creating visualizations...")
    cluster_map = {embryo_ids[i]: results['modal_cluster'][i] for i in range(len(embryo_ids))}
    df['dtw_cluster'] = df['embryo_id'].map(cluster_map)

    # [Create 3 plots as shown in Phase B section]

    # STEP 4: Analysis
    print("\n[Step 4/4] Computing cluster purity...")
    # [Confusion matrix, purity analysis as shown in Phase B section]

if __name__ == '__main__':
    main()
```

## 📝 Iteration Ideas for Phase B+

### To Improve Clustering
1. **More PCA dimensions** - Fit PCA with 5-10 components, use those for DTW
2. **Time windows** - Cluster early (11-40 hpf) vs late (60-125 hpf) separately
3. **Different metrics** - Use raw trajectory distances instead of delta
4. **K-selection** - Try k=3,4,5 and compare purity scores

### To Improve Visualizations
1. **Color by bootstrap support** - Shows which embryos are robustly clustered
2. **Split-view plots** - Side-by-side: clusters vs phenotypes
3. **Trajectory overlap** - Show how clusters move through PCA space

### If Results Are Poor
- Check if phenotypes actually differ in morphology (might need different metrics)
- Try non-Euclidean distance metrics (e.g., correlation-based)
- Separate wildtype from non-penetrant and re-cluster (might improve separation)

---

## 🔍 Quick Reference: Important Variables

**Column names:**
- `predicted_stage_hpf` - Developmental time
- `embryo_id` - Embryo identifier
- `phenotype` - Ground truth phenotype label
- `PCA_1`, `PCA_2`, `PCA_3` - Raw PCA coordinates
- `PCA_1_delta`, `PCA_2_delta`, `PCA_3_delta` - WT-subtracted coordinates

**Constants:**
- `PHENOTYPE_ORDER` = ['CE', 'HTA', 'BA_rescue', 'non_penetrant']
- `PHENOTYPE_COLORS` - Predefined color dict for phenotypes
- `TIME_COL` = 'predicted_stage_hpf'
- `EMBRYO_ID_COL` = 'embryo_id'

**Data summary:**
- 187 embryos, 23,074 timepoints
- 3 PCA components, 86% variance explained
- 35 WT reference embryos

---

## 🔗 References

- **Phenotype extraction:** `results/mcolon/20251219_b9d2_phenotype_extraction/`
- **DTW tutorial:** `results/mcolon/20251219_b9d2_phenotype_extraction/md_dtw_tutorial.ipynb`
- **Previous phenotype analysis:** `results/mcolon/20251228_b9d2_phenotype_comparisons/`
- **Plotting examples:** Check `plotting_3d.py` docstring for full API

---

## 📞 Contact & Status

**Status:** Phase A COMPLETE ✅ | Phase B READY 🚀
**Created:** 2025-12-30 by Claude Code
**Last Updated:** 2025-12-30 20:50 UTC

**What to do next:**
1. Create `dtw_clustering_analysis.py` using the template above
2. Run Phase B steps 1-4
3. Check `output/` for new visualizations and analysis results
4. Update this README with Phase B results

**Questions?**
- Check the error messages from the script - they're usually helpful
- Look at the Phase B code snippets for detailed comments
- Review `dtw_distance.py` and `plotting_3d.py` docstrings for function details
