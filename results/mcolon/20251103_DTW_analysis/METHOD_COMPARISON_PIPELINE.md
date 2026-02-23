# Clustering Method Comparison Pipeline

## Overview

This pipeline implements a forward-looking approach to clustering using **bootstrapped co-occurrence matrices** instead of single-run baseline clustering.

### The Philosophy Change

**Old Approach (Backwards):**
```
Baseline Clustering → Bootstrap for Validation → Compare → Use Baseline
```

**New Approach (Forward):**
```
Bootstrap Resampling → Co-occurrence Matrix → Multiple Clustering Methods → Select Best
```

## Scripts

### 1. `consensus_clustering.py`
**Standalone consensus clustering from bootstrapped co-occurrence**

Implements the simplest forward approach:
- Runs bootstrap resampling (100 iterations, 80% sampling)
- Extracts co-occurrence matrix (how often pairs cluster together)
- Applies hierarchical clustering to co-occurrence matrix
- Classifies membership based on stability

**Usage:**
```bash
python consensus_clustering.py
```

**Output:**
- `output/6_consensus/data/consensus_k{2,3,4,5}.pkl`
  - `labels`: Consensus cluster assignments
  - `coassoc`: Co-occurrence matrix
  - `membership`: Core/uncertain/outlier classification
  - Silhouette and ARI scores

### 2. `compare_clustering_methods.py`
**Compare 4 different clustering methods on the same data**

Tests hypothesis: **Consensus clustering produces more stable, confident clusters**

**Methods Compared:**

| Method | Input | Source | Type |
|--------|-------|--------|------|
| `kmedoids_dtw` | DTW distance matrix | Direct clustering | Baseline |
| `hierarchical_dtw` | DTW distance matrix | Direct clustering | Alternative |
| `kmedoids_consensus` | Co-occurrence (1-C) | Bootstrap-based | Consensus |
| `hierarchical_consensus` | Co-occurrence | Bootstrap-based | Consensus |

**Usage:**
```bash
python compare_clustering_methods.py
```

**Output:**

**16 Temporal Trends Plots** (4 methods × 4 k values):
```
output/7_method_comparison/plots/
  temporal_trends_kmedoids_dtw_k{2,3,4,5}.png
  temporal_trends_hierarchical_dtw_k{2,3,4,5}.png
  temporal_trends_kmedoids_consensus_k{2,3,4,5}.png
  temporal_trends_hierarchical_consensus_k{2,3,4,5}.png
```

Each plot shows:
- Individual trajectories (colored by membership: green=core, orange=uncertain, red=outlier)
- Cluster mean (black line)
- ±1 SD band (blue)
- Linear fit (red dashed)
- Per-cluster statistics (R², cluster size)

**Data File:**
```
output/7_method_comparison/data/method_comparison_all_k.pkl
  [k][method_name]:
    labels: cluster assignments
    silhouette: cluster quality score
    membership: core/uncertain/outlier classification
    medoids: (for k-medoids methods)
    n_core, n_uncertain, n_outlier: membership counts
    bootstrap_ari: stability metric
    ari_matrix: cross-method agreement (4×4)
    methods: list of method names
    coassoc: co-occurrence matrix
```

### 3. `visualize_method_comparison.py`
**Generate comparison visualizations**

Creates insights plots comparing methods across k values.

**Usage:**
```bash
python visualize_method_comparison.py
```

**Output Plots:**

#### 1. **Method Agreement Heatmaps** (4 total)
```
output/7_method_comparison/plots/
  method_agreement_k{2,3,4,5}.png
```
- 4×4 matrix showing ARI (Adjusted Rand Index) between method pairs
- Green = high agreement (methods find same clusters)
- Red = low agreement (methods disagree on structure)
- **Insight:** Identifies if consensus methods produce different structures

#### 2. **Aggregate Comparison (like membership_vs_k.png)**
```
output/7_method_comparison/plots/
  method_comparison_vs_k.png
```
3-panel figure showing:

**Panel 1: Silhouette Score** (cluster quality)
- Higher = better separated clusters
- Predicts: Direct methods may be sharper, consensus more stable

**Panel 2: Core Membership %** (confidence)
- Higher = more embryos in "core" category
- **Hypothesis:** Consensus methods have higher core %
- Indicates more confident, stable assignments

**Panel 3: Bootstrap Mean ARI** (stability)
- Higher = more consistent under resampling
- **Hypothesis:** Consensus methods more stable
- Tests robustness to data perturbations

#### 3. **Core Membership Distribution** (4 total)
```
output/7_method_comparison/plots/
  core_membership_by_method_k{2,3,4,5}.png
```
Stacked bar charts for each k showing:
- Green: Core members
- Orange: Uncertain members
- Red: Outliers
- **Insight:** Which methods identify more/fewer confident members

#### 4. **Summary Tables** (4 total)
```
output/7_method_comparison/plots/
  method_comparison_summary_k{2,3,4,5}.png
```
Table with for each method:
- Silhouette score
- Core membership %
- Bootstrap ARI
- Membership counts

---

## Hypothesis & Expected Results

### Main Hypothesis
**Consensus clustering (from bootstrapped co-occurrence) creates:**
1. **More stable clusters** - High bootstrap ARI
2. **More confident membership** - High % core members
3. **Comparable quality** - Similar or better silhouette scores

### Why Bootstrap-Based is Better

**Bootstrap Co-occurrence Matrix Captures:**
- Which embryos reliably cluster together (high C values)
- Which embryos are "on the boundary" (medium C values)
- Which embryos never cluster together (low C values)

**Membership Classification Reflects:**
- **Core**: High median C within cluster + good silhouette
- **Uncertain**: Medium C values
- **Outlier**: Very low total C values

**Result:** Outliers are identified from co-occurrence patterns, not post-hoc fit quality

### Metrics Explained

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **Silhouette** | [-1, 1] | Cluster separation quality. Higher = tighter, more separated clusters |
| **Core %** | [0, 100] | Percentage of embryos confidently assigned. Higher = more confident clustering |
| **Bootstrap ARI** | [-1, 1] | Agreement with bootstrap resamples. Higher = more stable under resampling |
| **ARI (between methods)** | [-1, 1] | Agreement between two clustering methods. 1.0 = identical clusters, 0 = random agreement |

---

## Workflow: How to Use

### Step 1: Run Method Comparison
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251103_DTW_analysis

# This generates 16 temporal trends plots (one per method/k combination)
# and saves method_comparison_all_k.pkl with all metrics
python compare_clustering_methods.py
```

**Expected output:**
- `output/7_method_comparison/plots/temporal_trends_*.png` (16 files)
- `output/7_method_comparison/data/method_comparison_all_k.pkl`

### Step 2: Generate Comparison Visualizations
```bash
# This loads the results and generates heatmaps, agreement plots, summary tables
python visualize_method_comparison.py
```

**Expected output:**
- `output/7_method_comparison/plots/method_agreement_k{2,3,4,5}.png` (4 files)
- `output/7_method_comparison/plots/method_comparison_vs_k.png` (1 file)
- `output/7_method_comparison/plots/core_membership_by_method_k{2,3,4,5}.png` (4 files)
- `output/7_method_comparison/plots/method_comparison_summary_k{2,3,4,5}.png` (4 files)

### Step 3: Interpret Results

Look for patterns:

1. **In temporal_trends plots:**
   - Do consensus methods show cleaner separation?
   - Do they have more/fewer outliers (red lines)?
   - Are core members (green lines) more stable?

2. **In method_agreement heatmap:**
   - Do direct methods (kmedoids_dtw vs hierarchical_dtw) agree?
   - Do consensus methods (kmedoids_consensus vs hierarchical_consensus) agree?
   - How much do direct vs consensus methods agree?

3. **In method_comparison_vs_k.png:**
   - **Panel 1 (Silhouette):** Which method has best cluster quality?
   - **Panel 2 (Core %):** Do consensus methods have higher confidence? ← **Key test of hypothesis**
   - **Panel 3 (Bootstrap ARI):** Are consensus methods more stable? ← **Key test of hypothesis**

4. **In core_membership plots:**
   - Which method has most core/fewest outliers?
   - Is distribution more balanced across methods?

---

## Implementation Details

### Bootstrap Parameters
From `config.py`:
- `N_BOOTSTRAP = 100` - 100 resampling iterations
- `BOOTSTRAP_FRAC = 0.8` - Each iteration samples 80% of embryos
- `CORE_THRESHOLD = 0.7` - Co-occurrence ≥0.7 = core member
- `OUTLIER_THRESHOLD = 0.3` - Total co-occurrence <0.3 = outlier

### Clustering Methods Implemented

**K-Medoids:**
```python
KMedoids(n_clusters=k, metric='precomputed', random_state=42)
```
- Returns actual embryo indices as cluster centers (interpretable)
- Works on distance matrices directly

**Hierarchical (Average Linkage):**
```python
AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
```
- Uses average pairwise distances between clusters (UPGMA)
- Builds dendrogram (shows hierarchical structure)

### Distance vs Similarity Conversion

- **DTW distance matrix D**: Already in distance form (higher = more different)
- **Co-occurrence matrix C**: Similarity form (higher = more similar)
- **For clustering C**: Convert to distance via `D_C = 1 - C`

---

## Key Insights & Interpretation

### What Each Method Captures

| Method | Captures | Interpretation |
|--------|----------|-----------------|
| **kmedoids_dtw** | Raw DTW distances | Which embryos have most different morphologies? |
| **hierarchical_dtw** | DTW distances + hierarchical structure | How do embryos hierarchically relate? |
| **kmedoids_consensus** | Stable co-occurrence patterns | Which embryos reliably group together? |
| **hierarchical_consensus** | Stable hierarchical structure from bootstrap | What's the stable biological structure? |

### Why Consensus Might Be Better

1. **Noise reduction**: Bootstrap averages out random variation
2. **Stability**: Co-occurrence reflects patterns across 100 resamples (robust)
3. **Natural outliers**: Embryos that never cluster together = true outliers
4. **Biological validity**: Captures reproducible patterns

### Why Direct Methods Might Be Better

1. **Fine structure**: May pick up subtle biological differences
2. **Deterministic**: No randomness, fully reproducible
3. **Interpretable medoids**: K-medoids gives actual exemplar embryos
4. **Simplicity**: No need for bootstrap, faster computation

---

## Expected Output Directory Structure

```
output/
├── 7_method_comparison/
│   ├── data/
│   │   └── method_comparison_all_k.pkl
│   └── plots/
│       ├── temporal_trends_kmedoids_dtw_k{2,3,4,5}.png
│       ├── temporal_trends_hierarchical_dtw_k{2,3,4,5}.png
│       ├── temporal_trends_kmedoids_consensus_k{2,3,4,5}.png
│       ├── temporal_trends_hierarchical_consensus_k{2,3,4,5}.png
│       ├── method_agreement_k{2,3,4,5}.png
│       ├── core_membership_by_method_k{2,3,4,5}.png
│       ├── method_comparison_summary_k{2,3,4,5}.png
│       └── method_comparison_vs_k.png
```

---

## References

- Bootstrap stability in clustering: Felsenstein (1985), Monti et al. (2003)
- Co-occurrence matrices in consensus clustering: Monti et al. (2003)
- ARI metric: Hubert & Arabie (1985)
- Silhouette score: Rousseeuw (1987)
