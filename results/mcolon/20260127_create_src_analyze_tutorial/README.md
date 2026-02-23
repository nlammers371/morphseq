# CEP290 Cross-Experiment Analysis Tutorial

**Purpose:** Complete morphseq analysis pipeline demonstrating CEP290 crispant penetrance trajectory analysis with cross-experiment projection.

**Data:** CEP290 crispant experiments (`20260122`, `20260124`)

**Output:** `output/figures/` and `output/results/`

---

## Key Finding

**Temporal coverage determines ability to detect penetrance dynamics.**

Cannot distinguish "Low_to_High" penetrance trajectories in experiments with limited temporal coverage (20260122: 11-47 hpf). Extended imaging to ~77 hpf (20260124: 27-77 hpf) is required to observe full penetrance progression from normal morphology to severe phenotypes.

**Implication:** When designing experiments to capture penetrance dynamics, ensure imaging window extends sufficiently late to observe phenotype emergence.

---

## Overview

This tutorial provides 6 executable Python scripts demonstrating the complete morphseq trajectory analysis pipeline. The canonical flow uses CEP290 data to demonstrate:

1. **Feature visualization** - Time series plotting of morphometric trajectories
2. **Dimensionality reduction** - PCA on VAE embeddings
3. **Within-experiment clustering** - DTW-based trajectory clustering
4. **Cross-experiment projection** - Projecting new experiments onto reference clusters
5. **Trajectory modeling** - Spline fits for projection-derived groups
6. **Statistical validation** - Classification tests on projected clusters

**Tutorial progression:**
```
01 → 02 → 03 → 04 → 05 → 06
```

---

## Tutorial Scripts

### 01: Feature Over Time (`01_feature_over_time.py`)

**Purpose:** Visualize morphometric trajectories over developmental time

**Key API:**
- `plot_feature_over_time()` with `facet_row`/`facet_col` parameters
- Multi-feature support: `feature=['feat1', 'feat2']`

**Data:**
- Experiments: 20260122 (11-47 hpf), 20260124 (27-77 hpf)
- Genotypes: `ab` (wildtype), `cep290_crispant`
- Features: `baseline_deviation_normalized` (curvature), `total_length_um` (length)

**Output:**
- `01_single_feature.html` - Basic single-feature plot
- `02a_baseline_deviation.html` - Curvature trajectories
- `02b_total_length.html` - Length trajectories
- `03_faceted_by_genotype.html` - Column faceting by genotype
- `04_multi_feature_by_genotype.html` - Multi-feature grid

**Key Observation:** CEP290 crispants show increased curvature over time, but temporal coverage affects when phenotypes are observable.

---

### 02: 3D Scatter + PCA (`02_3d_pca_scatter.py`)

**Purpose:** PCA dimensionality reduction and 3D trajectory visualization

**Key API:**
- `fit_transform_pca()` from `src.analyze.utils`
- `plot_3d_scatter()` with trajectory visualization options

**Demonstrates:**
- PCA on VAE embeddings (`z_mu_b` columns)
- Variance explained reporting
- 3D scatter with individual trajectories
- Mean trajectory per genotype
- 2D projections (PC1 vs PC2, PC2 vs PC3)

**Output:**
- `05_3d_scatter_points.html` - Points only
- `06_3d_scatter_trajectories.html` - With individual trajectories
- `07_3d_scatter_with_means.html` - With mean trajectories
- `08_2d_projection_pc1_pc2.html` - 2D projection
- `09_2d_projection_pc2_pc3.html` - Alternative 2D projection
- `pca_variance_explained.csv` - Variance explained per component

**Important:** `plot_3d_scatter()` uses `color_palette` parameter (not `color_lookup`).

---

### 03: DTW Clustering (`03_dtw_clustering.py`)

**Purpose:** Within-experiment DTW-based trajectory clustering

**Key API:**
- `extract_trajectories_df()` + `interpolate_to_common_grid_multi_df()`
- `filter_outliers_iqr()` for outlier removal
- `prepare_multivariate_array()` + `compute_md_dtw_distance_matrix()`
- `evaluate_k_range()` + `plot_k_selection()` for choosing k
- `run_bootstrap_hierarchical()` for robust clustering
- `analyze_bootstrap_results()` + `classify_membership_2d()`

**Pipeline:**
1. Extract and interpolate trajectories (curvature + length)
2. Filter outliers (IQR-based)
3. Compute multivariate DTW distance matrix
4. K-selection analysis (k=2 to k=7)
5. Bootstrap hierarchical clustering (k chosen based on metrics)
6. Posterior analysis and membership classification (core/uncertain/outlier)

**Output:**
- `10_k_selection.png` - K-selection metrics
- `dtw_distance_matrix.npy` - Distance matrix
- `embryo_ids.npy` - Embryo ID mapping
- `k_selection_results.csv` - K-selection metrics
- `bootstrap_results.pkl` - Full bootstrap clustering results
- `cluster_membership.csv` - Embryo cluster assignments with membership quality

**Note:** This demonstrates within-experiment clustering. For cross-experiment analysis, use projection approach (script 04).

---

### 04: Cluster Projection (`04_cluster_projection.py`)

**Purpose:** Project CEP290 experiments onto reference clusters and test for batch effects

**Key Methodology:**
- Bootstrap projection: Each experiment gets projected 100 times with resampled reference
- Posterior probability: Measures confidence in cluster assignments
- Batch effect testing: Chi-square test for experiment-cluster associations

**Reference Data:**
- 7 CEP290 experiments (20251229 analysis)
- Pre-computed clusters: "Not Penetrant", "Low_to_High", "High_to_Low"
- Path: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/`

**Input:**
- Reference: `embryo_cluster_labels.csv`, `embryo_data_with_labels.csv`
- Source: `qc_staged_20260122.csv`, `qc_staged_20260124.csv`

**Output:**
- `20260122_projection_bootstrap.csv` - Projection results for 20260122
- `20260124_projection_bootstrap.csv` - Projection results for 20260124
- `combined_projection_bootstrap.csv` - Combined results (KEY FILE for scripts 05-06)
- Distance diagnostics, posterior probabilities, frequency comparison tables

**Key Finding:**
- **Significant batch effects** (p = 0.0081) between experiments
- 20260122 (shorter window): 25.3% assigned to "High_to_Low"
- 20260124 (longer window): 7.1% assigned to "High_to_Low"
- This reflects temporal coverage differences, not true biological differences

**Interpretation:** Cannot reliably detect "Low_to_High" trajectories in 20260122 because imaging ends before phenotypes fully emerge.

---

### 05: Projection Splines (`05_projection_splines.py`)

**Purpose:** Fit spline trajectories to projection-derived cluster groups

**Key API:**
- Spline fitting on time series data
- Cluster-specific trajectory modeling

**Input:**
- `output/figures/04/projection_results/combined_projection_bootstrap.csv`

**Output:**
- `05_projection_splines_by_cluster.csv` - Spline coordinates
- `05_projection_splines_by_cluster.pkl` - Spline objects
- `05_projection_splines_by_cluster.png` - Visualization

**Purpose:** Model expected trajectories for each projected cluster to understand phenotype dynamics.

---

### 06: Projection Classification Test (`06_projection_classification_test.py`)

**Purpose:** Statistical validation of projection-derived cluster assignments

**Key API:**
- `run_classification_test()` for permutation-based testing

**Tests Performed:**

1. **One-vs-rest for cluster_label** (per experiment)
   - Tests if each cluster is distinguishable from others
   - Separate analysis for 20260122 and 20260124

2. **cep290_crispant vs ab** (per experiment)
   - Tests genotype distinguishability
   - Assesses penetrance (not all crispants show phenotype)

3. **20260122 only: Not Penetrant crispants vs Not Penetrant ab**
   - Tests if "Not Penetrant" crispants are truly indistinguishable from wildtype
   - Critical for validating penetrance classification

**Input:**
- `output/figures/04/projection_results/20260122_projection_bootstrap.csv`
- `output/figures/04/projection_results/20260124_projection_bootstrap.csv`

**Output:**
- `20260122_clusterlabel_ovr.csv` - One-vs-rest results for 20260122
- `20260124_clusterlabel_ovr.csv` - One-vs-rest results for 20260124
- `20260122_geno_crispant_vs_ab.csv` - Genotype comparison for 20260122
- `20260124_geno_crispant_vs_ab.csv` - Genotype comparison for 20260124
- `20260122_not_penetrant_crispant_vs_ab.csv` - Not Penetrant validation

**Interpretation:**
- AUROC > 0.7: Strong separability
- AUROC = 0.5: Indistinguishable (random chance)
- p_value < 0.05: Statistically significant
- Time-resolved: Shows WHEN phenotypes diverge

---

## Workflow Dependencies

**Linear Flow:**
```
01 → 02 → 03 (exploratory, establish baseline)
              ↓
             04 (KEY: project onto reference)
              ↓
         ┌────┴────┐
         ↓         ↓
        05        06
     (splines) (stats)
```

**Key Files:**
- Script 04 produces: `combined_projection_bootstrap.csv` (used by 05, 06)
- Scripts 01-03: Independent exploratory analysis
- Scripts 05-06: Depend on script 04 output

---

## Running the Tutorial

**Sequential execution:**

```bash
cd results/mcolon/20260127_create_src_analyze_tutorial/

# Exploratory analysis
python 01_feature_over_time.py
python 02_3d_pca_scatter.py
python 03_dtw_clustering.py

# Cross-experiment projection (KEY STEP)
python 04_cluster_projection.py

# Downstream analysis on projected clusters
python 05_projection_splines.py
python 06_projection_classification_test.py
```

---

## Key API Patterns

### Data Loading

```python
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'

df1 = pd.read_csv(meta_dir / 'qc_staged_20260122.csv')
df2 = pd.read_csv(meta_dir / 'qc_staged_20260124.csv')
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['use_embryo_flag']].copy()
```

### Feature Visualization

```python
from src.analyze.viz.plotting import plot_feature_over_time

# Single feature
fig = plot_feature_over_time(
    df,
    feature='baseline_deviation_normalized',
    color_by='genotype',
    color_lookup=color_dict,
    backend='plotly',  # 'plotly', 'matplotlib', 'both'
)

# Multi-feature
fig = plot_feature_over_time(
    df,
    feature=['baseline_deviation_normalized', 'total_length_um'],
    color_by='genotype',
    color_lookup=color_dict,
    facet_col='genotype',
)
```

### PCA and 3D Visualization

```python
from src.analyze.utils import fit_transform_pca
from src.analyze.viz.plotting import plot_3d_scatter

# PCA
df_pca, pca, scaler, z_mu_cols = fit_transform_pca(df, n_components=3)

# 3D scatter with trajectories
fig = plot_3d_scatter(
    df_pca,
    coords=['PCA_1', 'PCA_2', 'PCA_3'],
    color_by='genotype',
    color_palette=color_dict,
    line_by='embryo_id',
    show_lines=True,
    x_col='time_hpf',
    show_mean=True,
)
```

### DTW Clustering

```python
from src.analyze.trajectory_analysis.utilities import (
    extract_trajectories_df,
    interpolate_to_common_grid_multi_df,
)
from src.analyze.trajectory_analysis.distance import (
    prepare_multivariate_array,
    compute_md_dtw_distance_matrix,
)
from src.analyze.trajectory_analysis.clustering import (
    evaluate_k_range,
    run_bootstrap_hierarchical,
    analyze_bootstrap_results,
)

# Interpolate
df_interp = interpolate_to_common_grid_multi_df(
    df,
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    grid_step=0.5,
    time_col='predicted_stage_hpf',
)

# Distance matrix
X, embryo_ids, feature_names = prepare_multivariate_array(
    df_interp,
    features=['baseline_deviation_normalized', 'total_length_um'],
)
D = compute_md_dtw_distance_matrix(X, n_jobs=-1)

# Clustering
bootstrap_results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids)
posterior_results = analyze_bootstrap_results(bootstrap_results)
```

### Cross-Experiment Projection

See `04_cluster_projection.py` for complete projection methodology. Key steps:

1. Load reference clusters and data
2. For each source experiment:
   - Bootstrap resample reference
   - Compute DTW distances to reference clusters
   - Assign to nearest cluster
   - Aggregate posterior probabilities
3. Test for batch effects (chi-square on experiment × cluster contingency)

---

## Archive Directories

### `_archive_b9d2_tutorial/`

Original B9D2 exploratory analysis (scripts 04-09). Demonstrates within-experiment clustering workflow but no longer canonical. See archive README for details.

### `_investigation_dtw_methods/`

DTW method comparison investigation (scripts 04b-04h). Tested multivariate vs per-metric DTW for cross-experiment robustness. Key finding: **multivariate DTW wins** (11.6% disagreement vs 20.4%). See archive README for complete analysis.

---

## Data Sources

**CEP290 Experiments:**
- Path: `morphseq_playground/metadata/build04_output/`
- Files: `qc_staged_20260122.csv`, `qc_staged_20260124.csv`
- 20260122: 11-47 hpf (limited temporal coverage)
- 20260124: 27-77 hpf (extended temporal coverage)

**Reference Clusters:**
- Path: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/`
- Files: `embryo_cluster_labels.csv`, `embryo_data_with_labels.csv`
- 7 CEP290 experiments with pre-computed clusters

**Genotypes:**
- `ab`: Wildtype control
- `cep290_crispant`: CRISPR-induced cep290 mutants (incomplete penetrance)

**Key Features:**
- `baseline_deviation_normalized`: Normalized tail curvature (primary phenotype metric)
- `total_length_um`: Body axis length (secondary metric)
- `z_mu_b`: VAE embedding columns (for classification tests)

---

## Key Insights from CEP290 Analysis

### 1. Temporal Coverage is Critical

**Finding:** 20260122 (11-47 hpf) cannot reliably detect "Low_to_High" penetrance trajectories because imaging ends before phenotypes fully emerge.

**Evidence:**
- 20260122: 25.3% assigned to "High_to_Low" (unexpected)
- 20260124: 7.1% assigned to "High_to_Low" (expected)
- Significant batch effect (p = 0.0081)

**Interpretation:** The "High_to_Low" assignments in 20260122 are artifacts of incomplete temporal coverage. Embryos that would eventually show "Low_to_High" progression appear "High_to_Low" when imaging stops too early.

### 2. Incomplete Penetrance

Not all cep290 crispants show severe phenotypes. Script 06 validates that "Not Penetrant" crispants are genuinely indistinguishable from wildtype controls.

### 3. Cross-Experiment Projection is Robust

Despite batch effects from temporal coverage, the projection methodology provides interpretable results when properly contextualized. The key is understanding *why* batch effects occur (biological vs technical).

---

## Converting to Jupyter Notebooks

Each script can be converted to notebook format:

1. **Section headers** → Markdown cells
2. **Code blocks** → Code cells
3. **Figure outputs** → Replace `.write_html()` with `.show()`

**Example notebook pattern:**

```python
# Cell 1: Imports
import pandas as pd
from pathlib import Path
from src.analyze.viz.plotting import plot_feature_over_time

# Cell 2: Load data
project_root = Path.cwd().parents[3]
meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'
df1 = pd.read_csv(meta_dir / 'qc_staged_20260122.csv')
df2 = pd.read_csv(meta_dir / 'qc_staged_20260124.csv')
df = pd.concat([df1, df2])
df = df[df['use_embryo_flag']].copy()

# Cell 3: Plot
fig = plot_feature_over_time(df, feature='baseline_deviation_normalized', ...)
fig.show()  # Interactive in notebook
```

---

## File Structure

```
results/mcolon/20260127_create_src_analyze_tutorial/
├── README.md (this file)
├── README_tutorial_04.md (detailed documentation for script 04)
├── 01_feature_over_time.py
├── 02_3d_pca_scatter.py
├── 03_dtw_clustering.py
├── 04_cluster_projection.py (KEY: cross-experiment projection)
├── 05_projection_splines.py
├── 06_projection_classification_test.py
├── projection_utils.py (utilities for script 04)
├── _archive_b9d2_tutorial/
│   ├── README.md
│   ├── 04_cluster_labeling.py
│   ├── 05_faceted_feature_plots.py
│   ├── 06_proportions.py
│   ├── 07_spline_per_cluster.py
│   ├── 08_difference_detection.py
│   └── 09_plot_results.py
├── _investigation_dtw_methods/
│   ├── README.md
│   ├── FINAL_CONCLUSIONS.md
│   ├── README_04g_04h.md
│   ├── IMPLEMENTATION_STATUS.md
│   ├── RUN_04h.md
│   ├── 04b_compare_clustering_methods.py
│   ├── 04c_distance_matrix_test.py
│   ├── 04d_direct_distance_comparison.py
│   ├── 04e_normalization_alternatives.py
│   ├── 04f_clustering_normalization_test.py
│   ├── 04g_per_metric_dtw_combination.py
│   └── 04h_cross_experiment_validation.py
└── output/
    ├── figures/  # HTML (Plotly) and PNG (Matplotlib) outputs
    └── results/  # CSV and pickle files
```

---

## Next Steps

### For Production Pipelines

1. **Automated projection**: Use script 04 methodology to project new experiments onto established reference clusters
2. **Temporal coverage standardization**: Design experiments with sufficient imaging windows
3. **Batch effect monitoring**: Track experiment-cluster associations to detect technical artifacts

### For Analysis Extensions

1. **Multi-gene comparison**: Compare penetrance dynamics across different ciliopathy genes
2. **Dose-response analysis**: Relate crispant editing efficiency to phenotype severity
3. **Developmental timing**: Identify critical developmental windows for phenotype emergence

### For Claude Skills

Use this tutorial as API reference for building analysis skills:
1. **Data loading skill**: Load experiments by ID or date range
2. **Visualization skill**: Generate plots from user-specified features
3. **Clustering skill**: Run DTW clustering with user-specified parameters
4. **Projection skill**: Project new experiments onto reference clusters

---

## Related Analyses

- **TMEM67 cluster projection**: `results/mcolon/20260104_tmem67_cluster_projection_to_cep290/`
- **B9D2 phenotype comparison**: `results/mcolon/20251228_b9d2_phenotype_comparisons/`
- **CEP290 phenotype extraction**: `results/mcolon/20251229_cep290_phenotype_extraction/`

---

**Author:** Generated via Claude Code
**Date:** 2026-01-27 (updated 2026-02-05)
**Version:** 2.0 - CEP290 canonical flow
