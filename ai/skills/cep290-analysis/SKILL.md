# cep290-analysis — CEP290 Analysis Reference

## Overview

CEP290 (centrosomal protein 290) analysis in the morphseq pipeline. CEP290 mutations cause ciliopathy phenotypes detectable in zebrafish embryo morphology via the VAE-based morphseq system.

## Data Locations

### Build06 experiment CSVs
```
morphseq_playground/metadata/build06_output/df03_final_output_with_latents_{id}.csv
```

Known CEP290 experiment IDs: `20260208`, `20260210`.

### Reference clusters
```
results/mcolon/20251229_cep290_phenotype_extraction/final_data/
```

### Tutorial
```
src/analyze/tutorials/cep290_tutorial_notebook.ipynb
```

## Genotype Names (after normalization)

| Raw (common) | Normalized |
|---|---|
| CEP290_wildtype | cep290_wildtype |
| CEP290_heterozygous | cep290_heterozygous |
| CEP290_homozygous | cep290_homozygous |
| CEP290_unknown | cep290_unknown |
| CEP290_unkown | cep290_unknown (typo fix) |
| CEP290_homozyous | cep290_homozygous (typo fix) |

## Full Pipeline

### Phase 1: Genotype trends (quick screen)
- Feature-over-time plots (total_length_um, baseline_deviation_normalized)
- Raw genotype proportions
- AUROC classification (one-vs-all + each-vs-wildtype)
- Template: `results/mcolon/20260302_analyze_cep290_20260208_20260210/`

### Phase 2: PCA + deviation analysis
```python
from analyze.utils.pca import fit_transform_pca, compute_wt_reference_by_time, subtract_wt_reference

df, pca, scaler, pca_cols = fit_transform_pca(df, n_components=3)
wt_ref = compute_wt_reference_by_time(df, pca_cols, wt_genotype_pattern="wildtype")
df = subtract_wt_reference(df, wt_ref, pca_cols)
# Creates PCA1_delta, PCA2_delta, PCA3_delta
```

### Phase 3: DTW trajectory clustering
```python
from analyze.trajectory_analysis.utilities.dtw_utils import compute_trajectory_distances
from analyze.trajectory_analysis.clustering.k_selection import run_k_selection_with_plots

D, embryo_ids, _ = compute_trajectory_distances(
    df, metrics=pca_cols, time_col="predicted_stage_hpf",
    time_window=(24, 72), normalize=True, sakoe_chiba_radius=3,
)
k_results = run_k_selection_with_plots(
    df, D, embryo_ids, output_dir / "k_selection",
    plotting_metrics=pca_cols, k_range=[2, 3, 4, 5, 6], n_bootstrap=100,
)
```

### Phase 4: Bootstrap projection
```python
from analyze.trajectory_analysis.clustering.bootstrap_clustering import run_bootstrap_projection_with_plots

proj = run_bootstrap_projection_with_plots(
    source_df=target_df, reference_df=ref_df,
    output_dir=output_dir / "projection", run_name="cep290",
    labels_df=ref_labels, cluster_col="cluster",
    metrics=pca_cols, n_bootstrap=100, method="nearest_neighbor",
)
```

### Phase 5: Multi-feature classification
```python
feature_sets = {
    "embedding": "z_mu_b",
    "length": ["total_length_um"],
    "curvature": ["baseline_deviation_normalized"],
    "pca": pca_cols,
}
```

## Existing Scripts

| Script | Description |
|---|---|
| `01_plot_20260208_genotype_trends.py` | Single-experiment genotype trends for 20260208 |
| `02_plot_20260210_genotype_trends.py` | Single-experiment genotype trends for 20260210 |
| `03_plot_combined_genotype_trends.py` | Combined analysis of 20260208+20260210 |

All in `results/mcolon/20260302_analyze_cep290_20260208_20260210/`.
