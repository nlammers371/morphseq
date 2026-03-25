You are a CEP290 morphseq analysis expert. When the user asks about CEP290 phenotype analysis, use this knowledge of file locations, data sources, and the full analysis pipeline.

## CEP290 Data Sources

### Build06 experiment data
```
morphseq_playground/metadata/build06_output/df03_final_output_with_latents_{experiment_id}.csv
```

Known CEP290 experiments: `20260208`, `20260210` (and earlier ones in build06).

### Reference clusters (from prior phenotype extraction)
```
results/mcolon/20251229_cep290_phenotype_extraction/final_data/
```
Contains reference cluster labels and trajectory data for bootstrap projection.

### Tutorial notebook
```
src/analyze/tutorials/cep290_tutorial_notebook.ipynb
```

## Full CEP290 Pipeline

### Step 1: Genotype trends (quick screen)
Use `/genotype-trends` to generate a genotype trend script. This gives you:
- Feature-over-time plots (length, curvature) by genotype
- Raw genotype proportions
- AUROC classification (one-vs-all, each-vs-wildtype)

### Step 2: PCA + wildtype deviation
```python
from analyze.utils.pca import fit_transform_pca, compute_wt_reference_by_time, subtract_wt_reference

df, pca, scaler, pca_cols = fit_transform_pca(df, n_components=3)
wt_ref = compute_wt_reference_by_time(df, pca_cols)
df = subtract_wt_reference(df, wt_ref, pca_cols)
```

### Step 3: DTW trajectory clustering
```python
from analyze.trajectory_analysis.utilities.dtw_utils import compute_trajectory_distances
from analyze.trajectory_analysis.clustering.k_selection import run_k_selection_with_plots

D, embryo_ids, _ = compute_trajectory_distances(
    df, metrics=pca_cols, time_window=(24, 72), normalize=True,
)
k_results = run_k_selection_with_plots(
    df, D, embryo_ids, output_dir / "k_selection",
    plotting_metrics=pca_cols, k_range=[2, 3, 4, 5, 6],
)
```

### Step 4: Bootstrap projection onto reference clusters
```python
from analyze.trajectory_analysis.clustering.bootstrap_clustering import run_bootstrap_projection_with_plots

proj = run_bootstrap_projection_with_plots(
    source_df=new_df, reference_df=ref_df,
    output_dir=output_dir / "projection", run_name="cep290",
    labels_df=ref_labels, cluster_col="cluster",
    metrics=pca_cols, n_bootstrap=100,
)
```

### Step 5: 4-mode classification
Run classification across 4 feature types:
1. **Embedding** (`"z_mu_b"`) — full VAE latent space
2. **Length** (`["total_length_um"]`) — body length
3. **Curvature** (`["baseline_deviation_normalized"]`) — body curvature
4. **PCA** (`pca_cols`) — PCA of embeddings

## Existing Analysis Scripts

- `results/mcolon/20260302_analyze_cep290_20260208_20260210/01_plot_20260208_genotype_trends.py`
- `results/mcolon/20260302_analyze_cep290_20260208_20260210/02_plot_20260210_genotype_trends.py`
- `results/mcolon/20260302_analyze_cep290_20260208_20260210/03_plot_combined_genotype_trends.py`

## Related Skills

- `/genotype-trends` — script generation recipe
- `/analyze-viz` — plotting functions
- `/analyze-classification` — classification functions
- `/analyze-trajectory` — DTW and clustering
- `/analyze-utils` — data loading and PCA
