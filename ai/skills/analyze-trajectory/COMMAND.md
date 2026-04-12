You are a morphseq trajectory analysis expert. When the user asks about DTW clustering, k-selection, or bootstrap projection, use the `src/analyze/trajectory_analysis/` module. Follow these rules exactly.

## Setup

```python
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

project_root = Path(__file__).resolve().parents[N]  # adjust N
sys.path.insert(0, str(project_root / "src"))

from analyze.trajectory_analysis.utilities.dtw_utils import compute_trajectory_distances
from analyze.trajectory_analysis.clustering.k_selection import run_k_selection_with_plots
from analyze.trajectory_analysis.clustering.bootstrap_clustering import run_bootstrap_projection_with_plots
from analyze.trajectory_analysis.viz.styling import (
    get_color_for_genotype, sort_genotypes_by_suffix, extract_genotype_suffix,
)
```

## Pipeline Overview

1. **Compute DTW distances** — `compute_trajectory_distances(df, metrics, ...)`
2. **Select k** — `run_k_selection_with_plots(df, D, embryo_ids, output_dir, ...)`
3. **Bootstrap projection** — `run_bootstrap_projection_with_plots(source_df, reference_df, output_dir, run_name, ...)`

## Key Functions

### `compute_trajectory_distances(df, metrics, time_col=, embryo_id_col=, time_window=, normalize=True, sakoe_chiba_radius=3, n_jobs=-1)`

Returns `(D, embryo_ids, metrics_array)`:
- `D` — symmetric distance matrix (n_embryos × n_embryos)
- `embryo_ids` — ordered list matching D rows/cols
- `metrics_array` — not commonly used

**Metrics:** List of column names. For multi-dimensional DTW, pass multiple columns. For single-feature DTW, pass one.

### `run_k_selection_with_plots(df, D, embryo_ids, output_dir, ...)`

Runs silhouette/gap analysis over `k_range`, saves plots + CSVs. Returns dict with best k and labels.

Key params: `method="hierarchical"` (or `"kmedoids"`), `n_bootstrap=100`, `enable_stage1_filtering=True` (IQR-based outlier removal), `generate_cluster_flow=True` (sankey plot across k values).

### `run_bootstrap_projection_with_plots(source_df, reference_df, output_dir, run_name, ...)`

Projects `source_df` embryos onto `reference_df` clusters via bootstrapped cross-DTW.

Key params:
- `labels_df` — DataFrame with cluster labels for reference embryos
- `cluster_col` — column name in labels_df with cluster assignments
- `method="nearest_neighbor"` — projection method
- `n_bootstrap=100`, `frac=0.8` — bootstrap params
- `classification="2d"` — 2D or 3D classification

## Genotype Styling

```python
color_lookup = {gt: get_color_for_genotype(gt) for gt in genotypes}
sorted_gts = sort_genotypes_by_suffix(genotypes)
suffix = extract_genotype_suffix("cep290_homozygous")  # -> "homozygous"
```

Default colors: wildtype=#2166AC, het=#F7B267, homo=#B2182B, crispant=#9467bd, unknown=#808080.

## Common Pattern

```python
# 1. Compute DTW
D, embryo_ids, _ = compute_trajectory_distances(
    df, metrics=["PCA1", "PCA2", "PCA3"],
    time_col="predicted_stage_hpf",
    time_window=(24, 72), normalize=True, sakoe_chiba_radius=3,
)

# 2. K-selection
k_results = run_k_selection_with_plots(
    df, D, embryo_ids, output_dir=output_dir / "k_selection",
    plotting_metrics=["PCA1", "PCA2", "PCA3"],
    k_range=[2, 3, 4, 5, 6], n_bootstrap=100,
)

# 3. Bootstrap projection
proj_results = run_bootstrap_projection_with_plots(
    source_df=target_df, reference_df=ref_df,
    output_dir=output_dir / "projection", run_name="cep290_projection",
    labels_df=ref_labels, cluster_col="cluster",
    metrics=["PCA1", "PCA2", "PCA3"],
    n_bootstrap=100, method="nearest_neighbor",
)
```
