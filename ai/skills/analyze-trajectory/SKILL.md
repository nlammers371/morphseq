# analyze-trajectory — Trajectory Analysis Reference

## Module: `analyze.trajectory_analysis.utilities.dtw_utils`

### `compute_trajectory_distances`
**File:** `src/analyze/trajectory_analysis/utilities/dtw_utils.py`

```python
def compute_trajectory_distances(
    df: pd.DataFrame,
    metrics: List[str],
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    time_window: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
    sakoe_chiba_radius: int = 3,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]
```

**Returns:** `(D, embryo_ids, metrics_array)`
- `D` — symmetric pairwise DTW distance matrix (n × n)
- `embryo_ids` — list of IDs, order matches D rows/cols
- `metrics_array` — raw metric arrays (rarely used directly)

**Notes:**
- `metrics` can be a single column (univariate DTW) or multiple (multi-dimensional DTW)
- `time_window=(start, end)` filters to HPF range before computing
- `normalize=True` z-scores each metric before DTW
- `sakoe_chiba_radius` constrains the DTW warping path (higher = more flexible)

---

## Module: `analyze.trajectory_analysis.clustering.k_selection`

### `run_k_selection_with_plots`
**File:** `src/analyze/trajectory_analysis/clustering/k_selection.py`

```python
def run_k_selection_with_plots(
    df: pd.DataFrame,
    D: np.ndarray,
    embryo_ids: List[str],
    output_dir: Path,
    plotting_metrics: List[str] = None,
    k_range: List[int] = [2, 3, 4, 5, 6],
    n_bootstrap: int = 100,
    method: str = 'hierarchical',           # 'hierarchical' | 'kmedoids'
    x_col: str = 'predicted_stage_hpf',
    metric_labels: Optional[Dict[str, str]] = None,
    enable_stage1_filtering: bool = True,
    stage1_method: str = 'iqr',
    iqr_multiplier: float = 2,
    k_neighbors: int = 5,
    filtering_hist_bins: int = 30,
    generate_cluster_flow: bool = True,
    cluster_flow_k_range: Optional[List[int]] = None,
    cluster_flow_title: str = "Cluster Flow Across k Values",
    cluster_flow_filename: str = "cluster_flow_sankey.html",
    verbose: bool = True,
) -> Dict[str, Any]
```

**Returns dict with:**
- `best_k`, `labels`, `silhouette_scores`, `gap_statistics`
- Saves to `output_dir/`: silhouette plots, gap statistic plots, cluster assignment CSVs, sankey flow

**Stage 1 filtering:** `enable_stage1_filtering=True` removes outlier embryos via IQR before clustering. Set `False` to keep all embryos.

---

## Module: `analyze.trajectory_analysis.clustering.bootstrap_clustering`

### `run_bootstrap_projection_with_plots`
**File:** `src/analyze/trajectory_analysis/clustering/bootstrap_clustering.py`

```python
def run_bootstrap_projection_with_plots(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    output_dir: Path,
    run_name: str,
    labels_df: Optional[pd.DataFrame] = None,
    *,
    id_col: str = "embryo_id",
    time_col: Optional[str] = "predicted_stage_hpf",
    cluster_col: Optional[str] = None,
    category_col: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    plotting_metrics: Optional[List[str]] = None,
    sakoe_chiba_radius: int = 20,
    n_bootstrap: int = 100,
    frac: float = 0.8,
    bootstrap_on: str = "reference",   # "reference" | "source"
    method: str = "nearest_neighbor",
    k: int = 5,
    classification: str = "2d",        # "2d" | "3d"
    normalize: bool = True,
    verbose: bool = True,
    save_outputs: bool = True,
) -> Dict[str, Any]
```

**Workflow:**
1. Computes cross-DTW between source and reference embryos
2. For each bootstrap iteration, samples `frac` of reference, finds nearest neighbors
3. Assigns source embryos to reference clusters via majority vote
4. Produces confidence scores, confusion-style plots, trajectory overlays

---

## Module: `analyze.trajectory_analysis.viz.styling`

### `get_color_for_genotype(genotype: str, suffix_colors: Optional[Dict] = None) -> str`
Maps genotype name → hex color via suffix detection.

Default suffix→color map:
| Suffix | Color |
|---|---|
| wildtype | #2E7D32 |
| heterozygous | #F57C00 |
| homozygous | #D32F2F |
| crispant | #7B1FA2 |
| unknown | #9E9E9E |

### `sort_genotypes_by_suffix(genotypes: List[str], suffix_order: Optional[List] = None) -> List[str]`
Default order: wildtype → heterozygous → homozygous → crispant → unknown.

### `extract_genotype_suffix(genotype: str) -> str`
Returns canonical suffix. Recognizes abbreviations: `wt`, `het`, `homo`, `hom`, `crisp`, `crispant`, `wildtype`, `heterozygous`, `homozygous`.
