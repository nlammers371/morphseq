# analyze-viz — Visualization Reference

## Module: `analyze.viz.plotting`

### `plot_feature_over_time`
**File:** `src/analyze/viz/plotting/feature_over_time.py`

```python
def plot_feature_over_time(
    df: pd.DataFrame,
    features: Optional[Union[str, List[str]]] = None,
    time_col: str = 'predicted_stage_hpf',
    id_col: str = 'embryo_id',
    color_by: Optional[str] = None,
    color_lookup: Optional[Dict[Any, str]] = None,
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    layout: Optional[FacetSpec] = None,
    show_individual: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',          # 'iqr' | 'std' | 'ci95'
    trend_statistic: str = 'median',  # 'median' | 'mean'
    trend_smooth_sigma: float = 1.5,
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    backend: str = 'plotly',          # 'plotly' | 'matplotlib' | 'both'
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    style: Optional[StyleSpec] = None,
    color_palette: Optional[List[str]] = None,
    feature: Optional[Union[str, List[str]]] = None,  # deprecated alias for features
    # Visibility knobs (no StyleSpec import needed)
    repeat_xlabels: bool = False,
    repeat_ylabels: bool = False,
    repeat_xticklabels: bool = True,
    repeat_yticklabels: bool = True,
    # Manual axis limits (blanket applied to all subplots)
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Any
```

**Returns:**
- `backend="plotly"` → `plotly.graph_objects.Figure`
- `backend="matplotlib"` → `matplotlib.figure.Figure`
- `backend="both"` → `{"plotly": Figure, "matplotlib": Figure}`

**Faceting:**
- When `features` is a list, features become separate rows automatically
- `facet_row` / `facet_col` add additional faceting by a DataFrame column
- `layout` (FacetSpec) for advanced grid control

**Shared axes behavior:**
- `layout.sharex` and `layout.sharey` now actually share axes in both Matplotlib + Plotly.
- `sharey=True` shares y-scale *within each row* (so columns in the same row have the same y limits).

**Tick numbers vs axis titles:**
- Axis titles are shown only on the outer axes by default (no repeated `"curvature"`).
- Tick-label numbers are shown on every subplot by default (useful when axes are shared).

**Manual axis limits:**
To force consistent scaling without wiring per-facet logic, pass `xlim` and/or `ylim`:

```python
fig = plot_feature_over_time(
    df,
    features="curvature",
    facet_col="genotype",
    xlim=(24, 75),
    ylim=(0, 1.0),
    backend="matplotlib",
)
```

## Jupyter reload (after changing renderers)

If you changed the faceting engine renderers and your notebook still shows old behavior, reload the renderer modules too:

```python
import importlib
import analyze.viz.plotting.faceting_engine.renderers.matplotlib as mpl_renderer
import analyze.viz.plotting.faceting_engine.renderers.plotly as plotly_renderer
import analyze.viz.plotting.feature_over_time as feature_over_time

for m in (mpl_renderer, plotly_renderer, feature_over_time):
    importlib.reload(m)
plot_feature_over_time = feature_over_time.plot_feature_over_time
```



---

### `plot_proportions`
**File:** `src/analyze/viz/plotting/proportions.py`

```python
def plot_proportions(
    df: pd.DataFrame,
    color_by_grouping: str,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    count_by: str = 'embryo_id',
    facet_order: Optional[Dict[str, List]] = None,
    color_order: Optional[List] = None,
    color_palette: Optional[Dict] = None,
    normalize: bool = True,
    bar_mode: str = 'grouped',   # 'grouped' | 'stacked'
    height_per_row: int = 250,
    width_per_col: int = 400,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_counts: bool = True,
) -> plt.Figure
```

---

### `plot_3d_scatter`
**File:** `src/analyze/viz/plotting/plotting_3d.py`

```python
def plot_3d_scatter(
    df: pd.DataFrame,
    coords: List[str],                          # e.g. ['PCA1', 'PCA2', 'PCA3']
    color_by: str = 'group',
    color_palette: Optional[Dict[str, str]] = None,
    color_order: Optional[List[str]] = None,
    color_continuous: bool = False,
    group_by: Optional[str] = None,
    colorscale: str = 'Viridis',
    colorbar_title: Optional[str] = None,
    line_by: str = 'id',
    min_points_per_line: int = 20,
    filter_groups: Optional[List[Any]] = None,
    filter_by_col: Optional[str] = None,
    downsample_frac: Optional[Dict[str, float]] = None,
    show_lines: bool = False,
    x_col: Optional[str] = 'predicted_stage_hpf',
    line_opacity: float = 0.3,
    line_width: float = 1.5,
    show_mean: bool = False,
    mean_line_width: int = 6,
    point_opacity: float = 0.65,
    point_size: int = 4,
    hover_cols: Optional[List[str]] = None,
    title: str = "3D Scatter Plot",
    output_path: Optional[Path] = None,
    axis_labels: Optional[Dict[str, str]] = None,
) -> go.Figure
```

---

## Module: `analyze.viz.hpf_coverage`

### `plot_experiment_time_coverage`
```python
def plot_experiment_time_coverage(
    df: pd.DataFrame,
    experiment_col: str = "experiment_id",
    hpf_col: str = "predicted_stage_hpf",
    embryo_col: str = "embryo_id",
    bin_width: float = 0.5,
    min_embryos_per_bin: int = 1,
    hpf_min: Optional[float] = None,
    hpf_max: Optional[float] = None,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]
```

Returns `(bins_mid, cover_df, cov_count)`.

### `plot_hpf_overlap_quick`
```python
def plot_hpf_overlap_quick(
    bins_mid: np.ndarray,
    cov_count: np.ndarray,
    cover_df: Optional[pd.DataFrame] = None,
    min_experiments: int = 5,
    show_heatmap: bool = True,
    max_experiments_heatmap: int = 80,
    title_prefix: str = "HPF overlap",
    coverage_plot_path: Optional[Path] = None,
    heatmap_path: Optional[Path] = None,
    show: bool = True,
) -> Tuple[Optional[float], Optional[float]]
```

Returns `(hpf_start, hpf_end)` — the interval where >= `min_experiments` have coverage.

---

## Module: `analyze.trajectory_analysis.viz.styling`

### `get_color_for_genotype(genotype, suffix_colors=None) -> str`
Maps genotype suffix → hex color via `analyze.viz.styling.color_mapping_config.GENOTYPE_SUFFIX_COLORS`.
Current defaults: wildtype=#2166AC, heterozygous=#F7B267, homozygous=#B2182B, crispant=#9467bd, unknown=#808080.

### `sort_genotypes_by_suffix(genotypes, suffix_order=None) -> List[str]`
Sorts by: wildtype → heterozygous → homozygous → crispant → unknown.

### `extract_genotype_suffix(genotype) -> str`
Returns canonical suffix: 'wildtype', 'heterozygous', 'homozygous', 'crispant', or 'unknown'.
