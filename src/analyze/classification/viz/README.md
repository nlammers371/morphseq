# classification/viz

Plotting helpers for `ClassificationAnalysis` results. All functions are importable from `analyze.classification.viz` or directly from the submodule.

For the underlying compute API (producing the inputs to these plots), see [`../README.md`](../README.md).

---

## I want to see X → use Y

| Goal | Function | Required input | Notes |
|---|---|---|---|
| AUROC curves over time | `plot_aurocs_over_time` / `result.plot_aurocs()` | `result.scores` | Single or multiple curves; facetable; plotly or matplotlib backend. |
| Faceted AUROC heatmap (rows/cols × time) | `plot_auroc_heatmaps` | `result.scores` | Black-bordered cells mark significance. |
| Confusion matrices | `plot_confusion` / `result.plot_confusion()` | `result.layers["confusion"]` | Always available (confusion is always computed). |
| Per-embryo signed-margin trajectories | `plot_margin_trends` | `result.layers["predictions"]` (needs `save_predictions=True`) | Per-embryo "did the classifier get this embryo right, when?" lines colored by mean margin. |
| Per-embryo phenotype fingerprint (all-pairs heatmap) | `plot_pairwise_coordinate_heatmap` | `result.layers["raw_contrast_scores_long"]` (needs `save_contrast_coordinates=True`) | Where one embryo falls on every pairwise classifier simultaneously. |
| Phenotype emergence heatmap (static) | `plot_emergence_heatmap` | `EmergenceData` from `compute_emergence_data(result.scores, ...)` | Per-AUROC-level onset matrix. |
| Phenotype emergence — interactive explorer | `render_emergence_html` | `EmergenceData` | Self-contained HTML, no server; D3 + data inlined. |
| Wrong-rate heatmap (embryos × time) | `plot_wrongness_heatmap` | misclassification pipeline outputs | From misclass deep-dive pipeline. |
| Per-embryo deep dive (prediction timeline + probability traces) | `plot_embryo_deep_dive` | misclassification pipeline outputs | One embryo, full story. |
| Wrong-rate violins by group | `plot_wrong_rate_distributions` | misclassification pipeline outputs | |
| Confusion profile by true class | `plot_confusion_profile` | misclassification pipeline outputs | |
| Gallery of top-N misclassified embryos | `plot_flagged_embryo_gallery` | misclassification pipeline outputs | |
| Multi-metric overlay (low-level) | `plot_multiple_aurocs` / `plot_auroc_with_null` | Dict of AUROC DataFrames | Prefer `plot_aurocs_over_time` for most cases. |
| Side-by-side feature-type comparison | `plot_feature_comparison_grid` | `dict[str, MulticlassOVRResults]` | Legacy input type; still useful for multi-feature talk figures. |

---

## Submodule map

| File | Contains |
|---|---|
| `auroc_over_time.py` | `plot_aurocs_over_time` — faceted AUROC-vs-time with null bands and significance markers. |
| `heatmaps.py` | `plot_auroc_heatmaps` — faceted AUROC-as-heatmap with auto-inferred facet axes. |
| `confusion.py` | `plot_confusion` — confusion matrices from `layers["confusion"]`. |
| `misclassification.py` | `plot_margin_trends` + the misclassification deep-dive gallery (wrongness heatmap, embryo deep dive, wrong-rate violin, confusion profile, flagged gallery). |
| `pairwise_coordinates.py` | `plot_pairwise_coordinate_heatmap` — per-embryo all-pairs fingerprint. |
| `emergence.py` | `EmergenceData`, `compute_emergence_data`, `plot_emergence_heatmap`, `render_emergence_html`, `render_emergence_html_from_scores`. |
| `trajectory.py` | Cluster-level feature trends and rolling-window significance plots (used by misclass pipeline analysis). |
| `classification.py` | Low-level primitives (`plot_auroc_with_null`, `plot_multiple_aurocs`, `plot_feature_comparison_grid`, `plot_multiclass_ovr_aurocs`). |
| `utils.py` | Internal validators (`validate_required_columns`, `validate_margin_range`, `validate_unique_embryo_x`). |

---

## Margin conventions

Shared by `plot_margin_trends`, `plot_pairwise_coordinate_heatmap`, and the misclassification pipeline.

| Function | Range | Sign meaning |
|---|---|---|
| `class_signed_margin(p_pos)` | `[-1, 1]` | `+1` = strongest positive-class support. Formula: `2*p_pos - 1`. |
| `truth_signed_margin(p_pos, y_true)` | `[-1, 1]` | `+1` = most correct. |
| `coerce_margin_range(arr)` | `[-1, 1]` | Rescales legacy `[-0.5, 0.5]` saves. Pass loaded margins through this. |

`predictions.parquet` stores `truth_signed_margin`. `raw_contrast_scores_long` stores `class_signed_margin`.

---

## `plot_aurocs_over_time`

**File:** `auroc_over_time.py`

```python
plot_aurocs_over_time(
    results,                             # ClassificationAnalysis | pd.DataFrame
    *,
    time_col: str = "time_bin_center",
    auroc_col: str = "auroc_obs",
    curve_col: str = "positive",         # column that identifies each curve
    facet_row: str | None = None,
    facet_col: str | None = None,
    color_lookup: dict | None = None,
    show_null_band: bool = False,
    show_significance: bool = True,
    pval_col: str = "pval",
    sig_threshold: float = 0.01,
    show_chance_line: bool = True,
    ylim: tuple[float, float] = (0.3, 1.05),
    backend: str = "plotly",             # "plotly" | "matplotlib" | "both"
    output_path: str | Path | None = None,
) -> plotly.Figure | matplotlib.figure.Figure | dict
```

```python
from analyze.classification.viz import plot_aurocs_over_time

plot_aurocs_over_time(
    result,
    curve_col="comparison_id",
    sig_threshold=0.01,
    backend="matplotlib",
    output_path="aurocs.png",
)
```

---

## `plot_auroc_heatmaps`

**File:** `heatmaps.py`

Faceted heatmap of AUROC scores over time. Cells colored by AUROC; significant cells (p ≤ threshold) get a black border. Facet axes auto-inferred from data when not specified.

```python
plot_auroc_heatmaps(
    results,                               # ClassificationAnalysis | pd.DataFrame
    *,
    heatmap_row: str = "positive_label",
    heatmap_col: str = "time_bin_center",
    facet_row: str | None = None,           # None → auto-infer (feature_set if it varies)
    facet_col: str | None = None,           # None → auto-infer (negative_label if it varies)
    heatmap_row_order: Sequence | None = None,
    heatmap_col_order: Sequence | None = None,
    facet_row_order: Sequence | None = None,
    facet_col_order: Sequence | None = None,
    auroc_col: str = "auroc_obs",
    pval_col: str = "pval",
    show_significance: bool = True,
    sig_threshold: float = 0.01,
    cmap: str = "BuPu",
    vcenter: float | None = None,
    vmin: float = 0.4,
    vmax: float = 1.0,
    show_annotations: bool = False,
    title: str = "AUROC Heatmap",
    backend: str = "matplotlib",
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure | plotly.Figure | dict
```

```python
plot_auroc_heatmaps(result, output_path="auroc_heatmap.png")
plot_auroc_heatmaps(result, facet_row="feature_set", sig_threshold=0.05,
                    output_path="auroc_by_feature.png")
```

---

## `plot_confusion`

**File:** `confusion.py`

```python
plot_confusion(
    scores: pd.DataFrame,
    confusion: pd.DataFrame,
    *,
    feature_set: str | None = None,
    time_range: tuple[float, float] | None = None,
    backend: str = "matplotlib",
    output_path: str | Path | None = None,
    **kwargs,
) -> matplotlib.figure.Figure | plotly.Figure
```

```python
plot_confusion(
    result.scores, result.layers["confusion"],
    feature_set="emb", time_range=(24.0, 72.0),
    output_path="confusion.png",
)
```

---

## `plot_margin_trends`

**File:** `misclassification.py`

Per-embryo signed-margin trajectory for a binary comparison. Two panels: negative group left, positive right. Lines colored by mean margin on RdBu_r.

**Requires** `predictions.parquet` — set `save_predictions=True` (or a `save_dir`) on `run_classification`.

```python
plot_margin_trends(
    df,
    *,
    # Selector — exactly one mode
    comparison_id: str | None = None,        # pipeline mode: filters predictions by comparison_id
    positive_label: str | None = None,       # explicit mode: right panel
    negative_label: str | None = None,       # explicit mode: left panel
    # Secondary filter
    feature_col: str = "feature_set",
    feature_id: str | None = None,
    # Column names
    margin_col: str = "truth_signed_margin", # must be in [-1, 1]
    x_col: str = "time_bin_center",           # one value per (embryo, x)
    embryo_col: str = "embryo_id",
    # Display
    color_mode: str = "continuous",           # "continuous" | "discrete"
    discrete_class_lookup: dict | None = None,
    discrete_class_colors: dict | None = None,
    max_embryos: int = 30,
    vmin: float = -1.0,
    vmax: float = 1.0,
    time_window: tuple[float, float] | None = None,
    output_path: Path | None = None,
) -> matplotlib.figure.Figure
```

```python
from analyze.classification.viz.misclassification import plot_margin_trends

predictions = result.layers["predictions"]

# Pipeline mode
plot_margin_trends(predictions, comparison_id="inj_ctrl__vs__pbx4_crispant",
                   feature_id="vae", output_path="margin_trends.png")

# Explicit-label mode (ad-hoc df)
plot_margin_trends(df, positive_label="pbx4_crispant", negative_label="inj_ctrl",
                   output_path="margin_trends.png")

# Loop over all comparisons
for cid in sorted(predictions["comparison_id"].unique()):
    plot_margin_trends(predictions, comparison_id=cid, feature_id="vae",
                       output_path=figures_dir / f"margin_{cid}.png")
```

---

## `plot_pairwise_coordinate_heatmap`

**File:** `pairwise_coordinates.py`

Per-embryo phenotypic fingerprint: upper-triangle heatmap showing where one embryo falls on every pairwise classifier simultaneously. Rows = positive class (A), columns = negative class (B). Cell = time-averaged `class_signed_margin` on the A-vs-B classifier. Lower triangle and diagonal are masked.

**Requires** `raw_contrast_scores_long` — set `save_contrast_coordinates=True` on `run_classification`.

```python
plot_pairwise_coordinate_heatmap(
    df,                                         # long-form scores
    sample_id: str,
    *,
    id_col: str = "embryo_id",
    positive_label_col: str = "positive_label",
    negative_label_col: str = "negative_label",
    time_col: str = "time_bin",
    margin_col: str = "class_signed_margin",
    label_order: list[str] | None = None,       # None = alphabetical
    positive_labels: list[str] | None = None,
    negative_labels: list[str] | None = None,
    time_bins: list | None = None,
    vmin: float = -1.0,
    vcenter: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    title: str | None = None,
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure
```

```python
scores_long = result.layers["raw_contrast_scores_long"]
plot_pairwise_coordinate_heatmap(scores_long, "20260304_A01_e01",
                                 output_path="pairwise_coords.png")
```

---

## Emergence

Two-step API. Compute once, render as many views as you want.

### `compute_emergence_data`

**File:** `emergence.py`

Tidy pairwise `scores` DataFrame → `EmergenceData` (onset matrices across all AUROC levels). `class_order` is required and never inferred.

```python
compute_emergence_data(
    scores_df,
    class_order,                             # required — canonical ordered class list
    *,
    auroc_levels: Mapping[str, float] | None = None,   # {name: threshold}; default includes none/0.60/0.65/0.70
    p_sep: float = 0.05,
    p_ns: float = 0.10,
    subsequent_frac: float = 0.40,
    # Column-name overrides (defaults match run_classification output)
    time_col: str = "time_bin_center",
    positive_class_col: str = "positive_label",
    negative_class_col: str = "negative_label",
    auroc_col: str = "auroc_obs",
    pvalue_col: str = "pval",
) -> EmergenceData
```

`EmergenceData` is a frozen dataclass:

```python
@dataclass(frozen=True)
class EmergenceData:
    onset_matrices_by_level: dict[str, dict[str, dict[str, float | None]]]
    class_order: list[str]       # canonical axis order
    auroc_levels: list[str]      # ordered level names
    color_scale_min: float       # global min finite onset across levels
    color_scale_max: float
```

Invariants: matrices square and aligned to `class_order`; `float | None` values only; `color_scale_min <= color_scale_max`.

### `plot_emergence_heatmap` (static matplotlib)

```python
plot_emergence_heatmap(
    data: EmergenceData,
    level: str,                              # one of data.auroc_levels
    *,
    class_labels: Mapping[str, str] | None = None,
    title: str | None = None,
    cmap: str = "YlOrRd",
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure
```

### `render_emergence_html` (standalone interactive)

Writes a fully standalone interactive HTML — open in any browser, no server. D3 and data inlined.

```python
render_emergence_html(
    data: EmergenceData,
    *,
    class_labels: Mapping[str, str] | None = None,
    class_colors: Mapping[str, str] | None = None,   # None → tab10
    bin_width: float = 4.0,
    min_cross_support: float = 0.5,
    heatmap_font_scale: float = 1.0,
    output_path: str | Path | None = None,
) -> str
```

HTML controls: toggle which classes are included, pick the emergence reference (tree recomputes client-side), and switch AUROC threshold (none / 0.60 / 0.65 / 0.70).

### `render_emergence_html_from_scores` (convenience one-step)

```python
render_emergence_html_from_scores(
    scores_df,
    class_order,
    *,
    auroc_levels=..., p_sep=..., p_ns=..., subsequent_frac=...,
    class_labels=..., class_colors=...,
    bin_width=4.0, min_cross_support=0.5,
    heatmap_font_scale=1.0,
    time_col="time_bin_center",
    positive_class_col="positive_label",
    negative_class_col="negative_label",
    auroc_col="auroc_obs",
    pvalue_col="pval",
    output_path=None,
) -> str
```

### Typical pattern

```python
from analyze.classification.viz.emergence import (
    compute_emergence_data, plot_emergence_heatmap, render_emergence_html,
)

scores = result.scores[result.scores["feature_set"] == "vae"]
data = compute_emergence_data(scores, ALL_CLASSES)

plot_emergence_heatmap(data, level="0.70", output_path="onset_0p70.png")
render_emergence_html(
    data,
    class_labels={"inj_ctrl": "Inj. Ctrl", ...},
    class_colors={"inj_ctrl": "#2166AC", ...},
    output_path="emergence_explorer.html",
)
```

### Low-level algorithm (for direct use)

```python
from analyze.classification.emergence import (
    build_emergence_timeline,
    EmergenceTimeline, EmergenceScore, EmergenceBlock,
    ReferenceValidation, ResolutionNode,
)
```

See [`../emergence/ALGORITHM.md`](../emergence/ALGORITHM.md) and [`../emergence/DESIGN.md`](../emergence/DESIGN.md) for the full algorithm spec.

---

## Misclassification deep-dive

**File:** `misclassification.py`

Diagnostic plots for inspecting which embryos are misclassified and when. Typically driven by `run_misclassification_pipeline` (in `misclassification/`). Take the pipeline's outputs (`embryo_predictions`, `per_embryo_metrics`, `flagged_embryos`) and pass them in.

```python
plot_wrongness_heatmap(
    embryo_predictions, per_embryo_metrics, output_dir: Path,
    row_order: str = "wrong_rate",
    cmap: str = "Reds",
) -> Path

plot_embryo_deep_dive(
    embryo_predictions, embryo_id: str, output_dir: Path,
    class_colors: dict[str, str] | None = None,
) -> Path

plot_wrong_rate_distributions(
    per_embryo_metrics, output_dir: Path,
    group_by: str = "true_class", show_flagged: bool = True,
) -> Path

plot_confusion_profile(
    embryo_predictions, flagged_embryos, output_dir: Path,
) -> Path

plot_flagged_embryo_gallery(
    embryo_predictions, flagged_embryos, output_dir: Path, top_n: int = 20,
) -> list[Path]
```

---

## Trajectory diagnostics

**File:** `trajectory.py`

Cluster-level feature trends and rolling-window significance (used in misclassification pipeline analysis).

```python
save_pca_scatter(stage_table, color_col, output_path, title) -> Path
save_wrong_rate_null_diagnostics(stage_table, output_path, title="...") -> Path
save_rolling_window_significance_counts(rolling_df, output_path, title="...") -> Path
save_rolling_destination_significance_counts(rolling_df, output_path, title="...") -> Path
plot_cluster_feature_trends(
    raw_df, stage_table, cluster_col, output_path, features,
    time_col, embryo_id_col, group_color_by, facet_col_override=None,
) -> Path
```

---

## Low-level curve primitives

**File:** `classification.py`

Prefer `plot_aurocs_over_time` for most cases. These are the building blocks.

```python
plot_multiple_aurocs(
    auroc_dfs_dict: dict[str, pd.DataFrame],   # {label: df}
    colors_dict: dict[str, str],
    styles_dict: dict[str, str] | None = None,
    title: str = "AUROC Comparison",
    figsize: tuple = (14, 7),
    ylim: tuple = (0.3, 1.05),
    time_col: str = "time_bin_center",
    save_path: Path | None = None,
    ax: plt.Axes | None = None,
    sig_threshold: float = 0.01,
    show_null_band: bool = True,
    show_significance: bool = True,
    show_chance_line: bool = True,
    chance_y: float = 0.5,
    chance_label: str = "Chance (0.5)",
    chance_linestyle: str = ":",
    show_sig_legend: bool = True,
) -> plt.Figure

plot_auroc_with_null(
    ax, auroc_df, color, label,
    style: str = "-",
    time_col: str = "time_bin_center",
    show_null_band: bool = True,
    show_significance: bool = True,
    sig_threshold: float = 0.01,
    sig_marker_size: int = 200,
) -> None

plot_feature_comparison_grid(
    results_by_feature: dict[str, MulticlassOVRResults],
    feature_labels: dict[str, str],
    cluster_colors: dict[str, str],
    title: str = "",
    ylim: tuple = (0.3, 1.05),
    figsize_per_panel: tuple = (6, 5),
    save_path: Path | None = None,
    sig_threshold: float = 0.01,
) -> plt.Figure
```

---

## Validator utilities

**File:** `utils.py` — called internally by `plot_margin_trends`, available standalone.

```python
from analyze.classification.viz.utils import (
    validate_required_columns,   # raises if columns missing
    validate_margin_range,       # raises if values outside [-1, 1]
    validate_unique_embryo_x,    # raises if duplicate (embryo, x) pairs
)
```
