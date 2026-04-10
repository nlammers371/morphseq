# analyze-classification — Viz Reference

All functions are importable from `analyze.classification.viz` or directly from their submodule.

---

## Margin conventions

**File:** `src/analyze/classification/engine/margins.py`

| function | range | sign meaning |
|---|---|---|
| `class_signed_margin(p_pos)` | [-1, 1] | +1 = strongest support for positive class; sign relative to decision boundary, not truth |
| `truth_signed_margin(p_pos, y_true)` | [-1, 1] | +1 = most correct; sign relative to true label |
| `coerce_margin_range(arr)` | [-1, 1] | rescales legacy [-0.5, 0.5] arrays; pass all loaded margins through this |

`predictions.parquet` stores `truth_signed_margin`. `raw_contrast_scores_long` stores `class_signed_margin`.

---

## `plot_margin_trends`

**File:** `src/analyze/classification/viz/misclassification.py`

Per-embryo signed-margin trajectory plots for a binary comparison. One panel per group (negative left, positive right), lines colored by mean margin on RdBu_r.

**Data source:** `predictions.parquet` — requires `save_predictions=True`.

### Predictions parquet schema

| column | type | notes |
|---|---|---|
| `comparison_id` | str | e.g. `"inj_ctrl__vs__pbx4_crispant"` |
| `positive_label` | str | y_true == 1 |
| `negative_label` | str | y_true == 0 |
| `feature_set` | str | e.g. `"vae"` |
| `embryo_id` | str | |
| `time_bin` | int | bin start (hpf) |
| `time_bin_center` | float | bin midpoint — default x-axis |
| `bin_width` | float | |
| `n_positive` | int | embryos in positive class this bin |
| `n_negative` | int | embryos in negative class this bin |
| `auroc_obs` | float | per-bin AUROC |
| `y_true` | int | 0 or 1 |
| `p_pos` | float | classifier probability |
| `truth_signed_margin` | float | [-1, 1] |
| `y_pred` | int | |
| `is_correct` | bool | |

### API

```python
plot_margin_trends(
    df,
    *,
    # Selector — exactly one mode required
    comparison_id: str | None = None,        # pipeline mode: filter by comparison_id column
    positive_label: str | None = None,       # explicit mode: right panel
    negative_label: str | None = None,       # explicit mode: left panel

    # Secondary filter (optional)
    feature_col: str = "feature_set",
    feature_id: str | None = None,

    # Column names
    margin_col: str = "truth_signed_margin", # must be in [-1, 1]
    x_col: str = "time_bin_center",          # one value per (embryo, x) pair
    embryo_col: str = "embryo_id",

    # Display
    color_mode: str = "continuous",          # "continuous" | "discrete"
    discrete_class_lookup: dict | None = None,
    discrete_class_colors: dict | None = None,
    max_embryos: int = 30,
    vmin: float = -1.0,
    vmax: float = 1.0,
    time_window: tuple[float, float] | None = None,
    output_path: Path | None = None,
) -> matplotlib.figure.Figure
```

### Usage

```python
from analyze.classification.viz.misclassification import plot_margin_trends
predictions = pd.read_parquet("results/.../predictions.parquet")

# Pipeline mode
plot_margin_trends(predictions, comparison_id="inj_ctrl__vs__pbx4_crispant",
                   feature_id="vae", output_path="margin_trends.png")

# Explicit-label mode
plot_margin_trends(df, positive_label="pbx4_crispant", negative_label="inj_ctrl",
                   output_path="margin_trends.png")

# Loop over all comparisons
for cid in sorted(predictions["comparison_id"].unique()):
    plot_margin_trends(predictions, comparison_id=cid, feature_id="vae",
                       output_path=figures_dir / f"margin_{cid}.png")
```

---

## `plot_pairwise_coordinate_heatmap`

**File:** `src/analyze/classification/viz/pairwise_coordinates.py`

Per-embryo phenotypic fingerprint: square upper-triangle heatmap showing where one embryo falls on every pairwise classifier simultaneously. Blue-white-red (RdBu_r), diverging at 0.

- Rows = positive class (A), columns = negative class (B)
- Cell = time-averaged `class_signed_margin` on the A-vs-B classifier
- Lower triangle and diagonal are masked

**Data source:** `raw_contrast_scores_long` — requires `save_contrast_coordinates=True`.

```python
plot_pairwise_coordinate_heatmap(
    df,           # long-form scores DataFrame
    sample_id,    # str — value to match against id_col
    *,
    id_col: str = "embryo_id",                 # column identifying samples
    positive_label_col: str = "positive_label",
    negative_label_col: str = "negative_label",
    time_col: str = "time_bin",
    margin_col: str = "class_signed_margin",
    label_order: list[str] | None = None,      # explicit axis order; None = alphabetical
    positive_labels: list[str] | None = None,  # restrict rows (union logic)
    negative_labels: list[str] | None = None,  # restrict cols (union logic)
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
from analyze.classification.viz.pairwise_coordinates import plot_pairwise_coordinate_heatmap
scores_long = result.layers["raw_contrast_scores_long"]

plot_pairwise_coordinate_heatmap(scores_long, "20260304_A01_e01",
                                  output_path="pairwise_coords.png")
```

---

## `plot_auroc_heatmaps`

**File:** `src/analyze/classification/viz/heatmaps.py`

Faceted heatmap of AUROC scores over time. Cells colored by AUROC; significant cells (p ≤ threshold) get a black border. Auto-infers facet axes from data when not specified.

**Data source:** `result.scores` or any scores DataFrame — always available.

```python
plot_auroc_heatmaps(
    results,                               # ClassificationAnalysis or scores DataFrame
    *,
    heatmap_row: str = "positive_label",   # y-axis within each panel
    heatmap_col: str = "time_bin_center",  # x-axis within each panel
    facet_row: str | None = None,          # None = auto-infer (feature_set if varies)
    facet_col: str | None = None,          # None = auto-infer (negative_label if varies)
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
    backend: str = "matplotlib",           # "matplotlib" | "plotly" | "both"
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure | plotly.Figure | dict
```

```python
from analyze.classification.viz.heatmaps import plot_auroc_heatmaps

plot_auroc_heatmaps(result, output_path="auroc_heatmap.png")
plot_auroc_heatmaps(result, facet_row="feature_set", sig_threshold=0.05,
                    output_path="auroc_by_feature.png")
```

---

## `plot_aurocs_over_time`

**File:** `src/analyze/classification/viz/auroc_over_time.py`

AUROC curves over time for one or more comparisons. Optional null band, significance markers, chance line, and faceting.

**Data source:** `result.scores` or any scores DataFrame.

```python
plot_aurocs_over_time(
    results,                               # ClassificationAnalysis or scores DataFrame
    *,
    time_col: str = "time_bin_center",
    auroc_col: str = "auroc_obs",
    curve_col: str = "positive",           # column that identifies each curve
    facet_row: str | None = None,
    facet_col: str | None = None,
    color_lookup: dict | None = None,
    show_null_band: bool = False,
    show_significance: bool = True,
    pval_col: str = "pval",
    sig_threshold: float = 0.01,
    show_chance_line: bool = True,
    ylim: tuple[float, float] = (0.3, 1.05),
    backend: str = "plotly",               # "plotly" | "matplotlib" | "both"
    output_path: str | Path | None = None,
) -> plotly.Figure | matplotlib.figure.Figure | dict
```

---

## `plot_confusion`

**File:** `src/analyze/classification/viz/confusion.py`

Confusion matrices from a ClassificationAnalysis run.

**Data source:** `result.layers["confusion"]` — always computed.

```python
plot_confusion(
    scores: pd.DataFrame,
    confusion: pd.DataFrame,
    *,
    feature_set: str | None = None,
    time_range: tuple[float, float] | None = None,
    backend: str = "matplotlib",
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure | plotly.Figure
```

```python
from analyze.classification.viz import plot_confusion
plot_confusion(result.scores, result.layers["confusion"],
               feature_set="vae", time_range=(24.0, 72.0),
               output_path="confusion.png")
```

---

## `plot_multiple_aurocs` / `plot_auroc_with_null`

**File:** `src/analyze/classification/viz/classification.py`

Low-level matplotlib AUROC curve primitives. Use `plot_aurocs_over_time` for most cases.

```python
# Overlay multiple comparisons on one axis
plot_multiple_aurocs(
    auroc_dfs_dict: dict[str, pd.DataFrame],  # {label: df}
    colors_dict: dict[str, str],
    styles_dict: dict[str, str] | None = None,
    title: str = "AUROC Comparison",
    ylim: tuple = (0.3, 1.05),
    save_path: Path | None = None,
    ax: plt.Axes | None = None,
    show_null_band: bool = True,
    show_significance: bool = True,
    sig_threshold: float = 0.01,
) -> plt.Figure

# Single curve with null band on an existing axis
plot_auroc_with_null(
    ax: plt.Axes,
    auroc_df: pd.DataFrame,
    color: str,
    label: str,
    show_null_band: bool = True,
    show_significance: bool = True,
    sig_threshold: float = 0.01,
) -> None
```

---

## Misclassification deep-dive functions

**File:** `src/analyze/classification/viz/misclassification.py`

Diagnostic functions for inspecting which embryos are misclassified and when. Typically used together after running classification with predictions saved.

```python
# Wrong-rate heatmap: embryos × time bins
plot_wrongness_heatmap(
    embryo_predictions: pd.DataFrame,  # embryo_id, time_bin, pred_class, true_class
    per_embryo_metrics: pd.DataFrame,
    output_dir: Path,
    row_order: str = "wrong_rate",
    cmap: str = "Reds",
) -> Path

# Per-embryo prediction timeline + probability traces
plot_embryo_deep_dive(
    embryo_predictions: pd.DataFrame,
    embryo_id: str,
    output_dir: Path,
    class_colors: dict[str, str] | None = None,
) -> Path

# Wrong-rate distributions by group (violin)
plot_wrong_rate_distributions(
    per_embryo_metrics: pd.DataFrame,
    output_dir: Path,
    group_by: str = "true_class",
    show_flagged: bool = True,
) -> Path

# Confusion profile by true class (bar chart of misprediction targets)
plot_confusion_profile(
    embryo_predictions: pd.DataFrame,
    flagged_embryos: pd.DataFrame,
    output_dir: Path,
) -> Path

# Gallery of deep-dive plots for top-N most misclassified embryos
plot_flagged_embryo_gallery(
    embryo_predictions: pd.DataFrame,
    flagged_embryos: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> list[Path]
```

---

## Trajectory diagnostics

**File:** `src/analyze/classification/viz/trajectory.py`

Diagnostics for cluster-level feature trends and rolling-window significance. Used in misclassification pipeline analysis.

```python
# PCA scatter colored by a column
save_pca_scatter(
    stage_table: pd.DataFrame,
    color_col: str,
    output_path: Path,
    title: str,
) -> Path

# Wrong-rate observed vs permutation null
save_wrong_rate_null_diagnostics(
    stage_table: pd.DataFrame,
    output_path: Path,
    title: str = "Wrong-Rate Null Diagnostics",
) -> Path

# Significant embryo counts per rolling window
save_rolling_window_significance_counts(
    rolling_df: pd.DataFrame,
    output_path: Path,
    title: str = "Rolling-Window Wrong-Rate Significance",
) -> Path

# Destination-confusion significance counts by rolling window
save_rolling_destination_significance_counts(
    rolling_df: pd.DataFrame,
    output_path: Path,
    title: str = "Rolling Destination-Confusion Significance",
) -> Path

# Feature trends faceted by cluster
plot_cluster_feature_trends(
    raw_df: pd.DataFrame,
    stage_table: pd.DataFrame,
    cluster_col: str,
    output_path: Path,
    features: list[str],
    time_col: str,
    embryo_id_col: str,
    group_color_by: str,
    facet_col_override: str | None = None,
) -> Path
```

---

## Emergence timeline

**Algorithm module:** `src/analyze/classification/emergence/`
**Viz file:** `src/analyze/classification/viz/emergence.py` *(static renderer not yet implemented)*

Emergence answers: *in what order do classes become distinguishable from a reference set, and which classes co-emerge?* The algorithm takes a pairwise onset matrix (derived from `result.scores`) and produces a `EmergenceTimeline` — a reference-rooted tree of emergence blocks.

### Pipeline to generate an `EmergenceTimeline`

```python
from analyze.classification.emergence import (
    build_emergence_timeline,
)
from analyze.classification.emergence.onset import (
    OnsetParams,
    classify_pair_state_over_time,
    compute_pair_onsets,
    build_onset_matrix,
)

# 1. Load scores from a completed all-pairs classification run
scores = pd.read_parquet("results/.../scores.parquet")
scores = scores[scores["feature_set"] == "vae"].copy()

# 2. Classify each (pair, time_bin) into tri-state: separated / not_separated / ambiguous
params = OnsetParams(
    p_sep=0.01,        # pval threshold for "separated"
    auroc_sep=0.70,    # min AUROC for "separated"
    p_ns=0.10,         # pval threshold for "not_separated"
    subsequent_frac=0.75,  # fraction of remaining bins that must stay separated
)
classified = classify_pair_state_over_time(scores, params)

# 3. Compute durable onset time per pair
onset_df = compute_pair_onsets(classified, params)

# 4. Build symmetric onset matrix (index=class, columns=class, values=hpf)
all_classes = sorted(scores["positive_label"].unique())
onset_matrix = build_onset_matrix(onset_df, all_classes)

# 5. Build the emergence timeline
reference = ["inj_ctrl", "wik_ab"]   # classes treated as the temporal reference
timeline = build_emergence_timeline(
    onset_matrix,
    reference,
    bin_width=4.0,
    min_cross_support=0.5,
)
```

### `EmergenceTimeline` structure

```python
timeline.reference_validation   # ReferenceValidation — coherence of the reference set
timeline.scores                 # list[EmergenceScore] — per-class emergence timing
timeline.blocks                 # list[EmergenceBlock] — co-emergent groups
timeline.block_resolutions      # dict[block_id, ResolutionNode] — within-block tree
timeline.all_classes            # list[str]
timeline.reference              # list[str]
```

Key types:

| type | fields |
|---|---|
| `ReferenceValidation` | `status` ("valid"/"ambiguous"/"invalid"), `coherence_score`, `offending_pairs` |
| `EmergenceScore` | `class_name`, `emergence_time`, `emergence_min`, `emergence_max`, `n_resolved_refs` |
| `EmergenceBlock` | `block_id`, `members`, `emergence_time`, `bin_key` |
| `ResolutionNode` | `members`, `split_time`, `children`, `unresolved` |

### Interactive HTML explorer

The live emergence visualization is a fully standalone **HTML file with inline D3.js** — open in any browser, no server required.

The rendering script uses the `analyze.classification.emergence` package to compute onset matrices and serialize them to JSON, then a client-side D3 renderer draws the tree interactively.

**Controls:**
- **Included genotypes** — checklist to toggle which classes appear in the tree
- **Emergence reference** — checklist to switch which class set defines the baseline; tree recomputes instantly client-side
- **AUROC threshold** — radio buttons: none | 0.60 | 0.65 | 0.70

**Features:**
- All tree computation (blocks, bipartition, resolution) runs client-side → instant reference switching
- Two-layer tree: emergence from reference (blocks) + within-block resolution (recursive splits)
- Reference coherence badge (valid / ambiguous / invalid) in status bar
- Dashed borders mark unresolved composite blocks

**Pattern for generating the HTML:**
```python
from analyze.classification.emergence.transitivity import (
    TransitivityParams,
    classify_pair_state_over_time,
    compute_pair_onsets,
    build_onset_matrix,
)

# Compute onset matrices for each AUROC threshold level
params = TransitivityParams(p_sep=0.05, auroc_sep=0.70, p_ns=0.10, subsequent_frac=0.40)
classified = classify_pair_state_over_time(scores_df, params)
onset_df = compute_pair_onsets(classified, params)
onset_matrix = build_onset_matrix(onset_df, all_classes)

# Serialize onset_matrix to JSON → embed into HTML template with inline D3
```

The HTML template and D3 rendering logic live in the analysis script for this experiment. Future work should move the HTML renderer into `analyze.classification.viz.emergence` following the DESIGN.md boundary.

### Static rendering (placeholder)

`render_emergence_timeline_static` in `analyze.classification.viz.emergence` raises `NotImplementedError` — not yet implemented. Static matplotlib figures belong there when ready.

### Onset data artifacts

The script `12_phenotype_emergence.py` writes these to `results/.../emergence/`:

| file | content |
|---|---|
| `onset_matrix.csv` | symmetric onset matrix (hpf) |
| `onset_pairs.csv` | per-pair onset times + n_separated_bins |
| `coherent_partitions.csv` | partition history over time |
| `panel_A_onset_heatmap.png` | onset matrix as heatmap |
| `panel_B_relation_snapshots.png` | edge state snapshots at key time bins |
| `panel_C_partition_river.png` | alluvial partition river over time |

---

## Validator utilities

**File:** `src/analyze/classification/viz/utils.py`

```python
from analyze.classification.viz.utils import (
    validate_required_columns,   # raises if columns missing
    validate_margin_range,       # raises if values outside [-1, 1]
    validate_unique_embryo_x,    # raises if duplicate (embryo, x) pairs
)
```

Called internally by `plot_margin_trends` but usable standalone.
