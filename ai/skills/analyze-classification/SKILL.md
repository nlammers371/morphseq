# analyze-classification — Classification Reference

## Module: `analyze.classification`

**Note:** `analyze.difference_detection` is a deprecated shim that re-exports from `analyze.classification`. Prefer the canonical path.

---

## `run_classification` (primary entry point)

**File:** `src/analyze/classification/run_classification.py`

```python
def run_classification(
    df: pd.DataFrame,
    *,
    class_col: str,
    id_col: str,
    time_col: str,
    features: dict[str, str | list[str]],   # {"name": prefix_or_list}

    positive=None,       # str | tuple[str,...] | list[str | tuple]
    negative=None,       # str | tuple[str,...] | list[str | tuple]
    comparisons=None,    # "all_vs_rest" | "all_pairs" | DataFrame | list[dict]

    bin_width: float = 4.0,
    n_splits: int = 5,
    n_permutations: int = 100,
    n_jobs: int = 1,
    random_state: int = 42,
    min_samples_per_group: int = 3,
    min_samples_per_member: int = 2,
    save_predictions: bool = False,   # auto-True when save_dir is set
    save_multiclass_predictions: bool = False,
    save_null_arrays: bool = False,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> ClassificationAnalysis
```

**Parameters:**
- `save_dir`: If provided, automatically saves results to this directory at the end. Equivalent to calling `result.save(save_dir)` but done inline.

**Comparison modes:**

| `positive` | `negative` | `comparisons` | Mode |
|---|---|---|---|
| `None` | `None` | `None` | All-vs-rest (default) |
| `["A","B"]` | `None` | `None` | Each of A, B vs rest |
| `None` | `None` | `"all_pairs"` | Every C(n,2) pair |
| `"A"` | `"B"` | `None` | Single pair A vs B |
| `("A","B")` | `"C"` | `None` | Pooled (A+B) vs C |
| `None` | `None` | `[{"positive":"A","negative":"B"},...]` | Explicit design table |

**Feature detection:** `features={"emb": "z_mu_b"}` auto-expands to all columns matching `z_mu_b_*`. Pass a list for explicit columns. Multiple keys run all comparisons for each feature set.

**Performance:** `n_jobs=-1` for parallel. Typical: `n_permutations=100`, `bin_width=4.0`.

---

## `ClassificationAnalysis`

**File:** `src/analyze/classification/engine/analysis.py`

```python
@dataclass
class ClassificationAnalysis:
    scores: pd.DataFrame    # one row per (feature_set, comparison_id, time_bin)
    uns:    dict            # run metadata (class_col, comparisons, git_commit, ...)
    layers: _LazyLayers     # lazy artifact registry
```

**Scores required columns:** `feature_set`, `comparison_id`, `positive_label`, `negative_label`, `time_bin_center`, `auroc_obs`. Optional: `pval`, `n_permutations`, `n_pos`, `n_neg`.

**Properties:**
```python
result.feature_sets    # -> list[str]  sorted feature set names
result.comparison_ids  # -> list[str]  sorted comparison IDs
```

**Subsetting:**
```python
sub = result.subset(
    feature_set="emb",
    comparison_id="homo_vs_wildtype",
    positive_label="homo",
    time_range=(24.0, 72.0),
)
```

**Stacking (multi-metric accumulation):**
```python
res_emb   = run_classification(df, features={"emb": "z_mu_b"}, ...)
res_shape = run_classification(df, features={"shape": ["total_length_um"]}, ...)
combined  = res_emb.stack(res_shape)
combined.plot_aurocs(output_path="aurocs_by_metric.png")
```

**Layer access:**
```python
preds = result.layers["predictions"]     # KeyError if save_predictions=False
preds = result.layers.get("predictions") # None if missing
"predictions" in result.layers           # no disk load
result.layers.available()                # list of saved/cached keys
```

Layer keys: `"predictions"` (always saved when `save_dir` is set, or when `save_predictions=True`), `"multiclass_predictions"` (`save_multiclass_predictions=True`), `"confusion"`, `"null_full"` (`save_null_arrays=True`).

**`predictions` columns:** `embryo_id`, `time_bin_center`, `y_true`, `p_pos`, `y_pred`, `is_correct`, `truth_signed_margin` (range `[-1, 1]`, positive = correctly classified).

**Persistence:**
```python
result.save("results/my_run/")
loaded = ClassificationAnalysis.load("results/my_run/")
```

**Legacy migration:**
```python
result = ClassificationAnalysis.from_legacy(old_multiclass_ovr_results)
```

---

## Usage Examples

### All-vs-rest

```python
from analyze.classification import run_classification

result = run_classification(
    df,
    class_col="genotype",
    id_col="embryo_id",
    time_col="predicted_stage_hpf",
    features={"emb": "z_mu_b"},
)
print(result.scores[["comparison_id", "time_bin_center", "auroc_obs", "pval"]])
```

### Explicit pair with multiple feature sets

```python
result = run_classification(
    df,
    class_col="genotype",
    id_col="embryo_id",
    time_col="predicted_stage_hpf",
    positive="homo",
    negative="wildtype",
    features={
        "emb":   "z_mu_b",
        "shape": ["total_length_um", "yolk_area_um2"],
    },
    save_predictions=True,
)
result.plot_aurocs(output_path="aurocs.png")
```

### Pooled comparison

```python
result = run_classification(
    df,
    class_col="genotype",
    id_col="embryo_id",
    time_col="predicted_stage_hpf",
    positive=("het", "homo"),
    negative="wildtype",
    features={"emb": "z_mu_b"},
)
```

---

## Viz: `analyze.classification.viz`

### `plot_signed_margin_trends`

**File:** `src/analyze/classification/viz/misclassification.py`

Per-embryo signed-margin trajectory plots for a binary comparison. Each embryo is a line colored by mean `truth_signed_margin` (RdBu_r, vmin=-1, vmax=1).

```python
from analyze.classification.viz import plot_signed_margin_trends

fig = plot_signed_margin_trends(
    embryo_df,               # DataFrame with truth_signed_margin + true_label, or p_pos + y_true
    group1="inj_ctrl",
    group2="pbx4_crispant",
    max_embryos=-1,          # -1 = all embryos; positive int caps per genotype panel
    output_path="embryo_trajectories_signed_margin.png",
)
```

**`embryo_df` accepted columns (in priority order):**
1. `truth_signed_margin` + `true_label` — canonical; positive = correctly classified, range `[-1, 1]`
2. `signed_margin` + `true_label` — legacy; auto-coerced from `[-0.5, 0.5]` if needed
3. `p_pos` + `y_true` — raw classifier output; `truth_signed_margin` computed internally

**Margin conventions (`from analyze.classification import ...`):**
- `truth_signed_margin(p_pos, y_true)` — sign relative to true label; `+1` = most correct, `-1` = most wrong. Stored in `predictions.parquet`.
- `class_signed_margin(p_pos)` — sign relative to positive-class axis. Stored in `raw_contrast_scores_long` as `class_signed_margin`.
- `coerce_margin_range(arr)` — rescales legacy `[-0.5, 0.5]` arrays to `[-1, 1]` at load time.

---

### `plot_aurocs_over_time`

**File:** `src/analyze/classification/viz/auroc_over_time.py`

```python
from analyze.classification.viz import plot_aurocs_over_time

fig = plot_aurocs_over_time(
    scores_df,                    # result.scores or subset
    curve_col="comparison_id",    # column to split into separate curves
    sig_threshold=0.01,
    ylim=(0.3, 1.05),
    backend="matplotlib",
    output_path="aurocs.png",
)
```

### `plot_confusion`

**File:** `src/analyze/classification/viz/confusion.py`

```python
from analyze.classification.viz import plot_confusion

fig = plot_confusion(
    scores=result.scores,
    confusion=result.layers["confusion"],
    feature_set="emb",
    time_range=(24.0, 72.0),
    output_path="confusion.png",
)
```

### `plot_feature_comparison_grid`

```python
from analyze.classification.viz import plot_feature_comparison_grid

fig = plot_feature_comparison_grid(
    results_by_feature={"emb": result_emb, "shape": result_shape},
    feature_labels={"emb": "Embedding", "shape": "Shape"},
    cluster_colors={"homo_vs_wildtype": "#B2182B"},
    sig_threshold=0.01,
    save_path="grid.png",
)
```

### `plot_multiple_aurocs`

```python
from analyze.classification.viz import plot_multiple_aurocs

fig = plot_multiple_aurocs(
    auroc_dfs_dict={"homo": df_homo, "het": df_het},
    colors_dict={"homo": "#B2182B", "het": "#F7B267"},
    time_col="time_bin_center",
    sig_threshold=0.01,
    save_path="multi_auroc.png",
)
```

---

## Legacy API (FutureWarning)

The functions below still work but emit `FutureWarning`. Use `run_classification` for new code.

### `run_classification_test`

**File:** `src/analyze/classification/classification_test.py`

```python
def run_classification_test(
    df: pd.DataFrame,
    groupby: str,
    groups: Union[str, List[str]] = "all",
    reference: Union[str, List[Union[str, Tuple[str, ...]]]] = "rest",
    features: Union[str, List[str]] = "z_mu_b",
    time_col: str = "predicted_stage_hpf",
    embryo_id_col: str = "embryo_id",
    bin_width: float = 4.0,
    n_splits: int = 5,
    n_permutations: int = 100,
    n_jobs: int = 1,
    min_samples_per_class: int = 3,
    random_state: int = 42,
    verbose: bool = True,
) -> MulticlassOVRResults
```

### `MulticlassOVRResults`

**File:** `src/analyze/classification/results.py`

```python
res["positive_name", "negative_name"]  # -> DataFrame for that comparison
res.keys()      # -> List[Tuple[str, str]]
res.filter(positive="A", pval_lt=0.05, auroc_gt=0.7)
res.summary()   # -> summary DataFrame
res.save(path)
MulticlassOVRResults.from_dir(path)
```

### `ClassificationResults` (multi-metric accumulator)

**File:** `src/analyze/classification/classification_results.py`

```python
# Deprecated — use run_classification + .stack() instead
acc = ClassificationResults()
acc.add("emb", run_classification_test(df, ...))
acc.plot_aurocs_over_time(backend="matplotlib", output_path="aurocs.png")
```
