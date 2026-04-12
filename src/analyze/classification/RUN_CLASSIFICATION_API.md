# `run_classification` — Full API Reference

See [`README.md`](README.md) for overview, quick start, and core concepts.

## Scores Schema

Required columns (always present):

| Column | Type | Description |
|---|---|---|
| `feature_set` | str | Feature set name |
| `comparison_id` | str | Filesystem-safe comparison identifier |
| `positive_label` | str | Human-readable positive class label |
| `negative_label` | str | Human-readable negative class label |
| `time_bin_center` | float | Center of time bin in hpf |
| `auroc_obs` | float | Observed cross-validated AUROC |

Optional columns (present when computed):

| Column | Description |
|---|---|
| `pval` | Permutation p-value |
| `n_permutations` | Number of permutations |
| `n_pos` | Embryos in positive class |
| `n_neg` | Embryos in negative class |

### Layers (Opt-in Artifacts)

Artifacts are kept in memory during the run and persisted to disk via `result.save()`. Which artifacts are included depends on the `save_*` flags passed to `run_classification()`:

| Layer key | Flag | Content |
|---|---|---|
| `"predictions"` | `save_predictions=True` | Per-embryo predicted probabilities for binary resolved comparisons (tidy) |
| `"multiclass_predictions"` | `save_multiclass_predictions=True` | Wide-format one-vs-rest probabilities for multiclass runs |
| `"confusion"` | *(always computed)* | Per-bin confusion matrices, either multiclass or per comparison |
| `"null_full"` | `save_null_arrays=True` | Full null AUROC distributions (`NullDistributions`) |

**Save to disk (automatic):**
```python
result = run_classification(
    df, ...,
    save_predictions=True,
    save_null_arrays=True,
    save_dir="results/my_run/",  # auto-saves at the end
)
# Result is also returned, and has been saved to disk
```

**Save to disk (manual):**
```python
result = run_classification(df, ...)
result.save("results/my_run/")  # explicitly save later
```

**Load from disk:**
```python
loaded = ClassificationAnalysis.load("results/my_run/")
preds = loaded.layers["predictions"]   # loads predictions.parquet from disk
nulls = loaded.layers["null_full"]      # loads null_distributions.npz from disk
```

**Access in-memory (before save):**
```python
preds = result.layers["predictions"]   # raises KeyError if save_predictions=False
preds = result.layers.get("predictions")  # returns None if missing
"predictions" in result.layers         # True/False — no disk load
result.layers.available()              # list of keys in cache
```

## `run_classification()` Reference

```python
def run_classification(
    df: pd.DataFrame,
    *,
    # Required
    class_col: str,             # column with class labels
    id_col: str,                # column with embryo IDs
    time_col: str,              # column with developmental time (hpf)
    features: dict[str, str | list[str]],  # {"name": prefix_or_list}

    # Comparison selection (mutually exclusive rules apply)
    positive=None,              # str | tuple[str,...] | list[...]
    negative=None,              # str | tuple[str,...] | list[...]
    comparisons=None,           # "all_vs_rest" | "all_pairs" | DataFrame | list[dict]

    # Time binning
    bin_width: float = 4.0,     # hpf per bin

    # Classification
    n_splits: int = 5,          # cross-validation folds
    n_permutations: int = 100,  # label permutations for null
    n_jobs: int = 1,            # parallel workers (-1 = all)
    random_state: int = 42,

    # Sample size checks
    min_samples_per_group: int = 3,   # embryos per comparison side
    min_samples_per_member: int = 2,  # embryos per member of pooled group

    # Output control
    save_predictions: bool = False,
    save_multiclass_predictions: bool = False,
    save_null_arrays: bool = False,
    save_dir: str | Path | None = None,  # if set, auto-save results here
    verbose: bool = True,
) -> ClassificationAnalysis
```

## `ClassificationAnalysis` API

### Subsetting

```python
sub = result.subset(
    feature_set="embedding",          # or list of strings
    comparison_id="homo_vs_wildtype", # or list
    positive_label="homo",            # or list
    time_range=(24.0, 72.0),          # (lo, hi) hpf
)
```

### Stacking (multi-feature accumulation)

```python
res_emb   = run_classification(df, features={"emb": "z_mu_b"}, ...)
res_shape = run_classification(df, features={"shape": ["total_length_um"]}, ...)

combined = res_emb.stack(res_shape)
combined.plot_aurocs(output_path="combined.png")
```

### Visualization

```python
result.plot_aurocs(
    feature_set="emb",
    output_path="aurocs.png",
    backend="matplotlib",     # or "plotly"
)

result.plot_confusion(
    feature_set="emb",
    time_range=(24.0, 72.0),
    output_path="confusion.png",
)
```

### Persistence

```python
result.save("results/my_run/")  # writes scores.parquet, uns.json, layers/

loaded = ClassificationAnalysis.load("results/my_run/")
```

### Migration from legacy

```python
# Migrate a MulticlassOVRResults object
legacy = run_classification_test(df, groupby="genotype", ...)
result = ClassificationAnalysis.from_legacy(legacy)
```

## Architecture

```
classification/
├── run_classification.py          # Single entry point / orchestrator
├── engine/
│   ├── __init__.py
│   ├── comparison_resolution.py   # Types + resolve_comparisons() + check_min_samples()
│   ├── loop.py                    # Factory-line functions (build, bin, run, collect)
│   ├── null.py                    # NullDistributions dataclass
│   └── analysis.py               # ClassificationAnalysis + _LazyLayers
├── viz/
│   ├── __init__.py
│   ├── classification.py          # plot_auroc_with_null, plot_multiple_aurocs, ...
│   ├── auroc_over_time.py         # plot_aurocs_over_time
│   ├── confusion.py               # plot_confusion
│   ├── misclassification.py       # plot_margin_trends, 
│   └── trajectory.py              # Cluster trends, PCA scatter, rolling significance
├── misclassification/             # Misclassification pipeline (unchanged)
├── classification_test.py         # Legacy shim (FutureWarning)
├── results.py                     # Legacy shim (FutureWarning)
├── classification_results.py      # Legacy shim (FutureWarning)
└── tests/
    ├── test_comparison_resolution.py  # 22 tests
    ├── test_null_distributions.py     # 4 tests
    ├── test_analysis.py               # 15 tests
    └── test_run_classification.py     # 11 tests
```

## Migration from Legacy API

### All-vs-rest

```python
# Before
from analyze.classification import run_classification_test
res = run_classification_test(df, groupby="genotype", groups="all", reference="rest")
df_scores = res.comparisons

# After
from analyze.classification import run_classification
result = run_classification(df, class_col="genotype", id_col="embryo_id",
                             time_col="predicted_stage_hpf",
                             features={"emb": "z_mu_b"})
df_scores = result.scores
```

### Explicit pair

```python
# Before
res = run_classification_test(df, groupby="genotype", groups=["homo"], reference="wildtype")

# After
result = run_classification(df, class_col="genotype", id_col="embryo_id",
                             time_col="predicted_stage_hpf",
                             positive="homo", negative="wildtype",
                             features={"emb": "z_mu_b"})
```

### Multi-metric accumulation

```python
# Before
from analyze.classification import ClassificationResults
acc = ClassificationResults()
acc.add("emb",   run_classification_test(df, features="z_mu_b", ...))
acc.add("shape", run_classification_test(df, features=["total_length_um"], ...))

# After
res_emb   = run_classification(df, features={"emb": "z_mu_b"}, ...)
res_shape = run_classification(df, features={"shape": ["total_length_um"]}, ...)
combined  = res_emb.stack(res_shape)
combined.plot_aurocs(output_path="aurocs.png")
```

## Running Tests

```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  python -m pytest src/analyze/classification/tests/ -x -v \
  PYTHONPATH=src:$PYTHONPATH
```
