# ClassificationAnalysis — Output Specification

Complete reference for what `run_classification()` returns, the `scores` schema,
the optional layers, persistence layout, and the surrounding types
(`_LazyLayers`, `NullDistributions`). This is the **user contract** — the shape
of the data your code can rely on.

For the **design rationale** behind these shapes (why the API looks this way),
see [`DESIGN.md`](DESIGN.md). For a quick intro with examples, see the
[top-level README](../README.md).

---

## `ClassificationAnalysis` object

```python
@dataclass
class ClassificationAnalysis:
    scores: pd.DataFrame      # required — always eager
    uns:    dict              # required — always eager; treat as read-only
    layers: _LazyLayers       # optional artifacts — lazy from disk
```

### Derived properties

| Property | Returns | Notes |
|---|---|---|
| `feature_sets` | `list[str]` | Sorted unique `feature_set` values in `scores` |
| `comparison_ids` | `list[str]` | Sorted unique `comparison_id` values in `scores` |

### Methods

| Method | Purpose |
|---|---|
| `subset(feature_set=None, comparison_id=None, positive_label=None, time_range=None)` | Return a new `ClassificationAnalysis` with filtered `scores`. Layer cache is forked (subset and parent don't share state). |
| `stack(other, on_conflict="error")` | Merge `scores` and `uns["comparisons"]` from two runs. Layers are **not** merged (they're diagnostic, not canonically comparable across runs). |
| `plot_aurocs(**kwargs)` | Sugar over `plot_aurocs_over_time(self.scores, ...)`. |
| `plot_confusion(**kwargs)` | Sugar over `plot_confusion(self.scores, self.layers["confusion"], ...)`. Raises `KeyError` if confusion layer missing. |
| `save(path, overwrite=False)` | Persist `scores`, `uns`, and all cached layers to a directory. |
| `load(path)` *(classmethod)* | Load a previously saved run. Layers are lazy-loaded on access. |
| `from_legacy(path)` *(classmethod)* | Load a pre-`run_classification` save. Currently raises `NotImplementedError`. |

### Why `stack()` doesn't merge layers

Predictions, confusion, and null arrays are **diagnostic artifacts, not
canonical merge targets**. Raw prediction probabilities are not guaranteed to
be directly comparable across runs, because runs may differ in class
composition, comparison structure, feature sets, and training distributions.
Use stacked objects for summary-level `scores` comparison across runs; use the
original `ClassificationAnalysis` objects to inspect raw predictions or other
run-local diagnostics.

---

## `scores` DataFrame

Tidy long-form. One row per **(feature_set, comparison_id, time_bin_center)** —
this triple is the unique key.

### Required columns (always present)

| Column | Type | Description |
|---|---|---|
| `feature_set` | str | Feature set name (key of `features={}` dict) |
| `comparison_id` | str | Filesystem-safe identifier, e.g. `"homo__vs__wildtype"` |
| `positive_label` | str | Human-readable positive class |
| `negative_label` | str | Human-readable negative class |
| `time_bin_center` | float | Center of time bin (hpf) |
| `auroc_obs` | float | Observed cross-validated AUROC |

### Optional columns (present when relevant)

| Column | Description |
|---|---|
| `pval` | Permutation p-value |
| `n_permutations` | Number of permutations used |
| `n_positive`, `n_negative` | Sample counts per class in bin |

### Validation

`ClassificationAnalysis.__post_init__` enforces:
1. All required columns present.
2. No duplicate `(feature_set, comparison_id, time_bin_center)` rows.

---

## Layers — optional artifacts

Artifacts are produced on-demand (enabled by per-flag kwargs on
`run_classification()`) and persisted to `save_dir` or written later via
`result.save(path)`. Access is lazy: `result.layers[key]` loads from disk on
first read and caches.

### Flag → artifact matrix

| Layer key | Enabled by | Content |
|---|---|---|
| `predictions` | `save_predictions=True` *(auto-True when `save_dir` is set)* | Per-embryo probability + `truth_signed_margin` for binary comparisons (tidy long-form). |
| `multiclass_predictions` | `save_multiclass_predictions=True` | Wide-format per-class probabilities for the default multiclass path. |
| `confusion` | *always computed* | Per-bin confusion matrices. |
| `null_full` | `save_null_arrays=True` | Full null AUROC distributions (`NullDistributions` object). |
| `classifier_directions` | `save_classifier_directions=True` | Per-comparison coefficient vectors (`ClassifierDirections` artifact). Binary-path only. |

### Compatibility rules

- `save_classifier_directions=True` requires the **binary path** (explicit
    `positive=/negative=` or `comparisons="all_pairs"`). It is **not** allowed
    with the default multiclass path.

### Access patterns

```python
preds = result.layers["predictions"]        # KeyError if unavailable
preds = result.layers.get("predictions")    # None if unavailable
"predictions" in result.layers              # bool, no disk load
result.layers.available()                   # list of saved/cached keys
result.layers.cached()                      # list of in-memory keys
```

### Artifact tiers (what's on vs off by default)

| Artifact | Storage | Default | Rationale |
|---|---|---|---|
| Null stats (mean/std/n) | columns in `scores` | always | free, always useful |
| Raw null arrays | `null_distributions.npz` via `NullDistributions` | off | diagnostic only |
| Confusion profile | `confusion.parquet` | always | cheap; captures error asymmetry |
| Predictions (binary) | `predictions.parquet` | off | per-comparison diagnostics; can be large |
| Predictions (multiclass) | `multiclass_predictions.parquet` | off | required by misclassification pipeline |

### Misclassification pipeline — fail-loud contract

`run_misclassification_pipeline()` requires the `multiclass_predictions`
layer. If missing, it raises immediately:

```python
raise ValueError(
    "Misclassification pipeline requires the multiclass_predictions layer. "
    "Re-run run_classification() with save_multiclass_predictions=True."
)
```

This makes the dependency between `save_multiclass_predictions=False` (the
default) and `run_misclassification_pipeline()` explicit at runtime rather
than producing a cryptic `KeyError` downstream.

---

## `predictions.parquet` schema

Used by `plot_margin_trends`, `run_misclassification_pipeline`, and any
per-embryo analysis.

| Column | Notes |
|---|---|
| `comparison_id`, `positive_label`, `negative_label`, `feature_set` | Identifiers |
| `embryo_id` | From `id_col` |
| `time_bin`, `time_bin_center`, `bin_width` | Binning |
| `n_positive`, `n_negative`, `auroc_obs` | Per-bin stats |
| `y_true`, `p_pos`, `y_pred`, `is_correct` | Per-embryo prediction |
| `truth_signed_margin` | `[-1, 1]`; `+1` = most correct, `-1` = most wrong |

---

## Margin conventions

Defined in `src/analyze/classification/engine/margins.py`:

| Function | Range | Sign meaning |
|---|---|---|
| `class_signed_margin(p_pos)` | `[-1, 1]` | `+1` = strongest positive-class support; sign relative to decision boundary. Formula: `2 * p_pos - 1`. |
| `truth_signed_margin(p_pos, y_true)` | `[-1, 1]` | `+1` = most correct; sign relative to true label. |
| `coerce_margin_range(arr)` | `[-1, 1]` | Rescales legacy `[-0.5, 0.5]` arrays. Pass loaded-from-disk margins through this. |

`predictions.parquet` stores `truth_signed_margin`.
`raw_contrast_scores_long` stores `class_signed_margin`.

---

## `uns` structure

Run metadata — provenance, config, and comparison membership.

```python
uns = {
    # provenance
    "schema_version": "classification_v1",
    "created_at":     "2026-03-23T...",
    "git_commit":     "abc123",

    # run config
    "class_col":  "genotype",
    "id_col":     "embryo_id",
    "time_col":   "predicted_stage_hpf",
    "bin_width":  4.0,
    "n_permutations": 300,
    "feature_sets": {
        "embedding": {
            "spec":    "z_mu_b",                      # original user input
            "columns": ["z_mu_b_0", "z_mu_b_1", ...], # resolved columns used
        },
        "shape": {
            "spec":    ["total_length_um", "baseline_deviation_normalized"],
            "columns": ["total_length_um", "baseline_deviation_normalized"],
        },
    },

    # comparison membership
    "comparisons": {
        "homo__vs__wildtype_het": {
            "positive_members": ["homo"],
            "negative_members": ["wildtype", "het"],
            "positive_label":   "homo",
            "negative_label":   "wildtype+het",
        },
    },
}
```

Treat `uns` as read-only.

---

## Persistence

### Saving

```python
# Auto-save at run end (recommended):
result = run_classification(df, ..., save_dir="results/my_run/")

# Manual:
result = run_classification(df, ...)
result.save("results/my_run/", overwrite=False)
```

### Loading

```python
result = ClassificationAnalysis.load("results/my_run/")
preds  = result.layers["predictions"]   # lazy-loaded from parquet on first access
```

### On-disk layout

```
my_run/
├── scores.parquet                              ← always
├── metadata.json                               ← always (uns dict)
├── confusion.parquet                           ← always
├── predictions.parquet                         ← if save_predictions
├── multiclass_predictions.parquet              ← if save_multiclass_predictions
├── null_distributions.npz                      ← if save_null_arrays
├── classifier_directions.parquet               ← if save_classifier_directions
└── classifier_directions_vectors.npz           ← if save_classifier_directions
```

Contrast-coordinate parquets (`raw_contrast_scores_long`, `contrast_support_long`,
`contrast_specificity_by_timebin`, `raw_coordinates`, `shrunk_coordinates`,
`residual_coordinates`, `probe_index`) are also written when
`save_contrast_coordinates=True`, but that layer group is frozen pending a
shrinkage rewrite — see [`future_improvement.md`](future_improvement.md).
Prefer `classifier_directions` for new work.

`ClassificationAnalysis.from_legacy(...)` currently raises `NotImplementedError` —
use the legacy loader (`MulticlassOVRResults.from_dir`) for pre-`run_classification`
saves.

---

## `_LazyLayers` — internal type (FYI)

You normally interact with layers through `result.layers[...]`. Internally
it's a `_LazyLayers` handle keyed by layer name:

```python
class _LazyLayers:
    """Lazy-loading dict-like interface for optional artifacts."""

    def __init__(self, base_dir: Path | None) -> None: ...
    def __getitem__(self, key: str) -> Any: ...         # load from disk on miss; cache
    def get(self, key, default=None) -> Any: ...        # no-raise variant
    def __contains__(self, key: str) -> bool: ...       # existence check, no disk load
    def available(self) -> list[str]: ...               # keys that exist on disk
    def cached(self) -> list[str]: ...                  # keys currently in memory
    def store(self, key: str, data: Any) -> None: ...   # cache an in-memory artifact
```

Layer registry (keys ↔ on-disk filenames):

```python
_REGISTRY = {
    "predictions":             ("predictions.parquet",            "parquet"),
    "multiclass_predictions":  ("multiclass_predictions.parquet", "parquet"),
    "confusion":               ("confusion.parquet",              "parquet"),
    "null_full":               ("null_distributions.npz",         "nulls"),
    # directions + contrast layers registered alongside when enabled
}
```

### Contrast coordinates — in flux

The older contrast-coordinate artifacts remain part of the codebase, but they
are no longer promoted in the user contract. For current guidance and roadmap
context, see [`future_improvement.md`](future_improvement.md).

---

## `NullDistributions` — raw null arrays

Array-indexed handle for raw per-permutation AUROC null distributions. Avoids
delimiter hell by storing a parallel index rather than encoding keys into
strings.

```python
@dataclass
class NullDistributions:
    null_auc:        np.ndarray     # (N, P)  float32 — N cells × P permutations
    feature_set:     np.ndarray     # (N,)    str
    comparison_id:   np.ndarray     # (N,)    str
    time_bin_center: np.ndarray     # (N,)    float64

    # Lookup by (feature_set, comparison_id, time_bin_center):
    def get(self, feature_set, comparison_id, time_bin_center) -> np.ndarray: ...

    @property
    def index_df(self) -> pd.DataFrame: ...     # N-row DataFrame of cell identifiers

    @classmethod
    def load(cls, path: Path) -> "NullDistributions": ...
    def save(self, path: Path) -> None: ...
```

Access in a run:

```python
nd = result.layers["null_full"]
null_array = nd.get("embedding", "homo__vs__wildtype", 26.0)   # shape (P,)
nd.index_df.head()                                              # all available cells
```

---

## See also

- [`../README.md`](../README.md) — user-facing quick start and walkthrough.
- [`DESIGN.md`](DESIGN.md) — design rationale for the API shape and comparison-resolution semantics.
- [`../viz/README.md`](../viz/README.md) — plotting cookbook.
