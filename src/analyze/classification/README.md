# classification

Time-resolved classification with permutation testing. One entry point — **`run_classification()`** — runs cross-validated AUROC at each time bin across one or more feature sets and handles all-pairs, explicit pairs, pooled groups, and user-supplied design tables in a single call. It returns a `ClassificationAnalysis` object bundling scores, run metadata, and optional per-embryo artifacts (predictions, null distributions, classifier directions).

**When to use this module:** you have a dataframe of embryos × features × time and want to:
1. **Difference detection** — ask whether/when classes are distinguishable (per-bin AUROC + permutation p-values).
2. **Phenotype emergence** — order classes by when they first become separable from a reference.
3. **Extract directions / geometry** — pull interpretable axes (coefficient vectors) out of the trained pairwise classifiers for downstream use in `morphology_geometry/` and `trajectory_condensation/`.

**When NOT to use this module:** if you want the *ordering* of phenotype emergence rather than per-class AUROC, start in `emergence/`. If you want per-embryo deep dives on why a specific genotype gets misclassified, start in `misclassification/`.

---

## Import

```python
from analyze.classification import run_classification
from analyze.classification import ClassificationAnalysis   # for loading saved runs
```

---

## Two ways to use this module

1. **Phenotype emergence** — "when do classes become separable, and on which features?"
   → `run_classification(...)` → AUROC curves + plots.
2. **Extract classifier geometry** — "what combination of features separates A from B?"
   → `run_classification(..., save_classifier_directions=True)` → consumed by
   `morphology_geometry/` and `trajectory_condensation/`.

---

## Quick workflow

End-to-end: load a dataframe → run classification → inspect → plot → reload later.

```python
from analyze.classification import run_classification, ClassificationAnalysis

# 1. Load your features. Whatever loader you use, `df` must have:
#      - an id column (one row per embryo at one timepoint)
#      - a class column (e.g. genotype)
#      - a time column (continuous, e.g. predicted_stage_hpf)
#      - feature columns (numeric — VAE latents, shape metrics, etc.)
df = load_df(...)

# 2. Run classification: every pair of classes, VAE embedding.
result = run_classification(
    df,
    class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    comparisons="all_pairs",
    features={"emb": "z_mu_b"},    # prefix → all z_mu_b_* columns
    save_dir="results/my_run/",    # auto-saves scores, confusion, metadata
)

# 3. Inspect.
result.scores.head()            # one row per (feature_set, comparison, time_bin)
result.comparison_ids           # e.g. ["homo__vs__wildtype", "het__vs__wildtype", ...]

# 4. Plot.
result.plot_aurocs(output_path="aurocs.png")           # AUROC curves over time
from analyze.classification.viz import plot_auroc_heatmaps
plot_auroc_heatmaps(result.scores, output_path="heatmap.png")   # faceted heatmap

# 5. Later: reload without rerunning.
result = ClassificationAnalysis.load("results/my_run/")
```

Use case 1 below breaks this down into the four common variations (pooling features,
scoping classes, pooled sides, mixed specs).

---

## Which path does my call trigger?

`run_classification()` has two internal paths. Most downstream-artifact flags
(`save_classifier_directions`, etc.) require the **binary path**. The rule is simple:
**if you produce explicit (positive vs negative) pairs, you're on the binary path.**

| What you pass | Path |
|---|---|
| `comparisons="all_pairs"` | Binary (all C(n,2) pairs) |
| `comparisons="all_pairs", positive=["A","B","C"]` | Binary (pairs scoped to those classes) |
| `positive="A", negative="B"` | Binary (single pair) |
| `positive=("A","B"), negative="C"` | Binary (pooled side) |
| `comparisons=[{"positive":..., "negative":...}, ...]` | Binary (explicit design) |
| Everything omitted (`positive=None, negative=None, comparisons=None`) | Multiclass *(default)* |

Binary path unlocks `save_predictions`, `save_classifier_directions`, and `save_null_arrays`.
The default multiclass mode is mainly useful when you want a single multiclass model per bin
(see `save_multiclass_predictions`, used by the misclassification pipeline). Most users
should pass `comparisons="all_pairs"`.

Full semantics of how `positive`/`negative`/`comparisons` combine →
[`docs/DESIGN.md` § "How positive, negative, and comparisons interact"](docs/DESIGN.md).

---

## Use case 1 — Phenotype emergence

Most users land here. Run classification, get AUROC vs time, plot it. The snippets
below build from simplest to most specific.

### 1a. Simplest: all pairs, one feature set (VAE embedding)

```python
result = run_classification(
    df,
    class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    comparisons="all_pairs",
    features={"emb": "z_mu_b"},
)
# "z_mu_b" is a prefix — picks up all columns matching z_mu_b_* (e.g. z_mu_b_0, z_mu_b_1, ...).
# Pass a list instead (e.g. ["total_length_um"]) to use explicit columns.
```

### 1b. Pooling multiple feature sets (VAE + curvature + shape)

Each key in `features` becomes one `feature_set` row in `result.scores`:

```python
features={
    "emb":       "z_mu_b",
    "curvature": ["baseline_deviation_normalized"],
    "shape":     ["total_length_um", "baseline_deviation_normalized"],
}
```

### 1c. Subsetting classes — `positive` / `negative`

```python
# All pairs, but scoped to a subset of classes:
positive=["wildtype", "homo", "het"], comparisons="all_pairs"

# One specific pair:
positive="homo", negative="wildtype"
```

### 1d. Pooled sides, mixed specs

```python
# homo vs pooled {wildtype + het}:
positive="homo", negative=("wildtype", "het")

# Enumerate: two comparisons, both against the same pooled negative
positive=["homo", "het"], negative=("wildtype", "ctrl")
#   → homo vs {wildtype+ctrl}
#   → het  vs {wildtype+ctrl}

# Same positive compared against two different negatives (one pooled):
positive="homo", negative=["wildtype", ("wildtype", "het")]
#   → homo vs wildtype
#   → homo vs {wildtype+het}
```

More complex mixes, asymmetric designs, or hand-built design tables →
see [`docs/DESIGN.md` § "How positive, negative, and comparisons interact"](docs/DESIGN.md).

---

## Classification outputs — the `ClassificationAnalysis` object

`run_classification()` returns a single `ClassificationAnalysis`. It's a lightweight
container; the primary output is the **`scores` DataFrame** on `result.scores`.

```python
result.scores         # pd.DataFrame — THE main output (see below)
result.uns            # dict: run metadata (class_col, bin_width, git_commit, ...)
result.layers         # lazy artifact registry (predictions, confusion, nulls, ...)
result.feature_sets   # list[str] — keys of the `features={}` dict you passed
result.comparison_ids # list[str] — e.g. ["homo__vs__wildtype", "het__vs__wildtype"]
```

### `result.scores` — the main output

A tidy long-form DataFrame, one row per **(feature_set, comparison_id, time_bin_center)**.

| Column | Type | Description |
|---|---|---|
| `feature_set` | str | Feature set name (key of `features={}` dict) |
| `comparison_id` | str | Filesystem-safe identifier, e.g. `"homo__vs__wildtype"` |
| `positive_label` | str | Human-readable positive class |
| `negative_label` | str | Human-readable negative class |
| `time_bin_center` | float | Center of time bin (hpf) |
| `auroc_obs` | float | Observed cross-validated AUROC |

**Optional columns:** `pval`, `n_permutations`, `n_positive`, `n_negative`.

Example:

```
  feature_set       comparison_id  positive_label  negative_label  time_bin_center  auroc_obs   pval
0         emb  homo__vs__wildtype            homo        wildtype             26.0      0.612  0.18
1         emb  homo__vs__wildtype            homo        wildtype             30.0      0.781  0.02
2         emb    het__vs__wildtype             het        wildtype             26.0      0.549  0.31
...
```

Because it's a DataFrame, plain pandas works — filter, groupby, pivot, merge:

```python
# One comparison, one feature set, over time:
df = result.scores.query("comparison_id == 'homo__vs__wildtype' and feature_set == 'emb'")

# Peak AUROC per comparison:
result.scores.groupby("comparison_id")["auroc_obs"].max()
```

### Subsetting via the result object

`result.subset(...)` returns a new `ClassificationAnalysis` with filtered scores +
the same metadata. Use it when you want to carry a slice into downstream plotting
or save it as its own run:

```python
sub = result.subset(
    feature_set="emb",
    comparison_id="homo__vs__wildtype",
    time_range=(24.0, 72.0),
)
sub.plot_aurocs(output_path="homo_vs_wt_emb.png")
```

Stacking two runs (e.g. different feature sets run separately):

```python
combined = res_emb.stack(res_shape)   # merges scores; layers are NOT merged
```

### Common next steps

- **Plot** → `result.plot_aurocs()`, `plot_auroc_heatmaps(result.scores)` (see below).
- **Save / reload** → `result.save("path/")` / `ClassificationAnalysis.load("path/")`.
- **Slice** → `result.subset(feature_set=..., comparison_id=..., time_range=...)`.
- **Extract geometry** → re-run with `save_classifier_directions=True`
  (see [Workflow: extract classifier directions](#workflow-extract-classifier-directions)).

### Full spec

For the complete `ClassificationAnalysis` API (all methods, `_LazyLayers` internals,
`uns` structure, on-disk layout, save/load contract) see
[`docs/OUTPUT_SPEC.md`](docs/OUTPUT_SPEC.md).

---

## Plotting

### AUROC curves over time

```python
result.plot_aurocs(output_path="aurocs.png")
# Equivalent direct call:
from analyze.classification.viz import plot_aurocs_over_time
plot_aurocs_over_time(result.scores, output_path="aurocs.png")
```

### Faceted AUROC heatmap (all comparisons × feature sets at a glance)

```python
from analyze.classification.viz import plot_auroc_heatmaps
plot_auroc_heatmaps(result.scores, output_path="heatmap.png")
```

### Confusion matrices per bin

```python
result.plot_confusion(output_path="confusion.png")
```

Full plot catalogue → [`viz/README.md`](viz/README.md).

---

## Saving extra artifacts

Enable any of these flags to materialize additional layers. If `save_dir` is set,
they persist automatically; otherwise they live in-memory on `result.layers`.

```python
save_predictions=True            # per-embryo probabilities + truth_signed_margin (binary path)
save_multiclass_predictions=True # per-embryo probabilities (default multiclass mode; used by misclassification pipeline)
save_null_arrays=True            # full permutation null distributions (diagnostic)
save_dir="results/my_run/"       # auto-save at end; auto-enables save_predictions
```

Access pattern:

```python
preds = result.layers["predictions"]        # KeyError if unavailable
preds = result.layers.get("predictions")    # None if unavailable
"predictions" in result.layers              # bool, no disk load

# Reload later:
result = ClassificationAnalysis.load("results/my_run/")
```

Full layer reference (all flags, on-disk filenames, compatibility rules, margin conventions) →
[`docs/OUTPUT_SPEC.md`](docs/OUTPUT_SPEC.md).

---

## Workflow: extract classifier directions

When the question is not *"when are classes separable?"* but *"what axis in feature
space separates them?"*. End-to-end example — this is the stable entry point that
`morphology_geometry/` and `trajectory_condensation/` depend on.

```python
from analyze.classification import run_classification, ClassificationAnalysis

# 1. Load features (same df shape as Quick workflow):
#    id_col, class_col, time_col, feature columns.
df = load_df(...)

# 2. Run all-pairs classification with directions saved.
result = run_classification(
    df,
    class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    comparisons="all_pairs",                  # required — binary path
    features={"emb": "z_mu_b"},
    save_classifier_directions=True,          # REQUIRED for downstream geometry
    save_dir="results/my_geom_run/",
)

# 3. Access the directions artifact.
directions = result.layers["classifier_directions"]
# One coefficient vector per (feature_set, comparison, time_bin).

# 4. Reload without rerunning.
result = ClassificationAnalysis.load("results/my_geom_run/")
directions = result.layers["classifier_directions"]
```

**Downstream consumers** — these READMEs link back to this section. Don't rename the
section header or break the anchor `#workflow-extract-classifier-directions` without
coordinating:

- `src/analyze/morphology_geometry/` — uses directions as interpretable axes.
- `src/analyze/trajectory_condensation/` — uses directions for pathway fitting.

> **Note on contrast coordinates.** An earlier sibling API (`save_contrast_coordinates=True`)
> projected embryos onto these same axes, but its shrinkage math is being revised and the
> layer is effectively frozen — see [`docs/future_improvement.md`](docs/future_improvement.md).
> Classifier directions are the stable, recommended surface; use them for new work.

---

## Subpackage map

The module has several conceptual doors. Each subpackage answers a distinct question:

| Subpackage | What it answers | Entry point | Notes |
|---|---|---|---|
| `run_classification.py` | "When do classes become separable and on which features?" | `run_classification()` | The main orchestrator. Produces `ClassificationAnalysis` with `scores` + optional layers. |
| `engine/` | *(internal)* fit / CV / permutation / bin-and-aggregate machinery | — | Open when debugging comparison resolution or feature handling. |
| `viz/` | "Give me a figure of the result." | See [`viz/README.md`](viz/README.md) | Plots are organized by user goal, not function name. |
| `emergence/` | "In what order do classes become distinguishable from a reference, and which co-emerge?" | `compute_emergence_data()`, `render_emergence_html()` | Consumes `result.scores`. See [`emergence/ALGORITHM.md`](emergence/ALGORITHM.md) + [`emergence/DESIGN.md`](emergence/DESIGN.md). |
| `misclassification/` | "Which individual embryos get it wrong, when, and how?" | `run_misclassification_pipeline()` | Consumes the `predictions` layer. |
| `directions/` | "What axis in feature space separates A from B?" — interpretable geometry API | `save_classifier_directions=True` → `result.layers["classifier_directions"]` | Per-comparison coefficient vectors as interpretable axes. Binary path only. See [Workflow: extract classifier directions](#workflow-extract-classifier-directions). |
| Legacy shims (`classification_test.py`, `results.py`, `classification_results.py`) | *(deprecated)* pre-`run_classification` API | — | `FutureWarning` on call. `analyze.difference_detection` is a re-export shim to these. |

> Contrast coordinates (`save_contrast_coordinates=True`) are frozen pending a shrinkage rewrite — see [`docs/future_improvement.md`](docs/future_improvement.md). Use classifier directions.

---

## Reference docs in this directory

| File | Purpose |
|---|---|
| [`README.md`](README.md) | **This file.** User-facing landing page + walkthrough. |
| [`docs/OUTPUT_SPEC.md`](docs/OUTPUT_SPEC.md) | Full output contract — `ClassificationAnalysis`, `scores` schema, all layers, persistence, `uns`. |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Design rationale for the `run_classification` API and comparison resolution. Open when deciding *whether* to change the signature. |
| [`docs/future_improvement.md`](docs/future_improvement.md) | Historical roadmap snapshot (2026-03-05). Many P0 items have landed. |
| [`viz/README.md`](viz/README.md) | Plotting cookbook, organized by what you want to see. |
| [`emergence/ALGORITHM.md`](emergence/ALGORITHM.md) | Emergence algorithm spec. |
| [`emergence/DESIGN.md`](emergence/DESIGN.md) | Emergence package design notes. |

---

## Input contract

`run_classification(df, ...)` requires a pandas DataFrame with these columns:

| Column | Role | Notes |
|---|---|---|
| *`id_col`* (you name it, e.g. `embryo_id`) | Sample identity. | One embryo per value. Used for CV grouping. |
| *`class_col`* (e.g. `genotype`) | Class label. | Strings. Each id must map to exactly one class. |
| *`time_col`* (e.g. `predicted_stage_hpf`) | Continuous developmental time. | Binned internally using `bin_width`. |
| Feature columns | The numeric features to classify on. | Either a prefix (`features={"emb": "z_mu_b"}` → all `z_mu_b_*` cols) or an explicit list. |

**Where the df comes from.** In this repo, feature-labeled embryo dataframes are typically produced by `src/data_pipeline/feature_extraction/` + VAE embedding export. VAE latents are `z_mu_b_*`; shape features include `total_length_um`, `yolk_area_um2`, etc.

---

## Plot index (all layers)

Full catalogue → [`viz/README.md`](viz/README.md).

| Goal | Function | Layer required |
|---|---|---|
| AUROC curves over time | `plot_aurocs_over_time` / `result.plot_aurocs()` | `scores` |
| Faceted AUROC heatmap | `plot_auroc_heatmaps` | `scores` |
| Confusion matrices | `plot_confusion` / `result.plot_confusion()` | `confusion` |
| Per-embryo margin trajectory | `plot_margin_trends` | `predictions` |
| Phenotype emergence timeline (static + interactive HTML) | `compute_emergence_data` + `render_emergence_html` | `scores` |
| Misclassification deep-dive (embryo gallery) | `run_misclassification_pipeline` | `predictions` |

---

## Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `ValueError: No valid comparisons produced results.` | Every comparison fell below the sample-size floor. | Check `min_samples_per_group` / `min_samples_per_member`; verify `id_col` and `class_col` values are non-null and correctly spelled. |
| `KeyError: Layer 'predictions' was not computed during this run.` | `save_predictions=False` on the run, and no `save_dir` was set. | Re-run with `save_predictions=True` (or set `save_dir=...`, which auto-enables it). |
| `ValueError: save_classifier_directions=True is only supported for binary comparison runs.` | You're on the default multiclass mode. | Pass `comparisons="all_pairs"` or explicit `positive=/negative=` (see the path table above). |
| `FileExistsError` on save | `save_dir` already populated. | Pass `overwrite=True`, or point `save_dir` elsewhere. |
| AUROC pinned at 0.5 across all bins | Usually class imbalance collapsing CV folds, or features not actually predictive in that bin. | Inspect `n_positive`, `n_negative`; visualize raw features by class; consider widening `bin_width`. |

---

## Full `run_classification()` reference

```python
def run_classification(
    df: pd.DataFrame,
    *,
    # ── Data contract (required) ────────────────────────────────────────
    class_col: str,
    id_col: str,
    time_col: str,

    # ── Comparison spec ─────────────────────────────────────────────────
    positive: UserComparisonSpec | None = None,   # str | tuple[str,...] | list[str | tuple]
    negative: UserComparisonSpec | None = None,   # str | tuple[str,...] | list[str | tuple]
    comparisons: ComparisonScheme = None,          # "all_pairs" | DataFrame | list[dict] | None

    # ── Features (always named, always a dict) ─────────────────────────
    features: dict[str, str | list[str]],          # {"name": prefix_or_list}

    # ── Binning ─────────────────────────────────────────────────────────
    bin_width: float = 4.0,

    # ── Classifier / CV ─────────────────────────────────────────────────
    n_permutations: int = 100,
    n_splits: int = 5,
    min_samples_per_group: int = 3,
    min_samples_per_member: int = 2,
    n_jobs: int = 1,
    random_state: int = 42,
    class_weight: Any | None = "balanced",

    # ── Output / persistence ────────────────────────────────────────────
    verbose: bool = True,
    save_predictions: bool = False,
    save_multiclass_predictions: bool = False,
    save_null_arrays: bool = False,
    save_contrast_coordinates: bool = False,         # FROZEN — shrinkage rewrite pending; prefer save_classifier_directions
    save_classifier_directions: bool = False,        # binary-path only
    save_dir: str | Path | None = None,              # auto-saves at end; auto-enables save_predictions
    overwrite: bool = False,                         # pass-through to .save()
) -> ClassificationAnalysis
```

Validation runs before any computation — invalid comparison specs, missing df columns, or unresolvable labels all raise immediately.

---

## Architecture

```
classification/
├── README.md                       # this file — canonical user-facing reference
├── docs/
│   ├── OUTPUT_SPEC.md              # full ClassificationAnalysis + layers + persistence contract
│   ├── DESIGN.md                   # design rationale for run_classification
│   └── future_improvement.md       # historical roadmap snapshot (2026-03-05)
├── run_classification.py           # single entry-point orchestrator
├── engine/
│   ├── analysis.py                 # ClassificationAnalysis, _LazyLayers
│   ├── comparison_resolution.py    # resolve_comparisons, check_min_samples
│   ├── contrast_coordinates.py     # assemble_contrast_coordinates (legacy-but-supported)
│   ├── contrast_support.py         # contrast support masks
│   ├── data_prep.py                # _bin_and_aggregate, _build_binary_labels, _resolve_feature_columns
│   ├── loop.py                     # binary + multiclass fit loops + collectors
│   ├── margins.py                  # class_signed_margin, truth_signed_margin, coerce_margin_range
│   └── null.py                     # NullDistributions
├── directions/                     # preferred interpretable classifier geometry
│   ├── artifact.py                 # ClassifierDirections
│   ├── build_payload.py
│   ├── extract.py
│   ├── fit.py
│   └── ids.py
├── emergence/                      # phenotype emergence timelines (see ALGORITHM.md, DESIGN.md)
├── misclassification/              # embryo-level deep-dive pipeline
├── viz/                            # plotting (see viz/README.md)
├── classification_test.py          # legacy shim (FutureWarning)
├── results.py                      # legacy shim (FutureWarning)
├── classification_results.py       # legacy shim (FutureWarning)
└── tests/
```

---

## Running tests

```bash
PYTHONPATH=src:$PYTHONPATH conda run -n segmentation_grounded_sam --no-capture-output \
  python -m pytest src/analyze/classification/tests/ -x -v
```

---

## Appendix: legacy API and migration

The pre-`run_classification` functions (`run_classification_test`, `MulticlassOVRResults`, `ClassificationResults`) still work but emit `FutureWarning`. `analyze.difference_detection` is a re-export shim to these. Prefer the canonical path.

### Legacy all-vs-rest

```python
# Before
res = run_classification_test(df, groupby="genotype", groups="all", reference="rest")
df_scores = res.comparisons

# After
result = run_classification(
    df, class_col="genotype", id_col="embryo_id",
    time_col="predicted_stage_hpf",
    features={"emb": "z_mu_b"},
)
df_scores = result.scores
```

### Explicit pair

```python
# Before
res = run_classification_test(df, groupby="genotype", groups=["homo"], reference="wildtype")

# After
result = run_classification(
    df, class_col="genotype", id_col="embryo_id",
    time_col="predicted_stage_hpf",
    positive="homo", negative="wildtype",
    features={"emb": "z_mu_b"},
)
```

### Multi-metric accumulation

```python
# Before
acc = ClassificationResults()
acc.add("emb",   run_classification_test(df, features="z_mu_b", ...))
acc.add("shape", run_classification_test(df, features=["total_length_um"], ...))

# After
res_emb   = run_classification(df, features={"emb": "z_mu_b"}, ...)
res_shape = run_classification(df, features={"shape": ["total_length_um"]}, ...)
combined  = res_emb.stack(res_shape)
combined.plot_aurocs(output_path="aurocs.png")
```
