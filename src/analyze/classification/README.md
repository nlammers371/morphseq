# classification

Time-resolved binary and multiclass classification with permutation testing. Runs cross-validated AUROC at each developmental time bin and returns a single `ClassificationAnalysis` containing scores, metadata, and optional per-embryo artifacts. One entry point — `run_classification()` — handles default multiclass, all-pairs, explicit pairs, pooled groups, and user-supplied design tables, across one or multiple feature sets in a single call.

**When to use this module:** you have a dataframe of embryos × features × time and want to know when (and on which features) classes become distinguishable.

**When NOT to use this module:** if you want the *ordering* of phenotype emergence rather than per-class AUROC, start in `emergence/`. If you want per-embryo deep dives on why a specific genotype gets misclassified, start in `misclassification/`.

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
| `directions/` | "What axis in feature space separates A from B?" (*preferred* interpretable geometry API) | `save_classifier_directions=True` → `result.layers["classifier_directions"]` | Per-comparison coefficient vectors as interpretable axes. Binary-path only. |
| Contrast coordinates (in `engine/contrast_coordinates.py`) | "Where does each embryo sit along each pairwise separator?" (*legacy-but-supported* predecessor to `directions/`) | `save_contrast_coordinates=True` → 7 layer keys | Per-embryo projections. Feeds `plot_pairwise_coordinate_heatmap` and `trajectory_condensation`. Prefer `directions/` for new work. Binary-path only. |
| Legacy shims (`classification_test.py`, `results.py`, `classification_results.py`) | *(deprecated)* pre-`run_classification` API | — | `FutureWarning` on call. `analyze.difference_detection` is a re-export shim to these. |

### Directions and contrast coordinates — the relationship

Both derive from the **same trained pairwise binary classifiers**:

- **`directions/`** gives you the *axes*: one coefficient vector per (feature_set, comparison, time_bin). Answers *"what combination of features separates A from B?"*
- **Contrast coordinates** give you *embryos projected onto those axes*: per-embryo × comparison × time-bin values. Answers *"where does this specific embryo sit along each separator?"*

Contrast coordinates were the earlier API for interpretable classifier geometry. `directions/` is the cleaner replacement. Both remain supported and both can be enabled on the same run; they will be coherent because they're built from the same fitted models.

---

## Reference docs in this directory

| File | Purpose |
|---|---|
| [`README.md`](README.md) | **This file.** User-facing canonical reference. |
| [`DESIGN.md`](DESIGN.md) | Design rationale / approved spec for the `run_classification` API and comparison resolution contract. Open when deciding *whether* to change the signature. |
| [`future_improvement.md`](future_improvement.md) | Historical roadmap snapshot (2026-03-05). Many P0 items have landed. |
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

## Quick start

### Default multiclass (one-vs-rest reporting)

```python
from analyze.classification import run_classification

result = run_classification(
    df,
    class_col="genotype",
    id_col="embryo_id",
    time_col="predicted_stage_hpf",
    features={"emb": "z_mu_b"},
)
print(result.scores[["feature_set", "comparison_id", "time_bin_center", "auroc_obs", "pval"]].head())
# One row per (feature_set, comparison_id, time_bin_center).
# comparison_id is "<label>_vs_rest" in this mode.

result.plot_aurocs(output_path="aurocs.png")
```

### Explicit binary pair, multiple feature sets, saved to disk

```python
result = run_classification(
    df,
    class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    positive="homo", negative="wildtype",
    features={"emb": "z_mu_b", "shape": ["total_length_um", "yolk_area_um2"]},
    n_permutations=500,
    save_predictions=True,
    save_dir="results/homo_vs_wt/",    # auto-saves at end
)
```

### Pooled comparison (tuple = pooled group)

```python
result = run_classification(
    df,
    class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    positive=("het", "homo"),   # tuple → one pooled positive group
    negative="wildtype",
    features={"emb": "z_mu_b"},
)
```

---

## Choosing a comparison mode

The three parameters `positive`, `negative`, and `comparisons` control what gets compared. Rules:

- **Tuple = pooled.** `("het", "homo")` is *one* group pooling both labels.
- **List = enumerated.** `["het", "homo"]` runs *two* separate comparisons.
- **Any explicit `negative=`** keeps the run binary, even with pooled groups.
- **"one-vs-rest" is a reporting choice, not a fitting mode.** In default multiclass, the model is multiclass; results are reported per class as OvR.

| What you want | How to request it |
|---|---|
| All classes vs rest (default) | `positive=None, negative=None, comparisons=None` |
| All C(n,2) pairs | `comparisons="all_pairs"` |
| All pairs *within a scoped class list* | `comparisons="all_pairs", positive=["A", "B", "C"]` |
| Single pair A vs B | `positive="A", negative="B"` |
| Pooled positive vs single negative | `positive=("A", "B"), negative="C"` |
| Each non-WT vs one control | `positive=None, negative="wildtype"` (with `class_col` filtered upstream) |
| Explicit design table | `comparisons=[{"positive": "A", "negative": "B"}, ...]` |

**Forbidden combinations** (raise `ValueError`):
- `comparisons=DataFrame/list` with `positive` or `negative`
- `comparisons="all_pairs"` with `negative` or a scalar `positive`
- Pooled tuples combined with `"all_pairs"`

> **Confused by why a combination resolved the way it did?** The full semantics — role of each parameter (`positive` as scope filter vs. group definer, `negative` forcing explicit mode, etc.) and the complete mode-resolution table with "what happens" column — lives in [`DESIGN.md` § "How `positive`, `negative`, and `comparisons` interact"](DESIGN.md). Read that when the action table above doesn't answer your question.

---

## The result object: `ClassificationAnalysis`

```python
result.scores           # pd.DataFrame — one row per (feature_set, comparison_id, time_bin_center)
result.uns              # dict — run metadata (class_col, id_col, time_col, bin_width,
                        #        n_permutations, class_weight, feature_sets, comparisons, git_commit, ...)
result.layers           # lazy-loading artifact registry (see below)
result.feature_sets     # list[str]
result.comparison_ids   # list[str]
```

### `scores` schema

**Required columns** (always present):

| Column | Type | Description |
|---|---|---|
| `feature_set` | str | Feature set name (key of `features={}` dict) |
| `comparison_id` | str | Filesystem-safe identifier, e.g. `"homo__vs__wildtype"` |
| `positive_label` | str | Human-readable positive class |
| `negative_label` | str | Human-readable negative class |
| `time_bin_center` | float | Center of time bin (hpf) |
| `auroc_obs` | float | Observed cross-validated AUROC |

**Optional columns** (present when relevant):

| Column | Description |
|---|---|
| `pval` | Permutation p-value |
| `n_permutations` | Number of permutations used |
| `n_pos`, `n_neg` | Sample counts per class in bin |

`(feature_set, comparison_id, time_bin_center)` is the unique key.

### Subsetting and stacking

```python
sub = result.subset(
    feature_set="emb",
    comparison_id="homo__vs__wildtype",
    positive_label="homo",
    time_range=(24.0, 72.0),
)

res_emb   = run_classification(df, features={"emb": "z_mu_b"}, ...)
res_shape = run_classification(df, features={"shape": ["total_length_um"]}, ...)
combined  = res_emb.stack(res_shape)   # merges scores; layers are NOT merged
```

### Layers: flag → artifact matrix

Artifacts are produced on-demand and persisted to `save_dir` or written later via `result.save(path)`.

| Layer key | Enabled by | Content |
|---|---|---|
| `predictions` | `save_predictions=True` *(also auto-True when `save_dir` is set)* | Per-embryo probability + `truth_signed_margin` for binary comparisons (tidy long-form). |
| `multiclass_predictions` | `save_multiclass_predictions=True` | Wide-format OvR probabilities for the default multiclass path. |
| `confusion` | *always computed* | Per-bin confusion matrices. |
| `null_full` | `save_null_arrays=True` | Full null AUROC distributions (`NullDistributions` object). |
| `classifier_directions` | `save_classifier_directions=True` | Per-comparison coefficient vectors (`ClassifierDirections` artifact). Binary-path only. |
| `raw_contrast_scores_long` | `save_contrast_coordinates=True` | Per-embryo `class_signed_margin` on each pairwise classifier, long-form. |
| `contrast_support_long` | `save_contrast_coordinates=True` | Per-(embryo, classifier, bin) support indicator. |
| `contrast_specificity_by_timebin` | `save_contrast_coordinates=True` | Specificity metric per (embryo, bin). |
| `raw_coordinates` | `save_contrast_coordinates=True` | Embryo × classifier matrix of raw margins. |
| `shrunk_coordinates` | `save_contrast_coordinates=True` | `raw_coordinates` multiplied by per-(feature_set, comparison, bin) probe weight `clip((auroc_obs - null_mean) / 0.5, 0, 1)`. |
| `residual_coordinates` | `save_contrast_coordinates=True` | `raw_coordinates - shrunk_coordinates`. |
| `probe_index` | `save_contrast_coordinates=True` | Index over the probe dimension with feature-set + comparison + time-bin labels. |

Access patterns:

```python
preds = result.layers["predictions"]        # KeyError if unavailable
preds = result.layers.get("predictions")    # None if unavailable
"predictions" in result.layers              # bool, no disk load
result.layers.available()                    # list of saved/cached keys
```

`save_contrast_coordinates=True` requires `n_permutations > 0` and is **not** allowed with the multiclass fast path. Same for `save_classifier_directions=True`. Use explicit comparisons (binary path).

### `predictions` parquet columns

Used by `plot_margin_trends` and misclassification pipelines.

| Column | Notes |
|---|---|
| `comparison_id`, `positive_label`, `negative_label`, `feature_set` | Identifiers |
| `embryo_id` | From `id_col` |
| `time_bin`, `time_bin_center`, `bin_width` | Binning |
| `n_positive`, `n_negative`, `auroc_obs` | Per-bin stats |
| `y_true`, `p_pos`, `y_pred`, `is_correct` | Per-embryo prediction |
| `truth_signed_margin` | `[-1, 1]`; `+1` = most correct, `-1` = most wrong |

### Margin conventions

`src/analyze/classification/engine/margins.py`:

| Function | Range | Sign meaning |
|---|---|---|
| `class_signed_margin(p_pos)` | `[-1, 1]` | `+1` = strongest positive-class support; sign relative to decision boundary. Formula: `2 * p_pos - 1`. |
| `truth_signed_margin(p_pos, y_true)` | `[-1, 1]` | `+1` = most correct; sign relative to true label. |
| `coerce_margin_range(arr)` | `[-1, 1]` | Rescales legacy `[-0.5, 0.5]` arrays. Pass loaded-from-disk margins through this. |

`predictions.parquet` stores `truth_signed_margin`. `raw_contrast_scores_long` stores `class_signed_margin`.

### Persistence

```python
result = run_classification(df, ..., save_dir="results/my_run/")   # auto-save at run end
# -- or --
result = run_classification(df, ...)
result.save("results/my_run/", overwrite=False)                    # manual

loaded = ClassificationAnalysis.load("results/my_run/")
preds  = loaded.layers["predictions"]                              # lazy-loaded from parquet
```

On-disk layout:
```
results/my_run/
├── scores.parquet
├── metadata.json                              # result.uns
├── predictions.parquet                        # if save_predictions
├── multiclass_predictions.parquet             # if save_multiclass_predictions
├── confusion.parquet
├── null_distributions.npz                     # if save_null_arrays
├── classifier_directions.parquet              # if save_classifier_directions
├── classifier_directions_vectors.npz          # if save_classifier_directions
├── raw_contrast_scores_long.parquet           # if save_contrast_coordinates
├── contrast_support_long.parquet              # if save_contrast_coordinates
├── contrast_specificity_by_timebin.parquet    # if save_contrast_coordinates
├── raw_coordinates.parquet                    # if save_contrast_coordinates
├── shrunk_coordinates.parquet                 # if save_contrast_coordinates
├── residual_coordinates.parquet               # if save_contrast_coordinates
└── probe_index.parquet                        # if save_contrast_coordinates
```

`ClassificationAnalysis.from_legacy(...)` currently raises `NotImplementedError` — use the legacy loader (`MulticlassOVRResults.from_dir`) for pre-`run_classification` saves.

---

## Visualization

See [`viz/README.md`](viz/README.md) for full plot reference organized by "I want to see X".

One-line index of the most-used plots:

| Goal | Function | Layer required |
|---|---|---|
| AUROC curves over time | `plot_aurocs_over_time` / `result.plot_aurocs()` | `scores` |
| Faceted AUROC heatmap | `plot_auroc_heatmaps` | `scores` |
| Confusion matrices | `plot_confusion` / `result.plot_confusion()` | `confusion` |
| Per-embryo margin trajectory | `plot_margin_trends` | `predictions` |
| Per-embryo phenotype fingerprint | `plot_pairwise_coordinate_heatmap` | `raw_contrast_scores_long` |
| Phenotype emergence timeline (static + interactive HTML) | `compute_emergence_data` + `render_emergence_html` | `scores` |
| Misclassification deep-dive (embryo gallery) | `run_misclassification_pipeline` | `predictions` |

---

## Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `ValueError: No valid comparisons produced results.` | Every comparison fell below the sample-size floor. | Check `min_samples_per_group` / `min_samples_per_member`; verify `id_col` and `class_col` values are non-null and correctly spelled. |
| `KeyError: Layer 'predictions' was not computed during this run.` | `save_predictions=False` on the run, and no `save_dir` was set. | Re-run with `save_predictions=True` (or set `save_dir=...`, which auto-enables it). |
| `ValueError: save_contrast_coordinates=True is only supported for binary comparison runs.` | You're in the multiclass fast path. | Set an explicit `comparisons="all_pairs"` or `positive=/negative=`. Same applies to `save_classifier_directions`. |
| `ValueError: save_contrast_coordinates=True requires n_permutations > 0` | Shrinkage needs the null distribution. | Set `n_permutations >= 1` (typical: 100–500). |
| `FileExistsError` on save | `save_dir` already populated. | Pass `overwrite=True`, or point `save_dir` elsewhere. |
| AUROC pinned at 0.5 across all bins | Usually class imbalance collapsing CV folds, or features not actually predictive in that bin. | Inspect `n_pos`, `n_neg`; visualize raw features by class; consider widening `bin_width`. |

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
    save_contrast_coordinates: bool = False,         # binary-path only; requires n_permutations > 0
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
├── DESIGN.md                       # approved design spec for run_classification
├── future_improvement.md           # historical roadmap snapshot (2026-03-05)
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

### All-vs-rest

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
