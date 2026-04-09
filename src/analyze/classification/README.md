# Classification Module

Time-resolved binary and multiclass classification with permutation testing. Runs cross-validated AUROC at each developmental time bin, with optional null distributions from label permutation.

## Overview

A single `run_classification()` entry point first resolves the requested comparisons, then runs either the multiclass path or the binary path depending on that resolution. Returns a `ClassificationAnalysis` object containing scores, metadata, and optionally lazy-loaded artifacts (predictions, confusion matrices, null arrays). All comparison modes — default multiclass, all-pairs, explicit pairs, pooled groups, and design tables — are supported in one call. Multiple feature sets can be compared simultaneously.

**Full API reference:** see [`RUN_CLASSIFICATION_API.md`](RUN_CLASSIFICATION_API.md).

---

## Quick Start

```python
from analyze.classification import run_classification

# Default multiclass (reported one-vs-rest)
result = run_classification(
    df,
    class_col="genotype",
    id_col="embryo_id",
    time_col="predicted_stage_hpf",
    features={"emb": "z_mu_b"},
)
result.plot_aurocs(output_path="aurocs.png")

# Explicit binary pair, save predictions
result = run_classification(
    df,
    class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    positive="homo", negative="wildtype",
    features={"emb": "z_mu_b", "shape": ["total_length_um"]},
    n_permutations=500, save_predictions=True,
    save_dir="results/homo_vs_wt/",
)

# Pooled comparison (tuple = pooled, list = enumerated)
result = run_classification(
    df,
    class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    positive=("het", "homo"),  # treated as one pooled group
    negative="wildtype",
    features={"emb": "z_mu_b"},
)
```

---

## Core Concepts

### Comparison Resolution

The three parameters `positive`, `negative`, and `comparisons` control what gets compared. The most important rules:

- **Tuples mean pooled.** `("wik_ab", "inj_ctrl")` is one pooled group.
- **Lists mean enumerated.** `["wik_ab", "inj_ctrl"]` means two separate comparisons.
- **Any explicit `negative=`** keeps the run binary, even with pooled groups.
- **`one-vs-rest` is reporting, not a fitting mode.** The model is multiclass; results are reported per class.

| What you want | How to request it |
|---|---|
| All classes vs rest (default) | `positive=None, negative=None, comparisons=None` |
| All pairs | `comparisons="all_pairs"` |
| Single pair | `positive="A", negative="B"` |
| Pooled positive vs negative | `positive=("A","B"), negative="C"` |
| Each non-WT vs one control | `positive=None, negative="wildtype"` |
| Explicit design table | `comparisons=[{"positive":"A","negative":"B"}, ...]` |

Forbidden combinations (raise `ValueError`):
- `comparisons=DataFrame/list` with `positive` or `negative`
- `comparisons="all_pairs"` with `negative` or scalar `positive`

### Execution vs Reporting

Two paths:
- **Multiclass path** (default): one model per time bin, reported as one-vs-rest AUROC per class.
- **Binary path** (any explicit comparison): one binary model per resolved comparison, one AUROC series per pair.

`A vs B` → binary. `A vs rest` → multiclass reporting. `A vs (X+Y)` → binary (samples relabeled before fitting).

### ClassificationAnalysis

```python
result.scores      # pd.DataFrame — one row per (feature_set, comparison_id, time_bin)
result.uns         # dict — run metadata (class_col, comparisons, git_commit, ...)
result.layers      # lazy-loaded artifacts — predictions, confusion, null arrays
```

Key columns in `result.scores`: `feature_set`, `comparison_id`, `positive_label`, `negative_label`, `time_bin_center`, `auroc_obs`, `pval`.

### Artifacts and Saving

```python
# Auto-save at run time
result = run_classification(df, ..., save_predictions=True, save_dir="results/my_run/")
# Writes: scores.parquet, metadata.json, layers/ (predictions, etc.)

# Load later
loaded = ClassificationAnalysis.load("results/my_run/")
preds = loaded.layers["predictions"]   # predictions.parquet
```

Artifact keys: `"predictions"` (`save_predictions=True`), `"multiclass_predictions"`, `"confusion"` (always), `"null_full"` (`save_null_arrays=True`).

---

## Further Reading

- [`RUN_CLASSIFICATION_API.md`](RUN_CLASSIFICATION_API.md) — full function signature, argument semantics, schema tables, persistence, migration, architecture, tests
- Visualization references — see `src/analyze/classification/viz/` for plotting helpers such as `plot_margin_trends` (per-embryo margin trajectories) and `plot_aurocs_over_time` (AUROC curves)
