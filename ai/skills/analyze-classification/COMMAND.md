You are a morphseq classification expert. When the user asks about AUROC comparisons, genotype classification, or difference detection, use the `src/analyze/classification/` module. Follow these rules exactly.

**Full reference:** `src/analyze/classification/README.md` is the authoritative source. This COMMAND.md shows quick-start patterns.

**For margin trajectory plots:** see `VIZ.md` in this skill directory for `plot_margin_trends` API and usage examples.

**Important:** The canonical module is `analyze.classification`. The old `analyze.difference_detection` is a deprecated shim that re-exports from `analyze.classification` — it still works but prefer the canonical path.

## Setup

```python
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[N]  # adjust N
sys.path.insert(0, str(project_root / "src"))

from analyze.classification import run_classification_test, MulticlassOVRResults
from analyze.classification.viz.classification import (
    plot_feature_comparison_grid,
    plot_multiple_aurocs,
)
# Legacy import (still works):
# from analyze.difference_detection import run_classification_test, plot_feature_comparison_grid
```

## Key Functions

### `run_classification_test(df, groupby=, groups=, reference=, features=, ...)`

Runs time-binned AUROC classification with permutation testing.

**Mode control via `groups` + `reference`:**
- **One-vs-rest:** `groups="all", reference="rest"` — each group vs all others
- **Each-vs-ref:** `groups=non_wt_genotypes, reference="cep290_wildtype"` — each non-WT vs WT
- **Binary:** `groups=["A"], reference="B"` — single comparison

**Feature spec:** `features="z_mu_b"` auto-detects all `z_mu_b_*` columns. Or pass explicit list `["total_length_um"]`.

Returns `MulticlassOVRResults`.

### `MulticlassOVRResults`

Dict-like access to comparison DataFrames:
```python
res["cep290_homozygous", "cep290_wildtype"]  # -> DataFrame with time_bin, auroc, pval columns
res.summary()           # -> DataFrame with min_pval, max_auroc per comparison
res.comparisons         # -> full DataFrame of all bins × comparisons
res.save(path)          # -> saves to directory
res = MulticlassOVRResults.from_dir(path)  # -> load
```

### `plot_feature_comparison_grid(results_by_feature, feature_labels, cluster_colors, ...)`

Side-by-side AUROC panels comparing feature types. `results_by_feature = {"embedding": res1, "length": res2}`.

### `plot_multiple_aurocs(auroc_dfs_dict, colors_dict, ...)`

Overlay multiple AUROC curves on one axis.

## Standard pattern: multiple feature sets

```python
from analyze.classification import run_classification

results_by_feature = {}
for feat_key, feat_spec in {"vae": "z_mu_b", "shape": ["total_length_um"]}.items():
    res = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        features={feat_key: feat_spec},
        comparisons="all_pairs",
        bin_width=2.0,
        n_splits=5,
        n_permutations=500,
        n_jobs=-1,
        save_predictions=True,
        save_dir=results_dir / feat_key,
    )
    results_by_feature[feat_key] = res
    res.plot_aurocs(output_path=figures_dir / f"aurocs_{feat_key}.png")
```

## Generating predictions for `plot_margin_trends`

Use `run_classification` (not the legacy `run_classification_test`) with `save_predictions=True`:

```python
from analyze.classification import run_classification

result = run_classification(
    df,
    class_col="genotype",
    id_col="embryo_id",
    time_col="predicted_stage_hpf",
    positive="pbx4_crispant",
    negative="inj_ctrl",
    features={"vae": "z_mu_b", "shape": ["total_length_um"]},
    comparisons="all_pairs",
    bin_width=2.0,
    n_splits=5,
    n_permutations=500,
    n_jobs=-1,
    save_predictions=True,
    save_dir="results/my_run/",
)

# predictions.parquet is at: results/my_run/predictions.parquet
predictions = result.layers["predictions"]
```

Then plot directly — see `VIZ.md` for the full `plot_margin_trends` API.
