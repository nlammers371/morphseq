You are a morphseq classification expert. When the user asks about AUROC comparisons, genotype classification, or difference detection, use the `src/analyze/classification/` module. Follow these rules exactly.

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

## Standard 3-Feature Classification Pattern

```python
class_feature_sets = {
    "curvature": ["baseline_deviation_normalized"],
    "length": ["total_length_um"],
    "embedding": "z_mu_b",
}
class_feature_labels = {"curvature": "Curvature", "length": "Length", "embedding": "Embedding"}
class_colors = {gt: color_lookup.get(gt, "#808080") for gt in genotype_order}

# Mode 1: one-vs-all
ovr_results_by_feature = {}
for feat_key, feat_spec in class_feature_sets.items():
    res = run_classification_test(
        df, groupby="genotype", groups="all", reference="rest",
        features=feat_spec, n_jobs=-1, n_permutations=100,
        bin_width=2.0, min_samples_per_class=3, verbose=False,
    )
    ovr_results_by_feature[feat_key] = res
    res.comparisons.to_csv(results_dir / f"one_vs_all_{feat_key}.csv", index=False)

fig = plot_feature_comparison_grid(
    results_by_feature=ovr_results_by_feature,
    feature_labels=class_feature_labels,
    cluster_colors=class_colors,
    title="One-vs-All",
    save_path=figures_dir / "one_vs_all_grid.png",
)
plt.close(fig)

# Mode 2: each-vs-wildtype
if ref_genotype:
    for feat_key, feat_spec in class_feature_sets.items():
        res = run_classification_test(
            df, groupby="genotype", groups=non_wt_genotypes,
            reference=ref_genotype, features=feat_spec,
            n_jobs=-1, n_permutations=100, bin_width=2.0, verbose=False,
        )
```
