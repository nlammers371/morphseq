# genotype-trends — Genotype Trend Analysis Workflow

## Overview

Recipe for generating standardized genotype trend analysis scripts for morphseq experiments. Produces feature-over-time plots, proportion plots, and AUROC classification comparisons.

## Template Files

- **Single experiment:** `results/mcolon/20260302_analyze_cep290_20260208_20260210/01_plot_20260208_genotype_trends.py`
- **Combined experiments:** `results/mcolon/20260302_analyze_cep290_20260208_20260210/03_plot_combined_genotype_trends.py`

## Script Constants Block

```python
EXPERIMENT_IDS = ["20260208"]           # or ["20260208", "20260210"] for combined
EXPERIMENT_LABEL = "20260208"           # or "20260208_20260210"
FEATURES = ["total_length_um", "baseline_deviation_normalized"]
OVERLAP_FEATURE = "baseline_deviation_normalized"
```

## Genotype Normalization

```python
def _normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower()
    g = g.replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    # Known typo fixes (add per-experiment as discovered):
    g = g.replace("cep290_unkown", "cep290_unknown")
    g = g.replace("cep290_homozyous", "cep290_homozygous")
    return g
```

## Pipeline Steps

### 1. Load data
```python
from analyze.utils.data_loading import load_experiments
build_dir = project_root / "morphseq_playground" / "metadata" / "build06_output"
df = load_experiments(EXPERIMENT_IDS, build_dir)
```

Or manual loading (current template pattern):
```python
frames = []
for exp_id in EXPERIMENT_IDS:
    path = build_dir / f"df03_final_output_with_latents_{exp_id}.csv"
    part = pd.read_csv(path, low_memory=False)
    frames.append(part)
df = pd.concat(frames, ignore_index=True)
```

### 2. Filter & normalize
```python
if "use_embryo_flag" in df.columns:
    df = df[df["use_embryo_flag"]].copy()
df = df[df["embryo_id"].notna()].copy()
df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
```

### 3. Genotype proportions
```python
embryo_df = df.drop_duplicates(subset="embryo_id")[["embryo_id", "genotype"]].copy()
genotype_counts = embryo_df.groupby("genotype")["embryo_id"].nunique().rename("n_embryos")...
```

### 5. Color lookup
```python
from analyze.trajectory_analysis.viz.styling import get_color_for_genotype
color_lookup = {gt: get_color_for_genotype(gt) for gt in genotype_order}
```

### 6. Feature plots
```python
from analyze.viz.plotting import plot_feature_over_time, plot_proportions
# Faceted by genotype
figs = plot_feature_over_time(df, features=FEATURES, color_by="genotype",
    color_lookup=color_lookup, facet_col="genotype",
    show_individual=True, show_error_band=True, trend_statistic="median", backend="both",
    # Defaults: share y-scale within each row, show tick numbers on every facet.
    repeat_xlabels=False, repeat_ylabels=False,
    repeat_xticklabels=True, repeat_yticklabels=True,
)
# Overlapping single feature
overlap_figs = plot_feature_over_time(df, features=OVERLAP_FEATURE, color_by="genotype",
    color_lookup=color_lookup, show_individual=True, show_error_band=True, backend="both",
    repeat_xlabels=False, repeat_ylabels=False,
    repeat_xticklabels=True, repeat_yticklabels=True,
)
```

### 7. Classification
```python
from analyze.classification import run_classification_test
from analyze.classification.viz.classification import plot_feature_comparison_grid

class_feature_sets = {
    "curvature": ["baseline_deviation_normalized"],
    "length": ["total_length_um"],
    "embedding": "z_mu_b",
}
# Mode 1: one-vs-all
# Mode 2: each-vs-wildtype (if wildtype genotype detected)
```

### 8. Summary text
Write `summary_{LABEL}.txt` with experiment info, genotype proportions, top classification hits.

## Output Layout

```
results/mcolon/{date}_{description}/
  figures/
    {LABEL}_length_curvature_by_genotype.{html,png}
    {LABEL}_{OVERLAP}_overlap_by_genotype.{html,png}
    {LABEL}_raw_genotype_proportions.png
    classification/
      {LABEL}_one_vs_all_feature_grid.png
      {LABEL}_each_vs_wildtype_feature_grid.png
  results/
    raw_genotype_proportions_{LABEL}.csv
    raw_wildtype_vs_mutant_proportions_{LABEL}.csv
    summary_{LABEL}.txt
    classification/
      {LABEL}_*_comparisons.csv
      {LABEL}_*_summary.csv
```
