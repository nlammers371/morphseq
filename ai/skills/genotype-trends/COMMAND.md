You are a morphseq genotype trend analysis expert. When the user asks you to generate a genotype trend analysis script, follow this recipe exactly. The user will provide experiment IDs and you produce a complete, runnable script.

## Recipe

Generate a Python script that:
1. Loads experiment data from build06 CSVs
2. Normalizes genotype names (fix known typos)
3. Computes genotype proportions
4. Plots feature-over-time (faceted by genotype + overlapping)
5. Plots raw genotype proportions
6. Runs classification (one-vs-all + each-vs-wildtype)
7. Saves summary

## Script Structure

```python
"""
{GENE} genotype trend plots for experiment(s) {EXPERIMENT_LABEL}.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_IDS = ["{exp1}", "{exp2}"]      # user provides these
EXPERIMENT_LABEL = "{exp1}_{exp2}"          # joined with underscore
FEATURES = ["total_length_um", "baseline_deviation_normalized"]
OVERLAP_FEATURE = "baseline_deviation_normalized"


def _normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower()
    g = g.replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    # Add experiment-specific typo fixes here:
    g = g.replace("cep290_unkown", "cep290_unknown")
    g = g.replace("cep290_homozyous", "cep290_homozygous")
    return g
```

## Single-Experiment vs Combined

**Single experiment:** Standard `id_col="embryo_id"`.

**Combined (multiple experiments):**
- All embryo IDs are globally unique — no need for composite IDs
- Add batch-effect check: `facet_col="experiment_id"`

## Template Reference

Use `results/mcolon/20260302_analyze_cep290_20260208_20260210/01_plot_20260208_genotype_trends.py` as the single-experiment template and `03_plot_combined_genotype_trends.py` as the combined template.

## Output Layout

Scripts should output to:
```
results/mcolon/{date}_{description}/
  figures/
    {LABEL}_length_curvature_by_genotype.html
    {LABEL}_length_curvature_by_genotype.png
    {LABEL}_{OVERLAP_FEATURE}_overlap_by_genotype.html
    {LABEL}_{OVERLAP_FEATURE}_overlap_by_genotype.png
    {LABEL}_raw_genotype_proportions.png
    classification/
      {LABEL}_one_vs_all_feature_grid.png
      {LABEL}_each_vs_wildtype_feature_grid.png
  results/
    raw_genotype_proportions_{LABEL}.csv
    summary_{LABEL}.txt
    classification/
      {LABEL}_one_vs_all_summary_all_features.csv
      {LABEL}_each_vs_wildtype_summary_all_features.csv
```

## Classification Pattern

Always use the 3-feature comparison: `{"curvature": ["baseline_deviation_normalized"], "length": ["total_length_um"], "embedding": "z_mu_b"}`.

Standard params: `n_permutations=100`, `bin_width=2.0`, `min_samples_per_class=3`, `n_jobs=-1`.

## Related Skills

- `/analyze-viz` — full function signatures for plotting
- `/analyze-classification` — full function signatures for classification
- `/analyze-utils` — data loading functions
