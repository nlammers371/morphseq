# Refined Embedding and Metric Classification Analysis

## Overview

This analysis framework provides modular tools for comparing groups of embryos using classification-based difference detection (AUROC) and morphological metrics. The key innovation is **clean separation between data preprocessing and visualization**, enabling easy reuse and composition of plotting functions.

---

## Table of Contents

1. [Quick Start: Loading Data](#quick-start-loading-data)
2. [Plot Types and When to Use Them](#plot-types-and-when-to-use-them)
3. [Modular Plotting Architecture](#modular-plotting-architecture)
4. [Example Analyses](#example-analyses)
5. [Adding New Datasets](#adding-new-datasets)

---

## Quick Start: Loading Data

### CEP290 Data

```python
from pathlib import Path
import pandas as pd

# Load CEP290 data (ciliopathy with body axis curvature defects)
DATA_PATH = Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Genotype groups
genotype_mapping = {
    'cep290_wildtype': 'WT',
    'cep290_heterozygous': 'Het',
    'cep290_homozygous': 'Homo'
}
df['genotype_std'] = df['genotype'].map(genotype_mapping)

# Get embryo IDs per genotype
homo_ids = df[df['genotype_std'] == 'Homo']['embryo_id'].unique().tolist()
het_ids = df[df['genotype_std'] == 'Het']['embryo_id'].unique().tolist()
wt_ids = df[df['genotype_std'] == 'WT']['embryo_id'].unique().tolist()
```

### B9D2 Data (Coming Soon)

```python
# B9D2 data (multiple phenotypes)
# TODO: Add path once data is prepared
DATA_PATH = Path("results/mcolon/b9d2_phenotype_extraction/final_data/embryo_data_with_labels.csv")
df = pd.read_csv(DATA_PATH, low_memory=False)

# B9D2 has multiple phenotypes - group by phenotype column
phenotype_groups = df.groupby('phenotype')['embryo_id'].apply(list).to_dict()
```

**Key columns to expect in data:**
- `embryo_id`: Unique embryo identifier
- `predicted_stage_hpf`: Developmental time (hours post fertilization)
- `z_mu_b_*`: VAE latent features (80 columns)
- `baseline_deviation_normalized`: Curvature metric
- `total_length_um`: Length metric
- `genotype` or `phenotype`: Group labels

---

## Plot Types and When to Use Them

### 1. Single AUROC Plot with Null Distribution

**When to use:**
- Comparing **two groups** (e.g., mutant vs wildtype)
- Want to see **classification performance over developmental time**
- Need to show **statistical significance** and **null distribution**

**Example use case:** "Does the VAE embedding detect differences between homozygous and wildtype embryos?"

```python
from utils.preprocessing import prepare_auroc_data
from utils.plotting_functions import plot_multiple_aurocs

# After running compare_groups() for a single comparison
auroc_data = prepare_auroc_data(classification_df)

fig = plot_multiple_aurocs(
    auroc_dfs_dict={'Homo_vs_WT': auroc_data},
    colors_dict={'Homo_vs_WT': '#D32F2F'},
    title='Homozygous vs Wildtype Classification',
    save_path='output/homo_vs_wt.png'
)
```

**Output:** Single AUROC curve with:
- Shaded null distribution band (mean ± 1 SD)
- **Circles** marking p < 0.05 significance
- Reference line at 0.5 (chance)

---

### 2. Multiple Overlaid AUROC Comparisons

**When to use:**
- Comparing **multiple pairwise groups** simultaneously (e.g., Homo vs WT, Homo vs Het, Het vs WT)
- Want to see **relative timing** of when differences emerge
- Comparing **penetrance** across different comparisons

**Example use case:** "When do differences emerge between all three genotypes?"

```python
# After running compare_groups() for multiple comparisons
auroc_dfs_dict = {
    'Homo_vs_WT': prepare_auroc_data(results_homo_wt['classification']),
    'Homo_vs_Het': prepare_auroc_data(results_homo_het['classification']),
    'Het_vs_WT': prepare_auroc_data(results_het_wt['classification'])
}

comparison_colors = {
    'Homo_vs_WT': '#D32F2F',    # Red
    'Homo_vs_Het': '#9467BD',   # Purple
    'Het_vs_WT': '#888888',     # Gray
}

comparison_styles = {
    'Homo_vs_WT': '-',
    'Homo_vs_Het': '-',
    'Het_vs_WT': '--',
}

fig = plot_multiple_aurocs(
    auroc_dfs_dict=auroc_dfs_dict,
    colors_dict=comparison_colors,
    styles_dict=comparison_styles,
    title='Genotype Comparison',
    save_path='output/genotype_comparison.png'
)
```

**Output:** Multiple AUROC curves overlaid, easy to compare timing and strength of signals.

---

### 3. Feature Comparison Panel (1x3)

**When to use:**
- Comparing **different feature types** (curvature, length, VAE embeddings)
- Answering: "Which features best distinguish groups?"
- Understanding **when different signals emerge** (e.g., VAE detects cryptic phenotype before overt curvature)

**Example use case:** "Does the VAE embedding detect differences earlier than morphological metrics?"

```python
from utils.plotting_layouts import create_feature_comparison_panels

# Run classifications with different features
results_curvature = run_comparison_with_features(
    df, comparisons, features=['baseline_deviation_normalized']
)
results_length = run_comparison_with_features(
    df, comparisons, features=['total_length_um']
)
results_embedding = run_comparison_with_features(
    df, comparisons, features='z_mu_b'
)

# Create 1x3 panel
fig = create_feature_comparison_panels(
    results_curvature=results_curvature,
    results_length=results_length,
    results_embedding=results_embedding,
    colors_dict=comparison_colors,
    styles_dict=comparison_styles,
    title='Feature Comparison',
    save_path='output/feature_comparison.png'
)
```

**Output:**
- **Panel 1**: Curvature-based AUROC
- **Panel 2**: Length-based AUROC
- **Panel 3**: VAE embedding-based AUROC
- Each panel shows all comparisons overlaid

**Scientific interpretation:**
- If VAE AUROC > curvature AUROC at early timepoints → **cryptic phenotype** detected
- Compare relative strengths to understand which features are most informative

---

### 4. Three-Panel Comparison Figure

**When to use:**
- Want **comprehensive view** of a single comparison
- Need to show: classification, metric divergence, AND individual trajectories
- For **detailed biological interpretation** of a specific phenotype

**Example use case:** "Full characterization of penetrant vs control embryos"

```python
from utils.plotting_layouts import create_three_panel_comparison

fig = create_three_panel_comparison(
    auroc_data=auroc_data,                    # Pre-processed AUROC
    divergence_data=divergence_smoothed,      # Pre-smoothed divergence
    trajectory_data=trajectories_smoothed,    # Pre-smoothed trajectories
    group1_label='Penetrant',
    group2_label='Control',
    metric_cols=['baseline_deviation_normalized', 'total_length_um'],
    embedding_auroc_data=embedding_auroc,     # Optional overlay
    metric_labels={'baseline_deviation_normalized': 'Curvature'},
    save_path='output/full_comparison.png'
)
```

**Output:**
- **Panel A**: AUROC over time (with optional embedding overlay)
- **Panel B**: Metric divergence (z-scored, smoothed)
- **Panel C**: Individual trajectories + group means

---

## Modular Plotting Architecture

### Design Philosophy

**Key principle:** Plotting functions should be **pure visualization** - no data processing!

```
Raw Data → Preprocessing → Plot-Ready Data → Visualization → Figure
```

### Data Preprocessing Layer (`utils/preprocessing.py`)

All data transformations happen here:

```python
from utils.preprocessing import (
    prepare_auroc_data,        # Add significance flags
    smooth_divergence,         # Gaussian smoothing
    smooth_trajectories,       # Per-embryo smoothing
    limit_trajectories_per_group  # Subsample for cleaner plots
)

# Example workflow
auroc_data = prepare_auroc_data(classification_df)  # Adds is_significant_05, is_significant_01
divergence_smooth = smooth_divergence(divergence_df, sigma=1.5)
trajectories_smooth = smooth_trajectories(df_traj, metric_cols=['baseline_deviation_normalized'])
```

### Plotting Functions Layer (`utils/plotting_functions.py`)

Pure visualization - takes pre-processed data:

```python
from utils.plotting_functions import (
    plot_auroc_with_null,         # Single AUROC with null band
    plot_multiple_aurocs,         # Multiple AUROCs overlaid
    plot_divergence_timecourse,   # Metric divergence
    plot_raw_metric_timecourse    # Raw metric trajectories
)

# All functions expect plot-ready data
plot_auroc_with_null(ax, auroc_data, color='#D32F2F', label='Homo vs WT')
```

### Composition Layer (`utils/plotting_layouts.py`)

High-level functions that compose multiple plots:

```python
from utils.plotting_layouts import (
    create_feature_comparison_panels,  # 1x3 feature comparison
    create_three_panel_comparison,     # Full 3-panel figure
    create_auroc_only_figure          # Simple wrapper
)
```

### Benefits of This Architecture

1. **Reusability**: Mix and match plotting functions
2. **Testability**: Test data prep separately from visualization
3. **Performance**: Compute preprocessing once, use in multiple plots
4. **Flexibility**: Easy to add new plot types
5. **Clarity**: Explicit data flow

---

## Example Analyses

### Example 1: Genotype Comparison (CEP290)

**Goal:** Compare Homo vs WT, Homo vs Het, Het vs WT using three feature types

See: `genotype_comparison_analysis.py`

**Key steps:**
1. Load CEP290 data
2. Define genotype groups
3. Run classifications with curvature, length, and VAE features
4. Generate 1x3 feature comparison panel

**Outputs:**
- `genotype_comparison_auroc.png` - VAE only
- `genotype_comparison_by_feature.png` - 1x3 panel
- CSV files with AUROC results

---

### Example 2: Cryptic Phenotype Detection (CEP290)

**Goal:** Show that VAE detects differences before overt morphology

See: `cep290_analysis.py`

**Key insight:**
- If VAE AUROC becomes significant at ~18 hpf
- But curvature AUROC only significant at ~24 hpf
- → **Cryptic window**: 18-24 hpf where embedding detects subtle shape changes

**Outputs:**
- Full 3-panel comparison showing embedding vs metric timing
- Cryptic window analysis CSV

---

### Example 3: B9D2 Multi-Phenotype Analysis (Template)

**Goal:** Compare multiple B9D2 phenotypes

```python
# Load B9D2 data
df = pd.read_csv(B9D2_DATA_PATH)

# Define phenotype groups
phenotypes = df['phenotype'].unique()
phenotype_groups = {p: df[df['phenotype'] == p]['embryo_id'].unique().tolist()
                   for p in phenotypes}

# Compare each phenotype vs wildtype
comparisons = {}
wt_ids = phenotype_groups['wildtype']
for pheno in phenotypes:
    if pheno == 'wildtype':
        continue
    comparisons[f'{pheno}_vs_WT'] = (
        phenotype_groups[pheno],
        wt_ids,
        pheno,
        'WT'
    )

# Run feature comparison analysis
results_curvature = run_comparison_with_features(
    df, comparisons, features=['baseline_deviation_normalized']
)
results_embedding = run_comparison_with_features(
    df, comparisons, features='z_mu_b'
)

# Create comparison plot
fig = create_feature_comparison_panels(
    results_curvature=results_curvature,
    results_length={},  # Optional: skip length if not relevant
    results_embedding=results_embedding,
    colors_dict=phenotype_colors,
    save_path='output/b9d2_phenotype_comparison.png'
)
```

---

## Adding New Datasets

### Required Data Format

Your CSV must have these columns:

**Required:**
- `embryo_id` (str): Unique embryo identifier
- `predicted_stage_hpf` (float): Developmental time
- Group labels: `genotype`, `phenotype`, or custom column

**Feature columns (at least one):**
- `z_mu_b_00` through `z_mu_b_99`: VAE latent features (80 dimensions)
- `baseline_deviation_normalized`: Curvature metric
- `total_length_um`: Length metric
- Any custom morphological metrics

**Optional:**
- `experiment_id`, `well`, `treatment`, etc. for metadata

### Data Preparation Checklist

1. ✅ Load raw data and check for required columns
2. ✅ Define group labels (genotype/phenotype mapping)
3. ✅ Get embryo ID lists per group
4. ✅ Check data distribution (how many embryos per group?)
5. ✅ Verify feature columns exist

### Template Script

```python
#!/usr/bin/env python
"""
Template analysis script for new dataset.

Modify the paths and group definitions below.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import matplotlib
matplotlib.use('Agg')

from utils.preprocessing import prepare_auroc_data
from utils.plotting_layouts import create_feature_comparison_panels

# ============= MODIFY THESE =============
DATA_PATH = PROJECT_ROOT / "path/to/your/data.csv"
OUTPUT_DIR = Path(__file__).parent / "output" / "your_analysis_name"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define your groups
GROUP_COLUMN = 'phenotype'  # or 'genotype'
GROUP1_VALUE = 'your_phenotype'
GROUP2_VALUE = 'wildtype'
# ========================================

def main():
    # Load data
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df)} rows, {df['embryo_id'].nunique()} embryos")

    # Check group distribution
    print(f"\n{GROUP_COLUMN} distribution:")
    print(df.groupby(GROUP_COLUMN)['embryo_id'].nunique())

    # Get embryo IDs
    group1_ids = df[df[GROUP_COLUMN] == GROUP1_VALUE]['embryo_id'].unique().tolist()
    group2_ids = df[df[GROUP_COLUMN] == GROUP2_VALUE]['embryo_id'].unique().tolist()

    # Define comparisons
    comparisons = {
        f'{GROUP1_VALUE}_vs_{GROUP2_VALUE}': (group1_ids, group2_ids, GROUP1_VALUE, GROUP2_VALUE)
    }

    # Run analysis (add your analysis code here)
    # See genotype_comparison_analysis.py for full example

if __name__ == '__main__':
    main()
```

---

## Best Practices

### 1. Binning Strategy

**Time binning** aggregates data within time windows:
- **2-hour bins** (default): Good for most analyses, balances resolution vs sample size
- **4-hour bins**: Use if sample sizes are small (<50 embryos per group)
- **1-hour bins**: Use if data is dense and sample sizes are large (>200 embryos)

```python
compare_groups(df, bin_width=2.0, ...)  # 2-hour bins
```

### 2. Feature Selection

**When to use which features:**
- **VAE embeddings** (`z_mu_b`): Best for detecting subtle/cryptic phenotypes
- **Curvature** (`baseline_deviation_normalized`): For obvious body axis defects
- **Length** (`total_length_um`): For gross morphology changes
- **Custom metrics**: Any quantifiable morphological feature

### 3. Statistical Considerations

**P-value thresholds:**
- **p < 0.05**: Standard significance (shown as circles)
- Earlier work used p < 0.01 (stars), but simplified to single threshold

**Null distribution:**
- Generated via **permutation testing** (default: 500 permutations)
- Shown as shaded band (mean ± 1 SD)

**Within-bin time stratification:**
- Prevents confounding when groups have different age distributions within bins
- Enabled by default in `compare_groups()`

### 4. Colors and Styles

**Standard color scheme for genotypes:**
```python
comparison_colors = {
    'Homo_vs_WT': '#D32F2F',    # Red (strong effect)
    'Homo_vs_Het': '#9467BD',   # Purple (intermediate)
    'Het_vs_WT': '#888888',     # Gray (weak effect)
}

comparison_styles = {
    'Homo_vs_WT': '-',          # Solid
    'Homo_vs_Het': '-',         # Solid
    'Het_vs_WT': '--',          # Dashed (weaker signal)
}
```

Adapt for your phenotypes!

---

## Troubleshooting

### "Not enough values to unpack" error

**Cause:** Mismatch in `_run_classification()` return values

**Fix:** Ensure `src/analyze/difference_detection/comparison.py` has been updated to return 3 values:
```python
return df_results, df_embryo_probs, df_diagnostics
```

### "Column not found" errors

**Cause:** Data missing required columns

**Fix:** Check data has:
- `embryo_id`
- `predicted_stage_hpf`
- Feature columns (`z_mu_b_*`, metrics)

### "Empty DataFrame" in plots

**Cause:** No significant results or wrong group filtering

**Fix:**
- Check group IDs are correct
- Verify embryos exist in both groups
- Lower `min_samples_per_bin` if needed

---

## Contributing

When adding new analysis types:

1. **Data prep functions** → `utils/preprocessing.py`
2. **Plot functions** → `utils/plotting_functions.py`
3. **Layout functions** → `utils/plotting_layouts.py`
4. **Analysis scripts** → Top-level (e.g., `b9d2_analysis.py`)

**Keep plotting functions pure!** No data processing in visualization code.

---

## References

- **CEP290 analysis:** `genotype_comparison_analysis.py`, `cep290_analysis.py`
- **Plotting architecture:** `utils/plotting_*.py`
- **Classification API:** `src/analyze/difference_detection/comparison.py`

---

**Questions?** Check existing analysis scripts for examples, or ask!
