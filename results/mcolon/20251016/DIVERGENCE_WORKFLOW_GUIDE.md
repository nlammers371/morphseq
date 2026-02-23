# Divergence Analysis Workflow Guide

## Overview

A modular workflow for analyzing embryo divergence from reference distributions across developmental time. Supports both single-gene and cross-gene pooled comparisons.

---

## Architecture

### Core Modules

**`divergence_analysis/distance.py`**
- `mahalanobis_distance()` - Compute Mahalanobis distance
- `euclidean_distance()` - Compute Euclidean distance
- `compute_both_distances()` - Compute both metrics

**`divergence_analysis/workflow.py`**
- `bin_by_time()` - Bin embryos by developmental time
- `compute_reference_distribution()` - Pool genotypes into reference
- `compute_divergence_scores()` - Calculate divergence metrics

### Existing Utilities (Reused)
- `utils/data_loading.py` - Load experiments from build06
- `utils/binning.py` - Time binning logic
- `plot_embryo_trajectories.py` - Plotting functions

---

## Workflow Pattern

```python
from utils.data_loading import load_experiments
from divergence_analysis.workflow import (
    bin_by_time,
    compute_reference_distribution,
    compute_divergence_scores
)

# 1. Load raw data (can be used for other analyses too)
df_raw = load_experiments(experiment_ids, build_dir)

# 2. Bin by time
df_binned = bin_by_time(df_raw, bin_width=2.0)

# 3. Compute reference (single or pooled genotypes)
ref_stats = compute_reference_distribution(
    df_binned,
    reference_genotypes=['wik', 'wik-ab', 'ab']  # or ['cep290_wildtype']
)

# 4. Compute divergence
df_div = compute_divergence_scores(
    df_binned,
    ref_stats,
    test_genotypes=['cep290_homozygous']
)

# 5. Plot
plot_genotype_comparison(df_div)
```

---

## Running the Analyses

### Analysis 1: B9D2 Within-Gene Comparison

Compare b9d2_homozygous and b9d2_heterozygous to b9d2_wildtype reference.

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251016
python run_b9d2_trajectories.py
```

**Outputs:**
- `data/b9d2/divergence/binned_data.csv`
- `data/b9d2/divergence/hom_vs_wt_divergence.csv`
- `data/b9d2/divergence/het_vs_wt_divergence.csv`
- `plots/b9d2/divergence/*.png`

---

### Analysis 2: TMEM67 Within-Gene Comparison

Compare tmem67_homozygous and tmem67_heterozygote to tmem67_wildtype reference.

```bash
python run_tmem67_trajectories.py
```

**Outputs:**
- `data/tmem67/divergence/binned_data.csv`
- `data/tmem67/divergence/hom_vs_wt_divergence.csv`
- `data/tmem67/divergence/het_vs_wt_divergence.csv`
- `plots/tmem67/divergence/*.png`

---

### Analysis 3: Pooled Cross-Gene Comparison

Compare ALL homozygotes (cep290, b9d2, tmem67) and ALL heterozygotes to a SHARED wildtype pool (wik, wik-ab, ab).

```bash
python run_pooled_trajectories.py
```

**Purpose:** Test whether homozygotes share a common phenotype regardless of which gene is mutated.

**Outputs:**
- `data/pooled/divergence/binned_data.csv`
- `data/pooled/divergence/all_hom_vs_shared_wt_divergence.csv`
- `data/pooled/divergence/all_het_vs_shared_wt_divergence.csv`
- `plots/pooled/divergence/*.png`

---

## Key Design Features

### Modular Workflow
- Load raw data once, use for multiple analyses (binning, PCA, spline fitting, etc.)
- Reusable components: bin → reference → divergence → plot

### Flexible Reference
- **Single genotype**: `['cep290_wildtype']`
- **Pooled genotypes**: `['wik', 'wik-ab', 'ab']` (all pooled together)

### Distance Metrics
- **Mahalanobis**: Accounts for feature correlations and variance
- **Euclidean**: Simple L2 distance

### Binning
- Default: 2-hour bins
- Auto-detects latent features (`z_mu_b_*`)
- Averages embeddings per embryo × time bin

---

## Customization Examples

### Use different bin width
```python
df_binned = bin_by_time(df_raw, bin_width=3.0)  # 3-hour bins
```

### Use custom reference
```python
ref_stats = compute_reference_distribution(
    df_binned,
    reference_genotypes=['custom_genotype_1', 'custom_genotype_2']
)
```

### Compute only Euclidean distance
```python
df_div = compute_divergence_scores(
    df_binned, ref_stats, test_genotypes=[...],
    metrics=['euclidean']
)
```

---

## File Structure

```
results/mcolon/20251016/
├── divergence_analysis/
│   ├── distance.py           # Distance calculations
│   ├── workflow.py           # Main workflow functions
│   └── reference.py          # Reference stats (existing)
├── utils/
│   ├── data_loading.py       # Load experiments (existing)
│   └── binning.py            # Time binning (existing)
├── run_b9d2_trajectories.py     # B9D2 analysis
├── run_tmem67_trajectories.py   # TMEM67 analysis
├── run_pooled_trajectories.py   # Pooled analysis
└── plot_embryo_trajectories.py  # Plotting (existing)
```

---

## Data Flow

```
build06 CSV files
    ↓
load_experiments() → df_raw (raw embeddings + metadata)
    ↓
bin_by_time() → df_binned (averaged per embryo × time)
    ↓
compute_reference_distribution() → ref_stats (mean, cov per time bin)
    ↓
compute_divergence_scores() → df_divergence (distances per embryo × time)
    ↓
plot_genotype_comparison() → plots
```

---

## Notes

- **CEP290 analysis** already exists in `plot_embryo_trajectories.py`
- All three genes now have consistent analysis pipelines
- Pooled analysis enables cross-gene phenotype comparisons
- Raw data can be reused for other analyses (PCA, splines, etc.)
