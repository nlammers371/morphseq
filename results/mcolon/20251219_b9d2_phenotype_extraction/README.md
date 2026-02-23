# B9D2 Phenotype Extraction

**Date:** 2025-12-19 (Updated 2026-01-05)

## Overview

This directory contains manually curated phenotype lists for b9d2 mutant experiments and a pre-processed dataframe with phenotype labels applied.

## Files

### Phenotype Lists (`phenotype_lists/`)
- `b9d2-CE-phenotype.txt` - Convergent Extension defect embryos (53 embryos)
- `b9d2-HTA-embryos.txt` - Head-Trunk Angle phenotype (25 embryos)
- `b9d2-curved-rescue.txt` - Body Axis rescue phenotype (7 embryos)

### Data Output (`data/`)
- **`b9d2_labeled_data.csv`** - Complete labeled dataframe (ready for analysis)

---

## b9d2_labeled_data.csv

### Description

Complete morphological and embedding data from b9d2 experiments (20251121, 20251125) with phenotype labels pre-applied as `cluster_categories` column.

**Key features:**
- All timepoints for all embryos (no filtering)
- All original columns preserved (morphology + VAE embeddings + metadata)
- Ready to load and use - no need to re-apply phenotype labels

### Data Structure

| Category | Column Name | Description |
|----------|-------------|-------------|
| **Identifiers** | `embryo_id` | Unique embryo identifier (e.g., "20251121_A01_e01") |
| | `experiment_id` | Experiment ID ("20251121" or "20251125") |
| | `genotype` | Original genotype label (e.g., "b9d2_wildtype", "b9d2_mutant") |
| | **`cluster_categories`** | **Phenotype label: 'CE', 'HTA', 'BA_rescue', 'wildtype', or 'unlabeled'** |
| **Time** | `predicted_stage_hpf` | Developmental time (hours post fertilization) |
| | `frame_num` | Frame number in imaging sequence |
| **Morphology** | `total_length_um` | Total body length (micrometers) |
| | `baseline_deviation_um` | Body axis curvature (micrometers) |
| | `baseline_deviation_normalized` | Normalized curvature (baseline_deviation_um / total_length_um) |
| | `head_trunk_angle_deg` | Head-trunk angle (degrees) |
| | `tail_angle_deg` | Tail angle (degrees) |
| **VAE Embeddings** | `z_mu_b_0` ... `z_mu_b_79` | Biological latent features (80 dimensions) |
| | `z_mu_n_0` ... `z_mu_n_19` | Nuisance latent features (20 dimensions) |
| **Quality** | `use_embryo_flag` | Quality control flag (1=valid, 0=exclude) |

### cluster_categories Values

| Value | Description | N Embryos |
|-------|-------------|-----------|
| `'CE'` | Convergent Extension defect | 53 |
| `'HTA'` | Head-Trunk Angle phenotype | 25 |
| `'BA_rescue'` | Body Axis rescue phenotype | 7 |
| `'wildtype'` | Wild-type controls (b9d2_wildtype genotype, not in phenotype lists) | 35 |
| `'unlabeled'` | Other embryos (not classified) | Variable |

### How to Load

```python
import pandas as pd

# Load labeled data
df = pd.read_csv('results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv')

# Check structure
print(f"Total rows: {len(df)}")
print(f"Total embryos: {df['embryo_id'].nunique()}")
print(f"\nCluster categories:")
print(df.groupby('cluster_categories')['embryo_id'].nunique())

# Filter to specific phenotype
df_ce = df[df['cluster_categories'] == 'CE']

# Get VAE biological embeddings
z_cols = [col for col in df.columns if col.startswith('z_mu_b_')]
embeddings = df[z_cols].values
```

### Example Usage

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv')

# Compare CE vs wildtype length trajectories
fig, ax = plt.subplots(figsize=(10, 6))

for phenotype in ['CE', 'wildtype']:
    df_pheno = df[df['cluster_categories'] == phenotype]
    
    # Group by time and calculate mean
    df_grouped = df_pheno.groupby('predicted_stage_hpf')['total_length_um'].mean()
    
    ax.plot(df_grouped.index, df_grouped.values, label=phenotype, linewidth=2)

ax.set_xlabel('Time (hpf)')
ax.set_ylabel('Total Length (μm)')
ax.legend()
plt.show()
```

---

## Regenerating the Data

If phenotype lists are updated, regenerate the labeled dataframe:

```bash
cd results/mcolon/20251219_b9d2_phenotype_extraction
python save_b9d2_labeled_data.py
```

This will:
1. Load experiments 20251121 and 20251125
2. Parse phenotype lists from `phenotype_lists/`
3. Extract wildtype embryos (genotype='b9d2_wildtype', not in phenotype lists)
4. Apply `cluster_categories` labels to all rows
5. Save to `data/b9d2_labeled_data.csv`

---

## Notes

- **No filtering applied**: All timepoints and embryos are included (including `use_embryo_flag=0` if present)
- **Build06 data**: Uses latest data pipeline output (no need for use_embryo_flag filtering)
- **Column naming**: Uses `cluster_categories` (consistent with cep290 analysis convention)
- **Wildtype definition**: Embryos with genotype='b9d2_wildtype' that are NOT in any phenotype list
