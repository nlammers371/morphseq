# Multiclass Classification API - Usage Examples

This document shows how to use the new Scanpy-style comparison API with visualization code.

## Basic Usage

> **Class imbalance default:** the routed `run_classification_test()` path uses logistic regression with `class_weight='balanced'` for both binary and multiclass runs.
>
> **Verbose logging:** with `verbose=True` (default), the run now prints the balancing policy and per-bin class counts so balancing assumptions are explicit in logs.


```python
from analyze.difference_detection import run_classification_test

# Run analysis with flexible API
results = run_classification_test(
    df,
    groupby='cluster_categories',
    groups=['CE', 'HTA', 'Intermediate'],
    reference=['WT', ('WT', 'Het')],
    features='z_mu_b',
    bin_width=4.0,
    n_permutations=100
)
```

## Accessing Results

### Dict-like Access (Interactive Use)

```python
# Get specific comparison
ce_vs_wt = results['CE', 'WT']

# Iterate over all comparisons
for (pos, neg), df in results.iter_comparisons():
    print(f"{pos} vs {neg}: max AUROC = {df['auroc_obs'].max():.2f}")

# List all comparisons
print(results.keys())
# [('CE', 'WT'), ('CE', 'WT+Het'), ('HTA', 'WT'), ...]

# Nested iteration for faceted plots
for neg, sub in results.by_negative().items():
    print(f"=== Reference: {neg} ===")
    for pos, df in sub.items():
        print(f"  {pos}: {len(df)} time bins")
```

### DataFrame Access (Programmatic Use)

```python
# Get full long-format table
df = results.comparisons

# Standard pandas filtering
significant = df[df['pval'] < 0.01]
ce_at_16 = df[(df['positive'] == 'CE') & (df['time_bin'] == 16)]

# Groupby operations
df.groupby('positive')['auroc_obs'].max()
```

## Visualization

### With Existing Plotting Functions

Most existing plotting functions expect a dict-of-DataFrames format. Use `.to_dict_of_dfs()`:

```python
# Convert to legacy format
ovr_dict = results.to_dict_of_dfs()
# {('CE', 'WT'): DataFrame, ('CE', 'WT+Het'): DataFrame, ...}

# Pass to existing plotters
plot_multiclass_ovr_aurocs(ovr_dict, colors_dict=COLORS)
```

### With Seaborn (Recommended for New Code)

The long-format `.comparisons` DataFrame works directly with seaborn:

```python
import seaborn as sns

# Faceted AUROC plot
g = sns.FacetGrid(
    results.comparisons, 
    col='negative',  # Facet by reference group
    hue='positive',  # Color by target class
    col_wrap=3,
    height=4
)
g.map(sns.lineplot, 'time_bin_center', 'auroc_obs', marker='o')
g.add_legend()
```

### Custom Matplotlib Loop

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(results.by_negative()), figsize=(15, 5))

for ax, (neg, sub) in zip(axes, results.by_negative().items()):
    for pos, df in sub.items():
        ax.plot(df['time_bin_center'], df['auroc_obs'], 
                label=f'{pos}', marker='o')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'vs {neg}')
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('AUROC')
    ax.legend()

plt.tight_layout()
```

## Save & Load

```python
# Save results
results.save('results/my_analysis/multiclass_ovr/')
# Creates:
#   - comparisons.parquet
#   - metadata.json

# Load results
from analyze.difference_detection import MulticlassOVRResults
loaded = MulticlassOVRResults.from_dir('results/my_analysis/multiclass_ovr/')

# Metadata is preserved
print(loaded.metadata['groupby'])
print(loaded.metadata['n_permutations'])
```

## Advanced: Filtering Before Plotting

```python
# Filter to significant results only
sig_results = results.filter(pval_lt=0.01)

# Filter to specific time range
late_results = results.filter(time_bin=20)

# Combine filters
sig_ce = results.filter(positive='CE', pval_lt=0.05, auroc_gt=0.7)

# Plot filtered results
plot_auroc_curves(sig_ce)
```

## Comparison: Old vs New API

### Old API (Dict-based)

```python
from analyze.difference_detection import run_multiclass_classification_test

# Manual ID list construction
ce_ids = df[df['cluster'] == 'CE']['embryo_id'].unique().tolist()
wt_ids = df[df['cluster'] == 'WT']['embryo_id'].unique().tolist()
hta_ids = df[df['cluster'] == 'HTA']['embryo_id'].unique().tolist()

groups = {'CE': ce_ids, 'WT': wt_ids, 'HTA': hta_ids}

results = run_multiclass_classification_test(df, groups=groups)

# Returns dict with nested structure
ce_results = results['ovr_classification']['CE']  # CE vs Rest (all others)
```

### New API (Column-based)

```python
from analyze.difference_detection import run_classification_test

# Automatic group resolution from column
results = run_classification_test(
    df,
    groupby='cluster',
    groups=['CE', 'HTA'],
    reference='WT'  # Explicit reference, not "rest"
)

# Returns MulticlassOVRResults object
ce_results = results['CE', 'WT']  # CE vs WT specifically
```

## Migration Guide

If you have existing code using `run_multiclass_classification_test`, you can:

1. **Keep using the old API** - It still works and is not deprecated
2. **Gradual migration** - Use `.to_dict_of_dfs()` to bridge to new API:

```python
# Old code
results_old = run_multiclass_classification_test(df, groups=groups_dict)
ovr_dict = results_old['ovr_classification']

# New code (produces same structure)
results_new = run_classification_test(df, groupby='cluster', groups='all', reference='rest')
ovr_dict = results_new.to_dict_of_dfs()

# Your existing plotting code works unchanged
plot_multiclass_ovr_aurocs(ovr_dict)
```

## API Reference Quick Comparison

| Feature | Old API | New API |
|---------|---------|---------|
| Group specification | Dict of ID lists | Column name + values |
| Reference | Always "rest" | Flexible (single, list, tuple, rest) |
| Return type | Dict | MulticlassOVRResults object |
| Access pattern | `results['ovr_classification']['CE']` | `results['CE', 'WT']` |
| Serialization | Manual CSV per class | `.save()` / `.from_dir()` |
| Metadata | Separate dict | Built-in `.metadata` |
| Backward compat | N/A | `.to_dict_of_dfs()` |
