# Quick Migration Guide

This guide shows how to update existing code to use the refactored structure.

## Before → After

### Imports

**Before:**
```python
from classification import predictive_signal_test, compute_embryo_penetrance
import config
```

**After:**
```python
from difference_detection import run_classification_test
# OR for direct access:
from difference_detection.classification import predictive_signal_test, compute_embryo_penetrance
import config_new as config
```

### Configuration

**Before:**
```python
import config

n_perm = config.N_PERMUTATIONS
n_splits = config.N_CV_SPLITS
alpha = config.ALPHA
```

**After:**
```python
# Use function defaults or override inline
results = run_classification_test(
    df_binned,
    group1="wildtype",
    group2="homozygous",
    n_permutations=100,  # or use env var
    n_cv_splits=5,
    alpha=0.05
)

# Or override via environment variable
import os
os.environ["MORPHSEQ_N_PERMUTATIONS"] = "1000"
```

### Running Analysis

**Before:**
```python
# Manual workflow
df_results, df_probs = predictive_signal_test(
    df_binned,
    group_col="genotype",
    n_splits=config.N_CV_SPLITS,
    n_perm=config.N_PERMUTATIONS,
    return_embryo_probs=True,
    use_class_weights=config.USE_CLASS_WEIGHTS
)

# Filter to groups manually
df_subset = df_binned[df_binned['genotype'].isin(['wildtype', 'homozygous'])]

# Compute penetrance manually
df_penetrance = compute_embryo_penetrance(
    df_probs,
    confidence_threshold=config.CONFIDENCE_THRESHOLD
)

# Detect onset manually
df_sig = df_results[df_results['pval'] < config.ALPHA]
onset_time = df_sig['time_bin'].min() if not df_sig.empty else None
```

**After:**
```python
# Single function call with all the above
results = run_classification_test(
    df_binned,
    group1="wildtype",
    group2="homozygous"
    # Everything else uses sensible defaults
)

# Access results
df_results = results['time_results']
df_probs = results['embryo_probs']
df_penetrance = results['embryo_results']
onset_time = results['onset_info']['onset_time']
```

### Visualization

**Before:**
```python
from visualization.auroc_plots import plot_auroc_with_significance
```

**After:**
```python
from visualization import plot_auroc_with_significance
# OR
from visualization.classification_plots import plot_auroc_with_significance
```

## Common Patterns

### Pattern 1: Simple pairwise comparison

**Before:**
```python
import config
from classification import predictive_signal_test
from utils.binning import bin_by_embryo_time

df_binned = bin_by_embryo_time(df, bin_width=config.TIME_BIN_WIDTH)
df_subset = df_binned[df_binned['genotype'].isin(['wt', 'hom'])]
results, probs = predictive_signal_test(
    df_subset,
    n_splits=config.N_CV_SPLITS,
    n_perm=config.N_PERMUTATIONS
)
```

**After:**
```python
from difference_detection import run_classification_test
from utils.binning import bin_by_embryo_time

df_binned = bin_by_embryo_time(df, bin_width=2.0)
results = run_classification_test(df_binned, group1="wt", group2="hom")
```

### Pattern 2: Custom parameters

**Before:**
```python
results, probs = predictive_signal_test(
    df_subset,
    n_splits=10,
    n_perm=5000,
    random_state=123,
    use_class_weights=True
)
```

**After:**
```python
results = run_classification_test(
    df_binned,
    group1="wt",
    group2="hom",
    n_cv_splits=10,
    n_permutations=5000,
    random_state=123,
    use_class_weights=True
)
```

### Pattern 3: Environment-based configuration

**Before:**
```python
import os
import config

# Override via config
config.N_PERMUTATIONS = int(os.environ.get("N_PERM", 100))
```

**After:**
```python
import os

# Set before importing
os.environ["MORPHSEQ_N_PERMUTATIONS"] = "1000"

# Or override in function call
results = run_classification_test(..., n_permutations=1000)
```

## Step-by-Step Migration

1. **Test new imports**
   ```bash
   python3 test_refactor.py
   ```

2. **Update one analysis at a time**
   - Start with a notebook or script
   - Replace imports
   - Replace function calls
   - Test that results match

3. **Verify results**
   - New and old should give identical results
   - Check that onset times match
   - Verify plots look the same

4. **Clean up**
   - Once verified, can remove old imports
   - Update documentation
   - Update other analyses

## Backwards Compatibility

The old structure still works! You can:

- Keep using `config.py` and `classification/` if needed
- Migrate gradually, one file at a time
- Mix old and new in different notebooks
- Test new approach before committing

## Need Help?

- See `README_REFACTOR.md` for full documentation
- See `IMPLEMENTATION_SUMMARY.md` for what changed
- Run `test_refactor.py` to verify imports
- Use `run_classification.py` as a complete example

---

**TL;DR:** Import from `difference_detection`, use `run_classification_test()`, enjoy cleaner code! 🎯
