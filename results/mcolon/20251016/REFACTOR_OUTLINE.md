# Refactored Package Structure - Simple & Flexible

## Goal
Create a unified analysis package with two difference detection approaches that produce **comparable outputs** for downstream integration.

## Directory Structure

```
20251016/
├── config.py                          # SIMPLIFIED - just paths and experiment lists
│
├── utils/                             # Shared utilities (keep as-is)
│   ├── data_loading.py
│   ├── binning.py
│   └── file_utils.py
│
├── difference_detection/              # NEW: Unified parent module
│   │
│   ├── classification/                # Logistic regression approach
│   │   ├── __init__.py
│   │   ├── predictive_test.py        # Core classification test
│   │   └── penetrance.py             # Embryo-level metrics
│   │
│   ├── distribution/                  # Energy distance approach
│   │   ├── __init__.py
│   │   ├── energy_tests.py           # Energy distance & Hotelling's T²
│   │   ├── pairwise_tests.py         # Parallelized testing
│   │   ├── onset_detection.py        # FDR correction & onset finding
│   │   └── bootstrap.py              # Bootstrap/LOO robustness
│   │
│   └── __init__.py                    # Common interface
│
├── visualization/                     # Plotting functions
│   ├── classification_plots.py       # AUROC, trajectories, penetrance
│   └── distribution_plots.py         # Energy distance, bootstrap, LOO
│
├── run_classification.py              # Classification analysis script
├── run_distribution.py                # Distribution analysis script
└── compare_methods.py                 # NEW: Compare both approaches
```

## Key Design Principle: **Common Output Format**

Both approaches should return similar structures:

```python
# Common output for both methods:
{
    'time_results': pd.DataFrame,      # One row per time bin
                                       # Columns: time_bin, statistic, pvalue, ...

    'embryo_results': pd.DataFrame,    # One row per embryo (if applicable)
                                       # Columns: embryo_id, genotype, metric, ...

    'onset_info': dict,                # Onset detection results
                                       # Keys: onset_time, pvalue, ci_low, ci_high

    'comparison_info': dict            # Metadata about comparison
                                       # Keys: group1, group2, method, params
}
```

## Simplified Config

```python
# config.py - MUCH SIMPLER
"""Just the essentials - everything else is function defaults."""

import os

# Paths
RESULTS_DIR = "/path/to/results/20251016"
DATA_DIR = os.path.join(RESULTS_DIR, "data")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# Experiments
WT_EXPERIMENTS = ["20230615", "20230531", "20230525", "20250912"]
B9D2_EXPERIMENTS = ["20250519", "20250520"]
CEP290_EXPERIMENTS = ["20250305", "20250416", "20250512", "20250515_part2", "20250519"]
TMEM67_EXPERIMENTS = ["20250711"]

# Genotype groups
GENOTYPE_GROUPS = {
    "cep290": ["cep290_wildtype", "cep290_heterozygous", "cep290_homozygous"],
    "b9d2": ["b9d2_wildtype", "b9d2_heterozygous", "b9d2_homozygous"],
    "tmem67": ["tmem67_wildtype", "tmem67_heterozygote", "tmem67_homozygous"],
}

# That's it! Everything else is function defaults.
```

## Common Interface for Both Methods

```python
# difference_detection/__init__.py

def run_classification_test(df_binned, group1, group2, **kwargs):
    """
    Run classification-based difference detection.

    Returns
    -------
    dict with keys: time_results, embryo_results, onset_info, comparison_info
    """
    pass


def run_distribution_test(df_binned, group1, group2, **kwargs):
    """
    Run distribution-based difference detection (energy distance).

    Returns
    -------
    dict with keys: time_results, embryo_results, onset_info, comparison_info
    """
    pass
```

## Function Defaults Instead of Config

```python
# Example: Smart defaults in functions

def run_classification_test(
    df_binned,
    group1,
    group2,
    n_cv_splits=5,              # Sensible default
    n_permutations=100,         # Can override via env var or arg
    use_class_weights=True,     # Smart default
    random_state=42,
    **kwargs
):
    # Check environment variable for overrides
    n_permutations = int(os.environ.get("N_PERMUTATIONS", n_permutations))
    ...
```

## Implementation Priority

### Phase 1: Reorganize Existing (Quick)
1. Create `difference_detection/` directory
2. Move `classification/` module inside
3. Update imports

### Phase 2: Add Distribution Methods (Later)
1. Create `difference_detection/distribution/` module
2. Extract functions from detection script
3. Implement common output format

### Phase 3: Integration (Later)
1. Create `compare_methods.py`
2. Unified visualization comparing both approaches
3. Documentation

## Benefits of This Approach

✅ **Simple**: Config is just paths and experiment lists
✅ **Flexible**: Change parameters via function args or env vars
✅ **Comparable**: Common output format enables integration
✅ **Modular**: Easy to use either method independently
✅ **Testable**: Clear interfaces, easy to test
✅ **Not Overengineered**: Only what you need right now

## Next Steps

1. Simplify `config.py` (remove helper functions, constants)
2. Create `difference_detection/` structure
3. Move existing classification code
4. Add distribution code when ready
5. Build comparison tools when needed

No overengineering - just what you need to test things out! 🎯
