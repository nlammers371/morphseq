# Refactored Phenotype Emergence Analysis

**Date:** October 16, 2025  
**Status:** Phase 1 Complete ✓

## Overview

This package provides a unified interface for detecting phenotypic differences using two approaches:

1. **Classification-based**: Logistic regression with permutation testing
2. **Distribution-based**: Energy distance and Hotelling's T² (coming soon)

Both approaches produce **compatible outputs** for downstream integration.

## Directory Structure

```
20251016/
├── config_new.py                      # ✓ Simplified config (paths + experiments only)
│
├── utils/                             # ✓ Shared utilities (unchanged)
│   ├── data_loading.py
│   ├── binning.py
│   └── file_utils.py
│
├── difference_detection/              # ✓ NEW: Unified parent module
│   ├── __init__.py                    # ✓ Common interface
│   │
│   ├── classification/                # ✓ Classification approach
│   │   ├── __init__.py
│   │   ├── predictive_test.py        # Core classification test
│   │   └── penetrance.py             # Embryo-level metrics
│   │
│   └── distribution/                  # ⏳ Distribution approach (placeholder)
│       └── __init__.py
│
├── visualization/                     # ✓ Plotting functions
│   ├── classification_plots.py       # AUROC plots (renamed from auroc_plots)
│   ├── trajectory_plots.py           # Signed margin plots
│   └── penetrance_plots.py           # Penetrance distributions
│
├── run_classification.py              # ✓ NEW: Simplified run script
├── run_distribution.py                # ⏳ Placeholder
├── compare_methods.py                 # ⏳ Placeholder
│
└── README_REFACTOR.md                 # This file
```

## Quick Start

### Basic Usage

```python
from difference_detection import run_classification_test
from utils.data_loading import load_experiments
from utils.binning import bin_by_embryo_time
import config_new as config

# Load and bin data
df = load_experiments(config.CEP290_EXPERIMENTS, config.BUILD06_DIR)
df_binned = bin_by_embryo_time(df, time_col="predicted_stage_hpf", bin_width=2.0)

# Run test
results = run_classification_test(
    df_binned,
    group1="cep290_wildtype",
    group2="cep290_homozygous",
    n_cv_splits=5,
    n_permutations=100,
    use_class_weights=True
)

# Results structure
print(results.keys())
# dict_keys(['time_results', 'embryo_results', 'embryo_probs', 'onset_info', 'comparison_info'])
```

### Run Complete Analysis

```bash
# Use default settings (100 permutations)
python run_classification.py

# Override permutations via environment variable
MORPHSEQ_N_PERMUTATIONS=1000 python run_classification.py
```

## Common Output Format

Both detection methods return a dictionary with standardized keys:

```python
{
    'time_results': pd.DataFrame,      # One row per time bin
                                       # Columns: time_bin, statistic, pvalue, ...

    'embryo_results': pd.DataFrame,    # One row per embryo (if applicable)
                                       # Columns: embryo_id, genotype, metric, ...

    'embryo_probs': pd.DataFrame,      # Per-embryo predictions (classification only)

    'onset_info': dict,                # Onset detection results
                                       # Keys: onset_time, pvalue, is_significant

    'comparison_info': dict            # Metadata about comparison
                                       # Keys: group1, group2, method, params
}
```

## Configuration Philosophy

### Old Way (Complicated)
```python
# config.py had 186 lines with many parameters
N_PERMUTATIONS = 100
N_CV_SPLITS = 5
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.1
# ... many more ...
```

### New Way (Simple)
```python
# config_new.py has ~45 lines - just paths and experiments
RESULTS_DIR = "/path/to/results"
CEP290_EXPERIMENTS = ["20250305", "20250416", ...]
GENOTYPE_GROUPS = {"cep290": ["wildtype", "heterozygous", "homozygous"]}
# That's it!
```

**Parameters are now function defaults:**
```python
def run_classification_test(
    df_binned,
    group1,
    group2,
    n_cv_splits=5,              # Sensible default
    n_permutations=100,         # Override via env var if needed
    use_class_weights=True,     # Best practice default
    random_state=42,
    **kwargs
):
    # Can override via environment variable
    n_permutations = int(os.environ.get("MORPHSEQ_N_PERMUTATIONS", n_permutations))
    ...
```

## Implementation Status

### ✅ Phase 1: Reorganize Existing (Complete)
- [x] Create `difference_detection/` directory
- [x] Move `classification/` module inside
- [x] Create simplified `config_new.py`
- [x] Create common interface in `difference_detection/__init__.py`
- [x] Update visualization imports
- [x] Create simplified `run_classification.py`

### ⏳ Phase 2: Add Distribution Methods (Future)
- [ ] Create `difference_detection/distribution/` module
  - [ ] `energy_tests.py` - Energy distance & Hotelling's T²
  - [ ] `pairwise_tests.py` - Parallelized testing
  - [ ] `onset_detection.py` - FDR correction & onset finding
  - [ ] `bootstrap.py` - Bootstrap/LOO robustness
- [ ] Implement `run_distribution_test()`
- [ ] Create `visualization/distribution_plots.py`
- [ ] Update `run_distribution.py`

### ⏳ Phase 3: Integration (Future)
- [ ] Implement `compare_methods.py`
- [ ] Unified visualization comparing both approaches
- [ ] Documentation and examples

## Benefits

✅ **Simple**: Config is just paths and experiment lists  
✅ **Flexible**: Change parameters via function args or env vars  
✅ **Comparable**: Common output format enables integration  
✅ **Modular**: Easy to use either method independently  
✅ **Testable**: Clear interfaces, easy to test  
✅ **Not Overengineered**: Only what you need right now  

## Migration Guide

### Old code:
```python
from classification import predictive_signal_test
import config

results, probs = predictive_signal_test(
    df_binned,
    n_splits=config.N_CV_SPLITS,
    n_perm=config.N_PERMUTATIONS
)
```

### New code:
```python
from difference_detection import run_classification_test

results = run_classification_test(
    df_binned,
    group1="wildtype",
    group2="homozygous"
    # Uses sensible defaults, or override as needed
)
```

## Testing

```bash
# Quick test with minimal permutations
MORPHSEQ_N_PERMUTATIONS=10 python run_classification.py

# Standard analysis
python run_classification.py

# Publication-quality with many permutations
MORPHSEQ_N_PERMUTATIONS=5000 python run_classification.py
```

## Next Steps

1. Test the new structure with real data
2. Migrate existing analysis notebooks to use new interface
3. Add distribution methods when needed
4. Build comparison tools when both methods are ready

No overengineering - just what you need to test things out! 🎯
