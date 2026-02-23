# Migration Notes: Consensus Clustering to src/analyze Package

**Target Location:** `src/analyze/consensus_clustering/`
**Status:** Pre-migration planning document
**Date:** 2025-11-07

---

## Overview

When migrating the consensus clustering modules to the package, apply the following improvements to create a professional, maintainable API.

---

## Module Organization

### Target Structure
```
src/analyze/consensus_clustering/
├── __init__.py
├── bootstrap.py              # run_bootstrap_hierarchical() + run_bootstrap_kmedoids()
├── posteriors.py             # analyze_bootstrap_results(), compute_quality_metrics()
├── classification.py         # 2D gating and adaptive classification
├── plotting.py               # All visualization functions
├── data_loading.py           # Data loading utilities
└── utils.py                  # Helper functions
```

---

## API Improvements Before Migration

### 1. Consistent Parameter Naming

**Current inconsistencies:**
- `metric_col` vs `metric_name`
- `step` vs `grid_step`
- `embryo_col` (missing) vs `embryo_id` vs `embryo_id_col`
- `time_col` vs `hpf` vs `predicted_stage_hpf`

**Solution - Create wrapper module with standardized parameters:**

```python
# src/analyze/consensus_clustering/data_loading.py

def load_experiment_data(
    experiment_id: str,
    genotype: str,
    *,
    curv_dir: Optional[Path] = None,
    meta_dir: Optional[Path] = None,
    embryo_id_col: str = 'embryo_id',
    metric_col: str = 'normalized_baseline_deviation',
    time_col: str = 'predicted_stage_hpf',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load and merge curvature metrics with metadata.

    Uses consistent parameter naming:
    - embryo_id_col: Column name for embryo identifiers
    - metric_col: Column name for metric values
    - time_col: Column name for time points
    - genotype: Genotype filter

    Returns DataFrame with standardized columns:
    - embryo_id, time, metric, genotype
    """
    # Implementation
    pass


def extract_and_process_trajectories(
    df: pd.DataFrame,
    *,
    embryo_id_col: str = 'embryo_id',
    metric_col: str = 'normalized_baseline_deviation',
    time_col: str = 'predicted_stage_hpf',
    genotype: Optional[str] = None,
    min_timepoints: int = 3,
    grid_step: float = 0.5,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    Extract trajectories and interpolate to common grid.

    Standardized parameters:
    - embryo_id_col, metric_col, time_col for input columns
    - genotype for filtering
    - grid_step (not 'step') for interpolation

    Returns:
    - trajectories: Interpolated trajectory arrays
    - embryo_ids: Embryo identifiers in same order
    - common_grid: Time points (HPF values)
    """
    # 1. Call extract_trajectories with mapped column names
    # 2. Call interpolate_to_common_grid with grid_step
    # 3. Return only the three key items (hide orig_lens internally)
    pass
```

### 2. Explicit Return Value Documentation

**For ALL functions, document returns in this format:**

```python
def example_function(...) -> Tuple[Dict[str, Any], List[np.ndarray], np.ndarray]:
    """
    Brief description.

    Returns
    -------
    results : dict
        Complete results dictionary with keys:
        - 'posterior_analysis': dict
            - 'p_matrix': np.ndarray, shape (n_embryos, n_clusters)
            - 'max_p': np.ndarray, shape (n_embryos,)
            - 'entropy': np.ndarray, shape (n_embryos,)
            - 'modal_cluster': np.ndarray, shape (n_embryos,), dtype=int
        - 'classification': dict
            - 'category': np.ndarray of strings ('core'/'uncertain'/'outlier')
            - 'cluster': np.ndarray
            - 'thresholds': dict with threshold values used
    trajectories : list of np.ndarray
        Interpolated trajectories, variable lengths
    common_grid : np.ndarray, shape (n_timepoints,)
        Common time grid in HPF units

    Examples
    --------
    >>> results, trajs, grid = example_function(...)
    >>> classification = results['classification']
    >>> core_indices = np.where(classification['category'] == 'core')[0]
    """
```

### 3. Type Hints Everywhere

**Add full type hints to all functions:**

```python
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path

def run_bootstrap_hierarchical(
    D: np.ndarray,
    k: int,
    *,
    n_bootstrap: int = 100,
    frac: float = 0.8,
    random_state: int = 42,
    verbose: bool = False
) -> Dict[str, Any]:
    """Bootstrap hierarchical clustering."""

def analyze_bootstrap_results(
    bootstrap_results_dict: Dict[str, Any],
    return_aligned_labels: bool = False
) -> Dict[str, np.ndarray]:
    """Compute posteriors from bootstrap results."""

def classify_embryos_2d(
    max_p: np.ndarray,
    log_odds_gap: np.ndarray,
    modal_cluster: np.ndarray,
    *,
    threshold_max_p: float = 0.8,
    threshold_log_odds: float = 0.7,
    threshold_outlier_max_p: float = 0.5
) -> Dict[str, Any]:
    """2D gating classification."""
```

### 4. Example Usage in Docstrings

**Every public function should have usage examples:**

```python
def run_bootstrap_hierarchical(...) -> Dict[str, Any]:
    """
    Bootstrap hierarchical clustering.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n × n)
    k : int
        Number of clusters

    Returns
    -------
    dict
        Results with keys: 'reference_labels', 'bootstrap_results'

    Examples
    --------
    >>> from consensus_clustering import run_bootstrap_hierarchical
    >>> import numpy as np
    >>>
    >>> # Create dummy distance matrix
    >>> D = np.random.rand(20, 20)
    >>> D = (D + D.T) / 2  # Make symmetric
    >>> np.fill_diagonal(D, 0)
    >>>
    >>> # Run bootstrap
    >>> results = run_bootstrap_hierarchical(D, k=3, n_bootstrap=100)
    >>>
    >>> # Access results
    >>> labels = results['reference_labels']
    >>> print(f"Cluster assignments: {labels}")
    """
```

### 5. Configuration Module

**Create a centralized config for default values:**

```python
# src/analyze/consensus_clustering/config.py

# Bootstrap parameters
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42

# Data processing
MIN_TIMEPOINTS = 3
GRID_STEP = 0.5
DTW_WINDOW = 5

# Classification thresholds
THRESHOLD_MAX_P = 0.8
THRESHOLD_LOG_ODDS_GAP = 0.7
THRESHOLD_OUTLIER_MAX_P = 0.5

# Column names (standardized)
DEFAULT_EMBRYO_ID_COL = 'embryo_id'
DEFAULT_METRIC_COL = 'normalized_baseline_deviation'
DEFAULT_TIME_COL = 'predicted_stage_hpf'
DEFAULT_GENOTYPE_COL = 'genotype'
```

Then import in functions:
```python
from .config import THRESHOLD_MAX_P, THRESHOLD_LOG_ODDS_GAP, ...

def classify_embryos_2d(
    max_p: np.ndarray,
    log_odds_gap: np.ndarray,
    modal_cluster: np.ndarray,
    *,
    threshold_max_p: float = THRESHOLD_MAX_P,
    threshold_log_odds: float = THRESHOLD_LOG_ODDS_GAP,
    ...
) -> Dict[str, Any]:
    """2D gating classification."""
```

### 6. Package __init__.py

**Export public API clearly:**

```python
# src/analyze/consensus_clustering/__init__.py

from .bootstrap import (
    run_bootstrap_hierarchical,
    run_bootstrap_kmedoids,
)
from .posteriors import (
    analyze_bootstrap_results,
    compute_quality_metrics,
)
from .classification import (
    classify_embryos_2d,
    classify_embryos_adaptive,
    get_classification_summary,
)
from .plotting import (
    plot_posterior_heatmap,
    plot_2d_scatter,
    plot_trajectories_posterior,
    plot_trajectories_category,
)
from .data_loading import (
    load_experiment_data,
    extract_and_process_trajectories,
)

__all__ = [
    'run_bootstrap_hierarchical',
    'run_bootstrap_kmedoids',
    'analyze_bootstrap_results',
    'compute_quality_metrics',
    'classify_embryos_2d',
    'classify_embryos_adaptive',
    'get_classification_summary',
    'plot_posterior_heatmap',
    'plot_2d_scatter',
    'plot_trajectories_posterior',
    'plot_trajectories_category',
    'load_experiment_data',
    'extract_and_process_trajectories',
]
```

### 7. Tests

**Create test file to validate API:**

```python
# src/analyze/consensus_clustering/tests/test_api.py

def test_parameter_consistency():
    """Verify all functions use consistent parameter names."""
    # Ensure no 'step', always 'grid_step'
    # Ensure no 'metric_col'/'metric_name' inconsistency
    # All time columns called 'time_col'

def test_return_values():
    """Verify return values match documented types."""
    # Test analyze_bootstrap_results returns Dict with correct keys
    # Test posterior matrices have correct shapes
    # Test classification has all required keys

def test_example_usage():
    """Run examples from docstrings."""
    # Copy-paste every example and verify it works
```

---

## Checklist for Migration

- [ ] Rename all parameters to be consistent
- [ ] Add type hints to every function signature
- [ ] Expand all docstrings with explicit return value documentation
- [ ] Add usage examples to all public functions
- [ ] Create config.py with default values
- [ ] Create proper __init__.py with __all__
- [ ] Write tests validating API consistency
- [ ] Update README with new import paths
- [ ] Run examples from docstrings to verify they work
- [ ] Get code review from team

---

## Quick Reference: Old → New Parameter Names

| Old | New | Context |
|-----|-----|---------|
| `metric_col` | `metric_col` (standardize usage) | Column selection |
| `metric_name` | `metric_col` | Column selection |
| `step` | `grid_step` | Interpolation |
| `embryo_col` | `embryo_id_col` | Column selection |
| `time_col` | `time_col` (keep) | Column selection |
| `use_c` | (remove) | Not applicable |
| `hpf` | `time_col` | Column selection |

---

## Notes for Implementation

1. **Backward compatibility:** Keep old function signatures as deprecated wrappers that call new ones
2. **Migration path:** Users should update imports to new location but existing code can still work
3. **Documentation:** Update all examples in morphseq docs to use new API
4. **Release notes:** Clearly document what changed and why (API consistency)

---

## Files Ready for Migration

When ready to move, these files are ready:
- `bootstrap_posteriors.py` → `src/analyze/consensus_clustering/posteriors.py`
- `adaptive_classification.py` → `src/analyze/consensus_clustering/classification.py`
- `consensus_clustering_plotting.py` → `src/analyze/consensus_clustering/plotting.py`
- `run_hierarchical_posterior_analysis.py` → example/script (refactor as `example_usage.py`)

These need API improvements:
- `run_hierarchical_posterior_analysis.py` → extract core logic to `bootstrap.py`
- Create new `data_loading.py` with standardized wrappers
