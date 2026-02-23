# Permutation Testing Consolidation Proposal

## Current State

**What's actually used:** Only `predictive_signal_test()` (AUROC-based)
**What's unused:** `permutation_test_energy()`, `permutation_test_mmd()` - implemented but never called

### The Redundancy Problem

The same permutation test pattern is repeated 3x with slight variations:

| Aspect | AUROC | Energy | MMD |
|--------|-------|--------|-----|
| Shuffle | Label shuffle | Pool shuffle | Pool shuffle |
| P-value | `(count+1)/(n+1)` | `count/n` | `count/n` |
| Return | DataFrame | Dict | Dict |
| Status | **Used** | Unused | Unused |

---

## Proposed Options

### Option A: YAGNI - Delete Unused Code

**Philosophy:** You Ain't Gonna Need It. Energy/MMD aren't used, so delete them.

```
src/analyze/difference_detection/
├── __init__.py          # exports run_classification_test only
├── classification/
│   ├── predictive_test.py   # keep as-is
│   └── penetrance.py
└── (delete distribution/)
```

**Pros:**
- Simplest change
- No new abstractions
- Removes dead code

**Cons:**
- Loses potentially useful distribution tests
- May need to re-implement later

---

### Option B: Generic Permutation Framework (DRY)

**Philosophy:** Extract the common pattern into a reusable function.

```python
# src/analyze/difference_detection/permutation.py

def permutation_test(
    compute_statistic: Callable[[np.ndarray, np.ndarray], float],
    X1: np.ndarray,
    X2: np.ndarray,
    n_permutations: int = 100,
    shuffle_strategy: str = "pool",  # or "label"
    pseudo_count: bool = True,
    random_state: int = None
) -> dict:
    """
    Generic permutation test for any test statistic.

    Parameters
    ----------
    compute_statistic : callable
        Function that takes (X1, X2) and returns a scalar statistic
    shuffle_strategy : str
        "pool" - combine and redistribute (distribution tests)
        "label" - shuffle group labels (classification tests)
    """
    observed = compute_statistic(X1, X2)
    null_dist = []

    for _ in range(n_permutations):
        if shuffle_strategy == "pool":
            X1_perm, X2_perm = _pool_shuffle(X1, X2, rng)
        else:
            X1_perm, X2_perm = _label_shuffle(X1, X2, rng)

        null_dist.append(compute_statistic(X1_perm, X2_perm))

    pval = _compute_pvalue(observed, null_dist, pseudo_count)

    return {
        'statistic': observed,
        'pvalue': pval,
        'null_mean': np.mean(null_dist),
        'null_std': np.std(null_dist),
        'null_distribution': np.array(null_dist)
    }
```

Then each test becomes a thin wrapper:

```python
def test_energy_distance(X1, X2, **kwargs):
    return permutation_test(compute_energy_distance, X1, X2,
                           shuffle_strategy="pool", **kwargs)

def test_auroc(X, y, **kwargs):
    # Slightly different signature for classification
    return permutation_test_classification(compute_cv_auroc, X, y,
                                           shuffle_strategy="label", **kwargs)
```

**Pros:**
- Single source of truth for permutation logic
- Easy to add new metrics (ridge regression, curvature, etc.)
- Consistent p-value calculation

**Cons:**
- More abstraction
- Classification needs special handling (CV structure)

---

### Option C: Rename Module + Light Cleanup (KISS)

**Philosophy:** The name "classification" is misleading. It's really about "difference detection via predictive signal". Rename and standardize.

```
src/analyze/difference_detection/
├── __init__.py
├── predictive_test.py      # renamed from classification/
├── distribution_test.py    # consolidated energy + MMD
├── penetrance.py
└── utils.py                # shared: p-value calc, shuffle strategies
```

Changes:
1. Move `predictive_signal_test` up to main module level
2. Consolidate energy + MMD into one file
3. Extract shared utilities: `compute_pvalue()`, `pool_shuffle()`, `label_shuffle()`
4. Standardize return format across all tests

**Pros:**
- Minimal new abstraction
- Cleaner naming
- Keeps all code, just organized better

**Cons:**
- Still some repetition in test functions
- Doesn't fully unify the pattern

---

## My Recommendation: Option C with elements of B

1. **Keep all three test types** (they serve different purposes)
2. **Extract utilities** to `utils.py`:
   - `compute_permutation_pvalue(observed, null_dist, pseudo_count=True)`
   - `pool_shuffle(X1, X2, rng)`
   - `label_shuffle(y, rng)`
3. **Standardize return format** - all tests return same dict structure
4. **Flatten the module** - remove `classification/` and `distribution/` subdirs
5. **Rename for clarity** - "predictive_test" not "classification"

Final structure:
```
src/analyze/difference_detection/
├── __init__.py              # exports all public functions
├── predictive_test.py       # AUROC-based (label shuffle)
├── distribution_test.py     # Energy + MMD (pool shuffle)
├── penetrance.py            # embryo-level metrics
└── permutation_utils.py     # shared shuffle/pvalue logic
```

---

## Decision Matrix

| Option | Simplicity | DRY | Extensibility | Risk |
|--------|------------|-----|---------------|------|
| A: Delete unused | +++ | + | - | Low |
| B: Generic framework | + | +++ | +++ | Medium |
| C: Rename + cleanup | ++ | ++ | ++ | Low |

---

---

## SELECTED: Option C + B Flavor

**Decision:** Keep but consolidate, build for extensibility.

---

## Implementation Plan

### Step 1: Create `permutation_utils.py`

Core utilities that ALL tests will use:

```python
# src/analyze/difference_detection/permutation_utils.py

def compute_pvalue(
    observed: float,
    null_distribution: np.ndarray,
    alternative: str = "greater",
    pseudo_count: bool = True
) -> float:
    """
    Compute permutation p-value with consistent formula.

    Parameters
    ----------
    observed : float
        Observed test statistic
    null_distribution : array
        Null distribution from permutations
    alternative : str
        "greater" (default), "less", or "two-sided"
    pseudo_count : bool
        If True, use (k+1)/(n+1) formula to avoid zero p-values
    """
    null = np.asarray(null_distribution)
    n = len(null)

    if alternative == "greater":
        k = np.sum(null >= observed)
    elif alternative == "less":
        k = np.sum(null <= observed)
    else:  # two-sided
        k = np.sum(np.abs(null) >= np.abs(observed))

    if pseudo_count:
        return (k + 1) / (n + 1)
    return k / n


def pool_shuffle(X1: np.ndarray, X2: np.ndarray, rng) -> tuple:
    """
    Pool-and-redistribute shuffle for distribution tests.
    Null hypothesis: X1 and X2 come from same distribution.
    """
    combined = np.vstack([X1, X2])
    n1 = len(X1)
    perm_idx = rng.permutation(len(combined))
    return combined[perm_idx[:n1]], combined[perm_idx[n1:]]


def label_shuffle(y: np.ndarray, rng) -> np.ndarray:
    """
    Label shuffle for classification tests.
    Null hypothesis: labels are independent of features.
    """
    return rng.permutation(y)


class PermutationResult:
    """Standardized return format for all permutation tests."""

    def __init__(
        self,
        statistic_name: str,
        observed: float,
        pvalue: float,
        null_distribution: np.ndarray,
        **metadata
    ):
        self.statistic_name = statistic_name
        self.observed = observed
        self.pvalue = pvalue
        self.null_distribution = null_distribution
        self.null_mean = np.mean(null_distribution)
        self.null_std = np.std(null_distribution)
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            self.statistic_name: self.observed,
            'pvalue': self.pvalue,
            'null_mean': self.null_mean,
            'null_std': self.null_std,
            'null_distribution': self.null_distribution,
            **self.metadata
        }

    def __repr__(self):
        return f"PermutationResult({self.statistic_name}={self.observed:.4f}, p={self.pvalue:.4f})"
```

---

### Step 2: Refactor `distribution_test.py`

Consolidate energy + MMD, use shared utilities:

```python
# src/analyze/difference_detection/distribution_test.py

from .permutation_utils import compute_pvalue, pool_shuffle, PermutationResult

def permutation_test_distribution(
    X1: np.ndarray,
    X2: np.ndarray,
    statistic: str = "energy",  # or "mmd"
    n_permutations: int = 1000,
    random_state: int = None,
    **kwargs
) -> PermutationResult:
    """
    Distribution-based permutation test.

    Tests null hypothesis: X1 and X2 come from the same distribution.
    """
    rng = np.random.default_rng(random_state)

    # Select statistic function
    if statistic == "energy":
        compute_stat = compute_energy_distance
    elif statistic == "mmd":
        compute_stat = lambda a, b: compute_mmd(a, b, **kwargs)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Observed
    observed = compute_stat(X1, X2)

    # Null distribution via pool shuffle
    null_dist = []
    for _ in range(n_permutations):
        X1_perm, X2_perm = pool_shuffle(X1, X2, rng)
        null_dist.append(compute_stat(X1_perm, X2_perm))

    pvalue = compute_pvalue(observed, null_dist, pseudo_count=True)

    return PermutationResult(
        statistic_name=statistic,
        observed=observed,
        pvalue=pvalue,
        null_distribution=np.array(null_dist)
    )


# Keep individual functions as convenience wrappers
def permutation_test_energy(X1, X2, **kwargs):
    return permutation_test_distribution(X1, X2, statistic="energy", **kwargs)

def permutation_test_mmd(X1, X2, **kwargs):
    return permutation_test_distribution(X1, X2, statistic="mmd", **kwargs)
```

---

### Step 3: Refactor `predictive_test.py`

Use shared utilities while keeping CV structure:

```python
# src/analyze/difference_detection/predictive_test.py

from .permutation_utils import compute_pvalue, label_shuffle, PermutationResult

def predictive_signal_test(...) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    AUROC-based permutation test with cross-validation.

    (Keep existing API for backwards compatibility)
    """
    # ... existing code ...

    # CHANGE: Use shared p-value function
    pval = compute_pvalue(true_auc, null_aucs, pseudo_count=True)

    # ... rest of existing code ...
```

Minimal change - just swap in the shared `compute_pvalue()` function.

---

### Step 4: Add extensibility for new metrics

The key insight: **separate the statistic from the test**.

```python
# src/analyze/difference_detection/statistics.py

"""
Test statistics that can be plugged into permutation tests.
Each function takes (X1, X2) or (X, y) and returns a scalar.
"""

def compute_energy_distance(X1, X2) -> float:
    """Energy distance between two samples."""
    ...

def compute_mmd(X1, X2, bandwidth=None) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""
    ...

def compute_mean_distance(X1, X2) -> float:
    """Simple Euclidean distance between centroids."""
    return np.linalg.norm(X1.mean(axis=0) - X2.mean(axis=0))

def compute_ridge_coefficient(X, y, alpha=1.0) -> float:
    """Ridge regression coefficient magnitude."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha).fit(X, y)
    return np.linalg.norm(model.coef_)

def compute_cv_auroc(X, y, n_splits=5) -> float:
    """Cross-validated AUROC score."""
    ...
```

Then adding a new test is trivial:

```python
# To add ridge regression test:
from .statistics import compute_ridge_coefficient

def permutation_test_ridge(X, y, **kwargs):
    """Test if ridge coefficients are significant."""
    return permutation_test_supervised(
        compute_ridge_coefficient, X, y,
        shuffle_strategy="label", **kwargs
    )
```

---

### Step 5: Final Module Structure

```
src/analyze/difference_detection/
├── __init__.py              # Public API exports
├── permutation_utils.py     # Core: pvalue, shuffles, PermutationResult
├── statistics.py            # Test statistic functions (plug-and-play)
├── predictive_test.py       # AUROC test (uses label_shuffle)
├── distribution_test.py     # Energy/MMD tests (uses pool_shuffle)
├── penetrance.py            # Embryo-level penetrance metrics
└── plotting.py              # Visualization (if exists)
```

**Public API (`__init__.py`):**
```python
# High-level functions
from .predictive_test import predictive_signal_test, run_classification_test
from .distribution_test import permutation_test_energy, permutation_test_mmd

# Utilities for custom tests
from .permutation_utils import compute_pvalue, pool_shuffle, label_shuffle, PermutationResult
from .statistics import compute_energy_distance, compute_mmd, compute_mean_distance

# Penetrance
from .penetrance import compute_embryo_penetrance
```

---

### Step 6: Backwards Compatibility

Keep `classification/` as a thin re-export layer:

```python
# src/analyze/difference_detection/classification/__init__.py
# DEPRECATED: Use difference_detection directly

from ..predictive_test import predictive_signal_test
from ..penetrance import compute_embryo_penetrance

__all__ = ['predictive_signal_test', 'compute_embryo_penetrance']
```

This allows existing code to keep working while we migrate.

---

## Summary: What Changes

| File | Action |
|------|--------|
| `permutation_utils.py` | **NEW** - shared pvalue, shuffles |
| `statistics.py` | **NEW** - pluggable test statistics |
| `distribution_test.py` | **NEW** - consolidated energy + MMD |
| `predictive_test.py` | **MODIFY** - use shared pvalue |
| `distribution/energy.py` | **DELETE** (moved to distribution_test.py) |
| `distribution/mmd.py` | **DELETE** (moved to distribution_test.py) |
| `classification/__init__.py` | **DEPRECATE** - re-export only |

---

## Adding a New Metric (Example: Curvature Difference)

Once refactored, adding curvature difference detection is ~10 lines:

```python
# In statistics.py
def compute_curvature_difference(X1, X2) -> float:
    """Mean absolute curvature difference between groups."""
    return np.abs(X1.mean() - X2.mean())

# In distribution_test.py (or new file)
def permutation_test_curvature(curvatures_group1, curvatures_group2, **kwargs):
    return permutation_test_distribution(
        curvatures_group1.reshape(-1, 1),
        curvatures_group2.reshape(-1, 1),
        statistic="curvature",  # register in the function
        **kwargs
    )
```

**That's the power of DRY + KISS.**
