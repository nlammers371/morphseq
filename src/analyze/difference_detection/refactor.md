# Difference Detection Refactor Plan (Run Convention + Deviation)

This document captures the agreed naming and API refactor plan using a
run-oriented convention with "deviation" to capture biological intent
(mutants deviating from wildtype). The goal is a clear, package-friendly
surface for new users.

## Final Recommended Naming

### File Names

```
difference_detection/
├── classification_test.py              # Binary statistical testing (TODO)
├── classification_test_multiclass.py   # Multiclass statistical testing (TODO)
├── trajectory_deviation.py             # Direct metric comparison (TODO)
├── distribution_test.py                # Distribution-based permutation tests (IMPLEMENTED)
├── distance_metrics.py                 # Distance functions: energy, MMD (IMPLEMENTED)
├── penetrance_threshold.py             # Threshold-based penetrance workflow (IMPLEMENTED)
├── permutation_utils.py                # Shared permutation testing utilities (IMPLEMENTED)
└── compat/                             # Backward compatibility layer
    ├── __init__.py                     # Deprecation warnings
    ├── comparison.py                   # Old binary classification API
    └── comparison_multiclass.py        # Old multiclass API
```

### Function Names

#### Binary

```python
def run_binary_classification_test(
    df,
    group1: str,
    group2: str,
    metrics: Union[str, List[str]] = "auroc",  # "auroc", "f1", "precision", etc.
    n_permutations: int = 1000,
    time_col: str = "hpf",
    ...
) -> dict:
    """
    Run time-resolved binary classification test with label-shuffle permutation.

    Trains logistic regression classifiers at each timepoint to discriminate
    between two groups. Statistical significance assessed via permutation testing
    (shuffling group labels n_permutations times).

    Parameters
    ----------
    df : pd.DataFrame
        Data with embryo trajectories
    group1, group2 : str
        Group identifiers to classify
    metrics : str or list of str
        Classification metrics to compute. Options: "auroc", "f1", "precision",
        "recall", "accuracy". Default: "auroc"
    n_permutations : int
        Number of label shuffles for null distribution (default: 1000)

    Returns
    -------
    results : dict
        - "performance": Per-timepoint metric scores (observed data)
        - "p_values": Permutation p-values per timepoint
        - "null_distribution": Shuffled metric scores (n_permutations x n_timepoints)
        - "predictions": Per-embryo classification probabilities
        - "summary": Overall test statistics
        - "confusion_matrix": Per-timepoint confusion matrices (if applicable)

    Examples
    --------
    >>> results = run_binary_classification_test(
    ...     df,
    ...     group1="wildtype",
    ...     group2="mutant",
    ...     metrics=["auroc", "f1"],
    ...     n_permutations=1000
    ... )
    >>> print(results["p_values"])  # Timepoints with p < 0.05 are significant
    """
```

#### Multiclass

```python
def run_multiclass_classification_test(
    df,
    groups: List[str],
    metrics: Union[str, List[str]] = "auroc_ovr",  # "auroc_ovr", "f1_macro", etc.
    n_permutations: int = 1000,
    time_col: str = "hpf",
    ...
) -> dict:
    """
    Run time-resolved one-vs-rest multiclass classification test with permutation.

    Trains OvR classifiers at each timepoint to discriminate among multiple groups.
    Statistical significance assessed via label-shuffle permutation testing.

    Parameters
    ----------
    df : pd.DataFrame
        Data with embryo trajectories
    groups : list of str
        Group identifiers to classify (>=3 groups)
    metrics : str or list of str
        Classification metrics. Options: "auroc_ovr" (one-vs-rest AUROC),
        "f1_macro", "f1_weighted", "precision_macro", "accuracy".
        Default: "auroc_ovr"
    n_permutations : int
        Number of label shuffles for null distribution (default: 1000)

    Returns
    -------
    results : dict
        - "performance_per_class": Per-class per-timepoint metrics
        - "p_values_per_class": Permutation p-values per class per timepoint
        - "null_distribution": Shuffled scores per class
        - "predictions": Per-embryo class probabilities (OvR)
        - "confusion_matrices": Per-timepoint confusion matrices
        - "summary": Overall test statistics per class

    Examples
    --------
    >>> results = run_multiclass_classification_test(
    ...     df,
    ...     groups=["wt", "het", "hom"],
    ...     metrics="auroc_ovr",
    ...     n_permutations=1000
    ... )
    >>> # Check which classes separate significantly at each timepoint
    >>> print(results["p_values_per_class"])
    """
```

### Distribution Tests (IMPLEMENTED)

#### Unified Distribution Test API

```python
def permutation_test_distribution(
    X1: np.ndarray,
    X2: np.ndarray,
    statistic: str = "energy",  # "energy", "mmd", "mean"
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    **kwargs
) -> PermutationResult:
    """
    Run distribution-based permutation test.

    Tests null hypothesis: X1 and X2 come from the same distribution using
    pool-shuffle strategy (pool samples, permute, redistribute).

    Parameters
    ----------
    X1, X2 : np.ndarray
        Samples from two distributions
    statistic : str
        Test statistic to use:
        - "energy": Energy distance (Szekely-Rizzo)
        - "mmd": Maximum Mean Discrepancy with RBF kernel
        - "mean": Simple Euclidean distance between centroids
    n_permutations : int
        Number of permutations for null distribution
    **kwargs
        Additional args (e.g., bandwidth for MMD)

    Returns
    -------
    PermutationResult
        Object with observed, pvalue, null_distribution, etc.

    Examples
    --------
    >>> X1 = np.random.randn(50, 10)
    >>> X2 = np.random.randn(50, 10) + 0.3
    >>> result = permutation_test_distribution(X1, X2, statistic="energy")
    >>> print(result)
    PermutationResult(energy=0.2134, p=0.0099)
    """
```

#### Convenience Wrappers

```python
# Energy distance test
def permutation_test_energy(X1, X2, n_permutations=1000) -> PermutationResult:
    """Wrapper for permutation_test_distribution with statistic='energy'."""

# MMD test
def permutation_test_mmd(X1, X2, n_permutations=1000, bandwidth=None) -> PermutationResult:
    """Wrapper for permutation_test_distribution with statistic='mmd'."""
```

### Distance Metrics (IMPLEMENTED)

Pure geometric distance calculations (no statistical inference):

```python
def compute_energy_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Compute energy distance between two multivariate distributions.

    Energy distance equals zero if and only if distributions are identical.
    """

def compute_mmd(X1: np.ndarray, X2: np.ndarray, bandwidth: float = None) -> float:
    """
    Compute Maximum Mean Discrepancy between distributions.

    MMD with RBF kernel. Uses median heuristic for bandwidth if not specified.
    """

def compute_mean_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """Euclidean distance between centroids (simple baseline)."""
```

### Helper Functions

#### Group Assignment

```python
def assign_group_labels(
    df: pd.DataFrame,
    embryo_to_group: Dict[str, str] = None,
    cluster_results: dict = None,
    ...
) -> pd.DataFrame:
    """
    Assign group labels to embryos based on IDs or clustering results.

    Parameters
    ----------
    df : pd.DataFrame
        Data to annotate
    embryo_to_group : dict, optional
        Manual mapping {embryo_id: group_label}
    cluster_results : dict, optional
        Output from clustering (e.g., bootstrap results with "modal_cluster")

    Returns
    -------
    df : pd.DataFrame
        Input data with added "group" column
    """
```

#### Trajectory Deviation (Binary)

```python
def compute_trajectory_deviation(
    df: pd.DataFrame,
    group1: str,
    group2: str,
    metric_col: str,
    method: str = "absolute_difference",  # or "percent_change", "effect_size"
    ...
) -> pd.DataFrame:
    """
    Compute per-timepoint deviation of one group's trajectory from another.

    Measures how far group2 deviates from group1 (typically mutant from wildtype).
    Unlike classification tests, this directly compares metric values without
    building a classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Data with metric values per embryo per timepoint
    group1, group2 : str
        Groups to compare (e.g., 'wildtype', 'mutant')
    metric_col : str
        Column containing the metric to compare
    method : str
        How to quantify deviation:
        - "absolute_difference": |mean(group1) - mean(group2)|
        - "percent_change": (mean(group1) - mean(group2)) / mean(group2) * 100
        - "effect_size": Cohen's d effect size

    Returns
    -------
    deviation_df : pd.DataFrame
        Columns: [time_col, group1_mean, group2_mean, deviation]
    """
```

#### Group Differences (Multiclass)

```python
def compute_group_differences(
    df: pd.DataFrame,
    groups: List[str],
    metric_col: str,
    method: str = "absolute_difference",
    ...
) -> pd.DataFrame:
    """
    Compute pairwise differences between multiple groups over time.

    For each pair of groups, computes per-timepoint differences. Useful for
    comparing 3+ groups (e.g., wildtype, heterozygous, homozygous).

    Parameters
    ----------
    df : pd.DataFrame
        Data with metric values per embryo per timepoint
    groups : list of str
        Group identifiers to compare (>=2 groups)
    metric_col : str
        Column containing the metric to compare
    method : str
        How to quantify differences (same options as compute_trajectory_deviation)

    Returns
    -------
    differences_df : pd.DataFrame
        Columns: [time_col, group_pair, difference, group1_mean, group2_mean]
        One row per timepoint per group pair
    """
```

## Migration from Old Names

| Concept | Old Function | New Function | File | Status |
|---------|--------------|--------------|------|--------|
| **Binary Test** | `compare_groups()` | `run_binary_classification_test()` | `classification_test.py` | TODO (use compat.comparison) |
| **Multiclass Test** | `compare_groups_multiclass()` | `run_multiclass_classification_test()` | `classification_test_multiclass.py` | TODO (use compat.comparison_multiclass) |
| **Helper** | `add_group_column()` | `assign_group_labels()` | `classification_test.py` | TODO (use compat.comparison) |
| **Trajectory Comparison** | `compute_metric_divergence()` | `compute_trajectory_deviation()` | `trajectory_deviation.py` | TODO |
| **Multiclass Comparison** | *(new)* | `compute_group_differences()` | `trajectory_deviation.py` | TODO |
| **Distance Metric (Energy)** | `compute_energy_distance()` | `compute_energy_distance()` | `distance_metrics.py` (was `statistics.py`) | **DONE** |
| **Distance Metric (MMD)** | `compute_mmd()` | `compute_mmd()` | `distance_metrics.py` (was `statistics.py`) | **DONE** |
| **Distribution Test** | `permutation_test_distribution()` | `permutation_test_distribution()` | `distribution_test.py` | **DONE** |
| **Penetrance** | *(in research notebooks)* | `run_penetrance_threshold_analysis()` | `penetrance_threshold.py` | **DONE** |
| **Binning (aggregation)** | `bin_embryos_by_time()` | `bin_embryos_by_time()` | `utils.binning` | **DONE** |
| **Binning (labeling)** | *(new)* | `add_time_bins()` | `utils.binning` | **DONE** |
| **Legacy** | `predictive_signal_test()` | **Deprecated** (use `run_binary_classification_test()`) | *(remove)* | TODO |

## Why This Works

### Naming Conventions
- **"run"** matches common bioinformatics conventions (RunPCA, FindMarkers) and
  signals heavy computation (1000+ permutations)
- **"classification"** names the core operation (metric-agnostic: AUROC, F1, etc.)
- **"test"** signals hypothesis testing with null distribution and p-values
- **"compute"** signals simpler calculation without permutation testing
- **"deviation"** captures biological intuition (mutant deviating from wildtype)
- **"differences"** (plural) signals multiple comparisons across timepoints/groups

### API Design
- Binary/multiclass variants are explicit
- Metric-agnostic support covers AUROC, F1, precision, recall, accuracy
- Returns emphasize p-values and null distributions for classification tests
- Direct mean comparisons available via deviation/differences functions

### The One-Sentence Summary
You use `run_binary_classification_test()` to **prove** groups are distinct
(p-values), and `compute_trajectory_deviation()` to **measure** how far apart
they actually are (magnitude).

## Usage Examples

### Example 1: Binary Classification Test (with p-values)

```python
from src.analyze.difference_detection.classification_test import (
    run_binary_classification_test,
    assign_group_labels,
)

# Assign groups
df = assign_group_labels(
    df,
    embryo_to_group={"emb1": "wt", "emb2": "mut", ...},
)

# Run test with multiple metrics
results = run_binary_classification_test(
    df,
    group1="wildtype",
    group2="mutant",
    metrics=["auroc", "f1", "precision"],
    n_permutations=1000,
)

# Check significance
sig_timepoints = results["p_values"][results["p_values"]["auroc_pvalue"] < 0.05]
print(f"Significant separation at: {sig_timepoints['hpf'].tolist()}")

# Plot AUROC over time with permutation null
plot_classification_performance(results, metric="auroc")
```

### Example 2: Trajectory Deviation (direct comparison)

```python
from src.analyze.difference_detection.trajectory_deviation import (
    compute_trajectory_deviation,
)

# Compute how much mutant deviates from wildtype
deviation = compute_trajectory_deviation(
    df,
    group1="wildtype",
    group2="mutant",
    metric_col="area",
    method="percent_change",  # or "absolute_difference", "effect_size"
)

# Plot deviation over time
plt.plot(deviation["hpf"], deviation["deviation"])
plt.axhline(0, color="gray", linestyle="--")
plt.ylabel("% change from wildtype")
plt.xlabel("Hours post fertilization")
```

### Example 3: Multiclass Differences (pairwise)

```python
from src.analyze.difference_detection.trajectory_deviation import (
    compute_group_differences,
)

# Compare all pairs among wildtype, het, hom
differences = compute_group_differences(
    df,
    groups=["wildtype", "het", "hom"],
    metric_col="area",
    method="effect_size",
)

# Filter for large effect sizes
large_effects = differences[abs(differences["difference"]) > 0.8]
print(large_effects)
```

## Quick Reference

### Classification Tests (Statistical) - TODO
- **Binary:** `run_binary_classification_test()` in `classification_test.py`
- **Multiclass:** `run_multiclass_classification_test()` in `classification_test_multiclass.py`
- **Helper:** `assign_group_labels()` in `classification_test.py`
- **Temporary:** Use `compat.comparison.compare_groups()` (deprecated)

### Distribution Tests (Statistical) - IMPLEMENTED
- **Unified API:** `permutation_test_distribution(X1, X2, statistic='energy'|'mmd'|'mean')` in `distribution_test.py`
- **Energy Distance:** `permutation_test_energy(X1, X2)` in `distribution_test.py`
- **MMD:** `permutation_test_mmd(X1, X2, bandwidth=None)` in `distribution_test.py`
- **Distance Metrics:** `compute_energy_distance()`, `compute_mmd()`, `compute_mean_distance()` in `distance_metrics.py`

### Trajectory Comparisons (Descriptive) - TODO
- **Binary:** `compute_trajectory_deviation()` in `trajectory_deviation.py`
- **Multiclass:** `compute_group_differences()` in `trajectory_deviation.py`

### Penetrance Threshold Analysis (Threshold-Based) - IMPLEMENTED
- **Run:** `run_penetrance_threshold_analysis()` in `penetrance_threshold.py`
- **Helpers:** `compute_iqr_bounds()`, `compute_hybrid_iqr_bounds()`, `mark_threshold_violations()`, `compute_penetrance_by_time()`
- **Plotting:** `plot_penetrance_thresholds_by_category()` in `penetrance_threshold.py`

## Shared Utilities: `utils/binning.py`

**`utils/binning.py` provides ONLY data manipulation (no analysis, no thresholds, no metrics).**

### Pattern 1: Embryo-Level Aggregation
**Function:** `bin_embryos_by_time()`
**Output:** One row per (embryo_id, time_bin) with aggregated metrics
**Used by:** Classification tests, trajectory deviation, trajectory_analysis

### Pattern 2: Observation-Level Labeling
**Function:** `add_time_bins()`
**Output:** Original rows + time_bin column (no aggregation)
**Used by:** Penetrance threshold analysis

### Quality Filtering
**Function:** `filter_binned_data()`
**Purpose:** Remove embryos with too few time bins
**Used by:** All modules that need minimum bin requirements

### Module Hierarchy
```
src/analyze/
├── utils/
│   └── binning.py                    # ALL binning + threshold utilities (canonical)
├── trajectory_analysis/              # Uses utils.binning
└── difference_detection/
    ├── classification_test.py        # Uses utils.binning (aggregation)
    ├── trajectory_deviation.py       # Uses utils.binning (aggregation)
    └── penetrance_threshold.py       # Uses utils.binning (labeling + thresholds)
```

### Separation of Concerns
**`utils/binning.py` = Data manipulation ONLY**
- No analysis logic
- No threshold calculation
- No metric-specific operations
- Pure data transformation (binning, aggregation, filtering)

**`difference_detection/` modules = Analysis**
- `penetrance_threshold.py` computes thresholds and penetrance
- `classification_test.py` trains classifiers
- `trajectory_deviation.py` computes deviations

**Why move binning utilities here?**
1. **Binning is generic data prep** - used by trajectory_analysis, difference_detection, etc.
2. **Single source of truth** - no duplication across modules
3. **The `penetrance/` folder was over-engineered** - bin-width exploration belongs in research notebooks

## Penetrance Threshold Analysis (Lift Plan)

### Source Locations (current code)
- `results/mcolon/20260103_penetrance_trajectory_calculations/04_hybrid_threshold_48hpf.py`
  - Provides the hybrid IQR bounds logic (WT reference + category-specific outliers),
    threshold labeling, and category plots
- `results/mcolon/20260103_penetrance_trajectory_calculations/config.py`
  - Defines column names, categories, colors, and plotting defaults
- `results/mcolon/20260103_penetrance_trajectory_calculations/data_loading.py`
  - Loads the trajectory dataset and handles time binning for that workflow

### What to Extract and Where

**Extract from research notebooks:**
- `04_hybrid_threshold_48hpf.py` - Hybrid IQR bounds logic (WT + category-specific)
- `config.py` - Column names, categories, colors (inline into new module)
- `data_loading.py` - Binning logic (use `utils.binning` instead)

**Lift to `utils/binning.py` (data manipulation only):**
- `add_time_bins()` - Add time_bin column without aggregation (NEW, simplified from `penetrance.binwidth`)

**Create in `difference_detection/penetrance_threshold.py` (analysis):**
- `run_penetrance_threshold_analysis()` - Main entrypoint
- `compute_iqr_bounds()` - IQR-based threshold calculation (from `penetrance.binwidth`)
- `compute_hybrid_iqr_bounds()` - Hybrid bounds logic (from research notebook)
- `mark_threshold_violations()` - Binary violation marking (from `penetrance.binwidth`)
- `compute_penetrance_by_time()` - Per-bin penetrance counting (from `penetrance.binwidth`)
- `plot_penetrance_thresholds_by_category()` - Category-specific threshold plots (NEW)

**DELETE:**
- `src/analyze/penetrance/` folder entirely (over-engineered bin-width exploration)

### Rationale for Consolidation

**Why remove `src/analyze/penetrance/`?**
1. **Penetrance IS difference detection** - just threshold-based instead of model-based
2. **The folder had 435 lines** - mostly bin-width exploration plots (research, not production)
3. **Metric-based penetrance worked best** - simple IQR thresholds, not complex workflows
4. **Single source of truth** - `utils/binning.py` for ALL time operations
5. **Simpler is better** - difference_detection module contains all group comparison methods

**The three ways to detect differences:**
- `classification_test.py` - Model-based (can we predict group labels?)
- `trajectory_deviation.py` - Metric-based (how different are the means?)
- `penetrance_threshold.py` - Threshold-based (how many violate a threshold?)

All use the same binning utilities from `utils/binning.py`.

## Visualization Organization (Plotting Tweaks)

### Goals
- Keep analysis-specific plotting close to its domain (trajectory analysis).
- Keep truly generic plotting (e.g., PCA/embedding scatter) in a shared viz area.
- Split faceted (subplot) time-series plots from proportion plots.

### Proposed Structure

```
src/analyze/
├── utils/
│   └── splitting.py                         # Group-aware data splits (keep)
├── viz/
│   └── plotting/
│       └── scatter_3d.py                    # Generic 3D scatter (PCA/embedding)
└── trajectory_analysis/
    └── viz/
        └── plotting/
            ├── time_series.py               # From utils/plotting.py
            └── subplots/
                ├── time_series.py           # From utils/plotting_faceted.py
                ├── proportions.py           # From viz/plotting/faceted.py
                ├── shared.py                # IR + shared color helpers
                └── __init__.py              # Re-exports
```

### Source to Target (Plotting)

- `src/analyze/trajectory_analysis/viz/plotting/plotting_3d.py`
  - **Move to:** `src/analyze/viz/plotting/scatter_3d.py`
  - **Reason:** generic PCA/embedding visualization, not trajectory-specific
  - **Keep shims:** `src/analyze/trajectory_analysis/plotting_3d.py` and
    `src/analyze/trajectory_analysis/viz/plotting/plotting_3d.py` should re-export
    `plot_3d_scatter` for backward compatibility.

- `src/analyze/utils/plotting.py`
  - **Move to:** `src/analyze/trajectory_analysis/viz/plotting/time_series.py`
  - **Reason:** depends on DTW/DBA trajectory logic; not truly generic.

- `src/analyze/utils/plotting_faceted.py`
  - **Move to:** `src/analyze/trajectory_analysis/viz/plotting/subplots/time_series.py`
  - **Reason:** faceted time-series plotting is a subplot concern.

- `src/analyze/trajectory_analysis/viz/plotting/faceted.py`
  - **Split into:**
    - `subplots/time_series.py` (trajectory faceting)
    - `subplots/proportions.py` (bar/proportion faceting)
    - `subplots/shared.py` (IR + color helpers used by both)
  - **Keep shim:** `faceted.py` can re-export from `subplots` to avoid import breaks.

---

## Implementation Status (2025-01-16)

### ✅ Completed

#### 1. Distance Metrics Module (`distance_metrics.py`)
- **Action:** Renamed `statistics.py` → `distance_metrics.py`
- **Rationale:** Clarifies that this module contains geometric distances, not statistical inference
- **Contents:**
  - `compute_energy_distance()` - Energy distance between distributions
  - `compute_mmd()` - Maximum Mean Discrepancy with RBF kernel
  - `compute_rbf_kernel()` - RBF kernel computation
  - `estimate_bandwidth_median()` - Bandwidth estimation using median heuristic
  - `compute_mean_distance()` - Euclidean distance between centroids
- **Updated imports in:**
  - `distribution_test.py`
  - `__init__.py`

#### 2. Enhanced Binning Utilities (`utils/binning.py`)
- **Action:** Added `add_time_bins()` function
- **Purpose:** Observation-level time bin labeling without aggregation
- **Usage:** Penetrance threshold analysis (needs row-level bin membership)
- **Distinction:**
  - `add_time_bins()` - Labels each observation with bin (no aggregation)
  - `bin_embryos_by_time()` - Aggregates observations to embryo×bin level

#### 3. Penetrance Threshold Analysis (`penetrance_threshold.py`)
- **Action:** Created new module consolidating penetrance analysis
- **Source:** Extracted from research notebooks and `penetrance/binwidth.py`
- **Core Functions:**
  - `run_penetrance_threshold_analysis()` - Main workflow entrypoint
  - `compute_iqr_bounds()` - Basic IQR-based thresholds (Tukey fences)
  - `compute_hybrid_iqr_bounds()` - WT + category-specific thresholds
  - `mark_threshold_violations()` - Binary violation marking
  - `compute_penetrance_by_time()` - Embryo-level penetrance per time bin
  - `plot_penetrance_thresholds_by_category()` - Visualization
- **Deleted:** Over-engineered `src/analyze/penetrance/` folder (bin-width exploration belongs in notebooks)

#### 4. Backward Compatibility Layer (`compat/`)
- **Action:** Created `compat/` subdirectory for deprecated API
- **Contents:**
  - `compat/__init__.py` - Package-level deprecation warning
  - `compat/comparison.py` - Old binary classification API (moved from root)
  - `compat/comparison_multiclass.py` - Old multiclass API (moved from root)
- **Purpose:** Allows old code to keep working while emitting deprecation warnings
- **Migration Path:** Points users to new API in deprecation messages

#### 5. Documentation Updates (`refactor.md`)
- **Action:** Updated refactor documentation with implementation status
- **Added:**
  - Distribution test API documentation
  - Distance metrics API documentation
  - Migration table with completion status
  - Implementation status section (this section)
  - Updated quick reference with DONE/TODO markers

### 🔄 Not Implemented (Future Work)

The following modules from the original plan are NOT yet implemented:
- `classification_test.py` - Binary statistical testing with run_binary_classification_test()
- `classification_test_multiclass.py` - Multiclass statistical testing
- `trajectory_deviation.py` - Direct metric comparison without permutation testing

**Workaround:** Use `compat.comparison` and `compat.comparison_multiclass` for now (with deprecation warnings).

### 📁 File Organization Summary

```
src/analyze/
├── utils/
│   └── binning.py                          ✅ Enhanced with add_time_bins()
│
├── difference_detection/
│   ├── __init__.py                         ✅ Updated imports (distance_metrics)
│   ├── distance_metrics.py                 ✅ Renamed from statistics.py
│   ├── distribution_test.py                ✅ Updated imports
│   ├── penetrance_threshold.py             ✅ NEW - Threshold-based analysis
│   ├── permutation_utils.py                ✅ Existing (no changes needed)
│   ├── refactor.md                         ✅ Updated with implementation status
│   │
│   └── compat/                             ✅ NEW - Backward compatibility
│       ├── __init__.py                     ✅ Deprecation warnings
│       ├── comparison.py                   ✅ Moved from parent directory
│       └── comparison_multiclass.py        ✅ Moved from parent directory
│
└── penetrance/                             ❌ TO BE DELETED (over-engineered)
    ├── __init__.py
    └── binwidth.py
```

### 🧪 Testing Checklist

Run these import tests to verify the reorganization:

```bash
# Test distance metrics
python -c "from src.analyze.difference_detection.distance_metrics import compute_energy_distance, compute_mmd; print('✅ distance_metrics')"

# Test distribution tests
python -c "from src.analyze.difference_detection.distribution_test import permutation_test_distribution; print('✅ distribution_test')"

# Test permutation utils
python -c "from src.analyze.difference_detection.permutation_utils import compute_pvalue; print('✅ permutation_utils')"

# Test binning
python -c "from src.analyze.utils.binning import add_time_bins, bin_embryos_by_time; print('✅ binning')"

# Test penetrance threshold
python -c "from src.analyze.difference_detection.penetrance_threshold import run_penetrance_threshold_analysis; print('✅ penetrance_threshold')"

# Test backward compatibility (should show deprecation warning)
python -c "from src.analyze.difference_detection.compat.comparison import compare_groups; print('✅ compat layer')"
```

### 🎯 Key Improvements

1. **Naming Clarity**
   - `statistics.py` → `distance_metrics.py` removes ambiguity
   - Separates geometry (distances) from inference (p-values)

2. **Separation of Concerns**
   - Distance calculation: `distance_metrics.py`
   - Statistical inference: `permutation_utils.py`
   - Binning utilities: `utils/binning.py`
   - Analysis workflows: `penetrance_threshold.py`, `distribution_test.py`

3. **Code Consolidation**
   - Penetrance analysis extracted from research notebooks → production module
   - Deleted over-engineered `penetrance/` folder (435 lines → streamlined module)
   - Single source of truth for binning operations

4. **Backward Compatibility**
   - Old code keeps working via `compat/` layer
   - Clear deprecation warnings guide migration
   - No breaking changes for existing users

5. **Documentation**
   - Clear implementation status (DONE vs TODO)
   - Migration guide with status tracking
   - API documentation for all new modules
