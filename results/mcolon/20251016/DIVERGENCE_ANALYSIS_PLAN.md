# Morphological Divergence Analysis Plan

**Date:** October 16, 2025  
**Goal:** Quantify how much individual mutant embryos diverge from wild-type morphology

## Overview

Compute distance-based metrics to measure morphological divergence of each embryo from the wild-type reference distribution. This complements the classification and distribution tests by providing **embryo-level divergence scores**.

## Distance Metrics to Test

### 1. Mahalanobis Distance ⭐ (Primary)
- Accounts for correlations between latent dimensions
- Scale-invariant (handles different variances)
- Interpretation: How many standard deviations away from WT centroid?

**Formula:**
$$D_M(x) = \sqrt{(x - \mu_{WT})^T \Sigma_{WT}^{-1} (x - \mu_{WT})}$$

**Pros:**
- Gold standard for multivariate distance
- Accounts for feature correlations
- Statistical interpretation (chi-squared distributed under normality)

**Cons:**
- Requires enough WT samples to estimate covariance
- Assumes stable covariance structure

### 2. Euclidean Distance (Baseline)
- Simple L2 distance from WT centroid
- Easy to interpret, no assumptions

**Formula:**
$$D_E(x) = \sqrt{\sum_{i=1}^{n} (x_i - \mu_{WT,i})^2}$$

**Pros:**
- Simple, intuitive
- No assumptions about correlations
- Fast to compute

**Cons:**
- Doesn't account for different feature scales
- Ignores correlations

### 3. Cosine Distance (Directional)
- Measures angle between vectors, ignores magnitude
- Good for capturing "direction" of divergence

**Formula:**
$$D_{cos}(x, y) = 1 - \frac{x \cdot y}{||x|| \cdot ||y||}$$

**Pros:**
- Scale-invariant
- Captures directional differences

**Cons:**
- Ignores magnitude of divergence
- Less interpretable for morphology

### 4. Standardized Euclidean Distance
- Euclidean after z-scoring each dimension
- Middle ground between Euclidean and Mahalanobis

**Formula:**
$$D_{SE}(x) = \sqrt{\sum_{i=1}^{n} \frac{(x_i - \mu_{WT,i})^2}{\sigma_{WT,i}^2}}$$

**Pros:**
- Handles scale differences
- Simpler than Mahalanobis (doesn't need full covariance)

**Cons:**
- Still ignores correlations

## Proposed Module Structure

```
divergence_analysis/
├── __init__.py              # Main interface
├── distances.py             # Distance computation functions
├── reference.py             # Wild-type reference statistics
└── visualization.py         # Divergence plots
```

## Implementation Plan

### Phase 1: Core Distance Functions

**File: `divergence_analysis/distances.py`**

```python
def compute_mahalanobis_distance(
    X: np.ndarray,
    mu_ref: np.ndarray,
    cov_ref: np.ndarray
) -> np.ndarray:
    """Compute Mahalanobis distance from reference distribution."""
    
def compute_euclidean_distance(
    X: np.ndarray,
    mu_ref: np.ndarray
) -> np.ndarray:
    """Compute Euclidean distance from reference centroid."""
    
def compute_standardized_distance(
    X: np.ndarray,
    mu_ref: np.ndarray,
    std_ref: np.ndarray
) -> np.ndarray:
    """Compute standardized Euclidean distance."""
    
def compute_cosine_distance(
    X: np.ndarray,
    mu_ref: np.ndarray
) -> np.ndarray:
    """Compute cosine distance from reference."""
```

### Phase 2: Reference Distribution

**File: `divergence_analysis/reference.py`**

```python
def compute_wildtype_reference(
    df_binned: pd.DataFrame,
    wt_genotype: str = "wildtype",
    time_col: str = "time_bin",
    z_cols: Optional[list] = None
) -> Dict[str, Dict]:
    """
    Compute wild-type reference statistics for each time bin.
    
    Returns
    -------
    dict
        Keys are time bins, values are dicts with:
        - 'mean': WT centroid
        - 'cov': WT covariance matrix
        - 'std': WT standard deviations
        - 'n_samples': Number of WT embryos
    """
```

### Phase 3: Main Analysis Function

**File: `divergence_analysis/__init__.py`**

```python
def compute_divergence_scores(
    df_binned: pd.DataFrame,
    test_genotype: str,
    reference_genotype: str = "wildtype",
    metrics: List[str] = ["mahalanobis", "euclidean"],
    time_col: str = "time_bin",
    z_cols: Optional[list] = None,
    min_reference_samples: int = 10
) -> pd.DataFrame:
    """
    Compute divergence of each embryo from reference distribution.
    
    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data
    test_genotype : str
        Genotype to test (e.g., "cep290_homozygous")
    reference_genotype : str
        Reference genotype (default: "wildtype")
    metrics : list of str
        Distance metrics to compute:
        - "mahalanobis": Mahalanobis distance
        - "euclidean": Euclidean distance
        - "standardized": Standardized Euclidean
        - "cosine": Cosine distance
    time_col : str
        Time column name
    z_cols : list, optional
        Latent columns (auto-detected if None)
    min_reference_samples : int
        Minimum WT samples needed per time bin
    
    Returns
    -------
    pd.DataFrame
        One row per embryo-timepoint with columns:
        - embryo_id
        - time_bin
        - genotype
        - mahalanobis_distance (if requested)
        - euclidean_distance (if requested)
        - standardized_distance (if requested)
        - cosine_distance (if requested)
        - is_outlier_mahalanobis (if applicable)
    """
```

### Phase 4: Visualization

**File: `divergence_analysis/visualization.py`**

```python
def plot_divergence_over_time(
    df_divergence: pd.DataFrame,
    metric: str = "mahalanobis_distance"
) -> Figure:
    """Plot divergence trajectories over time."""
    
def plot_divergence_distribution(
    df_divergence: pd.DataFrame,
    metric: str = "mahalanobis_distance",
    by_genotype: bool = True
) -> Figure:
    """Plot distribution of divergence scores."""
    
def plot_divergence_heatmap(
    df_divergence: pd.DataFrame,
    metric: str = "mahalanobis_distance"
) -> Figure:
    """Heatmap of embryo divergence over time."""
    
def plot_metric_comparison(
    df_divergence: pd.DataFrame,
    metrics: List[str]
) -> Figure:
    """Compare different distance metrics."""
```

## Analysis Workflow

### Quick Test Script

**File: `test_divergence.py`**

```python
#!/usr/bin/env python3
"""
Quick test of divergence analysis with multiple distance metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import config_new as config
from utils.data_loading import load_experiments
from utils.binning import bin_by_embryo_time
from divergence_analysis import compute_divergence_scores
from divergence_analysis.visualization import (
    plot_divergence_over_time,
    plot_divergence_distribution,
    plot_metric_comparison
)

def main():
    print("="*80)
    print("MORPHOLOGICAL DIVERGENCE ANALYSIS")
    print("="*80)
    
    # Load CEP290 data
    print("\nLoading data...")
    df = load_experiments(config.CEP290_EXPERIMENTS, config.BUILD06_DIR)
    df_binned = bin_by_embryo_time(df, time_col="predicted_stage_hpf", bin_width=2.0)
    
    # Test all distance metrics
    print("\nComputing divergence scores...")
    print("  Reference: cep290_wildtype")
    print("  Test genotypes: cep290_heterozygous, cep290_homozygous")
    
    results = {}
    for genotype in ["cep290_heterozygous", "cep290_homozygous"]:
        print(f"\n  Computing for {genotype}...")
        
        df_div = compute_divergence_scores(
            df_binned,
            test_genotype=genotype,
            reference_genotype="cep290_wildtype",
            metrics=["mahalanobis", "euclidean", "standardized", "cosine"]
        )
        
        results[genotype] = df_div
        
        # Print summary
        print(f"    Mahalanobis: {df_div['mahalanobis_distance'].mean():.3f} ± {df_div['mahalanobis_distance'].std():.3f}")
        print(f"    Euclidean:   {df_div['euclidean_distance'].mean():.3f} ± {df_div['euclidean_distance'].std():.3f}")
    
    # Combine results
    df_all = pd.concat(results.values(), ignore_index=True)
    
    # Save results
    print("\nSaving results...")
    output_dir = Path(config.DATA_DIR) / "cep290" / "divergence"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for genotype, df in results.items():
        df.to_csv(output_dir / f"{genotype}_divergence.csv", index=False)
    
    # Create plots
    print("\nGenerating plots...")
    plot_dir = Path(config.PLOT_DIR) / "cep290" / "divergence"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot each metric
    for metric in ["mahalanobis_distance", "euclidean_distance"]:
        fig = plot_divergence_over_time(df_all, metric=metric)
        fig.savefig(plot_dir / f"{metric}_over_time.png", dpi=300, bbox_inches='tight')
        
        fig = plot_divergence_distribution(df_all, metric=metric)
        fig.savefig(plot_dir / f"{metric}_distribution.png", dpi=300, bbox_inches='tight')
    
    # Compare metrics
    fig = plot_metric_comparison(
        df_all,
        metrics=["mahalanobis_distance", "euclidean_distance", "standardized_distance"]
    )
    fig.savefig(plot_dir / "metric_comparison.png", dpi=300, bbox_inches='tight')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results: {output_dir}")
    print(f"Plots: {plot_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
```

## Integration with Existing Analyses

### Combined Analysis Script

Eventually we can combine all three approaches:

```python
# 1. Classification test
from difference_detection import run_classification_test
classification_results = run_classification_test(df_binned, "wt", "hom")

# 2. Distribution test (future)
from difference_detection import run_distribution_test
distribution_results = run_distribution_test(df_binned, "wt", "hom")

# 3. Divergence analysis (new)
from divergence_analysis import compute_divergence_scores
divergence_results = compute_divergence_scores(df_binned, "hom", "wt")

# All three provide complementary information!
```

## Key Questions to Answer

1. **Which distance metric is most sensitive?**
   - Compare onset times across metrics
   - Check which detects differences earliest

2. **Do different metrics capture different aspects?**
   - Mahalanobis: Overall divergence
   - Euclidean: Raw morphological difference
   - Cosine: Directional change

3. **How does divergence relate to classification?**
   - Do high-divergence embryos have high penetrance?
   - Does divergence predict classification confidence?

4. **Are there outlier embryos?**
   - Very high Mahalanobis distance = statistical outliers
   - Could identify mosaic or variable expressivity

## Implementation Priority

### Immediate (Tonight)
1. Create `divergence_analysis/` directory
2. Implement `distances.py` with core functions
3. Implement `reference.py` for WT statistics
4. Create basic `__init__.py` interface
5. Write `test_divergence.py` script

### Next Session
1. Implement visualization functions
2. Test on real CEP290 data
3. Compare metrics
4. Integrate with classification results

### Future
1. Statistical tests on divergence scores
2. Outlier detection algorithms
3. Temporal dynamics of divergence
4. Integration with penetrance analysis

## Expected Output Format

```python
# df_divergence structure:
{
    'embryo_id': ['emb_001', 'emb_001', ...],
    'time_bin': [24.0, 26.0, ...],
    'genotype': ['cep290_homozygous', ...],
    'mahalanobis_distance': [2.5, 3.1, ...],
    'euclidean_distance': [15.2, 18.3, ...],
    'standardized_distance': [2.3, 2.9, ...],
    'cosine_distance': [0.15, 0.21, ...],
    'is_outlier': [False, True, ...],  # Based on Mahalanobis > threshold
    'n_reference_samples': [45, 42, ...]  # WT samples at this time
}
```

## Benefits

✅ **Embryo-level granularity**: See which specific embryos diverge  
✅ **Multiple perspectives**: Different metrics capture different aspects  
✅ **Reference-based**: All comparisons to WT baseline  
✅ **Time-resolved**: Track divergence over development  
✅ **Outlier detection**: Statistical framework for finding extreme cases  
✅ **Complements classification**: Adds quantitative divergence to binary predictions  

## Next Steps

1. Create the module structure
2. Implement core distance functions
3. Write test script
4. Run on CEP290 data
5. Compare metrics and interpret results

Ready to implement? 🚀
