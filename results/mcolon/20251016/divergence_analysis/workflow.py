"""
Main workflow functions for divergence analysis.

Provides a modular pipeline for:
1. Computing reference distributions (single or pooled genotypes)
2. Calculating divergence scores (Mahalanobis and Euclidean distances)

For data loading and binning, import directly from:
- utils.data_loading (load_experiments)
- utils.binning (bin_embryos_by_time)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings

from divergence_analysis.distances import (
    compute_mahalanobis_distance,
    compute_euclidean_distance
)


def compute_reference_distribution(
    df_binned: pd.DataFrame,
    reference_genotypes: List[str],
    time_col: str = "time_bin",
    z_cols: Optional[List[str]] = None,
    min_samples: int = 10
) -> Dict[float, Dict]:
    """
    Compute reference distribution by pooling multiple genotypes.

    When given multiple genotypes (e.g., ['wik', 'wik-ab', 'ab']), all embryos
    from these genotypes are pooled together at each time bin to create a
    unified reference distribution.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data from bin_by_time()
    reference_genotypes : list of str
        Genotypes to pool as reference.
        Examples:
            - Single gene WT: ['cep290_wildtype']
            - Shared WT pool: ['wik', 'wik-ab', 'ab']
    time_col : str, default="time_bin"
        Column name for time bins
    z_cols : list of str, optional
        Latent feature columns. If None, auto-detects columns ending with '_binned'
    min_samples : int, default=10
        Minimum number of reference samples required per time bin

    Returns
    -------
    dict
        Dictionary mapping time_bin -> statistics dict.
        Each statistics dict contains:
        - 'mean': np.ndarray, reference centroid
        - 'cov': np.ndarray, reference covariance matrix
        - 'std': np.ndarray, reference standard deviations
        - 'n_samples': int, number of reference embryos
        - 'embryo_ids': list, IDs of reference embryos
        - 'genotypes': list, genotypes included in reference

    Examples
    --------
    >>> # Single genotype reference
    >>> ref_stats = compute_reference_distribution(
    ...     df_binned, reference_genotypes=['cep290_wildtype']
    ... )

    >>> # Pooled reference
    >>> ref_stats = compute_reference_distribution(
    ...     df_binned, reference_genotypes=['wik', 'wik-ab', 'ab']
    ... )
    """
    # Auto-detect latent columns if not specified
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
        if not z_cols:
            raise ValueError(
                "No latent columns found. Specify z_cols explicitly or ensure "
                "columns end with '_binned'"
            )

    # Filter to reference genotypes (POOL all together)
    df_ref = df_binned[df_binned['genotype'].isin(reference_genotypes)].copy()

    if df_ref.empty:
        raise ValueError(
            f"Reference genotypes {reference_genotypes} not found in data. "
            f"Available genotypes: {df_binned['genotype'].unique().tolist()}"
        )

    print(f"\nComputing reference distribution:")
    print(f"  Reference genotypes: {reference_genotypes}")
    print(f"  Total reference embryos: {df_ref['embryo_id'].nunique()}")
    print(f"  Total timepoints: {len(df_ref)}")

    # Compute statistics for each time bin
    reference_stats = {}
    skipped_bins = []

    for time_bin, group in df_ref.groupby(time_col):
        n_samples = len(group)

        # Check minimum sample size
        if n_samples < min_samples:
            skipped_bins.append((time_bin, n_samples))
            continue

        # Extract feature matrix
        X = group[z_cols].values

        # Compute statistics
        mean = X.mean(axis=0)
        cov = np.cov(X.T)
        std = X.std(axis=0)

        # Store results
        reference_stats[time_bin] = {
            'mean': mean,
            'cov': cov,
            'std': std,
            'n_samples': n_samples,
            'embryo_ids': group['embryo_id'].tolist(),
            'genotypes': group['genotype'].unique().tolist()
        }

    # Warn about skipped bins
    if skipped_bins:
        warnings.warn(
            f"Skipped {len(skipped_bins)} time bins with insufficient reference samples:\n" +
            "\n".join([f"  Time {t}: {n} samples (need {min_samples})"
                      for t, n in skipped_bins[:5]]) +
            (f"\n  ... and {len(skipped_bins) - 5} more" if len(skipped_bins) > 5 else "")
        )

    if not reference_stats:
        raise ValueError(
            f"No time bins have enough reference samples (need >= {min_samples})"
        )

    print(f"  Valid time bins: {len(reference_stats)}")
    print(f"  Time range: {min(reference_stats.keys())} - {max(reference_stats.keys())} hpf")

    return reference_stats


def compute_divergence_scores(
    df_binned: pd.DataFrame,
    reference_stats: Dict[float, Dict],
    test_genotypes: List[str],
    time_col: str = "time_bin",
    z_cols: Optional[List[str]] = None,
    metrics: List[str] = ['mahalanobis', 'euclidean']
) -> pd.DataFrame:
    """
    Compute divergence scores for test genotypes vs reference.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data from bin_by_time()
    reference_stats : dict
        Reference distribution from compute_reference_distribution()
    test_genotypes : list of str
        Genotypes to test (compute divergence for)
    time_col : str, default="time_bin"
        Column name for time bins
    z_cols : list of str, optional
        Latent feature columns. If None, auto-detects columns ending with '_binned'
    metrics : list of str, default=['mahalanobis', 'euclidean']
        Distance metrics to compute

    Returns
    -------
    pd.DataFrame
        Divergence scores with columns:
        - embryo_id
        - time_bin
        - genotype
        - mahalanobis_distance (if in metrics)
        - euclidean_distance (if in metrics)

    Examples
    --------
    >>> df_div = compute_divergence_scores(
    ...     df_binned, ref_stats,
    ...     test_genotypes=['cep290_homozygous', 'cep290_heterozygous']
    ... )
    """
    # Auto-detect latent columns if not specified
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
        if not z_cols:
            raise ValueError(
                "No latent columns found. Specify z_cols explicitly or ensure "
                "columns end with '_binned'"
            )

    # Filter to test genotypes
    df_test = df_binned[df_binned['genotype'].isin(test_genotypes)].copy()

    if df_test.empty:
        raise ValueError(
            f"Test genotypes {test_genotypes} not found in data. "
            f"Available genotypes: {df_binned['genotype'].unique().tolist()}"
        )

    print(f"\nComputing divergence scores:")
    print(f"  Test genotypes: {test_genotypes}")
    print(f"  Total test embryos: {df_test['embryo_id'].nunique()}")
    print(f"  Total timepoints: {len(df_test)}")

    # Compute divergence for each timepoint
    results = []
    n_skipped = 0

    for idx, row in df_test.iterrows():
        time_bin = row[time_col]

        # Get reference stats for this time bin
        if time_bin not in reference_stats:
            n_skipped += 1
            continue

        ref = reference_stats[time_bin]

        # Extract test point and ensure numpy arrays
        X = np.asarray(row[z_cols].values, dtype=np.float64).reshape(1, -1)
        mu_ref = np.asarray(ref['mean'], dtype=np.float64)
        cov_ref = np.asarray(ref['cov'], dtype=np.float64)

        # Compute distances
        result = {
            'embryo_id': row['embryo_id'],
            'time_bin': time_bin,
            'genotype': row['genotype']
        }

        if 'mahalanobis' in metrics:
            mahal = compute_mahalanobis_distance(X, mu_ref, cov_ref)
            result['mahalanobis_distance'] = mahal[0]

        if 'euclidean' in metrics:
            eucl = compute_euclidean_distance(X, mu_ref)
            result['euclidean_distance'] = eucl[0]

        if not any(m in metrics for m in ['mahalanobis', 'euclidean']):
            raise ValueError(f"Unknown metrics: {metrics}")

        results.append(result)

    if n_skipped > 0:
        warnings.warn(
            f"Skipped {n_skipped} timepoints with no matching reference time bin"
        )

    df_divergence = pd.DataFrame(results)

    print(f"  Computed divergence for {len(df_divergence)} timepoints")
    print(f"  Unique embryos: {df_divergence['embryo_id'].nunique()}")

    return df_divergence
