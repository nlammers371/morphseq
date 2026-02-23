"""
Reference distribution computation for divergence analysis.

Computes statistics (mean, covariance, std) for a reference genotype
at each time bin. The reference can be any genotype, not just wildtype.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import warnings


def compute_reference_statistics(
    df_binned: pd.DataFrame,
    reference_genotype: str,
    time_col: str = "time_bin",
    z_cols: Optional[List[str]] = None,
    min_samples: int = 10
) -> Dict[float, Dict]:
    """
    Compute reference distribution statistics for each time bin.
    
    Can use ANY genotype as reference (wildtype, heterozygous, etc.).
    Computes mean, covariance, and standard deviations needed for
    distance calculations.
    
    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data with latent features
    reference_genotype : str
        Genotype to use as reference (e.g., "cep290_wildtype",
        "cep290_heterozygous", or any other genotype)
    time_col : str, default="time_bin"
        Column name for time bins
    z_cols : list of str, optional
        Latent feature columns to use. If None, auto-detects columns
        ending with "_binned"
    min_samples : int, default=10
        Minimum number of reference samples required per time bin.
        Time bins with fewer samples will be skipped with a warning.
    
    Returns
    -------
    dict
        Dictionary mapping time_bin -> statistics dict.
        Each statistics dict contains:
        - 'mean': np.ndarray, shape (n_features,)
            Reference centroid
        - 'cov': np.ndarray, shape (n_features, n_features)
            Reference covariance matrix
        - 'std': np.ndarray, shape (n_features,)
            Reference standard deviations
        - 'n_samples': int
            Number of reference embryos at this time bin
        - 'embryo_ids': list of str
            IDs of reference embryos
    
    Raises
    ------
    ValueError
        If reference_genotype not found in data
        If no valid time bins have enough samples
    
    Examples
    --------
    >>> # Use wildtype as reference
    >>> ref_stats = compute_reference_statistics(
    ...     df_binned,
    ...     reference_genotype="cep290_wildtype"
    ... )
    
    >>> # Use heterozygous as reference
    >>> ref_stats = compute_reference_statistics(
    ...     df_binned,
    ...     reference_genotype="cep290_heterozygous"
    ... )
    
    >>> # Access statistics for a time bin
    >>> stats_24h = ref_stats[24.0]
    >>> mu = stats_24h['mean']
    >>> cov = stats_24h['cov']
    """
    # Auto-detect latent columns if not specified
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
        if not z_cols:
            raise ValueError(
                "No latent columns found. Specify z_cols explicitly or ensure "
                "columns end with '_binned'"
            )
    
    # Filter to reference genotype
    df_ref = df_binned[df_binned['genotype'] == reference_genotype].copy()
    
    if df_ref.empty:
        raise ValueError(
            f"Reference genotype '{reference_genotype}' not found in data. "
            f"Available genotypes: {df_binned['genotype'].unique().tolist()}"
        )
    
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
            'embryo_ids': group['embryo_id'].tolist() if 'embryo_id' in group.columns else []
        }
    
    # Warn about skipped bins
    if skipped_bins:
        warnings.warn(
            f"Skipped {len(skipped_bins)} time bins with insufficient reference samples:\n" +
            "\n".join([f"  Time {t}: {n} samples (need {min_samples})" 
                      for t, n in skipped_bins])
        )
    
    if not reference_stats:
        raise ValueError(
            f"No time bins have enough reference samples (need >= {min_samples})"
        )
    
    return reference_stats


def validate_reference_stats(
    reference_stats: Dict[float, Dict],
    verbose: bool = True
) -> Dict[str, any]:
    """
    Validate and summarize reference statistics.
    
    Parameters
    ----------
    reference_stats : dict
        Output from compute_reference_statistics()
    verbose : bool, default=True
        If True, print summary
    
    Returns
    -------
    dict
        Summary statistics including:
        - n_time_bins: Number of time bins
        - time_range: (min, max) time
        - total_samples: Total reference embryos
        - samples_per_bin: Mean ± std samples per bin
        - n_features: Number of features
    """
    time_bins = sorted(reference_stats.keys())
    n_samples = [stats['n_samples'] for stats in reference_stats.values()]
    n_features = reference_stats[time_bins[0]]['mean'].shape[0]
    
    summary = {
        'n_time_bins': len(time_bins),
        'time_range': (min(time_bins), max(time_bins)),
        'total_samples': sum(n_samples),
        'samples_per_bin_mean': np.mean(n_samples),
        'samples_per_bin_std': np.std(n_samples),
        'n_features': n_features
    }
    
    if verbose:
        print("Reference Statistics Summary")
        print("="*60)
        print(f"Time bins: {summary['n_time_bins']} bins from {summary['time_range'][0]} to {summary['time_range'][1]} hpf")
        print(f"Features: {summary['n_features']}")
        print(f"Total reference embryos: {summary['total_samples']}")
        print(f"Samples per bin: {summary['samples_per_bin_mean']:.1f} ± {summary['samples_per_bin_std']:.1f}")
        print("="*60)
    
    return summary


def get_reference_for_time(
    reference_stats: Dict[float, Dict],
    time_bin: float,
    allow_nearest: bool = True,
    max_distance: float = 2.0
) -> Optional[Dict]:
    """
    Get reference statistics for a specific time bin.
    
    Parameters
    ----------
    reference_stats : dict
        Output from compute_reference_statistics()
    time_bin : float
        Time bin to get statistics for
    allow_nearest : bool, default=True
        If True and exact time not found, use nearest time bin
    max_distance : float, default=2.0
        Maximum time distance to allow when using nearest
    
    Returns
    -------
    dict or None
        Reference statistics dict, or None if not found
    """
    # Exact match
    if time_bin in reference_stats:
        return reference_stats[time_bin]
    
    # Find nearest if allowed
    if allow_nearest:
        available_times = np.array(list(reference_stats.keys()))
        distances = np.abs(available_times - time_bin)
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] <= max_distance:
            nearest_time = available_times[nearest_idx]
            warnings.warn(
                f"Time bin {time_bin} not found. Using nearest: {nearest_time} "
                f"(distance: {distances[nearest_idx]:.1f} hours)"
            )
            return reference_stats[nearest_time]
    
    return None


def combine_references(
    reference_stats_list: List[Dict[float, Dict]],
    method: str = "mean"
) -> Dict[float, Dict]:
    """
    Combine multiple reference distributions (e.g., from different experiments).
    
    Useful for creating a pooled reference when you have wildtype embryos
    from multiple experimental batches.
    
    Parameters
    ----------
    reference_stats_list : list of dict
        List of reference statistics dicts from compute_reference_statistics()
    method : str, default="mean"
        How to combine:
        - "mean": Average the means, pool for covariance
        - "pool": Pool all samples and recompute
    
    Returns
    -------
    dict
        Combined reference statistics
    
    Notes
    -----
    Currently only "mean" method is implemented. "pool" would require
    access to original data.
    """
    if method != "mean":
        raise NotImplementedError("Only 'mean' method currently supported")
    
    # Find common time bins
    common_times = set(reference_stats_list[0].keys())
    for ref_stats in reference_stats_list[1:]:
        common_times &= set(ref_stats.keys())
    
    combined = {}
    for time_bin in common_times:
        # Collect statistics from all references
        means = [ref[time_bin]['mean'] for ref in reference_stats_list]
        covs = [ref[time_bin]['cov'] for ref in reference_stats_list]
        stds = [ref[time_bin]['std'] for ref in reference_stats_list]
        n_samples = [ref[time_bin]['n_samples'] for ref in reference_stats_list]
        
        # Combine
        combined[time_bin] = {
            'mean': np.mean(means, axis=0),
            'cov': np.mean(covs, axis=0),  # Simple average - could be weighted
            'std': np.mean(stds, axis=0),
            'n_samples': sum(n_samples),
            'embryo_ids': []  # Don't track IDs for combined
        }
    
    return combined
