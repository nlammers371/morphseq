"""
Generic statistical functions for trajectory analysis.

Pure numerical operations with no domain-specific logic.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from scipy.ndimage import gaussian_filter1d


def compute_trend_line(
    times: np.ndarray,
    values: np.ndarray,
    bin_width: float = 0.5,
    statistic: str = 'median',
    smooth_sigma: Optional[float] = 1.5,
) -> tuple[List[float], List[float]]:
    """
    Compute binned aggregate trend line from time-series data.

    This function bins temporal data, computes a statistic (mean or median) per bin,
    and optionally smooths the result with a Gaussian filter. It's commonly used
    to visualize central tendency over time for multiple trajectories.

    Parameters
    ----------
    times : np.ndarray
        Array of time values (e.g., hours post fertilization)
    values : np.ndarray
        Array of corresponding metric values
    bin_width : float, default=0.5
        Width of time bins in the same units as `times` (e.g., 0.5 hpf)
    statistic : str, default='median'
        Aggregation statistic to compute per bin. Options:
        - 'median': More robust to outliers (recommended for biological data)
        - 'mean': Standard average (sensitive to outliers)
    smooth_sigma : float or None, default=1.5
        Standard deviation for Gaussian kernel smoothing. If None, no smoothing
        is applied. A value of 1.5 smooths over ~3 time bins, which works well
        with the default bin_width of 0.5 hpf.

    Returns
    -------
    bin_times : List[float]
        Center times of each bin
    bin_stats : List[float]
        Computed statistic (mean or median) for each bin, optionally smoothed

    Examples
    --------
    >>> times = np.array([24.1, 24.3, 24.7, 24.9, 25.2, 25.5])
    >>> values = np.array([1.2, 1.3, 1.5, 1.4, 1.8, 1.9])
    >>>
    >>> # Compute median trend line with smoothing
    >>> bin_times, bin_medians = compute_trend_line(
    ...     times, values, bin_width=0.5, statistic='median', smooth_sigma=1.5
    ... )
    >>>
    >>> # Compute mean trend line without smoothing
    >>> bin_times, bin_means = compute_trend_line(
    ...     times, values, bin_width=0.5, statistic='mean', smooth_sigma=None
    ... )

    Notes
    -----
    - Median is recommended for biological data as it's more robust to outlier embryos
    - Gaussian smoothing reduces noise from sparse timepoints while preserving trends
    - Default sigma=1.5 matches the smoothing applied to individual trajectories
    - For very sparse data, consider increasing bin_width (e.g., to 2.0 hpf)
    - Empty bins are skipped automatically
    """
    if len(times) == 0 or len(values) == 0:
        return [], []

    # Remove NaNs (keeps trend robust to sparse/invalid points)
    mask = ~(np.isnan(times) | np.isnan(values))
    times = times[mask]
    values = values[mask]

    if len(times) == 0 or len(values) == 0:
        return [], []

    # Create time bins
    time_bins = np.arange(
        np.floor(times.min()),
        np.ceil(times.max()) + bin_width,
        bin_width
    )

    bin_stats = []
    bin_times = []

    # Compute statistic for each bin
    for i in range(len(time_bins) - 1):
        mask = (times >= time_bins[i]) & (times < time_bins[i + 1])
        if mask.sum() > 0:
            if statistic == 'median':
                bin_stats.append(np.median(values[mask]))
            elif statistic == 'mean':
                bin_stats.append(np.mean(values[mask]))
            else:
                raise ValueError(f"statistic must be 'mean' or 'median', got '{statistic}'")
            bin_times.append((time_bins[i] + time_bins[i + 1]) / 2)

    # Apply Gaussian smoothing if requested
    if smooth_sigma is not None and len(bin_stats) > 1:
        bin_stats = gaussian_filter1d(bin_stats, sigma=smooth_sigma)

    # Convert to list for return (handles both numpy arrays and lists)
    if isinstance(bin_stats, np.ndarray):
        bin_stats = bin_stats.tolist()

    return bin_times, bin_stats


def normalize_arbitrary_feature(
    values: Union[np.ndarray, "pd.Series"],
    *,
    low_percentile: float = 0.0,
    high_percentile: float = 95.0,
    low: Optional[float] = None,
    high: Optional[float] = None,
    clip: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize an arbitrary 1D feature to approximately [0, 1] using percentile scaling.

    Intended for plotting/feature engineering when a metric has an arbitrary scale
    (e.g., ~0..0.2) and you want a stable 0..1 range without being dominated by outliers.

    By default:
    - low is the 0th percentile (min)
    - high is the 95th percentile
    - values above `high` are clipped to 1.0 (if clip=True)

    Parameters
    ----------
    values
        1D array-like of feature values (NaNs allowed).
    low_percentile, high_percentile
        Percentiles used when `low`/`high` are not provided.
    low, high
        Optional explicit bounds. If provided, overrides the corresponding percentile.
        Common use: `low=0.0, high_percentile=95`.
    clip
        If True, clamp output to [0, 1].
    eps
        Minimum denominator to avoid division-by-zero.

    Returns
    -------
    np.ndarray
        Normalized float array (NaNs preserved).
    """
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return out

    finite_vals = arr[finite]

    lo = float(low) if low is not None else float(np.nanpercentile(finite_vals, float(low_percentile)))
    hi = float(high) if high is not None else float(np.nanpercentile(finite_vals, float(high_percentile)))

    if not np.isfinite(lo):
        lo = float(np.nanmin(finite_vals))
    if not np.isfinite(hi):
        hi = float(np.nanmax(finite_vals))

    denom = hi - lo
    if not np.isfinite(denom) or denom <= float(eps):
        out[finite] = 0.0
        return out

    scaled = (arr - lo) / denom
    if clip:
        scaled = np.clip(scaled, 0.0, 1.0)
    out[finite] = scaled[finite]
    return out
