"""
Generic statistical utilities for plotting.
"""

import numpy as np
from typing import Tuple, Optional

VALID_ERROR_TYPES = {
    'mean': ['sd', 'se'],
    'median': ['iqr', 'mad'],
}


def validate_error_type(trend_statistic: str, error_type: str) -> None:
    """
    Validate that error_type is compatible with trend_statistic.

    Parameters
    ----------
    trend_statistic : str
        Central tendency measure ('mean' or 'median')
    error_type : str
        Error measure ('sd', 'se' for mean; 'iqr', 'mad' for median)

    Raises
    ------
    ValueError
        If error_type is incompatible with trend_statistic
    """
    valid = VALID_ERROR_TYPES.get(trend_statistic, [])
    if error_type not in valid:
        raise ValueError(
            f"error_type='{error_type}' is incompatible with trend_statistic='{trend_statistic}'. "
            f"Valid options for '{trend_statistic}': {valid}"
        )


def compute_error_band(
    times: np.ndarray,
    metrics: np.ndarray,
    bin_width: float,
    statistic: str = 'median',
    error_type: str = 'iqr',
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute binned central tendency ± error for trajectories.

    Parameters
    ----------
    times : np.ndarray
        Time values (concatenated from all trajectories)
    metrics : np.ndarray
        Metric values (concatenated from all trajectories)
    bin_width : float
        Width of time bins for aggregation
    statistic : str
        Central tendency: 'mean' or 'median'
    error_type : str
        Error measure: 'sd'/'se' for mean, 'iqr'/'mad' for median

    Returns
    -------
    bin_times : np.ndarray or None
        Bin center times
    central_values : np.ndarray or None
        Mean or median per bin
    error_values : np.ndarray or None
        Error measure per bin (SD, SE, IQR/2, or MAD)
    """
    from scipy import stats as scipy_stats

    if len(times) == 0 or len(metrics) == 0:
        return None, None, None

    # Remove NaNs
    mask = ~(np.isnan(times) | np.isnan(metrics))
    times = times[mask]
    metrics = metrics[mask]

    if len(times) == 0:
        return None, None, None

    # Create bins
    t_min, t_max = times.min(), times.max()
    bins = np.arange(t_min, t_max + bin_width, bin_width)

    if len(bins) < 2:
        return None, None, None

    # Assign each point to a bin
    bin_indices = np.digitize(times, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    # Compute statistics per bin
    bin_times_list = []
    central_list = []
    error_list = []

    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        bin_values = metrics[bin_mask]

        if len(bin_values) < 2:
            continue

        bin_center = (bins[i] + bins[i + 1]) / 2
        bin_times_list.append(bin_center)

        if statistic == 'mean':
            central = np.mean(bin_values)
            if error_type == 'sd':
                error = np.std(bin_values, ddof=1)
            else:  # 'se'
                error = np.std(bin_values, ddof=1) / np.sqrt(len(bin_values))
        else:  # 'median'
            central = np.median(bin_values)
            if error_type == 'iqr':
                q75, q25 = np.percentile(bin_values, [75, 25])
                error = (q75 - q25) / 2  # Half IQR for symmetric band
            else:  # 'mad'
                error = np.median(np.abs(bin_values - central))

        central_list.append(central)
        error_list.append(error)

    if len(bin_times_list) == 0:
        return None, None, None

    return np.array(bin_times_list), np.array(central_list), np.array(error_list)


def compute_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Compute linear regression fit.

    Parameters
    ----------
    x : np.ndarray
        X values (typically bin times)
    y : np.ndarray
        Y values (typically trend line values)

    Returns
    -------
    x_fit : np.ndarray or None
        X values for fit line
    y_fit : np.ndarray or None
        Fitted Y values
    r_squared : float or None
        Coefficient of determination (R²)
    """
    from scipy.stats import linregress

    if x is None or y is None or len(x) < 2:
        return None, None, None

    # Remove NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return None, None, None

    try:
        result = linregress(x_clean, y_clean)
        y_fit = result.slope * x_clean + result.intercept
        r_squared = result.rvalue ** 2
        return x_clean, y_fit, r_squared
    except Exception:
        return None, None, None
