"""Data schemas for plotting functions.

Defines the expected structure of DataFrames passed to plotting functions.
This ensures clean separation between data preprocessing and visualization.
"""

from typing import List, Optional

# AUROC Plot Data Schema
# ======================
# Required columns for AUROC plotting functions:
AUROC_REQUIRED_COLUMNS = [
    'time_bin',              # Time bin identifier (float, e.g., 12.0, 16.0, 20.0)
    'auroc_observed',        # Observed AUROC value (float, 0-1)
    'auroc_null_mean',       # Mean of null distribution (float, 0-1)
    'auroc_null_std',        # Std of null distribution (float)
    'pval',                  # P-value from permutation test (float, 0-1)
]

AUROC_OPTIONAL_COLUMNS = [
    'time_bin_center',       # Preferred x-axis value (float, defaults to time_bin if missing)
    'bin_width',             # Width of time bin (float, for annotations)
    'is_significant',        # Pre-computed boolean: pval <= threshold
]

# Divergence Data Schema
# ======================
# Required columns for divergence/metric plotting functions:
DIVERGENCE_REQUIRED_COLUMNS = [
    'hpf',                   # Developmental time point (float)
    'metric',                # Metric name (str, e.g., 'baseline_deviation_normalized')
    'abs_difference',        # Raw absolute difference between groups (float)
    'group1_mean',           # Mean value for group 1 (float)
    'group2_mean',           # Mean value for group 2 (float)
]

DIVERGENCE_OPTIONAL_COLUMNS = [
    'group1_sem',            # Standard error for group 1 (float)
    'group2_sem',            # Standard error for group 2 (float)
    'abs_difference_smoothed',  # Gaussian-smoothed difference (float)
    'abs_difference_zscore',    # Z-scored difference (float)
    'abs_difference_zscore_smoothed',  # Smoothed z-score (float)
]

# Trajectory Data Schema
# ======================
# Required columns for trajectory plotting:
TRAJECTORY_REQUIRED_COLUMNS = [
    'group',                 # Group label (str, e.g., 'Penetrant', 'Control')
    'embryo_id',             # Unique embryo identifier (str)
    'predicted_stage_hpf',   # Developmental time point (float)
    # Plus at least one metric column (validated separately)
]

# Optional smoothed metric columns:
# {metric}_smoothed  - Gaussian-smoothed version of metric


def validate_auroc_data(df, required_columns: Optional[List[str]] = None):
    """Validate that DataFrame has required columns for AUROC plotting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list of str, optional
        Override default required columns

    Raises
    ------
    ValueError
        If required columns are missing
    """
    if required_columns is None:
        required_columns = AUROC_REQUIRED_COLUMNS

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"AUROC data missing required columns: {missing}\n"
            f"Required: {required_columns}\n"
            f"Available: {list(df.columns)}"
        )


def validate_divergence_data(df, required_columns: Optional[List[str]] = None):
    """Validate that DataFrame has required columns for divergence plotting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list of str, optional
        Override default required columns

    Raises
    ------
    ValueError
        If required columns are missing
    """
    if required_columns is None:
        required_columns = DIVERGENCE_REQUIRED_COLUMNS

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Divergence data missing required columns: {missing}\n"
            f"Required: {required_columns}\n"
            f"Available: {list(df.columns)}"
        )


def validate_trajectory_data(df, metric_cols: List[str]):
    """Validate that DataFrame has required columns for trajectory plotting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    metric_cols : list of str
        Metric column names to validate

    Raises
    ------
    ValueError
        If required columns are missing
    """
    required = TRAJECTORY_REQUIRED_COLUMNS.copy()
    required.extend(metric_cols)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Trajectory data missing required columns: {missing}\n"
            f"Required: {required}\n"
            f"Available: {list(df.columns)}"
        )
