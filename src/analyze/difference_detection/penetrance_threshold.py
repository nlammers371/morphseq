"""
Threshold-based penetrance analysis for difference detection.

This module provides functions for analyzing phenotype penetrance using
IQR-based thresholds. It represents a threshold-based approach to
difference detection (complementing distribution-based, classification-based,
and trajectory-based methods).

Key concepts:
1. Define "normal" range using WT IQR bounds (Tukey fences)
2. Mark observations outside this range as "penetrant"
3. Compute embryo-level penetrance (% embryos showing phenotype)
4. Track temporal dynamics of penetrance emergence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from ..utils.binning import add_time_bins


def compute_iqr_bounds(
    wt_df: pd.DataFrame,
    metric_col: str,
    k: float = 1.5,
) -> Dict[str, float]:
    """
    Compute IQR-based threshold bounds from wildtype data.

    Uses Tukey fence method: [Q1 - k*IQR, Q3 + k*IQR]
    where k=1.5 is the traditional outlier threshold.

    Parameters
    ----------
    wt_df : pd.DataFrame
        Wildtype reference data
    metric_col : str
        Column containing metric values
    k : float, default=1.5
        IQR multiplier (1.5 = traditional outlier threshold)

    Returns
    -------
    dict
        Dictionary with keys: low, high, median, mean, q1, q3, iqr, n_samples, k

    Examples
    --------
    >>> wt_data = pd.DataFrame({'metric': np.random.randn(100)})
    >>> bounds = compute_iqr_bounds(wt_data, 'metric')
    >>> bounds['low'], bounds['high']
    (-2.1, 2.1)  # Approximate values
    """
    values = pd.to_numeric(wt_df[metric_col], errors="coerce").dropna().to_numpy()
    if values.size == 0:
        raise ValueError(
            f"compute_iqr_bounds: no valid values in column '{metric_col}'. "
            f"Rows={len(wt_df)}, non-null=0."
        )

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    return {
        "low": q1 - k * iqr,
        "high": q3 + k * iqr,
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "n_samples": len(values),
        "k": k,
    }


def compute_hybrid_iqr_bounds(
    df: pd.DataFrame,
    metric_col: str,
    category_col: str,
    wt_category: str,
    k: float = 1.5,
    use_category_specific: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compute hybrid IQR bounds: WT global + category-specific fallbacks.

    For categories with sufficient data, computes category-specific bounds.
    Otherwise, falls back to WT bounds. This is useful when different
    genotypes may have distinct normal ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all categories
    metric_col : str
        Column containing metric values
    category_col : str
        Column containing category labels (e.g., 'genotype')
    wt_category : str
        Value in category_col representing wildtype
    k : float, default=1.5
        IQR multiplier
    use_category_specific : bool, default=True
        If True, compute category-specific bounds where possible

    Returns
    -------
    dict
        Dictionary mapping category -> bounds dict

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'genotype': ['wt']*50 + ['het']*50 + ['homo']*10,
    ...     'metric': np.random.randn(110)
    ... })
    >>> bounds = compute_hybrid_iqr_bounds(df, 'metric', 'genotype', 'wt')
    >>> 'wt' in bounds and 'het' in bounds
    True
    """
    # Compute global WT bounds as baseline
    wt_df = df[df[category_col] == wt_category]
    wt_bounds = compute_iqr_bounds(wt_df, metric_col, k=k)

    bounds_dict = {wt_category: wt_bounds}

    if use_category_specific:
        for category in df[category_col].unique():
            if category == wt_category:
                continue
            cat_df = df[df[category_col] == category]
            if len(cat_df) >= 30:  # Minimum sample size for reliable IQR
                bounds_dict[category] = compute_iqr_bounds(cat_df, metric_col, k=k)
            else:
                # Fallback to WT bounds for small categories
                bounds_dict[category] = wt_bounds.copy()

    return bounds_dict


def mark_threshold_violations(
    df: pd.DataFrame,
    bounds: Dict[str, float],
    metric_col: str,
    violation_col: str = "penetrant"
) -> pd.DataFrame:
    """
    Mark observations that fall outside threshold bounds.

    Adds a binary column indicating whether each observation violates
    the threshold (is "penetrant").

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    bounds : dict
        Bounds dictionary from compute_iqr_bounds()
    metric_col : str
        Column containing metric values
    violation_col : str, default="penetrant"
        Name for the new binary violation column

    Returns
    -------
    pd.DataFrame
        Copy of df with violation_col added (1=violates, 0=normal)

    Examples
    --------
    >>> df = pd.DataFrame({'metric': [-3, -1, 0, 1, 3]})
    >>> bounds = {'low': -2, 'high': 2}
    >>> df_marked = mark_threshold_violations(df, bounds, 'metric')
    >>> df_marked['penetrant'].tolist()
    [1, 0, 0, 0, 1]
    """
    df = df.copy()
    mask = (df[metric_col] < bounds["low"]) | (df[metric_col] > bounds["high"])
    df[violation_col] = mask.astype(int)
    return df


def compute_penetrance_by_time(
    df: pd.DataFrame,
    time_col: str = "predicted_stage_hpf",
    embryo_col: str = "embryo_id",
    violation_col: str = "penetrant",
    bin_col: str = "time_bin"
) -> pd.DataFrame:
    """
    Compute embryo-level penetrance per time bin.

    Penetrance = fraction of embryos showing at least one violation
    in that time bin.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time_bin and violation columns (from add_time_bins
        and mark_threshold_violations)
    time_col : str, default="predicted_stage_hpf"
        Time column name
    embryo_col : str, default="embryo_id"
        Embryo identifier column
    violation_col : str, default="penetrant"
        Binary violation column
    bin_col : str, default="time_bin"
        Time bin column

    Returns
    -------
    pd.DataFrame
        Per-bin penetrance statistics with columns:
        - time_bin: bin identifier
        - embryo_penetrance: fraction of penetrant embryos (0-1)
        - n_embryos: total embryos in bin
        - n_penetrant: number of penetrant embryos
        - se: binomial standard error

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'embryo_id': [1, 1, 2, 2, 3, 3],
    ...     'time_bin': [0, 0, 0, 0, 2, 2],
    ...     'penetrant': [0, 0, 1, 1, 1, 0]
    ... })
    >>> result = compute_penetrance_by_time(df)
    >>> result.loc[result['time_bin'] == 0, 'embryo_penetrance'].values[0]
    0.5  # 1 of 2 embryos penetrant in bin 0
    """
    results = []

    # Group by time bin
    for time_bin, bin_df in df.groupby(bin_col):
        # Get unique embryos in this bin
        embryos = bin_df[embryo_col].unique()

        # Identify penetrant embryos (any violation in this bin)
        penetrant_embryos = bin_df.loc[
            bin_df[violation_col] == 1, embryo_col
        ].unique()

        n_embryos = len(embryos)
        n_penetrant = len(penetrant_embryos)
        penetrance = n_penetrant / n_embryos if n_embryos > 0 else 0.0

        # Binomial standard error
        se = (
            np.sqrt(penetrance * (1 - penetrance) / n_embryos)
            if n_embryos > 0
            else 0.0
        )

        results.append({
            "time_bin": float(time_bin),
            "embryo_penetrance": float(penetrance),
            "n_embryos": int(n_embryos),
            "n_penetrant": int(n_penetrant),
            "se": float(se),
        })

    return pd.DataFrame(results).sort_values("time_bin").reset_index(drop=True)


def plot_penetrance_thresholds_by_category(
    df: pd.DataFrame,
    bounds_dict: Dict[str, Dict[str, float]],
    metric_col: str,
    category_col: str,
    time_col: str = "predicted_stage_hpf",
    figsize: Tuple[float, float] = (14, 5)
) -> plt.Figure:
    """
    Plot metric distributions with IQR threshold bands per category.

    Creates one subplot per category showing:
    - Scatter of all observations
    - IQR threshold bands (shaded)
    - Median/mean lines

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    bounds_dict : dict
        Category -> bounds mapping from compute_hybrid_iqr_bounds()
    metric_col : str
        Metric column to plot
    category_col : str
        Category column (e.g., 'genotype')
    time_col : str, default="predicted_stage_hpf"
        Time column for x-axis
    figsize : tuple, default=(14, 5)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure with threshold visualization

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'genotype': ['wt']*100 + ['het']*100,
    ...     'metric': np.random.randn(200),
    ...     'predicted_stage_hpf': np.random.uniform(0, 20, 200)
    ... })
    >>> bounds = compute_hybrid_iqr_bounds(df, 'metric', 'genotype', 'wt')
    >>> fig = plot_penetrance_thresholds_by_category(df, bounds, 'metric', 'genotype')
    """
    categories = sorted(df[category_col].unique())
    n_cats = len(categories)

    fig, axes = plt.subplots(1, n_cats, figsize=figsize, sharey=True)
    if n_cats == 1:
        axes = [axes]

    for ax, category in zip(axes, categories):
        cat_df = df[df[category_col] == category]
        bounds = bounds_dict.get(category, bounds_dict[categories[0]])

        # Scatter plot
        ax.scatter(
            cat_df[time_col],
            cat_df[metric_col],
            alpha=0.3,
            s=10,
            label=f'{category} (n={len(cat_df)})'
        )

        # Threshold bands
        time_range = [cat_df[time_col].min(), cat_df[time_col].max()]
        ax.axhline(bounds['median'], color='black', linestyle='--', label='Median')
        ax.axhline(bounds['low'], color='red', linestyle=':', alpha=0.7, label='IQR bounds')
        ax.axhline(bounds['high'], color='red', linestyle=':', alpha=0.7)
        ax.fill_between(
            time_range,
            bounds['low'],
            bounds['high'],
            alpha=0.1,
            color='green',
            label='Normal range'
        )

        ax.set_title(f'{category}', fontweight='bold')
        ax.set_xlabel('Time (hpf)')
        ax.set_ylabel(metric_col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('IQR Threshold Bands by Category', y=1.02, fontweight='bold')
    fig.tight_layout()
    return fig


def run_penetrance_threshold_analysis(
    df: pd.DataFrame,
    metric_col: str,
    category_col: str,
    wt_category: str,
    time_col: str = "predicted_stage_hpf",
    embryo_col: str = "embryo_id",
    bin_width: float = 2.0,
    k: float = 1.5,
    use_category_specific: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Run complete penetrance threshold analysis workflow.

    This is the main entry point for threshold-based difference detection.
    It performs:
    1. IQR threshold computation (WT or category-specific)
    2. Time binning of observations
    3. Threshold violation marking
    4. Embryo-level penetrance calculation per time bin and category

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with observations, categories, and time
    metric_col : str
        Column containing the metric to threshold
    category_col : str
        Column containing category labels (e.g., 'genotype')
    wt_category : str
        Value in category_col representing wildtype reference
    time_col : str, default="predicted_stage_hpf"
        Time column for binning
    embryo_col : str, default="embryo_id"
        Embryo identifier column
    bin_width : float, default=2.0
        Time bin width (in units of time_col)
    k : float, default=1.5
        IQR multiplier for threshold (1.5 = traditional outlier fence)
    use_category_specific : bool, default=False
        If True, compute category-specific thresholds instead of using
        only WT thresholds for all categories

    Returns
    -------
    dict
        Dictionary with keys:
        - 'bounds': Category -> bounds dict
        - 'df_marked': Full dataframe with time_bin and penetrant columns
        - 'penetrance_by_time': Dict mapping category -> penetrance DataFrame

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'genotype': ['wt']*200 + ['het']*200,
    ...     'metric': np.concatenate([np.random.randn(200), np.random.randn(200) + 0.5]),
    ...     'predicted_stage_hpf': np.random.uniform(0, 20, 400),
    ...     'embryo_id': np.repeat(np.arange(40), 10)
    ... })
    >>> results = run_penetrance_threshold_analysis(
    ...     df, 'metric', 'genotype', 'wt', bin_width=2.0
    ... )
    >>> 'bounds' in results and 'penetrance_by_time' in results
    True
    """
    # 1. Compute bounds
    if use_category_specific:
        bounds_dict = compute_hybrid_iqr_bounds(
            df, metric_col, category_col, wt_category, k=k
        )
    else:
        wt_df = df[df[category_col] == wt_category]
        wt_bounds = compute_iqr_bounds(wt_df, metric_col, k=k)
        # Use WT bounds for all categories
        bounds_dict = {cat: wt_bounds for cat in df[category_col].unique()}

    # 2. Add time bins (observation-level)
    df_binned = add_time_bins(df, time_col=time_col, bin_width=bin_width)

    # 3. Mark violations (per-category bounds)
    dfs_marked = []
    for category in df[category_col].unique():
        cat_df = df_binned[df_binned[category_col] == category].copy()
        bounds = bounds_dict[category]
        cat_df = mark_threshold_violations(cat_df, bounds, metric_col)
        dfs_marked.append(cat_df)

    df_marked = pd.concat(dfs_marked, ignore_index=True)

    # 4. Compute penetrance by time for each category
    penetrance_by_time = {}
    for category in df[category_col].unique():
        cat_df = df_marked[df_marked[category_col] == category]
        penetrance_by_time[category] = compute_penetrance_by_time(
            cat_df,
            time_col=time_col,
            embryo_col=embryo_col,
            violation_col="penetrant",
            bin_col="time_bin"
        )

    return {
        'bounds': bounds_dict,
        'df_marked': df_marked,
        'penetrance_by_time': penetrance_by_time,
    }


__all__ = [
    'compute_iqr_bounds',
    'compute_hybrid_iqr_bounds',
    'mark_threshold_violations',
    'compute_penetrance_by_time',
    'plot_penetrance_thresholds_by_category',
    'run_penetrance_threshold_analysis',
]
