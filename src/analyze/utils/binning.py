"""
Time binning utilities for embryo morphological data.

This module provides functions for binning VAE embeddings by developmental time
and embryo identity, enabling temporal analysis of phenotype emergence.

Contract
--------
Binning is a preprocessing step, not an analysis step. Call ``add_time_bins``
at the boundary where ``bin_width`` is decided, then pass prepared DataFrames
downstream. Analysis code should consume the bin column and avoid re-flooring
``time / bin_width`` outside this module.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def add_time_bins(
    df: pd.DataFrame,
    time_col: str = "predicted_stage_hpf",
    bin_width: float = 2.0,
    bin_col: str = "time_bin"
) -> pd.DataFrame:
    """
    Add time_bin column without aggregating (observation-level labeling).

    This function labels each observation with its time bin membership but
    does NOT aggregate data. Useful for penetrance analysis and other methods
    that need row-level bin assignments.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing time column.
    time_col : str, default="predicted_stage_hpf"
        Column name to bin by.
    bin_width : float, default=2.0
        Width of time bins (same units as time_col, usually hours).
    bin_col : str, default="time_bin"
        Name for the new bin column.

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with time_bin column added.

    Examples
    --------
    >>> df = pd.DataFrame({'predicted_stage_hpf': [5.2, 7.8, 8.1, 10.3]})
    >>> df_binned = add_time_bins(df, bin_width=2.0)
    >>> df_binned['time_bin'].tolist()
    [4, 6, 8, 10]
    """
    df = df.copy()
    times = df[time_col]
    bins = np.floor(times / bin_width) * bin_width

    if times.isna().any():
        bins = pd.Series(bins, index=df.index)
        if float(bin_width).is_integer():
            bins = bins.astype("Int64")
        df[bin_col] = bins
    else:
        df[bin_col] = bins.astype(int)

    return df


def bin_embryos_by_time(
    df: pd.DataFrame,
    time_col: str = "predicted_stage_hpf",
    z_cols: Optional[List[str]] = None,
    bin_width: float = 2.0,
    suffix: str = "_binned",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Bin any numeric columns by predicted time and embryo.

    Aggregates values per embryo_id × time_bin, keeping all non-aggregated
    metadata columns (e.g., genotype).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'embryo_id', time column, and numeric columns.
    time_col : str, default="predicted_stage_hpf"
        Column name to bin by.
    z_cols : list of str or None
        Columns to aggregate. If None, auto-detects columns containing 'z_mu_b'
        (VAE latent vectors). For scalar metrics, pass z_cols explicitly.
    bin_width : float, default=2.0
        Width of time bins (same units as time_col, usually hours).
    suffix : str, default="_binned"
        Suffix to append to aggregated column names.
    agg : str, default="mean"
        Aggregation function to apply. Must be 'mean' or 'median'.

    Returns
    -------
    pd.DataFrame
        One row per (embryo_id, time_bin) containing aggregated columns
        and preserved metadata.

    Raises
    ------
    ValueError
        If no columns are found and z_cols is None, or if agg is not supported.

    Examples
    --------
    >>> df_binned = bin_embryos_by_time(df, bin_width=2.0)
    >>> df_binned.columns
    Index(['embryo_id', 'time_bin', 'z_mu_b0_binned', 'z_mu_b1_binned', ...])
    """
    agg_fn = {"mean": "mean", "median": "median"}.get(agg)
    if agg_fn is None:
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")

    df = df.copy()

    # Detect latent columns
    if z_cols is None:
        z_cols = [c for c in df.columns if "z_mu_b" in c]
        if not z_cols:
            raise ValueError(
                "No latent columns found matching pattern 'z_mu_b'. "
                "Please specify z_cols explicitly."
            )

    # Create time bins
    df["time_bin"] = (np.floor(df[time_col] / bin_width) * bin_width).astype(int)

    # Aggregate columns per embryo × time_bin
    grouped = (
        df.groupby(["embryo_id", "time_bin"], as_index=False)[z_cols]
        .agg(agg_fn)
    )

    # Rename aggregated columns
    grouped.rename(columns={c: f"{c}{suffix}" for c in z_cols}, inplace=True)

    # Merge back non-latent metadata (take first unique per embryo)
    # Exclude time_bin and time_col from meta_cols to avoid conflicts
    meta_cols = [c for c in df.columns if c not in z_cols + [time_col, "time_bin"]]
    meta_df = (
        df[meta_cols]
        .drop_duplicates(subset=["embryo_id"])
    )

    # Merge metadata back in
    out = grouped.merge(meta_df, on="embryo_id", how="left")

    # Ensure sorting
    out = out.sort_values(["embryo_id", "time_bin"]).reset_index(drop=True)

    return out


def bins_in_time_window(
    bin_t_min: "np.ndarray | pd.Series",
    bin_t_max: "np.ndarray | pd.Series",
    time_window: "tuple[float, float]",
    *,
    closed: str = "both",
) -> "np.ndarray":
    """Return a boolean mask of bins whose center falls within *time_window*.

    A bin is in scope iff its center time satisfies the window condition.
    Center-of-bin inclusion avoids partial-overlap weirdness when bins straddle
    a window edge.

    Parameters
    ----------
    bin_t_min, bin_t_max : array-like of float
        Left and right edges of each bin (same length).
    time_window : (t_min, t_max)
        Inclusive window boundaries in the same units as bin edges.
    closed : {"both"}
        Only "both" (inclusive on both ends) is supported for now.

    Returns
    -------
    in_scope : np.ndarray of bool
        True for each bin whose center is in [t_min, t_max].

    Examples
    --------
    >>> edges_lo = np.array([22., 24., 26., 28.])
    >>> edges_hi = np.array([24., 26., 28., 30.])
    >>> bins_in_time_window(edges_lo, edges_hi, (24.5, 27.5))
    array([False,  True,  True, False])
    """
    if closed != "both":
        raise ValueError(f"closed={closed!r} is not supported; only 'both'.")
    t_min, t_max = float(time_window[0]), float(time_window[1])
    centers = (np.asarray(bin_t_min, dtype=float) + np.asarray(bin_t_max, dtype=float)) / 2.0
    return (centers >= t_min) & (centers <= t_max)


def filter_binned_data(
    df_binned: pd.DataFrame,
    min_time_bins: int = 3,
    min_embryos: int = 5
) -> pd.DataFrame:
    """
    Filter binned data to remove embryos with too few time bins.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data from bin_embryos_by_time().
    min_time_bins : int, default=3
        Minimum number of time bins required per embryo.
    min_embryos : int, default=5
        Minimum number of embryos required overall.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    # Count time bins per embryo
    embryo_counts = df_binned.groupby('embryo_id').size()
    valid_embryos = embryo_counts[embryo_counts >= min_time_bins].index

    df_filtered = df_binned[df_binned['embryo_id'].isin(valid_embryos)].copy()

    if len(df_filtered['embryo_id'].unique()) < min_embryos:
        print(f"Warning: Only {len(df_filtered['embryo_id'].unique())} embryos "
              f"remain after filtering (minimum: {min_embryos})")

    return df_filtered
