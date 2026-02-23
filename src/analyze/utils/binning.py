"""
Time binning utilities for embryo morphological data.

This module provides functions for binning VAE embeddings by developmental time
and embryo identity, enabling temporal analysis of phenotype emergence.
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
    suffix: str = "_binned"
) -> pd.DataFrame:
    """
    Bin VAE embeddings by predicted time and embryo.

    Always averages embeddings per embryo_id × time_bin, keeping all
    non-latent metadata columns (e.g., genotype).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'embryo_id', time column, and latent columns.
    time_col : str, default="predicted_stage_hpf"
        Column name to bin by.
    z_cols : list of str or None
        Columns to average. If None, auto-detect those containing 'z_mu_b'.
    bin_width : float, default=2.0
        Width of time bins (same units as time_col, usually hours).
    suffix : str, default="_binned"
        Suffix to append to averaged latent column names.

    Returns
    -------
    pd.DataFrame
        One row per (embryo_id, time_bin) containing averaged latent columns
        and preserved metadata.

    Raises
    ------
    ValueError
        If no latent columns are found and z_cols is None.

    Examples
    --------
    >>> df_binned = bin_embryos_by_time(df, bin_width=2.0)
    >>> df_binned.columns
    Index(['embryo_id', 'time_bin', 'z_mu_b0_binned', 'z_mu_b1_binned', ...])
    """
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

    # Average latent vectors per embryo × time_bin
    agg = (
        df.groupby(["embryo_id", "time_bin"], as_index=False)[z_cols]
        .mean()
    )

    # Rename averaged latent columns
    agg.rename(columns={c: f"{c}{suffix}" for c in z_cols}, inplace=True)

    # Merge back non-latent metadata (take first unique per embryo)
    # Exclude time_bin and time_col from meta_cols to avoid conflicts
    meta_cols = [c for c in df.columns if c not in z_cols + [time_col, "time_bin"]]
    meta_df = (
        df[meta_cols]
        .drop_duplicates(subset=["embryo_id"])
    )

    # Merge metadata back in
    out = agg.merge(meta_df, on="embryo_id", how="left")

    # Ensure sorting
    out = out.sort_values(["embryo_id", "time_bin"]).reset_index(drop=True)

    return out


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
