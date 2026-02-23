"""
Data loading utilities for trajectory-specific penetrance analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

from config import (
    EMBRYO_DATA_PATH,
    WT_GENOTYPE,
    TIME_COL,
    EMBRYO_COL,
    GENOTYPE_COL,
    METRIC_NAME,
    TIME_BIN_WIDTH,
)

warnings.filterwarnings('ignore')


def load_trajectory_data():
    """
    Load embryo data with trajectory cluster labels.

    Returns
    -------
    df : pd.DataFrame
        Complete embryo data with cluster assignments
    """
    print(f"Loading data from: {EMBRYO_DATA_PATH}")
    df = pd.read_csv(EMBRYO_DATA_PATH)

    print(f"  Loaded {len(df):,} frames from {df[EMBRYO_COL].nunique():,} embryos")
    print(f"  Time range: {df[TIME_COL].min():.1f} - {df[TIME_COL].max():.1f} hpf")
    print(f"  Genotypes: {df[GENOTYPE_COL].value_counts().to_dict()}")

    return df


def extract_wt_data(df):
    """
    Extract wildtype embryo data for threshold calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Full embryo dataset

    Returns
    -------
    wt_df : pd.DataFrame
        WT embryo data only
    """
    wt_df = df[df[GENOTYPE_COL] == WT_GENOTYPE].copy()

    print(f"\nWT data:")
    print(f"  {wt_df[EMBRYO_COL].nunique()} embryos")
    print(f"  {len(wt_df):,} frames")
    print(f"  {METRIC_NAME} range: {wt_df[METRIC_NAME].min():.4f} - {wt_df[METRIC_NAME].max():.4f}")

    return wt_df


def bin_data_by_time(df, bin_width=TIME_BIN_WIDTH, time_col=TIME_COL):
    """
    Bin data by developmental time.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    bin_width : float
        Width of time bins in hpf
    time_col : str
        Column containing time values

    Returns
    -------
    df : pd.DataFrame
        Dataframe with 'time_bin' column added
    bin_centers : np.ndarray
        Array of bin center values
    """
    df = df.copy()
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    bin_edges = np.arange(
        np.floor(min_time / bin_width) * bin_width,
        np.ceil(max_time / bin_width) * bin_width + bin_width,
        bin_width
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    df['time_bin'] = pd.cut(df[time_col], bins=bin_edges, labels=bin_centers)
    df['time_bin'] = df['time_bin'].astype(float)

    return df, bin_centers


def get_genotype_short_name(genotype):
    """Get short name for genotype."""
    if genotype == 'cep290_wildtype':
        return 'WT'
    elif genotype == 'cep290_heterozygous':
        return 'Het'
    elif genotype == 'cep290_homozygous':
        return 'Homo'
    else:
        return genotype
