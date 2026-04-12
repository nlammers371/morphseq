"""
Data loading utilities for WT quantile envelope penetrance pipeline.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Make src importable
_morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_morphseq_root / "src"))

from analyze.utils.binning import add_time_bins

from config import (
    EMBRYO_DATA_PATH,
    WT_GENOTYPE,
    HET_GENOTYPE,
    TIME_COL,
    EMBRYO_COL,
    GENOTYPE_COL,
    METRIC_NAME,
    TIME_BIN_WIDTH,
)

warnings.filterwarnings("ignore")


def load_data():
    """
    Load embryo data with cluster labels.

    Returns
    -------
    df : pd.DataFrame
        Full dataset with time_bin column added.
    time_bins : np.ndarray
        Sorted unique bin values present in the data.
    """
    print(f"Loading data from: {EMBRYO_DATA_PATH}")
    df = pd.read_csv(EMBRYO_DATA_PATH)

    print(f"  {len(df):,} frames from {df[EMBRYO_COL].nunique():,} embryos")
    print(f"  Time range: {df[TIME_COL].min():.1f} – {df[TIME_COL].max():.1f} hpf")
    print(f"  Genotypes: {df[GENOTYPE_COL].value_counts().to_dict()}")

    df = add_time_bins(df, time_col=TIME_COL, bin_width=TIME_BIN_WIDTH)
    time_bins = np.sort(df["time_bin"].dropna().unique())

    return df, time_bins


def split_by_genotype(df):
    """
    Return (wt_df, het_df) subsets.
    """
    wt_df = df[df[GENOTYPE_COL] == WT_GENOTYPE].copy()
    het_df = df[df[GENOTYPE_COL] == HET_GENOTYPE].copy()

    print(f"\nWT:  {wt_df[EMBRYO_COL].nunique()} embryos, {len(wt_df):,} frames")
    print(f"Het: {het_df[EMBRYO_COL].nunique()} embryos, {len(het_df):,} frames")
    print(f"  {METRIC_NAME} range (WT):  {wt_df[METRIC_NAME].min():.4f} – {wt_df[METRIC_NAME].max():.4f}")

    return wt_df, het_df


def get_genotype_short_name(genotype):
    short = {"cep290_wildtype": "WT", "cep290_heterozygous": "Het", "cep290_homozygous": "Homo"}
    return short.get(genotype, genotype)
