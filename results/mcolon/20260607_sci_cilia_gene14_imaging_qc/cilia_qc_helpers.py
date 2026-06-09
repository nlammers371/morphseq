"""
Shared helpers for the sci cilia gene14 imaging-QC scripts (steps 0-6).

Small, pure, dependency-light functions used by more than one numbered script live here
so there is a single source of truth (no copy-paste drift, no importing across the
digit-prefixed script filenames).
"""

from __future__ import annotations

import pandas as pd


def select_for_label_transfer(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row-source per physical embryo: TIMESERIES if it exists, else SNAPSHOT.

    The LOUD plate01 rule, generalized. A plate01 48 hpf physical embryo has BOTH a
    `_sci_` timeseries and a redundant `_t02` snapshot backup; they share a
    `physical_embryo_id`. Label transfer must use the timeseries and drop the backup.
    plate02 (snapshot only) and crispants (no timeseries) keep their snapshots.
    """
    # physical embryos that have any timeseries row
    physical_with_timeseries = set(
        df.loc[df["data_source"] == "timeseries", "physical_embryo_id"].dropna()
    )
    # drop snapshot rows ONLY for physical embryos that also have a timeseries
    redundant_snapshot = (
        (df["data_source"] == "snapshot")
        & df["physical_embryo_id"].isin(physical_with_timeseries)
    )
    return df[~redundant_snapshot].copy()
