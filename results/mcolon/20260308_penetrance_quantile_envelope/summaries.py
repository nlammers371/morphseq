"""
Summary-table utilities for penetrance outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EMBRYO_COL, GENOTYPE_COL, HET_GENOTYPE, WT_GENOTYPE
from penetrance_plots import summarize_binary_penetrance


def compute_penetrance_by_group_and_time(
    df: pd.DataFrame,
    group_col: str,
    *,
    unit_col: str | None,
    count_col_name: str,
) -> pd.DataFrame:
    """
    Compute penetrance from binary calls, optionally collapsing repeated rows per unit.
    """
    if unit_col is None:
        rows = []
        for (group, tb), grp in df.groupby([group_col, "time_bin"]):
            valid = grp.dropna(subset=["penetrant"])
            if valid.empty:
                continue
            unit_flags = valid["penetrant"].astype(float)
            n_units = len(unit_flags)
            if n_units == 0:
                continue
            penetrance = float(unit_flags.mean())
            se = np.sqrt(penetrance * (1 - penetrance) / n_units)
            rows.append(
                {
                    "group": group,
                    "time_bin": tb,
                    "penetrance": penetrance,
                    count_col_name: n_units,
                    "n_penetrant": int(unit_flags.sum()),
                    "se": se,
                    "q25": float(np.quantile(unit_flags, 0.25)),
                    "q75": float(np.quantile(unit_flags, 0.75)),
                }
            )
        return pd.DataFrame(rows).sort_values(["group", "time_bin"]).reset_index(drop=True)

    summary = summarize_binary_penetrance(
        df,
        group_col=group_col,
        bin_col="time_bin",
        penetrant_col="penetrant",
        unit_col=unit_col,
        value_scale=1.0,
    ).rename(columns={"n_units": count_col_name})
    return summary.sort_values(["group", "time_bin"]).reset_index(drop=True)


def compute_penetrance_consistency(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Per-embryo fraction of supported bins that are penetrant.
    """
    rows = []
    for embryo, grp in df.groupby(EMBRYO_COL):
        group = grp[group_col].iloc[0]
        valid = grp["penetrant"].dropna()
        n_total_bins = len(valid)
        n_penetrant_bins = int(valid.sum()) if n_total_bins > 0 else 0
        frac_penetrant = n_penetrant_bins / n_total_bins if n_total_bins > 0 else np.nan
        rows.append(
            {
                "embryo_id": embryo,
                "group": group,
                "n_total_bins": n_total_bins,
                "n_penetrant_bins": n_penetrant_bins,
                "frac_penetrant": frac_penetrant,
            }
        )

    return pd.DataFrame(rows).sort_values(["group", "embryo_id"]).reset_index(drop=True)


def compute_calibration(df: pd.DataFrame, *, count_col_name: str):
    """
    Compute outside-envelope rate for WT and Het, overall and by time bin.
    """
    cal_df = df[df[GENOTYPE_COL].isin([WT_GENOTYPE, HET_GENOTYPE])].copy()

    overall_rows = []
    time_rows = []
    for geno, grp in cal_df.groupby(GENOTYPE_COL):
        valid = grp["penetrant"].dropna()
        overall_rows.append({"genotype": geno, "outside_rate": valid.mean(), count_col_name: len(valid)})

        for tb, tgrp in grp.groupby("time_bin"):
            tv = tgrp["penetrant"].dropna()
            time_rows.append({"genotype": geno, "time_bin": tb, "outside_rate": tv.mean(), count_col_name: len(tv)})

    overall = pd.DataFrame(overall_rows)
    by_time = pd.DataFrame(time_rows).sort_values(["genotype", "time_bin"]).reset_index(drop=True)
    return overall, by_time
