from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.quality_control.config import get_qc_defaults
from data_pipeline.quality_control.surface_area_outlier_detection import compute_sa_outlier_flag
from ._shared import align_to_universe, assert_no_duplicate_columns


def compute_surface_area_qc_flags(
    mask_geometry_df: pd.DataFrame,
    snip_universe_df: pd.DataFrame,
    *,
    sa_reference_path: Path,
    k_upper: float | None = None,
    k_lower: float | None = None,
) -> pd.DataFrame:
    defaults = get_qc_defaults("surface_area_qc")
    if k_upper is None:
        k_upper = float(defaults["k_upper"])
    if k_lower is None:
        k_lower = float(defaults["k_lower"])

    required = {"snip_id", "area_um2"}
    missing = sorted(required - set(mask_geometry_df.columns))
    if missing:
        raise ValueError(f"surface_area_qc: missing required columns {missing}")
    if "predicted_stage_hpf" not in snip_universe_df.columns:
        raise ValueError("surface_area_qc: snip universe must include predicted_stage_hpf")

    assert_no_duplicate_columns(mask_geometry_df, "surface_area_qc input")

    merged = snip_universe_df[["snip_id", "predicted_stage_hpf"]].merge(
        mask_geometry_df[["snip_id", "area_um2"]],
        on="snip_id",
        how="left",
        validate="one_to_one",
    )
    if merged.isna().any(axis=None):
        raise ValueError("surface_area_qc: merge against mask geometry introduced null rows")

    out = compute_sa_outlier_flag(merged[["snip_id", "predicted_stage_hpf", "area_um2"]], sa_reference_path=sa_reference_path, k_upper=k_upper, k_lower=k_lower, stage_col="predicted_stage_hpf", sa_col="area_um2")
    out = out[["snip_id", "sa_outlier_flag"]].copy()
    out["sa_outlier_flag"] = out["sa_outlier_flag"].astype(bool)
    aligned = align_to_universe(snip_universe_df, out, "surface_area_qc")
    return aligned[["snip_id", "sa_outlier_flag"]]


compute_surface_area_qc = compute_surface_area_qc_flags
