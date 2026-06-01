from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.quality_control.config import get_qc_defaults
from ._shared import align_to_universe, assert_no_duplicate_columns


def compute_viability_qc_flags(
    mask_geometry_df: pd.DataFrame,
    snip_universe_df: pd.DataFrame,
    *,
    min_mask_size_px: int | None = None,
    aspect_ratio_threshold: float | None = None,
) -> pd.DataFrame:
    defaults = get_qc_defaults("viability_qc")
    if min_mask_size_px is None:
        min_mask_size_px = int(defaults["min_mask_size_px"])
    if aspect_ratio_threshold is None:
        aspect_ratio_threshold = float(defaults["aspect_ratio_threshold"])

    required = {"snip_id", "area_um2", "length_um", "width_um"}
    missing = sorted(required - set(mask_geometry_df.columns))
    if missing:
        raise ValueError(f"viability_qc: missing required columns {missing}")

    assert_no_duplicate_columns(mask_geometry_df, "viability_qc input")

    area = pd.to_numeric(mask_geometry_df["area_um2"], errors="coerce").to_numpy()
    width = pd.to_numeric(mask_geometry_df["width_um"], errors="coerce").to_numpy()
    height = pd.to_numeric(mask_geometry_df["length_um"], errors="coerce").to_numpy()
    min_dim = np.minimum(width, height)
    max_dim = np.maximum(width, height)
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_ratio = np.divide(max_dim, min_dim, out=np.full_like(max_dim, np.inf), where=min_dim > 0)

    viability_flag = (
        (area < float(min_mask_size_px))
        | ~np.isfinite(aspect_ratio)
        | (aspect_ratio > aspect_ratio_threshold)
    )

    qc_df = pd.DataFrame(
        {
            "snip_id": mask_geometry_df["snip_id"].astype(str),
            "viability_flag": viability_flag.astype(bool),
        }
    )
    aligned = align_to_universe(snip_universe_df, qc_df, "viability_qc")
    return aligned[["snip_id", "viability_flag"]]


compute_viability_qc = compute_viability_qc_flags
