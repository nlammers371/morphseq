from __future__ import annotations

import pandas as pd

from data_pipeline.schemas.segmentation import REQUIRED_COLUMNS_SEGMENTATION_TRACKING
from ._shared import validate_schema


def build_segmentation_tracking_contract(mask_rle_df: pd.DataFrame, *, well_index: int) -> pd.DataFrame:
    """
    Build the legacy-ish `segmentation_tracking.csv` contract from mask_rle rows.

    Downstream pipelines primarily consume this file.
    """
    df = mask_rle_df.copy()
    if "well_index" not in df.columns:
        df["well_index"] = int(well_index)

    # Map names expected by REQUIRED_COLUMNS_SEGMENTATION_TRACKING.
    rename = {
        "mask_confidence": "mask_confidence",
    }
    df = df.rename(columns=rename)

    # Ensure required columns exist; extra columns are allowed.
    validate_schema(df, REQUIRED_COLUMNS_SEGMENTATION_TRACKING, stage_name="segmentation_tracking")
    df = df.sort_values(["well_id", "frame_index", "image_id", "embryo_id"]).reset_index(drop=True)
    return df

