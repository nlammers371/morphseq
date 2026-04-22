from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.frame_contract import REQUIRED_COLUMNS_FRAME_CONTRACT
from data_pipeline.schemas.auxiliary_masks import REQUIRED_COLUMNS_AUXILIARY_MASKS
from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
from data_pipeline.schemas.segmentation import REQUIRED_COLUMNS_SEGMENTATION_TRACKING
from data_pipeline.schemas.snip_processing import REQUIRED_COLUMNS_SNIP_MANIFEST


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_optional_table(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return load_table(path)


def load_segmentation_tracking(path: Path) -> pd.DataFrame:
    df = load_table(path)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_SEGMENTATION_TRACKING, "segmentation_tracking.csv")
    return df


def load_frame_contract(path: Path) -> pd.DataFrame:
    df = load_table(path)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_FRAME_CONTRACT, "frame_contract.csv")
    return df


def load_snip_manifest(path: Path) -> pd.DataFrame:
    df = load_table(path)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_SNIP_MANIFEST, "snip_manifest.csv")
    return df


def load_plate_metadata(path: Path) -> pd.DataFrame:
    df = load_table(path)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_PLATE_METADATA, "plate_metadata.csv")
    return df


def load_auxiliary_masks_manifest(path: Path) -> pd.DataFrame:
    df = load_table(path)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_AUXILIARY_MASKS, "auxiliary_masks.csv")
    return df


def merge_tracking_with_frame_contract(
    tracking_df: pd.DataFrame,
    frame_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    preferred_keys = ["experiment_id", "well_id", "channel_id", "time_int"]
    join_keys = [key for key in preferred_keys if key in tracking_df.columns and key in frame_contract_df.columns]

    if not join_keys:
        raise ValueError("No shared keys between segmentation tracking and frame contract")

    merged = tracking_df.merge(
        frame_contract_df,
        on=join_keys,
        how="left",
        suffixes=("", "_frame"),
        validate="many_to_one",
    )

    if "micrometers_per_pixel" not in merged.columns and "source_micrometers_per_pixel" in merged.columns:
        merged["micrometers_per_pixel"] = pd.to_numeric(
            merged["source_micrometers_per_pixel"], errors="coerce"
        )

    if "temperature" not in merged.columns and "temperature_c" in merged.columns:
        merged["temperature"] = pd.to_numeric(merged["temperature_c"], errors="coerce")

    if "experiment_time_s" not in merged.columns and "time_int" in merged.columns:
        merged["experiment_time_s"] = pd.to_numeric(merged["time_int"], errors="coerce")

    return merged
