from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.features import REQUIRED_COLUMNS_FEATURES
from data_pipeline.schemas.quality_control import REQUIRED_COLUMNS_QC


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_features_table(path: Path) -> pd.DataFrame:
    df = load_table(path)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_FEATURES, "consolidated_snip_features.csv")
    return df


def load_segmentation_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)


def load_viability_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)


def load_death_detection_flags(path: Path) -> pd.DataFrame:
    return load_table(path)


def load_surface_area_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)


def load_auxiliary_masks(path: Path) -> pd.DataFrame:
    return load_table(path)


def load_focus_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)


def load_motion_qc_flags(path: Path) -> pd.DataFrame:
    return load_table(path)


def load_qc_table(path: Path) -> pd.DataFrame:
    df = load_table(path)
    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_QC,
        "qc_flags.csv",
        nullable_columns=["death_inflection_time_int", "death_predicted_stage_hpf"],
    )
    return df
