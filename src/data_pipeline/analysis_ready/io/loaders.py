from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.features import REQUIRED_COLUMNS_FEATURES
from data_pipeline.schemas.quality_control import REQUIRED_COLUMNS_QC


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_analysis_ready_features(path: Path) -> pd.DataFrame:
    df = load_table(path)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_FEATURES, "consolidated_snip_features.csv")
    return df


def load_analysis_ready_qc_flags(path: Path) -> pd.DataFrame:
    df = load_table(path)
    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_QC,
        "qc_flags.csv",
        nullable_columns=["death_inflection_time_int", "death_predicted_stage_hpf"],
    )
    return df
