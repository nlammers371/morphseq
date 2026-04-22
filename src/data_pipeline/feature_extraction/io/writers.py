from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_pipeline.io.savers import save_csv
from data_pipeline.schemas.features import REQUIRED_COLUMNS_FEATURES
from .paths import feature_sentinel_path, schema_sidecar_path


def write_feature_table(df: pd.DataFrame, output_csv: Path) -> None:
    save_csv(df, output_csv)
    feature_sentinel_path(output_csv).write_text("ok\n")


def write_consolidated_features_contract(
    df: pd.DataFrame,
    output_csv: Path,
    *,
    schema_version: int = 1,
) -> None:
    save_csv(df, output_csv)
    feature_sentinel_path(output_csv).write_text("ok\n")
    schema_sidecar_path(output_csv).write_text(
        json.dumps(
            {
                "schema_name": "consolidated_snip_features",
                "schema_version": schema_version,
                "required_columns": REQUIRED_COLUMNS_FEATURES,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

