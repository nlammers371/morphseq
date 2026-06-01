from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_pipeline.analysis_ready.io.paths import analysis_ready_schema_path, analysis_ready_sentinel_path
from data_pipeline.analysis_ready.validators import validate_analysis_ready
from data_pipeline.io.savers import save_csv
from data_pipeline.schemas.analysis_ready import REQUIRED_COLUMNS_ANALYSIS_READY


def write_analysis_ready_contract(
    df: pd.DataFrame,
    output_csv: Path,
    *,
    schema_json_path: Path | None = None,
) -> None:
    validate_analysis_ready(df)
    save_csv(df, output_csv)
    target_schema_path = schema_json_path or analysis_ready_schema_path(output_csv)
    target_schema_path.write_text(
        json.dumps(
            {
                "schema_name": "analysis_ready",
                "required_columns": REQUIRED_COLUMNS_ANALYSIS_READY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    analysis_ready_sentinel_path(output_csv).write_text("ok\n")
