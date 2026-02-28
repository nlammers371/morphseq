from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema


def validate_provenance(records: Iterable, *, stage_name: str) -> None:
    for r in records:
        if not getattr(r, "source_backend", ""):
            raise ValueError(f"{stage_name}: missing source_backend on record {r}")
        if not getattr(r, "run_id", ""):
            raise ValueError(f"{stage_name}: missing run_id on record {r}")


def require_unique(df: pd.DataFrame, unique_key: list[str], *, stage_name: str) -> None:
    dup = df.duplicated(subset=unique_key, keep=False)
    if dup.any():
        preview = df.loc[dup, unique_key].head(10).to_dict(orient="records")
        raise ValueError(f"{stage_name}: duplicate keys detected for {unique_key}: {preview}")


def dumps_json(value) -> str:
    return json.dumps(value, sort_keys=True)


def validate_schema(df: pd.DataFrame, required_columns: list[str], *, stage_name: str) -> None:
    validate_dataframe_schema(df, required_columns, stage_name)

