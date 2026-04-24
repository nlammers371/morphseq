"""Shared helpers for resolving and validating path-based contracts.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_ROOT_NAME = "data_pipeline_output"


def resolve_data_root_relative_path(value: object, *, data_root_name: str = DATA_ROOT_NAME) -> object:
    """Resolve a path string against the pipeline data root when needed."""
    try:
        if value is None or pd.isna(value):
            return value
    except Exception:
        if value is None:
            return value

    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == data_root_name:
        return path
    return Path(data_root_name) / path


def require_existing_path(
    value: object,
    *,
    context: str,
    field_name: str | None = None,
    row_id: str | None = None,
    data_root_name: str = DATA_ROOT_NAME,
) -> Path:
    """Return a resolved path or fail loudly if it does not exist."""
    resolved = resolve_data_root_relative_path(value, data_root_name=data_root_name)
    if resolved is None or (isinstance(resolved, float) and pd.isna(resolved)):
        where = f" for {row_id}" if row_id else ""
        field = f" field={field_name}" if field_name else ""
        raise ValueError(f"{context}: missing required path{field}{where}")

    path = Path(resolved)
    if not path.exists():
        where = f" for {row_id}" if row_id else ""
        field = f" field={field_name}" if field_name else ""
        raise FileNotFoundError(f"{context}: missing required file{field}{where}: {path}")
    return path
