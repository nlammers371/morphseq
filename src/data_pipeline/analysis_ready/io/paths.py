from __future__ import annotations

from pathlib import Path


def analysis_ready_sentinel_path(table_path: Path) -> Path:
    return table_path.with_suffix(table_path.suffix + ".validated")


def analysis_ready_schema_path(table_path: Path) -> Path:
    return table_path.with_suffix("").with_suffix(".schema.json")
