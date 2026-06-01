from __future__ import annotations

from pathlib import Path


def qc_sentinel_path(table_path: Path) -> Path:
    return table_path.with_suffix(table_path.suffix + ".validated")
