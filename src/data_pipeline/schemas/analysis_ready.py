"""Schema definition for analysis-ready table.

This module defines the final flat table contract after feature and QC
consolidation.
"""

from __future__ import annotations

from .features import REQUIRED_COLUMNS_FEATURES
from .plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
from .quality_control import REQUIRED_COLUMNS_QC
from .scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA

def _ordered_unique(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            ordered.append(column)
    return ordered


REQUIRED_COLUMNS_ANALYSIS_READY = _ordered_unique(
    REQUIRED_COLUMNS_FEATURES
    + REQUIRED_COLUMNS_QC
    + REQUIRED_COLUMNS_PLATE_METADATA
    + REQUIRED_COLUMNS_SCOPE_METADATA
    + ["embedding_calculated"]
)
