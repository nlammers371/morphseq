"""Schema definition for analysis-ready table."""

from __future__ import annotations

from .features import REQUIRED_COLUMNS_FEATURES
from .quality_control import QC_OUTPUT_COLUMNS

ANALYSIS_READY_QC_COLUMNS = [
    *[column for column in QC_OUTPUT_COLUMNS if column != "fraction_alive"],
]


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
    + ANALYSIS_READY_QC_COLUMNS
    + ["embedding_calculated"]
)
