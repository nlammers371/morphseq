from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LevelName = Literal["binned", "raw", "bin_meta", "embryo_meta", "cross_bin"]


@dataclass(frozen=True)
class FeatureSpec:
    """Description of how to materialize a binned feature from raw inputs."""

    feature_id: str
    source_columns: tuple[str, ...]
    within_bin_reducer: str = "mean"
    notes: str | None = None
