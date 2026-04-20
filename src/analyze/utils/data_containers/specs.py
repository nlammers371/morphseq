from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


LevelName = Literal["binned", "raw", "bin_meta", "embryo_meta", "cross_bin"]


@dataclass(frozen=True)
class InputRef:
    """Reference to a declared input on a specific data level."""

    level: LevelName
    key: str


@dataclass(frozen=True)
class FeatureSpec:
    """Description of how to materialize a binned feature from raw inputs."""

    feature_id: str
    source_columns: tuple[str, ...]
    within_bin_reducer: str = "mean"
    notes: str | None = None


ReducerFunc = Callable[["pd.DataFrame", dict[str, Any]], dict[str, Any]]


@dataclass
class ReducerSpec:
    """Declarative contract for a cross-bin reducer."""

    name: str
    consumes: tuple[InputRef, ...]
    output_schema: tuple[str, ...]
    math_min_bins: int
    func: ReducerFunc | None = field(default=None, repr=False)
    version: str = "1"
    notes: str | None = None


def normalize_consumes(consumes: list[InputRef] | tuple[InputRef, ...]) -> tuple[InputRef, ...]:
    return tuple(consumes)
