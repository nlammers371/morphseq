from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SupportReport:
    """Audit report emitted alongside each cross-bin reduction."""

    time_window: tuple[float, float]
    selected_bin_ids: tuple[Any, ...]
    selected_bin_centers: tuple[float, ...]
    bins_in_scope: int
    required_bins: int
    bin_fract: float
    min_bins: int | None
    math_min_bins: int
    kept_embryos: tuple[Any, ...]
    dropped_embryos: tuple[Any, ...]
    drop_reasons: dict[Any, str] = field(default_factory=dict)
    confounding_threshold: float = 0.15
    confounding_warning: str | None = None
    reducer_name: str = ""
    consumed_inputs: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_window": self.time_window,
            "selected_bin_ids": self.selected_bin_ids,
            "selected_bin_centers": self.selected_bin_centers,
            "bins_in_scope": self.bins_in_scope,
            "required_bins": self.required_bins,
            "bin_fract": self.bin_fract,
            "min_bins": self.min_bins,
            "math_min_bins": self.math_min_bins,
            "kept_embryos": self.kept_embryos,
            "dropped_embryos": self.dropped_embryos,
            "drop_reasons": self.drop_reasons,
            "confounding_threshold": self.confounding_threshold,
            "confounding_warning": self.confounding_warning,
            "reducer_name": self.reducer_name,
            "consumed_inputs": self.consumed_inputs,
            "provenance": self.provenance,
        }
