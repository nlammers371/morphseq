from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizeContext:
    source_backend: str
    source_model: str
    model_release: str
    run_id: str

    def __post_init__(self) -> None:
        if not self.model_release:
            object.__setattr__(self, "model_release", "unknown")

    def stamp(self, records: list) -> list:
        for r in records:
            r.source_backend = self.source_backend
            r.source_model = self.source_model
            r.model_release = self.model_release
            r.run_id = self.run_id
        return records

