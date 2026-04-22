"""Experiment identity resolution helpers.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.shared.identifiers import sanitize_experiment_id


def resolve_experiment_id(
    raw_source_path: str | Path,
    microscope: str,
    explicit_experiment_id: str | None = None,
) -> str:
    """Resolve the canonical experiment ID once at ingest."""
    if explicit_experiment_id:
        return sanitize_experiment_id(explicit_experiment_id)

    raw_path = Path(raw_source_path)
    if microscope == "Keyence":
        candidates = [raw_path.name, raw_path.parent.name]
    else:
        candidates = [raw_path.stem, raw_path.parent.name, raw_path.name]

    for candidate in candidates:
        cleaned = sanitize_experiment_id(candidate)
        if cleaned:
            return cleaned

    raise ValueError(f"Could not resolve experiment_id from {raw_source_path!s} for microscope={microscope!r}")

