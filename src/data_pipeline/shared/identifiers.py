"""Canonical identifier helpers for the pipeline.

See docs/refactors/streamline-snakemake/identifier_and_wildcard_contract.md.
"""

from __future__ import annotations

import re


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SEP_RE = re.compile(r"[_-]{2,}")
_LOCAL_ID_RE = re.compile(r"(\d+)$")


def sanitize_experiment_id(value: str) -> str:
    """Return a deterministic, path-safe experiment identifier."""
    cleaned = str(value).strip()
    cleaned = cleaned.replace(" ", "_")
    cleaned = _SANITIZE_RE.sub("_", cleaned)
    cleaned = _SEP_RE.sub("_", cleaned)
    return cleaned.strip("_-")


def build_well_id(well_index: str) -> str:
    """Return the canonical plate-local well ID, e.g. A01."""
    return str(well_index).strip()


def build_image_id(experiment_id: str, well_id: str, channel_id: str, time_int: int) -> str:
    """Return the canonical image ID for one channel at one timepoint."""
    return f"{sanitize_experiment_id(experiment_id)}_{build_well_id(well_id)}_{str(channel_id)}_t{int(time_int):04d}"


def build_embryo_id(experiment_id: str, well_id: str, local_track_id: int) -> str:
    """Return the canonical embryo ID for one tracked embryo within one well."""
    return f"{sanitize_experiment_id(experiment_id)}_{build_well_id(well_id)}_e{int(local_track_id):02d}"


def normalize_embryo_local_track_id(value: object) -> int:
    """Normalize tracker-native embryo IDs like ``embryo_0`` to an integer local track id."""
    if isinstance(value, bool):
        raise ValueError("embryo local track id cannot be boolean")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    match = _LOCAL_ID_RE.search(text)
    if not match:
        raise ValueError(f"Could not parse embryo local track id from {value!r}")
    return int(match.group(1))


def build_snip_id(embryo_id: str, time_int: int) -> str:
    """Return the canonical snip ID for one embryo at one timepoint."""
    return f"{str(embryo_id)}_t{int(time_int):04d}"

