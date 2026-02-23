"""
Shared helpers for stitched FF image layout.

All image-building modules should route well/image ID generation and
output path construction through these helpers so the directory layout
matches the expectations documented in processing_files_pipeline_structure_and_plan.md.
"""

from __future__ import annotations

from pathlib import Path

from identifiers import parsing


def make_well_id(experiment_id: str, well_index: str) -> str:
    """Canonical well_id derived from experiment + well index."""
    return parsing.make_well_id(experiment_id, well_index)


def make_image_id(
    experiment_id: str,
    well_index: str,
    channel: str,
    frame_index: int,
) -> str:
    """Canonical image_id incorporating normalized channel and frame index."""
    return parsing.make_image_id(experiment_id, well_index, channel, frame_index)


def make_image_path(
    experiment_id: str,
    well_index: str,
    channel: str,
    frame_index: int,
    root: Path,
) -> Path:
    """
    Resolve the filesystem path for a stitched FF image.

    Directory structure:
        {root}/{well_id}/{channel}/{image_id}.tif
    where IDs are derived via parsing helpers.
    """
    well_id = make_well_id(experiment_id, well_index)
    image_id = make_image_id(experiment_id, well_index, channel, frame_index)
    return root / well_id / channel / f"{image_id}.tif"
