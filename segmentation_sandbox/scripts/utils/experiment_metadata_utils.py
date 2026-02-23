#!/usr/bin/env python3
"""
Minimal Experiment Metadata Utils

Provides small shims used by SAM2 and GroundedDINO utilities:
- load_experiment_metadata(path) -> ExperimentMetadata instance
- get_image_id_paths(image_ids, metadata_or_path) -> list[(image_id, full_path)]
- get_video_info(video_id, metadata_or_path) -> dict video metadata

These wrap the existing ExperimentMetadata class to avoid duplicating logic.
"""

from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

try:
    # Prefer intra-sandbox import path
    from scripts.metadata.experiment_metadata import ExperimentMetadata
except ImportError:
    # Fallback for direct execution
    from segmentation_sandbox.scripts.metadata.experiment_metadata import ExperimentMetadata


def _ensure_manager(metadata_or_path: Union[str, Path, ExperimentMetadata]) -> ExperimentMetadata:
    if isinstance(metadata_or_path, ExperimentMetadata):
        return metadata_or_path
    return ExperimentMetadata(str(metadata_or_path), verbose=False)


def load_experiment_metadata(path: Union[str, Path]) -> ExperimentMetadata:
    """Load ExperimentMetadata manager from a JSON path."""
    return ExperimentMetadata(str(path), verbose=False)


def get_image_id_paths(image_ids: List[str], metadata_or_path: Union[str, Path, ExperimentMetadata]) -> List[Tuple[str, Path]]:
    """Resolve a list of image_ids to list of (image_id, full_path) tuples."""
    manager = _ensure_manager(metadata_or_path)
    pairs: List[Tuple[str, Path]] = []
    for image_id in image_ids:
        try:
            full_path = manager.get_image_path(image_id)
            pairs.append((image_id, full_path))
        except Exception:
            # Skip if resolution fails
            continue
    return pairs


def get_video_info(video_id: str, metadata_or_path: Union[str, Path, ExperimentMetadata]) -> Dict[str, Any]:
    """Return the video metadata dict for a given video_id, or {} if not found."""
    manager = _ensure_manager(metadata_or_path)
    meta = manager.metadata
    experiments = meta.get("experiments", {})
    for exp_id, exp_data in experiments.items():
        videos = exp_data.get("videos", {})
        if video_id in videos:
            return videos[video_id]
    return {}

