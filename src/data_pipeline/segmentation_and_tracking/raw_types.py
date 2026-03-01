from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RawDetection:
    """One detected object in one frame, from any detector backend."""

    frame_index: int
    image_id: str
    box_xyxy_norm: list[float]
    confidence: float

    # Canonical in contracts (absolute pixels).
    box_xyxy_abs: list[float] | None = None
    image_height_px: int | None = None
    image_width_px: int | None = None

    # Backend-specific optional fields.
    phrase: str | None = None
    category_id: int | None = None

    # Provenance (stamped by NormalizeContext).
    source_backend: str = ""
    source_model: str = ""
    model_release: str = ""
    run_id: str = ""


@dataclass
class RawTrack:
    """One tracked object in one frame, from any tracker backend."""

    frame_index: int
    image_id: str
    embryo_id: str
    mask: Any  # np.ndarray (H,W) binary
    bbox_xyxy_abs: list[float]
    area_px: float
    confidence: float

    # Tracker-native ID (backend/debug). Optional because not all trackers expose this.
    embryo_local_id: str = ""
    channel_id: str = ""
    is_seed_frame: bool = False
    propagation_direction: str | None = None

    # Provenance (stamped by NormalizeContext).
    source_backend: str = ""
    source_model: str = ""
    model_release: str = ""
    run_id: str = ""


@dataclass
class RawMask:
    """One RLE-encoded mask for one embryo in one frame."""

    frame_index: int
    image_id: str
    embryo_id: str
    mask_type: str
    mask_rle: dict[str, Any]
    area_px: float
    bbox_xyxy_abs: list[float]
    centroid_x_px: float
    centroid_y_px: float
    confidence: float

    # Tracker-native ID (backend/debug). Optional because not all trackers expose this.
    embryo_local_id: str = ""
    channel_id: str = ""
    is_seed_frame: bool = False
    exported_mask_path: str = ""
    source_image_path: str = ""

    # Provenance (stamped by NormalizeContext).
    source_backend: str = ""
    source_model: str = ""
    model_release: str = ""
    run_id: str = ""


@dataclass
class SeedSelection:
    """Record of which frame was selected as SAM seed and why."""

    experiment_id: str
    well_id: str
    video_id: str
    seed_frame_index: int
    seed_image_id: str
    num_detections: int
    avg_confidence: float
    selection_reason: str
    candidate_frames_evaluated: int
    selected_detection_indices: list[int]
    detector_backend: str = ""
    run_id: str = ""
