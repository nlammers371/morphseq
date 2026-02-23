"""Canonical data models for segmentation video rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnnotationRecord:
    """Single object annotation for one frame.

    `bbox_mode` values:
      - `xyxy_norm`: [x1, y1, x2, y2] normalized to [0, 1]
      - `xyxy_abs`: [x1, y1, x2, y2] absolute pixels
      - `xywh_abs`: [x, y, w, h] absolute pixels
    """

    annotation_id: str
    segmentation: Any | None = None
    bbox: list[float] | None = None
    bbox_mode: str = "xyxy_norm"
    score: float | None = None
    label: str | None = None
    category_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameRecord:
    """All overlays for one image/frame."""

    image_id: str
    image_path: Path
    frame_index: int | None = None
    annotations: list[AnnotationRecord] = field(default_factory=list)
    qc_flags: list[str] = field(default_factory=list)


@dataclass
class VideoRecord:
    """A renderable video collection."""

    experiment_id: str
    video_id: str
    frames: list[FrameRecord] = field(default_factory=list)
