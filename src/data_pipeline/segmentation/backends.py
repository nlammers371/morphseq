"""Segmentation backend selection helpers for detector/tracker swapability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Literal, Any


DetectorBackend = Literal["groundingdino", "detectron2"]
TrackerBackend = Literal["sam2", "sam3"]

SUPPORTED_DETECTOR_BACKENDS = {"groundingdino", "detectron2"}
SUPPORTED_TRACKER_BACKENDS = {"sam2", "sam3"}


@dataclass(frozen=True)
class SegmentationBackendsConfig:
    detector_backend: DetectorBackend = "groundingdino"
    tracker_backend: TrackerBackend = "sam2"


def _normalize_choice(value: str | None, *, valid: set[str], field_name: str, default: str) -> str:
    choice = (value or default).strip().lower()
    if choice not in valid:
        allowed = ", ".join(sorted(valid))
        raise ValueError(f"Invalid {field_name}={choice!r}. Allowed values: {allowed}")
    return choice


def load_segmentation_backends_config(config: Mapping[str, Any] | None) -> SegmentationBackendsConfig:
    """
    Parse segmentation backend settings from pipeline config mapping.

    Expected shape:
    {
      "segmentation": {
        "detector_backend": "groundingdino|detectron2",
        "tracker_backend": "sam2|sam3",
      }
    }
    """
    raw = (config or {}).get("segmentation", {})
    detector = _normalize_choice(
        raw.get("detector_backend"),
        valid=SUPPORTED_DETECTOR_BACKENDS,
        field_name="detector_backend",
        default="groundingdino",
    )
    tracker = _normalize_choice(
        raw.get("tracker_backend"),
        valid=SUPPORTED_TRACKER_BACKENDS,
        field_name="tracker_backend",
        default="sam2",
    )
    return SegmentationBackendsConfig(
        detector_backend=detector,  # type: ignore[arg-type]
        tracker_backend=tracker,  # type: ignore[arg-type]
    )
