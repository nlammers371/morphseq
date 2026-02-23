"""Mask decoding utilities for GroundedSAM2 and COCO segmentations."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np


def _decode_uncompressed_coco_rle(counts: list[int], size: list[int]) -> np.ndarray:
    """Decode COCO uncompressed RLE counts into a binary mask.

    COCO RLE is run-length encoded in column-major order.
    """
    if len(size) < 2:
        raise ValueError(f"Invalid COCO RLE size: {size}")
    h, w = int(size[0]), int(size[1])
    total = h * w

    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    value = 0
    for run in counts:
        run_i = int(run)
        if run_i < 0:
            raise ValueError("RLE run length must be non-negative")
        end = idx + run_i
        if end > total:
            raise ValueError("RLE run exceeds mask size")
        if value == 1:
            flat[idx:end] = 1
        idx = end
        value = 1 - value

    if idx != total:
        raise ValueError("RLE decoding did not fill expected number of pixels")

    return flat.reshape((h, w), order="F")


def _decode_coco_polygons(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if not polygon or len(polygon) < 6:
            continue
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _decode_rle_with_pycocotools(rle_data: dict[str, Any]) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required to decode compressed RLE masks"
        ) from exc

    rle_copy = dict(rle_data)
    counts = rle_copy.get("counts")
    if isinstance(counts, str):
        # GroundedSAM2 commonly stores base64-encoded bytes.
        try:
            rle_copy["counts"] = base64.b64decode(counts)
        except Exception:
            # If this is not base64, assume raw compressed string is accepted by pycocotools.
            rle_copy["counts"] = counts

    size = rle_copy.get("size")
    if isinstance(size, (tuple, list)) and len(size) > 2:
        rle_copy["size"] = list(size)[-2:]

    mask = mask_utils.decode(rle_copy)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def decode_grounded_sam2_segmentation(segmentation: Any) -> np.ndarray:
    """Decode segmentation payload used by GroundedSAM2 outputs."""
    if not isinstance(segmentation, dict):
        raise ValueError(f"GroundedSAM2 segmentation must be dict, got {type(segmentation)}")

    fmt = str(segmentation.get("format", "rle")).lower()
    if "rle" not in fmt:
        raise ValueError(f"Unsupported GroundedSAM2 segmentation format: {fmt}")

    if "counts" not in segmentation or "size" not in segmentation:
        raise ValueError("GroundedSAM2 segmentation missing `counts` or `size`")

    return _decode_rle_with_pycocotools(segmentation)


def decode_coco_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    """Decode COCO segmentation in polygon or RLE form."""
    if segmentation is None:
        raise ValueError("Missing COCO segmentation")

    # Polygon format: list[list[float]] or list[float]
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)
        if segmentation and isinstance(segmentation[0], (int, float)):
            polygons = [segmentation]
        else:
            polygons = segmentation
        return _decode_coco_polygons(polygons, height, width)

    # RLE dict format
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        size = segmentation.get("size", [height, width])

        if isinstance(counts, list):
            return _decode_uncompressed_coco_rle(counts, size)

        if isinstance(counts, (str, bytes)):
            rle = {
                "counts": counts,
                "size": size,
            }
            return _decode_rle_with_pycocotools(rle)

    raise ValueError(f"Unsupported COCO segmentation type: {type(segmentation)}")


def decode_mask(segmentation: Any, *, source_format: str, height: int | None = None, width: int | None = None) -> np.ndarray:
    """Decode a mask from either GroundedSAM2 or COCO payloads."""
    fmt = source_format.lower()
    if fmt == "grounded_sam2":
        return decode_grounded_sam2_segmentation(segmentation)
    if fmt == "coco":
        if height is None or width is None:
            raise ValueError("COCO decoding requires frame height and width")
        return decode_coco_segmentation(segmentation, int(height), int(width))
    raise ValueError(f"Unknown source_format: {source_format}")
