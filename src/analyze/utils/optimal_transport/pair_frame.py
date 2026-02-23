"""Pair-frame geometry computation for UOT pipeline."""

from typing import Tuple
import numpy as np
from .config import BoxYX, PairFrameGeometry


def _compute_tight_bbox(mask: np.ndarray) -> BoxYX:
    """Compute tight bounding box of nonzero pixels."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("Cannot compute bbox of empty mask")
    return BoxYX(
        y0=int(ys.min()),
        y1=int(ys.max()) + 1,  # Half-open
        x0=int(xs.min()),
        x1=int(xs.max()) + 1,
    )


def _union_bbox(bbox_a: BoxYX, bbox_b: BoxYX) -> BoxYX:
    """Compute union of two bboxes."""
    return BoxYX(
        y0=min(bbox_a.y0, bbox_b.y0),
        y1=max(bbox_a.y1, bbox_b.y1),
        x0=min(bbox_a.x0, bbox_b.x0),
        x1=max(bbox_a.x1, bbox_b.x1),
    )


def _expand_bbox_with_padding(
    bbox: BoxYX,
    canvas_shape_hw: Tuple[int, int],
    padding_px: int
) -> BoxYX:
    """Expand bbox by padding, clamped to canvas."""
    canvas_h, canvas_w = canvas_shape_hw
    return BoxYX(
        y0=max(0, bbox.y0 - padding_px),
        y1=min(canvas_h, bbox.y1 + padding_px),
        x0=max(0, bbox.x0 - padding_px),
        x1=min(canvas_w, bbox.x1 + padding_px),
    )


# Note: We do NOT need _pad_to_divisible for bbox manipulation!
# Instead, we compute padding amounts and apply them to cropped arrays in memory.


def create_pair_frame_geometry(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    downsample_factor: int,
    padding_px: int = 16,
    px_size_um: float = 7.8,
    crop_policy: str = "union",
) -> PairFrameGeometry:
    """
    Factory method for PairFrameGeometry following spec Section 2.1.

    CRITICAL DESIGN: The pair_crop_box_yx is ALWAYS a real region in canonical
    space. If the crop dimensions are not divisible by downsample_factor, we
    compute padding amounts (crop_pad_hw) that will be applied to the CROPPED
    ARRAYS in memory, not to the crop coordinates themselves.

    Args:
        mask_a: Binary mask in canonical space
        mask_b: Binary mask in canonical space (same shape as mask_a)
        downsample_factor: Downsampling factor (s >= 1)
        padding_px: Padding around union bbox (in canonical pixels)
        px_size_um: Physical pixel size in canonical space
        crop_policy: Only "union" supported in MVP

    Returns:
        Frozen PairFrameGeometry object

    Raises:
        ValueError: If masks have different shapes or if empty
        NotImplementedError: If crop_policy is not "union"
    """
    if crop_policy != "union":
        raise NotImplementedError(f"crop_policy='{crop_policy}' not yet supported (only 'union')")

    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shapes must match: {mask_a.shape} vs {mask_b.shape}")

    canon_shape_hw = mask_a.shape

    # Step 1: Compute tight bboxes (spec step 1)
    bbox_a = _compute_tight_bbox(mask_a)
    bbox_b = _compute_tight_bbox(mask_b)

    # Step 2: Union + padding, clamped to canvas (spec step 2)
    union_bbox = _union_bbox(bbox_a, bbox_b)
    pair_crop_box = _expand_bbox_with_padding(union_bbox, canon_shape_hw, padding_px)

    # Step 3: Compute padding needed to make crop divisible (spec step 3)
    # This will be applied to cropped arrays, NOT to the crop box itself
    crop_h, crop_w = pair_crop_box.h, pair_crop_box.w
    pad_h = (downsample_factor - (crop_h % downsample_factor)) % downsample_factor
    pad_w = (downsample_factor - (crop_w % downsample_factor)) % downsample_factor
    crop_pad_hw = (pad_h, pad_w)

    # Step 4: Compute work shape after padding + downsampling (spec step 4)
    padded_h = crop_h + pad_h
    padded_w = crop_w + pad_w
    work_h = padded_h // downsample_factor
    work_w = padded_w // downsample_factor
    work_shape_hw = (work_h, work_w)

    # Golden test 6.1: Pair crop contains both individual bboxes
    assert pair_crop_box.contains(bbox_a), "Pair crop must contain mask A bbox"
    assert pair_crop_box.contains(bbox_b), "Pair crop must contain mask B bbox"

    # Sanity check: padded dimensions are divisible
    assert padded_h % downsample_factor == 0
    assert padded_w % downsample_factor == 0

    return PairFrameGeometry(
        canon_shape_hw=canon_shape_hw,
        pair_crop_box_yx=pair_crop_box,
        crop_pad_hw=crop_pad_hw,
        downsample_factor=downsample_factor,
        work_shape_hw=work_shape_hw,
        px_size_um=px_size_um,
        work_valid_box_yx=None,  # No bucketing in MVP
        work_pad_offsets_yx=(0, 0),
    )
