"""Pair-frame geometry computation for UOT pipeline."""

import numpy as np
from analyze.utils.coord.types import BoxYX
from .config import PairFrameGeometry


def _compute_tight_bbox(mask: np.ndarray) -> BoxYX:
    """Compute tight bounding box of nonzero pixels."""
    bbox = BoxYX.from_mask(mask)
    if bbox is None:
        raise ValueError("Cannot compute bbox of empty mask")
    return bbox


def create_pair_frame_geometry(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    downsample_factor: int,
    padding_px: int = 16,
    px_size_um: float = 7.8,
    crop_policy: str = "union",
    *,
    bbox_a: BoxYX | None = None,
    bbox_b: BoxYX | None = None,
) -> PairFrameGeometry:
    """
    Factory method for PairFrameGeometry following spec Section 2.1.

    Args:
        mask_a: Binary mask in canonical space
        mask_b: Binary mask in canonical space (same shape as mask_a)
        downsample_factor: Downsampling factor (s >= 1)
        padding_px: Padding around union bbox (in canonical pixels)
        px_size_um: Physical pixel size in canonical space
        crop_policy: Only "union" supported in MVP
        bbox_a: Pre-computed tight bbox for mask_a (skips pixel scan if given)
        bbox_b: Pre-computed tight bbox for mask_b (skips pixel scan if given)

    Returns:
        Frozen PairFrameGeometry object
    """
    if crop_policy != "union":
        raise NotImplementedError(f"crop_policy='{crop_policy}' not yet supported (only 'union')")

    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shapes must match: {mask_a.shape} vs {mask_b.shape}")

    canon_shape_hw = mask_a.shape

    # Step 1: Compute tight bboxes (use pre-computed if available)
    if bbox_a is None:
        bbox_a = _compute_tight_bbox(mask_a)
    if bbox_b is None:
        bbox_b = _compute_tight_bbox(mask_b)

    # Step 2: Union + padding, clamped to canvas
    pair_crop_box = (
        bbox_a
        .union(bbox_b)
        .pad(padding_px, padding_px)
        .clamp(*canon_shape_hw)
    )

    # Step 3: Compute padding needed to make crop divisible
    crop_h, crop_w = pair_crop_box.h, pair_crop_box.w
    pad_h = (downsample_factor - (crop_h % downsample_factor)) % downsample_factor
    pad_w = (downsample_factor - (crop_w % downsample_factor)) % downsample_factor
    crop_pad_hw = (pad_h, pad_w)

    # Step 4: Compute work shape after padding + downsampling
    padded_h = crop_h + pad_h
    padded_w = crop_w + pad_w
    work_h = padded_h // downsample_factor
    work_w = padded_w // downsample_factor
    work_shape_hw = (work_h, work_w)

    # Pair crop contains both individual bboxes
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
        work_valid_box_yx=None,
        work_pad_offsets_yx=(0, 0),
    )
