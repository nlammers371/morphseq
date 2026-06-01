from __future__ import annotations

import pandas as pd

from data_pipeline.quality_control.config import get_qc_defaults
from data_pipeline.quality_control.segmentation_qc.segmentation_quality_qc import (
    check_discontinuous_mask,
    check_overlapping_masks_per_image,
    check_mask_on_edge,
    decode_mask_rle,
    _HAS_IMAGE_LIBS,
)
from ._shared import align_to_universe, assert_no_duplicate_columns


def compute_segmentation_qc_flags(
    segmentation_tracking_df: pd.DataFrame,
    snip_universe_df: pd.DataFrame,
    *,
    margin_pixels: int | None = None,
    iou_threshold: float | None = None,
    min_component_fraction: float | None = None,
) -> pd.DataFrame:
    defaults = get_qc_defaults("segmentation_qc")
    if margin_pixels is None:
        margin_pixels = int(defaults["edge_margin_pixels"])
    if iou_threshold is None:
        iou_threshold = float(defaults["max_mask_overlap_fraction"])
    if min_component_fraction is None:
        min_component_fraction = 0.05

    required = {"snip_id", "image_id", "mask_rle"}
    missing = sorted(required - set(segmentation_tracking_df.columns))
    if missing:
        raise ValueError(f"segmentation_qc: missing required columns {missing}")

    assert_no_duplicate_columns(segmentation_tracking_df, "segmentation_qc input")

    qc_df = segmentation_tracking_df[["snip_id"]].copy()
    qc_df["edge_flag"] = False
    qc_df["discontinuous_mask_flag"] = False
    qc_df["overlapping_mask_flag"] = False

    edge_count = 0
    discontinuous_count = 0
    for _, row in segmentation_tracking_df.iterrows():
        snip_id = row["snip_id"]
        mask_rle = row["mask_rle"]
        if pd.isna(mask_rle):
            continue
        mask = decode_mask_rle(mask_rle)
        if any(check_mask_on_edge(mask, margin_pixels).values()):
            qc_df.loc[qc_df["snip_id"] == snip_id, "edge_flag"] = True
            edge_count += 1
        if _HAS_IMAGE_LIBS and check_discontinuous_mask(mask, min_component_fraction)["is_discontinuous"]:
            qc_df.loc[qc_df["snip_id"] == snip_id, "discontinuous_mask_flag"] = True
            discontinuous_count += 1

    overlap_count = 0
    for image_id in segmentation_tracking_df["image_id"].dropna().unique():
        image_snips = segmentation_tracking_df[segmentation_tracking_df["image_id"] == image_id]
        image_masks = {}
        for _, row in image_snips.iterrows():
            mask_rle = row["mask_rle"]
            if pd.isna(mask_rle):
                continue
            image_masks[str(row["snip_id"])] = decode_mask_rle(mask_rle)
        overlaps = check_overlapping_masks_per_image(image_masks, iou_threshold)
        for overlap in overlaps:
            qc_df.loc[qc_df["snip_id"].isin([overlap["snip_id1"], overlap["snip_id2"]]), "overlapping_mask_flag"] = True
            overlap_count += 2

    qc_df["edge_flag"] = qc_df["edge_flag"].astype(bool)
    qc_df["discontinuous_mask_flag"] = qc_df["discontinuous_mask_flag"].astype(bool)
    qc_df["overlapping_mask_flag"] = qc_df["overlapping_mask_flag"].astype(bool)

    aligned = align_to_universe(snip_universe_df, qc_df, "segmentation_qc")
    return aligned[["snip_id", "edge_flag", "discontinuous_mask_flag", "overlapping_mask_flag"]]


compute_segmentation_qc = compute_segmentation_qc_flags
