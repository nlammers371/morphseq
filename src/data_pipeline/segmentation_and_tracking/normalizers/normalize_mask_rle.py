from __future__ import annotations

import pandas as pd

from data_pipeline.schemas.segmentation import (
    REQUIRED_COLUMNS_MASK_RLE,
    UNIQUE_KEY_MASK_RLE,
)

from ..raw_types import RawMask
from ._shared import dumps_json, require_unique, validate_provenance, validate_schema


def normalize_mask_rle(
    masks: list[RawMask],
    *,
    experiment_id: str,
    well_id: str,
    video_id: str,
) -> pd.DataFrame:
    validate_provenance(masks, stage_name="mask_rle")
    well_slug = str(well_id).split("_")[-1]
    rows = []
    for m in masks:
        x0, y0, x1, y1 = [float(v) for v in (m.bbox_xyxy_abs or [0, 0, 0, 0])]
        # snip_id is a human-facing identifier; keep it stable even if well_id is experiment-qualified.
        snip_id = f"{experiment_id}_{well_slug}_{m.embryo_id}_f{int(m.frame_index):04d}"
        rows.append(
            {
                "experiment_id": str(experiment_id),
                "video_id": str(video_id),
                "well_id": str(well_id),
                "image_id": str(m.image_id),
                "embryo_id": str(m.embryo_id),
                "snip_id": str(snip_id),
                "frame_index": int(m.frame_index),
                "mask_type": str(m.mask_type),
                "mask_rle": dumps_json(m.mask_rle),
                "area_px": float(m.area_px),
                "bbox_x_min": float(x0),
                "bbox_y_min": float(y0),
                "bbox_x_max": float(x1),
                "bbox_y_max": float(y1),
                "centroid_x_px": float(m.centroid_x_px),
                "centroid_y_px": float(m.centroid_y_px),
                "mask_confidence": float(m.confidence),
                "is_seed_frame": bool(m.is_seed_frame),
                "source_image_path": str(m.source_image_path),
                "exported_mask_path": str(m.exported_mask_path),
                "source_backend": str(m.source_backend),
                "source_model": str(m.source_model),
                "model_release": str(m.model_release),
                "run_id": str(m.run_id),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS_MASK_RLE)
    df = df.sort_values(["well_id", "frame_index", "image_id", "embryo_id"]).reset_index(drop=True)
    validate_schema(df, REQUIRED_COLUMNS_MASK_RLE, stage_name="mask_rle")
    require_unique(df, UNIQUE_KEY_MASK_RLE, stage_name="mask_rle")
    return df
