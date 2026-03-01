from __future__ import annotations

import pandas as pd

from data_pipeline.schemas.segmentation import (
    REQUIRED_COLUMNS_FRAME_DETECTIONS,
    UNIQUE_KEY_FRAME_DETECTIONS,
)

from ..raw_types import RawDetection
from ._shared import require_unique, validate_provenance, validate_schema


def normalize_frame_detections(
    detections: list[RawDetection],
    *,
    experiment_id: str,
    well_id: str,
    video_id: str,
) -> pd.DataFrame:
    validate_provenance(detections, stage_name="frame_detections")

    rows = []
    # detection_index is defined per (experiment_id, well_id, image_id) by deterministic ordering
    by_image: dict[str, list[RawDetection]] = {}
    for d in detections:
        by_image.setdefault(d.image_id, []).append(d)

    for image_id, dets in by_image.items():
        dets_sorted = sorted(dets, key=lambda r: (-float(r.confidence), float(r.box_xyxy_abs[0]), float(r.box_xyxy_abs[1])))
        for idx, d in enumerate(dets_sorted):
            x0, y0, x1, y1 = [float(v) for v in (d.box_xyxy_abs or [0, 0, 0, 0])]
            rows.append(
                {
                    "experiment_id": str(experiment_id),
                    "well_id": str(well_id),
                    "video_id": str(video_id),
                    "image_id": str(image_id),
                    "frame_index": int(d.frame_index),
                    "detection_index": int(idx),
                    "detection_instance_id": f"{str(image_id)}_det{int(idx):03d}",
                    "box_x_min_abs": float(x0),
                    "box_y_min_abs": float(y0),
                    "box_x_max_abs": float(x1),
                    "box_y_max_abs": float(y1),
                    "detection_confidence": float(d.confidence),
                    "image_height_px": int(d.image_height_px or 0),
                    "image_width_px": int(d.image_width_px or 0),
                    "source_backend": str(d.source_backend),
                    "source_model": str(d.source_model),
                    "model_release": str(d.model_release),
                    "run_id": str(d.run_id),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS_FRAME_DETECTIONS)
    df = df.sort_values(["well_id", "frame_index", "image_id", "detection_index"]).reset_index(drop=True)
    validate_schema(df, REQUIRED_COLUMNS_FRAME_DETECTIONS, stage_name="frame_detections")
    require_unique(df, UNIQUE_KEY_FRAME_DETECTIONS, stage_name="frame_detections")
    return df
