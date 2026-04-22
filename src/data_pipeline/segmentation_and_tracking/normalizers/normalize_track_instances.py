from __future__ import annotations

import pandas as pd

from data_pipeline.schemas.segmentation import (
    REQUIRED_COLUMNS_TRACK_INSTANCES,
    UNIQUE_KEY_TRACK_INSTANCES,
)

from ..raw_types import RawTrack
from ._shared import require_unique, validate_provenance, validate_schema


def _centroid_from_bbox(bbox: list[float]) -> tuple[float, float]:
    if len(bbox) < 4:
        return 0.0, 0.0
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    return float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)


def normalize_track_instances(
    tracks: list[RawTrack],
    *,
    experiment_id: str,
    well_id: str,
    well_index: int,
    video_id: str,
) -> pd.DataFrame:
    validate_provenance(tracks, stage_name="track_instances")
    rows = []
    for t in tracks:
        x0, y0, x1, y1 = [float(v) for v in (t.bbox_xyxy_abs or [0, 0, 0, 0])]
        cx, cy = _centroid_from_bbox([x0, y0, x1, y1])
        instance_id = str(t.embryo_local_id or "")
        if not instance_id:
            instance_id = str(t.embryo_id)
        rows.append(
            {
                "experiment_id": str(experiment_id),
                "video_id": str(video_id),
                "well_id": str(well_id),
                "well_index": int(well_index),
                "image_id": str(t.image_id),
                "embryo_id": str(t.embryo_id),
                "embryo_local_id": str(t.embryo_local_id),
                "channel_id": str(t.channel_id),
                "instance_id": instance_id,
                "time_int": int(t.time_int),
                "bbox_x_min": float(x0),
                "bbox_y_min": float(y0),
                "bbox_x_max": float(x1),
                "bbox_y_max": float(y1),
                "area_px": float(t.area_px),
                "mask_confidence": float(t.confidence),
                "centroid_x_px": float(cx),
                "centroid_y_px": float(cy),
                "is_seed_frame": bool(t.is_seed_frame),
                "source_backend": str(t.source_backend),
                "source_model": str(t.source_model),
                "model_release": str(t.model_release),
                "run_id": str(t.run_id),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS_TRACK_INSTANCES)
    df = df.sort_values(["well_id", "time_int", "image_id", "embryo_id"]).reset_index(drop=True)
    validate_schema(df, REQUIRED_COLUMNS_TRACK_INSTANCES, stage_name="track_instances")
    require_unique(df, UNIQUE_KEY_TRACK_INSTANCES, stage_name="track_instances")
    return df
