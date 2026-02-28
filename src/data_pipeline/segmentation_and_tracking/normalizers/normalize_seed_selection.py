from __future__ import annotations

import pandas as pd

from data_pipeline.schemas.segmentation import (
    REQUIRED_COLUMNS_SEED_SELECTION,
    UNIQUE_KEY_SEED_SELECTION,
)

from ..raw_types import SeedSelection
from ._shared import dumps_json, require_unique, validate_schema


def normalize_seed_selection(seeds: list[SeedSelection]) -> pd.DataFrame:
    rows = []
    for s in seeds:
        rows.append(
            {
                "experiment_id": str(s.experiment_id),
                "well_id": str(s.well_id),
                "video_id": str(s.video_id),
                "seed_frame_index": int(s.seed_frame_index),
                "seed_image_id": str(s.seed_image_id),
                "num_detections": int(s.num_detections),
                "avg_confidence": float(s.avg_confidence),
                "selection_reason": str(s.selection_reason),
                "candidate_frames_evaluated": int(s.candidate_frames_evaluated),
                "selected_detection_indices": dumps_json(list(s.selected_detection_indices)),
                "detector_backend": str(s.detector_backend),
                "run_id": str(s.run_id),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS_SEED_SELECTION)
    validate_schema(df, REQUIRED_COLUMNS_SEED_SELECTION, stage_name="seed_selection")
    require_unique(df, UNIQUE_KEY_SEED_SELECTION, stage_name="seed_selection")
    return df

