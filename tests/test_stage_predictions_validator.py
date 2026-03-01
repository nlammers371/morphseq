from __future__ import annotations

import pandas as pd

from data_pipeline.feature_extraction.pipelines.validate_stage_predictions import validate_stage_predictions


def test_validate_stage_predictions_accepts_valid_table() -> None:
    df = pd.DataFrame(
        [
            {
                "experiment_id": "20240418",
                "well_id": "20240418_A01",
                "well_index": "A01",
                "image_id": "20240418_A01_BF_f0000",
                "embryo_id": "embryo_0",
                "frame_index": 0,
                "snip_id": "20240418_A01_embryo_0_BF_f0000",
                "elapsed_time_s": 0.0,
                "start_age_hpf": 24.0,
                "temperature": 28.5,
                "developmental_rate_hpf_per_h": 0.0,
                "predicted_stage_hpf": 24.0,
                "stage_confidence": 1.0,
                "stage_model": "kimmel1995_temp_rate",
                "pipeline_version": "deadbeef",
            }
        ]
    )
    validate_stage_predictions(df)
