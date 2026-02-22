from __future__ import annotations

import pandas as pd

from data_pipeline.metadata_ingest.time_helpers import add_elapsed_time_columns
from data_pipeline.metadata_ingest.time_helpers import add_experiment_time_cols
from data_pipeline.metadata_ingest.time_helpers import ensure_frame_time_alias


def test_ensure_frame_time_alias_backfills_and_int_casts() -> None:
    df = pd.DataFrame(
        {
            "experiment_id": ["exp1", "exp1"],
            "well_id": ["exp1_A01", "exp1_A01"],
            "channel_id": ["BF", "BF"],
            "frame_index": [0.0, 1.0],
        }
    )
    out = ensure_frame_time_alias(df, stage_name="unit_test")
    assert list(out["frame_index"]) == [0, 1]
    assert list(out["time_int"]) == [0, 1]


def test_add_elapsed_time_columns_prefers_experiment_time() -> None:
    df = pd.DataFrame(
        {
            "experiment_id": ["exp1", "exp1", "exp1"],
            "well_id": ["exp1_A01", "exp1_A01", "exp1_A01"],
            "channel_id": ["BF", "BF", "BF"],
            "frame_index": [0, 1, 2],
            "time_int": [0, 1, 2],
            "experiment_time_s": [10.0, 40.0, 70.0],
            "frame_interval_s": [999.0, 999.0, 999.0],
        }
    )
    out = add_elapsed_time_columns(df, group_cols=["experiment_id", "well_id", "channel_id"])
    assert list(out["elapsed_time_s"]) == [0.0, 30.0, 60.0]
    assert list(out["elapsed_time_min"]) == [0.0, 0.5, 1.0]


def test_add_elapsed_time_columns_falls_back_to_frame_interval() -> None:
    df = pd.DataFrame(
        {
            "experiment_id": ["exp1", "exp1"],
            "well_id": ["exp1_A01", "exp1_A01"],
            "channel_id": ["BF", "BF"],
            "frame_index": [3, 4],
            "time_int": [3, 4],
            "frame_interval_s": [600.0, 600.0],
        }
    )
    out = add_elapsed_time_columns(df, group_cols=["experiment_id", "well_id", "channel_id"])
    assert list(out["elapsed_time_s"]) == [0.0, 600.0]
    assert list(out["elapsed_time_hr"]) == [0.0, 600.0 / 3600.0]


def test_add_experiment_time_cols_derives_min_and_hr() -> None:
    df = pd.DataFrame({"experiment_time_s": [0.0, 600.0, 1800.0]})
    out = add_experiment_time_cols(df)
    assert list(out["experiment_time_min"]) == [0.0, 10.0, 30.0]
    assert list(out["experiment_time_hr"]) == [0.0, 600.0 / 3600.0, 0.5]
