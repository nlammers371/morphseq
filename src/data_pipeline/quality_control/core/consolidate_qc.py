from __future__ import annotations

import pandas as pd

from data_pipeline.schemas.quality_control import REQUIRED_COLUMNS_QC, SNIP_EXCLUSION_FLAGS, SNIP_INFORMATIONAL_FLAGS
from ._shared import assert_exact_snip_id_set, assert_no_duplicate_columns, assert_unique_snip_id


def _exact_columns(df: pd.DataFrame, expected: list[str], stage_name: str) -> pd.DataFrame:
    assert_no_duplicate_columns(df, stage_name)
    missing = [column for column in expected if column not in df.columns]
    extra = [column for column in df.columns if column not in expected]
    if missing or extra:
        raise ValueError(f"{stage_name}: column mismatch (missing={missing}, extra={extra})")
    return df[expected].copy()


def consolidate_qc_flags(
    snip_universe_df: pd.DataFrame,
    segmentation_qc_df: pd.DataFrame,
    viability_qc_df: pd.DataFrame,
    death_detection_df: pd.DataFrame,
    surface_area_qc_df: pd.DataFrame,
    auxiliary_mask_qc_df: pd.DataFrame,
    focus_qc_df: pd.DataFrame,
    motion_qc_df: pd.DataFrame,
) -> pd.DataFrame:
    assert_unique_snip_id(snip_universe_df, "snip universe")
    base = snip_universe_df[["snip_id", "predicted_stage_hpf"]].copy()

    stage_frames = {
        "segmentation_qc": _exact_columns(
            segmentation_qc_df,
            ["snip_id", "edge_flag", "discontinuous_mask_flag", "overlapping_mask_flag"],
            "segmentation_qc",
        ),
        "viability_qc": _exact_columns(viability_qc_df, ["snip_id", "viability_flag"], "viability_qc"),
        "death_detection": _exact_columns(
            death_detection_df,
            ["snip_id", "dead_flag", "death_inflection_time_int", "death_predicted_stage_hpf"],
            "death_detection",
        ),
        "surface_area_qc": _exact_columns(surface_area_qc_df, ["snip_id", "sa_outlier_flag"], "surface_area_qc"),
        "auxiliary_mask_qc": _exact_columns(
            auxiliary_mask_qc_df,
            ["snip_id", "yolk_flag", "bubble_flag"],
            "auxiliary_mask_qc",
        ),
        "focus_qc": _exact_columns(focus_qc_df, ["snip_id", "focus_flag"], "focus_qc"),
        "motion_qc": _exact_columns(motion_qc_df, ["snip_id", "motion_flag"], "motion_qc"),
    }

    expected_ids = snip_universe_df["snip_id"].astype(str)
    for stage_name, frame in stage_frames.items():
        assert_unique_snip_id(frame, stage_name)
        assert_exact_snip_id_set(frame, expected_ids, stage_name)

    consolidated = base
    for frame in stage_frames.values():
        consolidated = consolidated.merge(frame, on="snip_id", how="left", validate="one_to_one")

    if consolidated.isna().any(axis=None):
        raise ValueError("consolidate_qc: merge introduced null values")

    consolidated["use_snip"] = ~consolidated[SNIP_EXCLUSION_FLAGS].any(axis=1)
    consolidated["use_snip"] = consolidated["use_snip"].astype(bool)
    consolidated["death_predicted_stage_hpf"] = consolidated["predicted_stage_hpf"].where(consolidated["dead_flag"].astype(bool))
    ordered = consolidated[["snip_id", "use_snip", *SNIP_EXCLUSION_FLAGS, *SNIP_INFORMATIONAL_FLAGS, "death_inflection_time_int", "death_predicted_stage_hpf"]].copy()

    for flag in SNIP_EXCLUSION_FLAGS + SNIP_INFORMATIONAL_FLAGS + ["use_snip"]:
        ordered[flag] = ordered[flag].astype(bool)

    ordered["death_inflection_time_int"] = ordered["death_inflection_time_int"].astype("Int64")
    ordered["death_predicted_stage_hpf"] = ordered["death_predicted_stage_hpf"].astype("Float64")
    return ordered[REQUIRED_COLUMNS_QC]
