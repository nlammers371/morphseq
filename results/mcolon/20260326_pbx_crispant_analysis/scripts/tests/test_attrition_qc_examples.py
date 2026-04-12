from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from attrition_qc.examples import (
    build_flag_example_frame_summary,
    build_flag_example_summary,
    select_flag_example_frames,
    select_flag_examples,
)


def _make_example_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame_index, stage, no_yolk, use_flag in [
        (0, 21.0, False, True),
        (1, 22.0, True, False),
        (2, 23.0, True, False),
    ]:
        rows.append(
            {
                "experiment_id": "exp_a",
                "source_experiment_id": "20260304",
                "experiment_date": "20260304",
                "genotype": "inj_ctrl",
                "embryo_id": "inj_1",
                "snip_id": f"inj_1_t{frame_index:04d}",
                "frame_index": frame_index,
                "predicted_stage_hpf": stage,
                "use_embryo_flag": use_flag,
                "dead_flag": False,
                "dead_flag2": False,
                "sa_outlier_flag": False,
                "sam2_qc_flag": False,
                "sam2_qc_flags": "",
                "frame_flag": False,
                "no_yolk_flag": no_yolk,
                "focus_flag": False,
                "bubble_flag": False,
            }
        )
    for frame_index, stage, frame_flag, use_flag in [
        (0, 41.0, True, False),
        (1, 42.0, True, False),
    ]:
        rows.append(
            {
                "experiment_id": "exp_b",
                "source_experiment_id": "20260306",
                "experiment_date": "20260306",
                "genotype": "pbx4_crispant",
                "embryo_id": "pbx4_1",
                "snip_id": f"pbx4_1_t{frame_index:04d}",
                "frame_index": frame_index,
                "predicted_stage_hpf": stage,
                "use_embryo_flag": use_flag,
                "dead_flag": False,
                "dead_flag2": False,
                "sa_outlier_flag": False,
                "sam2_qc_flag": False,
                "sam2_qc_flags": "",
                "frame_flag": frame_flag,
                "no_yolk_flag": False,
                "focus_flag": False,
                "bubble_flag": False,
            }
        )
    for frame_index, stage in [(0, 45.0), (1, 46.0)]:
        rows.append(
            {
                "experiment_id": "exp_b",
                "source_experiment_id": "20260306",
                "experiment_date": "20260306",
                "genotype": "pbx4_crispant",
                "embryo_id": "pbx4_2",
                "snip_id": f"pbx4_2_t{frame_index:04d}",
                "frame_index": frame_index,
                "predicted_stage_hpf": stage,
                "use_embryo_flag": False,
                "dead_flag": False,
                "dead_flag2": False,
                "sa_outlier_flag": True,
                "sam2_qc_flag": False,
                "sam2_qc_flags": "",
                "frame_flag": True,
                "no_yolk_flag": False,
                "focus_flag": False,
                "bubble_flag": False,
            }
        )
    for frame_index, stage in [(0, 61.0), (1, 62.0)]:
        rows.append(
            {
                "experiment_id": "exp_c",
                "source_experiment_id": "20260306",
                "experiment_date": "20260306",
                "genotype": "wik_ab",
                "embryo_id": "wik_1",
                "snip_id": f"wik_1_t{frame_index:04d}",
                "frame_index": frame_index,
                "predicted_stage_hpf": stage,
                "use_embryo_flag": False,
                "dead_flag": False,
                "dead_flag2": False,
                "sa_outlier_flag": False,
                "sam2_qc_flag": True,
                "sam2_qc_flags": "MASK_ON_EDGE",
                "frame_flag": False,
                "no_yolk_flag": False,
                "focus_flag": False,
                "bubble_flag": False,
            }
        )
    return pd.DataFrame(rows)


def test_build_flag_example_summary_tracks_only_target_status():
    df = _make_example_rows()
    summary = build_flag_example_summary(df, target_flag="frame_flag", bin_width=4.0, time_col="predicted_stage_hpf")
    assert len(summary) == 2
    only_target = summary.loc[summary["embryo_id"] == "pbx4_1", "only_target_canonical"].iloc[0]
    with_other = summary.loc[summary["embryo_id"] == "pbx4_2", "only_target_canonical"].iloc[0]
    assert bool(only_target)
    assert not bool(with_other)


def test_select_flag_examples_prefers_only_target_and_excluded_bins():
    df = _make_example_rows()
    summary = build_flag_example_summary(df, target_flag="frame_flag", bin_width=4.0, time_col="predicted_stage_hpf")
    selected = select_flag_examples(summary, max_examples_per_genotype=1)
    assert len(selected) == 1
    assert selected.iloc[0]["embryo_id"] == "pbx4_1"


def test_select_flag_examples_can_require_bin_level_family_exclusivity():
    df = _make_example_rows()
    summary = build_flag_example_summary(df, target_flag="frame_flag", bin_width=4.0, time_col="predicted_stage_hpf")
    assert bool(summary.loc[summary["embryo_id"] == "pbx4_1", "target_family_only_in_bin"].iloc[0])
    assert not bool(summary.loc[summary["embryo_id"] == "pbx4_2", "target_family_only_in_bin"].iloc[0])
    selected = select_flag_examples(
        summary,
        max_examples_per_genotype=2,
        require_target_family_only_in_bin=True,
    )
    assert len(selected) == 1
    assert selected.iloc[0]["embryo_id"] == "pbx4_1"


def test_build_flag_example_frame_summary_tracks_frame_level_family_exclusivity():
    df = _make_example_rows()
    summary = build_flag_example_frame_summary(df, target_flag="frame_flag", time_col="predicted_stage_hpf")
    only_target = summary.loc[summary["embryo_id"] == "pbx4_1", "target_family_only_in_frame"].unique().tolist()
    with_other = summary.loc[summary["embryo_id"] == "pbx4_2", "target_family_only_in_frame"].unique().tolist()
    assert only_target == [True]
    assert with_other == [False]


def test_select_flag_example_frames_filters_to_unique_alive_excluded_frames():
    df = _make_example_rows()
    summary = build_flag_example_frame_summary(df, target_flag="frame_flag", time_col="predicted_stage_hpf")
    selected = select_flag_example_frames(
        summary,
        max_examples_per_genotype=5,
        require_target_family_only_in_frame=True,
    )
    assert len(selected) == 1
    assert set(selected["embryo_id"]) == {"pbx4_1"}
