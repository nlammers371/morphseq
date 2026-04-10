from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from attrition_qc.data import build_embryo_bin_status, normalize_build04_genotype
from attrition_qc.summaries import (
    summarize_alive_only_qc,
    summarize_attrition,
    summarize_exclusionary_flag_rates,
    summarize_overall_attrition,
)


def _make_raw_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in [21.0, 23.0]:
        rows.append(
            {
                "experiment_id": "exp_a",
                "source_experiment_id": "20260304",
                "experiment_date": "20260304",
                "genotype": "inj_ctrl",
                "embryo_id": "inj_1",
                "predicted_stage_hpf": stage,
                "use_embryo_flag": True,
                "dead_flag": False,
                "dead_flag2": False,
                "sa_outlier_flag": False,
                "sam2_qc_flag": False,
                "frame_flag": False,
                "no_yolk_flag": False,
                "focus_flag": True,
                "bubble_flag": False,
            }
        )
    for stage in [21.0, 23.0]:
        rows.append(
            {
                "experiment_id": "exp_a",
                "source_experiment_id": "20260304",
                "experiment_date": "20260304",
                "genotype": "pbx4_crispant",
                "embryo_id": "pbx4_1",
                "predicted_stage_hpf": stage,
                "use_embryo_flag": False,
                "dead_flag": False,
                "dead_flag2": False,
                "sa_outlier_flag": False,
                "sam2_qc_flag": True,
                "frame_flag": False,
                "no_yolk_flag": False,
                "focus_flag": False,
                "bubble_flag": False,
            }
        )
    for stage in [41.0, 43.0]:
        rows.append(
            {
                "experiment_id": "exp_b",
                "source_experiment_id": "20260306",
                "experiment_date": "20260306",
                "genotype": "pbx1b_pbx4_crispant",
                "embryo_id": "dbl_1",
                "predicted_stage_hpf": stage,
                "use_embryo_flag": False,
                "dead_flag": False,
                "dead_flag2": True,
                "sa_outlier_flag": True,
                "sam2_qc_flag": False,
                "frame_flag": False,
                "no_yolk_flag": False,
                "focus_flag": False,
                "bubble_flag": False,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_build04_genotype():
    assert normalize_build04_genotype("wik-ab_inj_ctrl") == "inj_ctrl"
    assert normalize_build04_genotype("wik-ab") == "wik_ab"
    assert normalize_build04_genotype("pbx4_crispant") == "pbx4_crispant"


def test_build_embryo_bin_status_and_overall_summary():
    raw = _make_raw_rows()
    status = build_embryo_bin_status(raw, bin_width=4.0, time_col="predicted_stage_hpf")

    assert len(status) == 3
    assert set(status["time_bin_start"]) == {20, 40}
    assert (status["embryo_present"]).all()
    assert not status["excluded_other"].any()

    inj = status[status["embryo_id"] == "inj_1"].iloc[0]
    qc = status[status["embryo_id"] == "pbx4_1"].iloc[0]
    death_qc = status[status["embryo_id"] == "dbl_1"].iloc[0]

    assert bool(inj["included"])
    assert not bool(inj["excluded"])
    assert bool(qc["excluded_qc_only"])
    assert bool(death_qc["excluded_dead_and_qc"])

    overall = summarize_overall_attrition(status)
    assert set(overall["genotype"]) == {"inj_ctrl", "pbx4_crispant", "pbx1b_pbx4_crispant"}
    assert int(overall.loc[overall["genotype"] == "inj_ctrl", "embryos_present"].iloc[0]) == 1
    assert float(overall.loc[overall["genotype"] == "inj_ctrl", "fraction_ever_included"].iloc[0]) == 1.0


def test_attrition_summary_invariants_and_alive_only_rates():
    raw = _make_raw_rows()
    status = build_embryo_bin_status(raw, bin_width=4.0, time_col="predicted_stage_hpf")

    attrition = summarize_attrition(status, group_cols=["genotype", "time_bin_start", "time_bin_center"])
    assert (
        attrition["embryos_present"]
        == attrition["embryos_included"] + attrition["embryos_excluded"]
    ).all()
    assert (
        attrition["embryos_excluded"]
        == attrition["excluded_qc_only"] + attrition["excluded_dead_only"] + attrition["excluded_dead_and_qc"] + attrition["excluded_other"]
    ).all()
    assert (
        attrition["excluded_death_involved"]
        == attrition["excluded_dead_only"] + attrition["excluded_dead_and_qc"]
    ).all()

    alive = summarize_alive_only_qc(status, group_cols=["genotype", "time_bin_start", "time_bin_center"])
    assert set(alive["genotype"]) == {"inj_ctrl", "pbx4_crispant"}
    assert (alive["alive_use_embryo_pass_rate"] >= 0.0).all()
    assert (alive["alive_use_embryo_pass_rate"] <= 1.0).all()
    inj_alive = alive[alive["genotype"] == "inj_ctrl"].iloc[0]
    pbx4_alive = alive[alive["genotype"] == "pbx4_crispant"].iloc[0]
    assert float(inj_alive["alive_use_embryo_pass_rate"]) == 1.0
    assert float(pbx4_alive["alive_use_embryo_pass_rate"]) == 0.0
    assert float(pbx4_alive["alive_sam2_qc_flag_rate"]) == 1.0


def test_exclusionary_flag_fractions_track_present_and_excluded_denominators():
    raw = _make_raw_rows()
    status = build_embryo_bin_status(raw, bin_width=4.0, time_col="predicted_stage_hpf")
    summary = summarize_exclusionary_flag_rates(status, group_cols=["genotype", "time_bin_start", "time_bin_center"])

    inj = summary[summary["genotype"] == "inj_ctrl"].iloc[0]
    pbx4 = summary[summary["genotype"] == "pbx4_crispant"].iloc[0]
    dbl = summary[summary["genotype"] == "pbx1b_pbx4_crispant"].iloc[0]

    assert int(inj["embryos_present"]) == 1
    assert float(inj["frame_flag_fraction_present"]) == 0.0
    assert float(inj["dead_flag2_fraction_present"]) == 0.0

    assert int(pbx4["embryos_excluded"]) == 1
    assert float(pbx4["sam2_qc_flag_fraction_present"]) == 1.0
    assert float(pbx4["sam2_qc_flag_fraction_excluded"]) == 1.0

    assert int(dbl["embryos_excluded"]) == 1
    assert float(dbl["dead_flag2_fraction_present"]) == 1.0
    assert float(dbl["sa_outlier_flag_fraction_excluded"]) == 1.0
