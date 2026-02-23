"""Tests for OT pair metrics/artifact storage helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyze.utils.optimal_transport import UOTResultCanonical, UOTResultWork
from analyze.optimal_transport_morphometrics.uot_masks.feature_compaction.storage import (
    apply_contract_dtypes,
    build_pair_metrics_record,
    compute_barycentric_projection,
    save_pair_artifacts,
    upsert_ot_pair_metrics_parquet,
    upsert_pair_metrics,
)


def _make_results() -> tuple[UOTResultWork, UOTResultCanonical]:
    coupling = np.array(
        [
            [0.6, 0.4],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    work = UOTResultWork(
        cost=1.23,
        coupling=coupling,
        mass_created_work=np.zeros((2, 2), dtype=np.float32),
        mass_destroyed_work=np.zeros((2, 2), dtype=np.float32),
        velocity_work_px_per_step_yx=np.zeros((2, 2, 2), dtype=np.float32),
        support_src_yx=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        support_tgt_yx=np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        weights_src=np.array([1.0, 1.0], dtype=np.float32),
        weights_tgt=np.array([1.0, 1.0], dtype=np.float32),
        diagnostics={"metrics": {"total_transport_cost": 1.23, "created_mass_pct": 3.0}},
        work_shape_hw=(2, 2),
        work_um_per_px=40.0,
    )
    canon = UOTResultCanonical(
        cost=1.23,
        mass_created_canon=np.zeros((4, 4), dtype=np.float32),
        mass_destroyed_canon=np.zeros((4, 4), dtype=np.float32),
        velocity_canon_px_per_step_yx=np.zeros((4, 4, 2), dtype=np.float32),
        canonical_shape_hw=(4, 4),
        canonical_um_per_px=10.0,
        diagnostics=work.diagnostics,
    )
    return work, canon


def test_upsert_pair_metrics_idempotent():
    existing = pd.DataFrame(
        [
            {"run_id": "r1", "pair_id": "p1", "backend": "POT", "metric": "sqeuclidean", "cost": 1.0},
        ]
    )
    incoming = pd.DataFrame(
        [
            {"run_id": "r1", "pair_id": "p1", "backend": "POT", "metric": "sqeuclidean", "cost": 2.0},
            {"run_id": "r1", "pair_id": "p2", "backend": "OTT", "metric": "sqeuclidean", "cost": 3.0},
            {"run_id": "r1", "pair_id": "p2", "backend": "OTT", "metric": "sqeuclidean", "cost": 4.0},
        ]
    )
    merged = upsert_pair_metrics(existing, incoming)
    assert len(merged) == 2
    assert float(merged.loc[merged["pair_id"] == "p1", "cost"].iloc[0]) == 2.0
    assert float(merged.loc[merged["pair_id"] == "p2", "cost"].iloc[0]) == 4.0


def test_apply_contract_dtypes_sets_categories():
    df = pd.DataFrame(
        [
            {"backend": "POT", "metric": "sqeuclidean", "canonical_grid_align_mode": "yolk", "success": True},
            {"backend": "OTT", "metric": "sqeuclidean", "canonical_grid_align_mode": "yolk", "success": False},
        ]
    )
    out = apply_contract_dtypes(df)
    assert str(out["backend"].dtype) == "category"
    assert str(out["metric"].dtype) == "category"
    assert str(out["canonical_grid_align_mode"].dtype) == "category"
    assert str(out["success"].dtype) == "boolean"


def test_apply_contract_dtypes_normalizes_identifier_strings():
    df = pd.DataFrame(
        [
            {"run_id": "r1", "pair_id": "p1", "src_experiment_id": 20251113, "src_experiment_date": 20251113},
            {"run_id": "r2", "pair_id": "p2", "src_experiment_id": "20251114", "src_experiment_date": "20251114"},
        ]
    )
    out = apply_contract_dtypes(df)
    assert str(out["run_id"].dtype) == "string"
    assert str(out["pair_id"].dtype) == "string"
    assert str(out["src_experiment_id"].dtype) == "string"
    assert str(out["src_experiment_date"].dtype) == "string"
    assert out["src_experiment_id"].tolist() == ["20251113", "20251114"]


def test_compute_barycentric_projection_dense():
    result_work, _result_canon = _make_results()
    bary = compute_barycentric_projection(result_work)
    np.testing.assert_allclose(bary["barycentric_tgt_yx"][0], np.array([0.4, 1.4]), atol=1e-6)
    np.testing.assert_allclose(bary["barycentric_tgt_yx"][1], np.array([1.0, 2.0]), atol=1e-6)
    np.testing.assert_allclose(bary["barycentric_velocity_yx"][0], np.array([0.4, 1.4]), atol=1e-6)
    np.testing.assert_allclose(bary["barycentric_velocity_yx"][1], np.array([0.0, 1.0]), atol=1e-6)


def test_save_pair_artifacts_writes_npz(tmp_path):
    result_work, result_canon = _make_results()
    paths = save_pair_artifacts(
        result_work=result_work,
        result_canon=result_canon,
        artifact_root=tmp_path,
        pair_id="pair_a",
        float_dtype="float32",
        include_barycentric=True,
    )
    assert paths["fields"].exists()
    assert paths["metadata"].exists()
    assert paths["barycentric"].exists()


def test_upsert_ot_pair_metrics_parquet(tmp_path):
    parquet_path = tmp_path / "ot_pair_metrics.parquet"
    row1 = {
        "run_id": "run_x",
        "pair_id": "pair_1",
        "backend": "POT",
        "metric": "sqeuclidean",
        "cost": 1.0,
    }
    row2 = {
        "run_id": "run_x",
        "pair_id": "pair_1",
        "backend": "POT",
        "metric": "sqeuclidean",
        "cost": 2.0,
    }
    upsert_ot_pair_metrics_parquet(parquet_path, [row1])
    out = upsert_ot_pair_metrics_parquet(parquet_path, [row2])
    assert len(out) == 1
    assert float(out["cost"].iloc[0]) == 2.0


def test_upsert_ot_pair_metrics_parquet_handles_mixed_identifier_types(tmp_path):
    parquet_path = tmp_path / "ot_pair_metrics.parquet"
    row1 = {
        "run_id": "run_x",
        "pair_id": "pair_1",
        "src_experiment_id": 20251113,
        "tgt_experiment_id": 20251113,
        "src_experiment_date": 20251113,
        "tgt_experiment_date": 20251113,
        "cost": 1.0,
    }
    row2 = {
        "run_id": "run_y",
        "pair_id": "pair_2",
        "src_experiment_id": "20251114",
        "tgt_experiment_id": "20251114",
        "src_experiment_date": "20251114",
        "tgt_experiment_date": "20251114",
        "cost": 2.0,
    }
    upsert_ot_pair_metrics_parquet(parquet_path, [row1])
    out = upsert_ot_pair_metrics_parquet(parquet_path, [row2])
    assert len(out) == 2
    assert str(out["src_experiment_id"].dtype) == "string"
    assert str(out["src_experiment_date"].dtype) == "string"
    assert out["src_experiment_id"].tolist() == ["20251113", "20251114"]


def test_build_pair_metrics_record_includes_core_fields():
    result, _canon = _make_results()
    src_meta = {"embryo_id": "src_e", "frame_index": 10, "relative_time_s": 100.0}
    tgt_meta = {"embryo_id": "tgt_e", "frame_index": 11, "relative_time_s": 130.0}
    rec = build_pair_metrics_record(
        run_id="run1",
        pair_id="pair1",
        result=result,
        src_meta=src_meta,
        tgt_meta=tgt_meta,
        config={"epsilon": 1e-4, "metric": "sqeuclidean", "mass_mode": "uniform"},
        backend="POT",
        runtime_sec=2.5,
    )
    assert rec["run_id"] == "run1"
    assert rec["pair_id"] == "pair1"
    assert rec["src_embryo_id"] == "src_e"
    assert rec["tgt_embryo_id"] == "tgt_e"
    assert rec["delta_time_s"] == 30.0
    assert rec["backend"] == "POT"
    assert rec["cost"] == 1.23
