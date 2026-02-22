from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.metadata_ingest.frame_manifest.build_frame_manifest import build_frame_manifest
from data_pipeline.metadata_ingest.stitched_index.validate_stitched_image_index import (
    validate_stitched_image_index,
)


def _write_stitched_row(
    stitched_csv: Path,
    *,
    frame_index: int,
    include_time_int: bool,
    time_int: int | None = None,
) -> None:
    row = {
        "experiment_id": "exp1",
        "microscope_id": "Keyence",
        "well_id": "A01",
        "well_index": "A01",
        "channel_id": "BF",
        "frame_index": frame_index,
        "image_id": f"exp1_A01_BF_t{frame_index:04d}",
        "stitched_image_path": "built_image_data/exp1/stitched_ff_images/A01/BF/exp1_A01_BF_t0000.jpg",
        "materialization_status": "created",
        "source_artifact_path": "raw_image_data/Keyence/exp1/W001/P00001",
        "source_artifact_kind": "keyence_tiff_stitched_tiles_log",
    }
    if include_time_int:
        row["time_int"] = frame_index if time_int is None else time_int
    pd.DataFrame([row]).to_csv(stitched_csv, index=False)


def test_validate_stitched_index_accepts_frame_index_without_time_int(tmp_path: Path) -> None:
    data_root = tmp_path / "data_pipeline_output"
    exp_dir = data_root / "experiment_metadata" / "exp1"
    exp_dir.mkdir(parents=True)
    image_path = data_root / "built_image_data" / "exp1" / "stitched_ff_images" / "A01" / "BF" / "exp1_A01_BF_t0000.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"test")

    stitched_csv = exp_dir / "stitched_image_index.csv"
    validation_flag = exp_dir / ".stitched_image_index.validated"
    _write_stitched_row(stitched_csv, frame_index=0, include_time_int=False)

    validated = validate_stitched_image_index(stitched_csv, validation_flag)
    assert len(validated) == 1
    assert validation_flag.exists()


def test_validate_stitched_index_rejects_time_int_frame_index_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "data_pipeline_output"
    exp_dir = data_root / "experiment_metadata" / "exp1"
    exp_dir.mkdir(parents=True)
    image_path = data_root / "built_image_data" / "exp1" / "stitched_ff_images" / "A01" / "BF" / "exp1_A01_BF_t0000.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"test")

    stitched_csv = exp_dir / "stitched_image_index.csv"
    validation_flag = exp_dir / ".stitched_image_index.validated"
    _write_stitched_row(stitched_csv, frame_index=1, include_time_int=True, time_int=0)

    with pytest.raises(ValueError, match="frame_index != time_int"):
        validate_stitched_image_index(stitched_csv, validation_flag)


def test_build_frame_manifest_joins_on_frame_index_and_emits_time_alias(tmp_path: Path) -> None:
    data_root = tmp_path / "data_pipeline_output"
    exp_dir = data_root / "experiment_metadata" / "exp1"
    exp_dir.mkdir(parents=True)

    stitched_csv = exp_dir / "stitched_image_index.csv"
    scope_csv = exp_dir / "scope_and_plate_metadata.csv"
    output_csv = exp_dir / "frame_manifest.csv"

    _write_stitched_row(stitched_csv, frame_index=3, include_time_int=False)

    scope_row = {
        "experiment_id": "exp1",
        "well_id": "A01",
        "well_index": "A01",
        "channel_id": "BF",
        "time_int": 3,
        "channel_name_raw": "BF",
        "micrometers_per_pixel": 1.23,
        "frame_interval_s": 600,
        "absolute_start_time": "2026-01-01T00:00:00",
        "experiment_time_s": 1800,
        "image_width_px": 1440,
        "image_height_px": 3420,
        "objective_magnification": 4,
        "genotype": "wt",
        "treatment": "control",
        "medium": "egg_water",
        "temperature": 28.5,
        "start_age_hpf": 24,
        "embryos_per_well": 1,
    }
    pd.DataFrame([scope_row]).to_csv(scope_csv, index=False)

    manifest = build_frame_manifest(
        stitched_index_csv=stitched_csv,
        scope_and_plate_csv=scope_csv,
        output_csv=output_csv,
    )
    assert output_csv.exists()
    assert manifest.loc[0, "frame_index"] == 3
    assert manifest.loc[0, "time_int"] == 3
