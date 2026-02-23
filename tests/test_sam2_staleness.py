from __future__ import annotations

import os
from pathlib import Path

from src.build.pipeline_objects import Experiment


def _set_mtime(path: Path, timestamp: float) -> None:
    os.utime(path, (timestamp, timestamp))


def _make_stub_file(path: Path, content: bytes = b"stub") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_needs_sam2_not_triggered_when_metadata_older(tmp_path):
    exp_name = "20250101_test"

    # Minimal directory layout for Experiment helpers
    ff_file = _make_stub_file(
        tmp_path
        / "built_image_data"
        / "stitched_FF_images"
        / exp_name
        / "sample_stitch.jpg"
    )
    meta_file = _make_stub_file(
        tmp_path / "metadata" / "built_metadata_files" / f"{exp_name}_metadata.csv",
        b"medium,genotype\n",
    )
    sam2_csv = _make_stub_file(
        tmp_path / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{exp_name}.csv"
    )

    # Seed distinct mtimes (ff < metadata < sam2)
    _set_mtime(ff_file, 1_000)
    _set_mtime(meta_file, 1_500)
    _set_mtime(sam2_csv, 2_000)

    exp = Experiment(date=exp_name, data_root=tmp_path)

    assert exp.needs_sam2 is False


def test_needs_sam2_reacts_to_newer_metadata(tmp_path):
    exp_name = "20250101_test"

    ff_file = _make_stub_file(
        tmp_path
        / "built_image_data"
        / "stitched_FF_images"
        / exp_name
        / "sample_stitch.jpg"
    )
    meta_file = _make_stub_file(
        tmp_path / "metadata" / "built_metadata_files" / f"{exp_name}_metadata.csv"
    )
    sam2_csv = _make_stub_file(
        tmp_path / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{exp_name}.csv"
    )

    _set_mtime(ff_file, 1_000)
    _set_mtime(meta_file, 1_500)
    _set_mtime(sam2_csv, 2_000)

    exp = Experiment(date=exp_name, data_root=tmp_path)

    assert exp.needs_sam2 is False

    # Make metadata newer than the SAM2 CSV; property should flip to True
    _set_mtime(meta_file, 3_000)

    assert exp.needs_sam2 is True
