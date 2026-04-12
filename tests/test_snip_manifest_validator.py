from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.snip_processing.validate_snip_manifest import validate_snip_manifest


def test_validate_snip_manifest_happy_path(tmp_path: Path) -> None:
    img = tmp_path / "img.tif"
    mask = tmp_path / "mask.png"
    out = tmp_path / "snip.jpg"
    raw = tmp_path / "raw.tif"
    img.write_bytes(b"x")
    mask.write_bytes(b"x")
    out.write_bytes(b"x")
    raw.write_bytes(b"x")

    df = pd.DataFrame(
        [
            {
                "schema_version": 2,
                "snip_id": "emb1_BF_f0003",
                "experiment_id": "E",
                "well_id": "E_A01",
                "well_index": "A01",
                "image_id": "E_A01_BF_t0003",
                "frame_index": 3,
                "channel_id": "BF",
                "embryo_id": "emb1",
                "instance_id": "emb1",
                "source_image_path": str(img),
                "embryo_mask_path": str(mask),
                "yolk_mask_path": None,
                "source_micrometers_per_pixel": 0.65,
                "frame_snapshot_hash": "0123456789abcdef",
                "processed_snip_path": str(out),
                "raw_crop_path": str(raw),
                "target_pixel_size_um": 1.0,
                "output_height_px": 512,
                "output_width_px": 512,
                "blend_radius_um": 30.0,
                "background_mean": 128.0,
                "background_std": 30.0,
                "rotation_angle_rad": 0.1,
                "rotation_angle_deg": 5.7,
                "rotation_used_yolk": False,
                "snip_processing_run_id": "r1",
                "snip_processing_version": "abc123",
            }
        ]
    )
    validate_snip_manifest(df)


def test_validate_snip_manifest_missing_file_fails(tmp_path: Path) -> None:
    img = tmp_path / "img.tif"
    mask = tmp_path / "mask.png"
    out = tmp_path / "snip.jpg"
    img.write_bytes(b"x")
    mask.write_bytes(b"x")
    out.write_bytes(b"x")

    df = pd.DataFrame(
        [
            {
                "schema_version": 2,
                "snip_id": "emb1_BF_f0003",
                "experiment_id": "E",
                "well_id": "E_A01",
                "well_index": "A01",
                "image_id": "E_A01_BF_t0003",
                "frame_index": 3,
                "channel_id": "BF",
                "embryo_id": "emb1",
                "instance_id": "emb1",
                "source_image_path": str(img),
                "embryo_mask_path": str(mask),
                "yolk_mask_path": None,
                "source_micrometers_per_pixel": 0.65,
                "frame_snapshot_hash": "0123456789abcdef",
                "processed_snip_path": str(tmp_path / "missing.jpg"),
                "raw_crop_path": None,
                "target_pixel_size_um": 1.0,
                "output_height_px": 512,
                "output_width_px": 512,
                "blend_radius_um": 30.0,
                "background_mean": 128.0,
                "background_std": 30.0,
                "rotation_angle_rad": 0.1,
                "rotation_angle_deg": 5.7,
                "rotation_used_yolk": False,
                "snip_processing_run_id": "r1",
                "snip_processing_version": "abc123",
            }
        ]
    )
    with pytest.raises(ValueError):
        validate_snip_manifest(df)

