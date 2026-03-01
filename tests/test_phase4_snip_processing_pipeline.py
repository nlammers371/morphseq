from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write_u8_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:  # pragma: no cover
        import imageio  # type: ignore
    imageio.imwrite(path, arr.astype(np.uint8))


def test_phase4_snip_processing_well_smoke(tmp_path: Path) -> None:
    # Arrange a tiny "output_root" with one image and one mask.
    output_root = tmp_path / "data_pipeline_output"
    exp = "exp1"
    well = f"{exp}_A01"

    image_id = f"{well}_BF_f0000"
    snip_id = f"{well}_embryo_0_BF_f0000"

    source_rel = f"built_image_data/{exp}/stitched_ff_images/A01/BF/{image_id}.jpg"
    mask_rel = f"segmentation_and_tracking/{exp}/per_well/{well}/masks/embryo_mask/{snip_id}_mask.png"

    # Create a simple synthetic image and a centered mask.
    img = np.zeros((128, 128), dtype=np.uint8)
    img[32:96, 32:96] = 180
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:88, 40:88] = 255

    _write_u8_png(output_root / source_rel, img)
    _write_u8_png(output_root / mask_rel, mask)

    seg_csv = output_root / "segmentation_and_tracking" / exp / "per_well" / well / "contracts" / "segmentation_tracking.csv"
    seg_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "experiment_id": exp,
                "well_id": well,
                "well_index": 1,
                "image_id": image_id,
                "embryo_id": "embryo_0",
                "snip_id": snip_id,
                "frame_index": 0,
                "mask_type": "embryo",
                "source_image_path": source_rel,
                "exported_mask_path": mask_rel,
            }
        ]
    ).to_csv(seg_csv, index=False)

    frame_manifest_csv = output_root / "experiment_metadata" / exp / "frame_manifest.csv"
    frame_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "experiment_id": exp,
                "well_id": well,
                "well_index": 1,
                "frame_index": 0,
                "image_id": image_id,
                "micrometers_per_pixel": 2.0,
            }
        ]
    ).to_csv(frame_manifest_csv, index=False)

    # Act
    from data_pipeline.snip_processing.pipelines.snip_processing import run_snip_processing_well

    cfg = {
        "snip_processing": {
            "enabled": True,
            "mask_type": "embryo",
            "target_pixel_size_um": 7.8,
            "output_shape_hw": [64, 64],
            "blend_radius_um": 30.0,
            "save_raw_crops": True,
            "yolk_mask": {"enabled": False},
            "background_stats": {"mode": "fixed", "fixed": {"mean": 128.0, "std": 30.0}, "definition": "full_frame_outside_embryo"},
            "overwrite": True,
            "skip_existing": True,
        }
    }

    out = run_snip_processing_well(
        output_root=output_root,
        experiment_id=exp,
        well_id=well,
        frame_manifest_csv=frame_manifest_csv,
        segmentation_tracking_csv=seg_csv,
        pipeline_config=cfg,
        verbose=False,
    )

    # Assert outputs exist and manifest is readable.
    assert Path(out["validated_flag"]).exists()
    manifest_pq = Path(out["manifest_parquet"])
    assert manifest_pq.exists()
    df = pd.read_parquet(manifest_pq)
    assert len(df) == 1
    assert df.loc[0, "snip_id"] == snip_id
    assert df.loc[0, "is_valid"] in (True, False)
    # Processed snip should exist for this smoke test.
    processed_rel = str(df.loc[0, "processed_snip_path"])
    assert (output_root / processed_rel).exists()
