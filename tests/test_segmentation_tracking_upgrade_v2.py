from __future__ import annotations

import pandas as pd

from data_pipeline.segmentation.grounded_sam2.csv_formatter import mint_segmentation_tracking_snapshot


def test_mint_segmentation_tracking_snapshot_stamps_snapshot_and_snip_id() -> None:
    v1 = pd.DataFrame(
        [
            {
                "experiment_id": "E",
                "video_id": "E_A01",
                "well_id": "E_A01",
                "well_index": "A01",
                "image_id": "E_A01_BF_t0003",
                "embryo_id": "emb1",
                "snip_id": "legacy",
                "frame_index": 3,
                "time_int": 3,
                "mask_rle": '{"counts":"eJzt","size":[2,2]}',
                "area_px": 1.0,
                "bbox_x_min": 0,
                "bbox_y_min": 0,
                "bbox_x_max": 1,
                "bbox_y_max": 1,
                "mask_confidence": 0.9,
                "centroid_x_px": 0.5,
                "centroid_y_px": 0.5,
                "is_seed_frame": False,
                "source_image_path": "/tmp/img.tif",
                "exported_mask_path": "/tmp/mask.png",
            }
        ]
    )

    snap = pd.DataFrame(
        [
            {
                "image_id": "E_A01_BF_t0003",
                "source_image_path": "/tmp/img.tif",
                "channel_id": "BF",
                "micrometers_per_pixel": 0.65,
                "image_width_px": 10,
                "image_height_px": 20,
            }
        ]
    )

    v2 = mint_segmentation_tracking_snapshot(v1, frame_snapshot_df=snap)
    assert int(v2["schema_version"].iloc[0]) == 2
    assert v2["channel_id"].iloc[0] == "BF"
    assert float(v2["source_micrometers_per_pixel"].iloc[0]) == 0.65
    assert v2["snip_id"].iloc[0] == "emb1_BF_f0003"
    assert v2["instance_id"].iloc[0] == "emb1"
    assert "frame_snapshot_hash" in v2.columns
    assert isinstance(v2["frame_snapshot_hash"].iloc[0], str) and len(v2["frame_snapshot_hash"].iloc[0]) == 16
    assert v2["embryo_mask_rle"].iloc[0] == v2["mask_rle"].iloc[0]
    # embryo_mask_path is intentionally empty so downstream snip processing materializes a per-snip binary PNG from RLE.
    assert v2["embryo_mask_path"].iloc[0] == ""
