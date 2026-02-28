from __future__ import annotations

from pathlib import Path

import numpy as np

from data_pipeline.segmentation_and_tracking.ingestors.sam2_ingestor import (
    ingest_propagation,
    tracks_to_raw_masks,
)


def test_tracks_to_raw_masks_encodes_rle_and_exports_png(tmp_path: Path) -> None:
    # One frame, one embryo.
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    results = {
        0: {
            "embryo_0": {
                "mask": mask,
                "bbox": [3, 2, 6, 4],
                "area": float(mask.sum()),
                "confidence": 0.9,
            }
        }
    }
    image_id_by_frame_index = {0: "exp_A01_BF_f0000"}
    tracks = ingest_propagation(
        results,
        image_id_by_frame_index=image_id_by_frame_index,
        seed_frame_index=0,
        seed_image_id="exp_A01_BF_f0000",
    )
    assert len(tracks) == 1

    exported_mask_dir = tmp_path / "masks"
    masks = tracks_to_raw_masks(
        tracks,
        source_image_path_by_image_id={"exp_A01_BF_f0000": "built_image_data/exp/.../f0000.jpg"},
        exported_mask_dir=exported_mask_dir,
        exported_mask_rel_prefix="segmentation_and_tracking/exp/per_well/A01",
        experiment_id="exp",
        well_id="A01",
        mask_type="embryo",
    )
    assert len(masks) == 1
    m = masks[0]
    assert m.mask_type == "embryo"
    assert "counts" in m.mask_rle and "size" in m.mask_rle
    assert (exported_mask_dir / "exp_A01_embryo_0_f0000_mask.png").exists()

