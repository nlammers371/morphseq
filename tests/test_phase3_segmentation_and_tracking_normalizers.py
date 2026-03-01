from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.segmentation_and_tracking.raw_types import RawDetection, RawMask, RawTrack, SeedSelection
from data_pipeline.segmentation_and_tracking.normalizers import (
    normalize_frame_detections,
    normalize_seed_selection,
    normalize_track_instances,
    normalize_mask_rle,
)


def test_normalize_frame_detections_requires_provenance() -> None:
    det = RawDetection(
        frame_index=0,
        image_id="exp_A01_BF_f0000",
        box_xyxy_norm=[0.1, 0.1, 0.2, 0.2],
        confidence=0.9,
        box_xyxy_abs=[10, 10, 20, 20],
        image_height_px=100,
        image_width_px=100,
    )
    with pytest.raises(ValueError, match="missing source_backend"):
        normalize_frame_detections([det], experiment_id="exp", well_id="A01", video_id="exp_A01")


def test_normalize_frame_detections_outputs_schema() -> None:
    dets = [
        RawDetection(
            frame_index=0,
            image_id="exp_A01_BF_f0000",
            box_xyxy_norm=[0.1, 0.1, 0.2, 0.2],
            confidence=0.9,
            box_xyxy_abs=[10, 10, 20, 20],
            image_height_px=100,
            image_width_px=100,
            source_backend="groundingdino",
            source_model="SwinT_OGC",
            model_release="unknown",
            run_id="r1",
        ),
        RawDetection(
            frame_index=0,
            image_id="exp_A01_BF_f0000",
            box_xyxy_norm=[0.3, 0.3, 0.4, 0.4],
            confidence=0.8,
            box_xyxy_abs=[30, 30, 40, 40],
            image_height_px=100,
            image_width_px=100,
            source_backend="groundingdino",
            source_model="SwinT_OGC",
            model_release="unknown",
            run_id="r1",
        ),
    ]
    df = normalize_frame_detections(dets, experiment_id="exp", well_id="A01", video_id="exp_A01")
    assert len(df) == 2
    assert set(["experiment_id", "well_id", "image_id", "detection_index"]).issubset(df.columns)
    assert df["detection_index"].tolist() == [0, 1]


def test_normalize_seed_selection_outputs_schema() -> None:
    seed = SeedSelection(
        experiment_id="exp",
        well_id="A01",
        video_id="exp_A01",
        seed_frame_index=0,
        seed_image_id="exp_A01_BF_f0000",
        num_detections=2,
        avg_confidence=0.85,
        selection_reason="highest_avg_confidence",
        candidate_frames_evaluated=3,
        selected_detection_indices=[0, 1],
        detector_backend="groundingdino",
        run_id="r1",
    )
    df = normalize_seed_selection([seed])
    assert len(df) == 1
    assert df.loc[0, "seed_image_id"] == "exp_A01_BF_f0000"


def test_normalize_tracks_and_masks_round_trip(tmp_path: Path) -> None:
    t = RawTrack(
        frame_index=0,
        image_id="exp_A01_BF_f0000",
        embryo_id="embryo_0",
        embryo_local_id="embryo_0",
        channel_id="BF",
        mask=None,
        bbox_xyxy_abs=[1, 2, 3, 4],
        area_px=10,
        confidence=0.7,
        is_seed_frame=True,
        source_backend="sam2",
        source_model="hiera_large",
        model_release="unknown",
        run_id="r1",
    )
    tracks_df = normalize_track_instances([t], experiment_id="exp", well_id="A01", well_index=1, video_id="exp_A01")
    assert len(tracks_df) == 1
    assert bool(tracks_df.loc[0, "is_seed_frame"]) is True

    m = RawMask(
        frame_index=0,
        image_id="exp_A01_BF_f0000",
        embryo_id="embryo_0",
        embryo_local_id="embryo_0",
        channel_id="BF",
        mask_type="sam",
        mask_rle={"counts": "abc", "size": [5, 5]},
        area_px=10,
        bbox_xyxy_abs=[1, 2, 3, 4],
        centroid_x_px=2.0,
        centroid_y_px=3.0,
        confidence=0.7,
        is_seed_frame=True,
        exported_mask_path="segmentation_and_tracking/exp/per_well/A01/masks/x.png",
        source_image_path="built_image_data/exp/...",
        source_backend="sam2",
        source_model="hiera_large",
        model_release="unknown",
        run_id="r1",
    )
    masks_df = normalize_mask_rle([m], experiment_id="exp", well_id="A01", video_id="exp_A01", channel_id="BF")
    assert len(masks_df) == 1
    assert masks_df.loc[0, "mask_type"] == "sam"
