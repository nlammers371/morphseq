from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyze.viz.embryo_renderer import (
    EmbryoTrack,
    blend_frame_pair,
    build_embryo_track,
    build_snip_path,
    export_embryo_video,
    render_embryo_sequence,
    resolve_playback_frame_count,
    resolve_frame_schedule,
)


def test_build_embryo_track_deduplicates_identical_times():
    df = pd.DataFrame(
        {
            "embryo_id": ["emb1", "emb1", "emb1"],
            "experiment_date": ["20260101", "20260101", "20260101"],
            "predicted_stage_hpf": [24.0, 24.0, 26.0],
            "frame_index": [10, 11, 20],
            "well": ["A01", "A01", "A01"],
        }
    )

    track = build_embryo_track(df, well_col="well")

    assert track.embryo_id == "emb1"
    assert track.experiment_date == "20260101"
    assert track.well == "A01"
    assert track.times_hpf.tolist() == [24.0, 26.0]
    assert track.frame_indices.tolist() == [10, 20]


def test_resolve_frame_schedule_interpolates_between_timepoints():
    track = EmbryoTrack(
        embryo_id="emb1",
        experiment_date="20260101",
        well="A01",
        times_hpf=np.array([24.0, 26.0, 28.0], dtype=float),
        frame_indices=np.array([100, 200, 300], dtype=int),
    )
    t_out = np.array([23.0, 25.0, 27.0, 29.0], dtype=float)

    fi0, fi1, alpha = resolve_frame_schedule(track, t_out)

    assert fi0.tolist() == [100, 100, 200, 300]
    assert fi1.tolist() == [100, 200, 300, 300]
    assert alpha.tolist() == [0.0, 0.5, 0.5, 0.0]


def test_blend_frame_pair_handles_fallbacks_and_crossfade():
    img0 = np.zeros((2, 2, 3), dtype=np.uint8)
    img1 = np.full((2, 2, 3), 200, dtype=np.uint8)

    assert np.array_equal(blend_frame_pair(img0, None, 0.5), img0)
    assert np.array_equal(blend_frame_pair(None, img1, 0.5), img1)

    blended = blend_frame_pair(img0, img1, 0.25)
    assert blended is not None
    assert blended.shape == (2, 2, 3)
    assert int(blended[0, 0, 0]) == 50


def test_build_snip_path_uses_canonical_naming():
    path = build_snip_path(Path("/tmp/snips"), "20260101", "emb1", 42)
    assert str(path) == "/tmp/snips/20260101/emb1_t0042.jpg"


def test_resolve_playback_frame_count_accepts_exactly_one_mode():
    assert resolve_playback_frame_count(fps=20, video_duration_s=2.0) == 40
    assert resolve_playback_frame_count(fps=20, n_frames_out=55) == 55

    with pytest.raises(ValueError, match="exactly one"):
        resolve_playback_frame_count(fps=20, video_duration_s=2.0, n_frames_out=55)

    with pytest.raises(ValueError, match="exactly one"):
        resolve_playback_frame_count(fps=20)


def test_render_embryo_sequence_uses_duration_mode(tmp_path: Path):
    import cv2

    snip_root = tmp_path / "snips"
    exp_dir = snip_root / "20260101"
    exp_dir.mkdir(parents=True)
    img0 = np.zeros((3, 5, 3), dtype=np.uint8)
    img1 = np.full((3, 5, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(exp_dir / "emb1_t0010.jpg"), img0)
    cv2.imwrite(str(exp_dir / "emb1_t0020.jpg"), img1)

    track = EmbryoTrack(
        embryo_id="emb1",
        experiment_date="20260101",
        times_hpf=np.array([10.0, 12.0], dtype=float),
        frame_indices=np.array([10, 20], dtype=int),
    )

    sequence = render_embryo_sequence(
        track,
        snip_root=snip_root,
        start_hpf=10.0,
        end_hpf=12.0,
        fps=10,
        video_duration_s=1.0,
    )

    assert len(sequence.frames) == 10
    assert sequence.fps == 10
    assert sequence.times_hpf[0] == pytest.approx(10.0)
    assert sequence.times_hpf[-1] == pytest.approx(12.0)
    assert sequence.frames[0].shape == (4, 6, 3)


def test_export_embryo_video_writes_mp4(tmp_path: Path):
    import cv2

    snip_root = tmp_path / "snips"
    exp_dir = snip_root / "20260101"
    exp_dir.mkdir(parents=True)
    img0 = np.zeros((4, 6, 3), dtype=np.uint8)
    img1 = np.full((4, 6, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(exp_dir / "emb1_t0010.jpg"), img0)
    cv2.imwrite(str(exp_dir / "emb1_t0020.jpg"), img1)

    track = EmbryoTrack(
        embryo_id="emb1",
        experiment_date="20260101",
        times_hpf=np.array([10.0, 12.0], dtype=float),
        frame_indices=np.array([10, 20], dtype=int),
    )
    sequence = render_embryo_sequence(
        track,
        snip_root=snip_root,
        start_hpf=10.0,
        end_hpf=12.0,
        fps=8,
        n_frames_out=8,
    )

    out_mp4 = tmp_path / "emb1.mp4"
    width, height = export_embryo_video(sequence, out_mp4)

    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 0
    assert (width, height) == (6, 4)
