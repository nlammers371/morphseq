import math
import numpy as np

from src.build.qc_utils import (
    compute_fraction_alive,
    compute_qc_flags,
    compute_speed,
)


def test_compute_fraction_alive_basic():
    emb = np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    via = np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    # emb has 3 pixels; via overlaps 1 pixel => alive = 2/3
    frac = compute_fraction_alive(emb, via)
    assert math.isclose(frac, 2.0 / 3.0, rel_tol=1e-6)


def test_compute_fraction_alive_nan_without_via():
    emb = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    frac = compute_fraction_alive(emb, None)
    assert math.isnan(frac)


def test_compute_qc_flags_overlap_and_frame():
    # Embryo centered, 5x5, no proximity masks
    emb = np.zeros((5, 5), dtype=np.uint8)
    emb[2, 1:4] = 1
    px_dim = 1.0  # 1 um/px

    # Yolk overlaps
    yolk = np.zeros_like(emb)
    yolk[2, 2] = 1

    # Use a small QC scale so the interior crop does not truncate the embryo
    flags = compute_qc_flags(emb, px_dim_um=px_dim, qc_scale_um=1, yolk_mask=yolk)
    assert flags["no_yolk_flag"] is False
    assert flags["frame_flag"] is False
    assert flags["focus_flag"] is False
    assert flags["bubble_flag"] is False


def test_compute_qc_flags_no_yolk_and_proximity():
    emb = np.zeros((9, 9), dtype=np.uint8)
    emb[4, 3:6] = 1  # center
    px_dim = 1.0

    # No yolk
    yolk = np.zeros_like(emb)

    # Focus near embryo (within 2*qc_scale_px, qc_scale_um=2 => thresh=4 px)
    focus = np.zeros_like(emb)
    focus[4, 0] = 1  # 3 px away horizontally from left edge of embryo segment

    # Bubble far away
    bubble = np.zeros_like(emb)
    bubble[0, 0] = 1

    flags = compute_qc_flags(
        emb,
        px_dim_um=px_dim,
        qc_scale_um=2,
        yolk_mask=yolk,
        focus_mask=focus,
        bubble_mask=bubble,
    )
    assert flags["no_yolk_flag"] is True
    assert flags["focus_flag"] is True
    assert flags["bubble_flag"] is False


def test_compute_qc_flags_frame_truncation():
    # Embryo touches left border; with qc_scale_um=2 and px_dim=1, inner crop removes embryo
    emb = np.zeros((7, 7), dtype=np.uint8)
    emb[3, 0:3] = 1  # touches left border
    flags = compute_qc_flags(emb, px_dim_um=1.0, qc_scale_um=2)
    assert flags["frame_flag"] is True


def test_compute_speed_basic():
    prev_xy = (10.0, 10.0)
    curr_xy = (13.0, 14.0)
    prev_t = 0.0
    curr_t = 2.0
    px_dim = 2.0  # um/px
    # distance in px = sqrt(3^2 + 4^2)=5 px => 10 um over 2 s => 5 um/s
    v = compute_speed(prev_xy, prev_t, curr_xy, curr_t, px_dim)
    assert math.isclose(v, 5.0, rel_tol=1e-6)


def test_compute_speed_nan_cases():
    assert math.isnan(compute_speed(None, 0.0, (0, 0), 1.0, 1.0))
    assert math.isnan(compute_speed((0, 0), None, (0, 0), 1.0, 1.0))
    assert math.isnan(compute_speed((0, 0), 1.0, (0, 0), 0.5, 1.0))
