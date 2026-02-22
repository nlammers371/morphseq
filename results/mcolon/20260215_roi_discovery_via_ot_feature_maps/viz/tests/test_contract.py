"""
Tests for viz.contract — structural, geometry, and strict-mode.

Run from the ROI directory:
    pytest viz/tests/test_contract.py -v
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the ROI dir is on the path so we can import viz
_ROI_DIR = Path(__file__).resolve().parents[2]
if str(_ROI_DIR) not in sys.path:
    sys.path.insert(0, str(_ROI_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Structural test — no direct matplotlib calls in viz/phase0.py or viz/qc.py
# ---------------------------------------------------------------------------

BANNED = [
    "ax.contour(",
    "ax.contourf(",
    "ax.imshow(",
    "plt.contour(",
    "plt.contourf(",
    "plt.imshow(",
]


def test_no_direct_ax_calls():
    """viz/phase0.py and viz/qc.py must not call ax.imshow/contour directly."""
    import viz.phase0
    import viz.qc

    for mod in [viz.phase0, viz.qc]:
        src = inspect.getsource(mod)
        for bad in BANNED:
            assert bad not in src, (
                f"Direct {bad!r} found in {mod.__name__} — use viz.contract instead"
            )


# ---------------------------------------------------------------------------
# Geometry test — contour paths align with imshow pixel coordinates
# ---------------------------------------------------------------------------

def _make_l_mask(H=100, W=100):
    """
    Asymmetric L-shaped mask:
      filled cols [5, 95), rows [10, 40) — horizontal bar
      filled cols [5, 30), rows [5, 20) — vertical bump

    Pixel-center boundary expectations:
      x-left ≈ 4.5, x-right ≈ 94.5
      y-top ≈ 4.5,  y-bottom ≈ 39.5
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[10:40, 5:95] = 1   # horizontal bar
    mask[5:20, 5:30] = 1    # vertical bump
    return mask


def test_contour_aligns_with_imshow():
    """
    Contour paths must lie within ~1 px of the mask pixel-center boundaries.
    This catches origin= double-flip bugs.
    """
    from viz import contract

    mask = _make_l_mask()
    fig, ax = plt.subplots()

    contract.imshow(ax, mask.astype(float), cmap="gray")
    cs = contract.contour(ax, mask.astype(float), levels=[0.5])

    # cs.get_paths() is available in mpl 3.8+; fall back to cs.collections for older
    try:
        paths = cs.get_paths()
    except AttributeError:
        paths = [p for coll in cs.collections for p in coll.get_paths()]
    all_verts = np.concatenate([p.vertices for p in paths if len(p.vertices)])
    assert len(all_verts) > 0, "No contour paths found"

    # x-axis: leftmost boundary should be near x=4.5, rightmost near x=94.5
    assert all_verts[:, 0].min() >= 4.0, (
        f"x-left contour vertex {all_verts[:, 0].min():.2f} is too far left of expected ~4.5"
    )
    assert all_verts[:, 0].max() <= 95.0, (
        f"x-right contour vertex {all_verts[:, 0].max():.2f} is too far right of expected ~94.5"
    )

    # y-axis: topmost boundary near y=4.5, bottommost near y=39.5
    assert all_verts[:, 1].min() >= 4.0, (
        f"y-top contour vertex {all_verts[:, 1].min():.2f} is too far above expected ~4.5"
    )
    assert all_verts[:, 1].max() <= 40.0, (
        f"y-bottom contour vertex {all_verts[:, 1].max():.2f} is too far below expected ~39.5"
    )

    plt.close(fig)


def test_imshow_sets_axis_limits():
    """contract.imshow must set pixel-edge axis limits."""
    from viz import contract

    H, W = 80, 120
    img = np.zeros((H, W))
    fig, ax = plt.subplots()
    contract.imshow(ax, img)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert abs(xlim[0] - (-0.5)) < 1e-6, f"xlim left expected -0.5, got {xlim[0]}"
    assert abs(xlim[1] - (W - 0.5)) < 1e-6, f"xlim right expected {W-0.5}, got {xlim[1]}"
    # y reversed: top limit is H-0.5, bottom limit is -0.5
    assert abs(ylim[0] - (H - 0.5)) < 1e-6, f"ylim[0] expected {H-0.5}, got {ylim[0]}"
    assert abs(ylim[1] - (-0.5)) < 1e-6, f"ylim[1] expected -0.5, got {ylim[1]}"

    plt.close(fig)


# ---------------------------------------------------------------------------
# Strict-mode test — MORPHSEQ_VIZ_STRICT=1 raises ValueError on origin=
# ---------------------------------------------------------------------------

def test_strict_mode_rejects_origin_on_contour():
    """With MORPHSEQ_VIZ_STRICT=1, passing origin= to contract.contour raises ValueError."""
    os.environ["MORPHSEQ_VIZ_STRICT"] = "1"
    import viz.contract
    importlib.reload(viz.contract)

    mask = np.zeros((10, 10))
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="origin="):
            viz.contract.contour(ax, mask, origin="upper")
    finally:
        os.environ["MORPHSEQ_VIZ_STRICT"] = "0"
        importlib.reload(viz.contract)
        plt.close(fig)


def test_strict_mode_rejects_origin_on_imshow():
    """With MORPHSEQ_VIZ_STRICT=1, passing origin= to contract.imshow raises ValueError."""
    os.environ["MORPHSEQ_VIZ_STRICT"] = "1"
    import viz.contract
    importlib.reload(viz.contract)

    img = np.zeros((10, 10))
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="origin="):
            viz.contract.imshow(ax, img, origin="lower")
    finally:
        os.environ["MORPHSEQ_VIZ_STRICT"] = "0"
        importlib.reload(viz.contract)
        plt.close(fig)


def test_silent_mode_strips_origin():
    """With MORPHSEQ_VIZ_STRICT=0 (default), origin= is silently stripped without error."""
    os.environ["MORPHSEQ_VIZ_STRICT"] = "0"
    import viz.contract
    importlib.reload(viz.contract)

    mask = np.zeros((10, 10))
    fig, ax = plt.subplots()
    # Should not raise
    viz.contract.contour(ax, mask, levels=[0.5], origin="upper")
    plt.close(fig)
