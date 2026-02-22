"""
viz.contract — Single source of truth for the matplotlib coordinate contract.

Pixel-edge coordinate frame:
  x ∈ [-0.5, W-0.5], y ∈ [-0.5, H-0.5], row 0 at top.

Rules enforced here (not by comments):
  - imshow: always origin="upper", always sets axis limits
  - contour/contourf: never pass origin=; use explicit x/y meshgrids
  - MORPHSEQ_VIZ_STRICT=1 turns silent stripping into ValueError

DO NOT call ax.imshow / ax.contour / plt.imshow / plt.contour directly in
viz/phase0.py or viz/qc.py — use these wrappers instead.
"""

from __future__ import annotations

import os
import numpy as np

_ORIGIN = "upper"
STRICT = bool(int(os.getenv("MORPHSEQ_VIZ_STRICT", "0")))


def _check_no_origin(kwargs: dict, fn_name: str) -> None:
    if "origin" in kwargs:
        bad_val = kwargs.pop("origin")
        if STRICT:
            raise ValueError(
                f"viz.contract.{fn_name}: do not pass origin={bad_val!r}; "
                "the contract sets it. Set MORPHSEQ_VIZ_STRICT=0 to silence."
            )


def imshow(ax, img, *, extent=None, **kwargs):
    """
    Project-standard imshow.

    Enforces origin="upper" and pixel-edge axis limits so that
    contour() pixel-center grids align exactly.

    extent : (xmin, xmax, ymin, ymax) in data coords, optional.
    """
    _check_no_origin(kwargs, "imshow")
    H, W = img.shape[:2]
    result = ax.imshow(img, origin=_ORIGIN, extent=extent, **kwargs)
    if extent is None:
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
    else:
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)   # reversed: row 0 at top
    return result


def _make_grids(z, extent):
    H, W = z.shape[:2]
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        xs = np.linspace(xmin, xmax, W)
        ys = np.linspace(ymin, ymax, H)
    else:
        xs = np.arange(W)
        ys = np.arange(H)
    return np.meshgrid(xs, ys, indexing="xy")


def contour(ax, z, *, extent=None, **kwargs):
    """
    Project-standard contour.

    Uses explicit pixel-center x/y grids matched to the imshow pixel-edge
    axes, so the 0.5 level traces mask boundaries at half-integer coords.
    Never pass origin= — it would double-flip the data in mpl 3.10.
    """
    _check_no_origin(kwargs, "contour")
    X_grid, Y_grid = _make_grids(z, extent)
    return ax.contour(X_grid, Y_grid, z, **kwargs)


def contourf(ax, z, *, extent=None, **kwargs):
    """Project-standard contourf. Same contract as contour()."""
    _check_no_origin(kwargs, "contourf")
    X_grid, Y_grid = _make_grids(z, extent)
    return ax.contourf(X_grid, Y_grid, z, **kwargs)


def embryo_outline(mask_ref: np.ndarray, ax, *, color: str = "white", lw: float = 1.0):
    """Draw embryo boundary using the contract contour wrapper."""
    contour(ax, mask_ref.astype(float), levels=[0.5],
            colors=[color], linewidths=lw, linestyles="-")


__all__ = ["imshow", "contour", "contourf", "embryo_outline", "STRICT"]
