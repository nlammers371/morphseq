"""
io.py
=====
Save and load metric grids. One .npz file per source frame.

Naming convention (caller's responsibility):
    <date>_<well>_t<tttt>_motion_qc.npz

Contents of each .npz:
    ncc_grid       : (Z-1, Ny, Nx) float32
    entropy_grid   : (Z,   Ny, Nx) float32
    tile_size      : scalar int
    stride         : scalar int
    stack_shape_y  : scalar int   (original Y dimension)
    stack_shape_x  : scalar int   (original X dimension)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np


def save_grids(
    path: str | Path,
    ncc_grid: np.ndarray,
    entropy_grid: np.ndarray,
    tile_size: int,
    stride: int,
    stack_shape_yx: tuple[int, int],
) -> None:
    np.savez_compressed(
        str(path),
        ncc_grid=ncc_grid.astype(np.float32),
        entropy_grid=entropy_grid.astype(np.float32),
        tile_size=np.int32(tile_size),
        stride=np.int32(stride),
        stack_shape_y=np.int32(stack_shape_yx[0]),
        stack_shape_x=np.int32(stack_shape_yx[1]),
    )


def load_grids(path: str | Path) -> dict:
    """
    Returns a dict with keys:
        ncc_grid, entropy_grid, tile_size, stride, stack_shape_yx
    """
    data = np.load(str(path))
    return {
        "ncc_grid":      data["ncc_grid"],
        "entropy_grid":  data["entropy_grid"],
        "tile_size":     int(data["tile_size"]),
        "stride":        int(data["stride"]),
        "stack_shape_yx": (int(data["stack_shape_y"]), int(data["stack_shape_x"])),
    }
