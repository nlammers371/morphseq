"""
grids.py
========
Per-stack grid builders.

Each function takes a single Z-stack of shape (Z, Y, X) and returns a
metric grid. Tiling logic lives here; pure metric kernels live in kernels.py.

Output shapes:
  NCC grid     : (Z-1, Ny, Nx)   — pairwise across adjacent Z slices
  entropy grid : (Z,   Ny, Nx)   — per slice
  laplacian grid: (Z,  Ny, Nx)   — per slice

Tile coordinates can be recovered from (tile_size, stride, stack shape).
"""

from __future__ import annotations
import numpy as np
from .kernels import ncc, shannon_entropy, laplacian_var


def _tile_coords(length: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    coords = []
    start = 0
    while start < length:
        end = min(start + tile_size, length)
        coords.append((start, end))
        if end == length:
            break
        start += stride
    return coords


def compute_local_ncc_grid(
    stack_zyx: np.ndarray,
    tile_size: int = 128,
    stride: int | None = None,
) -> np.ndarray:
    """
    NCC between adjacent Z-slice pairs, computed per tile.

    Args:
        stack_zyx : (Z, Y, X) float array
        tile_size : tile side length in pixels
        stride    : tile stride (defaults to tile_size → non-overlapping)

    Returns:
        ncc_grid  : (Z-1, Ny, Nx) float32
                    NaN where the tile had zero variance in either slice.
    """
    stride = stride or tile_size
    Z, Y, X = stack_zyx.shape
    y_coords = _tile_coords(Y, tile_size, stride)
    x_coords = _tile_coords(X, tile_size, stride)
    Ny, Nx = len(y_coords), len(x_coords)
    grid = np.full((Z - 1, Ny, Nx), np.nan, dtype=np.float32)

    for z in range(Z - 1):
        s0 = stack_zyx[z].astype(np.float64)
        s1 = stack_zyx[z + 1].astype(np.float64)
        for iy, (y0, y1) in enumerate(y_coords):
            for ix, (x0, x1) in enumerate(x_coords):
                grid[z, iy, ix] = ncc(s0[y0:y1, x0:x1], s1[y0:y1, x0:x1])

    return grid


def compute_local_entropy_grid(
    stack_zyx: np.ndarray,
    tile_size: int = 128,
    stride: int | None = None,
    n_bins: int = 64,
) -> np.ndarray:
    """
    Shannon entropy per tile per Z slice.

    Returns:
        entropy_grid : (Z, Ny, Nx) float32
    """
    stride = stride or tile_size
    Z, Y, X = stack_zyx.shape
    y_coords = _tile_coords(Y, tile_size, stride)
    x_coords = _tile_coords(X, tile_size, stride)
    Ny, Nx = len(y_coords), len(x_coords)
    grid = np.full((Z, Ny, Nx), np.nan, dtype=np.float32)

    for z in range(Z):
        sl = stack_zyx[z].astype(np.float64)
        for iy, (y0, y1) in enumerate(y_coords):
            for ix, (x0, x1) in enumerate(x_coords):
                grid[z, iy, ix] = shannon_entropy(sl[y0:y1, x0:x1], n_bins=n_bins)

    return grid


def compute_local_laplacian_grid(
    stack_zyx: np.ndarray,
    tile_size: int = 128,
    stride: int | None = None,
) -> np.ndarray:
    """
    Laplacian variance per tile per Z slice — focus sharpness measure.

    Returns:
        lap_grid : (Z, Ny, Nx) float32
    """
    stride = stride or tile_size
    Z, Y, X = stack_zyx.shape
    y_coords = _tile_coords(Y, tile_size, stride)
    x_coords = _tile_coords(X, tile_size, stride)
    Ny, Nx = len(y_coords), len(x_coords)
    grid = np.full((Z, Ny, Nx), np.nan, dtype=np.float32)

    for z in range(Z):
        sl = stack_zyx[z].astype(np.float64)
        for iy, (y0, y1) in enumerate(y_coords):
            for ix, (x0, x1) in enumerate(x_coords):
                grid[z, iy, ix] = laplacian_var(sl[y0:y1, x0:x1])

    return grid


def tile_origin_coords(
    stack_shape_yx: tuple[int, int],
    tile_size: int = 128,
    stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y_origins, x_origins) arrays giving the top-left corner of each
    tile, matching the grid layout produced by the compute_* functions.
    Useful for post-hoc mask intersection.
    """
    stride = stride or tile_size
    Y, X = stack_shape_yx
    y_coords = _tile_coords(Y, tile_size, stride)
    x_coords = _tile_coords(X, tile_size, stride)
    y_origins = np.array([y0 for y0, _ in y_coords], dtype=np.int32)
    x_origins = np.array([x0 for x0, _ in x_coords], dtype=np.int32)
    return y_origins, x_origins
