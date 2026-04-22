"""
embryo_qc.py
============
Post-segmentation, mask-aware embryo-level QC.

Given a saved metric grid (loaded via io.load_grids) and a binary embryo
mask (H, W bool array), derive per-embryo scalar summaries and a QC flag.

This is the only layer that knows about embryo masks. Everything above it
is mask-agnostic.
"""

from __future__ import annotations
import numpy as np
from .grids import tile_origin_coords
from .summaries import ncc_stack_summary, rel_entropy_summary


def _mask_tile_weights(
    mask: np.ndarray,
    y_origins: np.ndarray,
    x_origins: np.ndarray,
    tile_size: int,
    min_coverage: float = 0.10,
) -> np.ndarray:
    """
    For each tile (iy, ix), compute the fraction of pixels inside `mask`.
    Returns a (Ny, Nx) float32 weight array; tiles below min_coverage → 0.
    """
    Ny, Nx = len(y_origins), len(x_origins)
    weights = np.zeros((Ny, Nx), dtype=np.float32)
    for iy, y0 in enumerate(y_origins):
        y1 = min(y0 + tile_size, mask.shape[0])
        for ix, x0 in enumerate(x_origins):
            x1 = min(x0 + tile_size, mask.shape[1])
            frac = mask[y0:y1, x0:x1].mean()
            if frac >= min_coverage:
                weights[iy, ix] = frac
    return weights


def embryo_ncc_summary(
    ncc_grid: np.ndarray,
    mask: np.ndarray,
    tile_size: int,
    stride: int,
    bad_thresh: float = 0.90,
    min_tile_coverage: float = 0.10,
) -> dict:
    """
    NCC summary for a single embryo, using only tiles that overlap the mask.

    Args:
        ncc_grid  : (Z-1, Ny, Nx) from load_grids
        mask      : (H, W) bool array at original image resolution
        tile_size, stride : must match the values used when the grid was built
        bad_thresh : NCC below this = bad pair

    Returns dict with: ncc_mean, ncc_min, bad_pair_frac, local_ncc_std_mean, n_tiles
    """
    y_origins, x_origins = tile_origin_coords(mask.shape, tile_size, stride)
    weights = _mask_tile_weights(mask, y_origins, x_origins, tile_size, min_tile_coverage)
    valid = weights > 0

    if not valid.any():
        return {"ncc_mean": np.nan, "ncc_min": np.nan,
                "bad_pair_frac": np.nan, "local_ncc_std_mean": np.nan, "n_tiles": 0}

    masked_grid = ncc_grid[:, valid]  # (Z-1, n_valid_tiles)
    pair_means  = np.nanmean(masked_grid, axis=1)
    pair_stds   = np.nanstd(masked_grid, axis=1)

    return {
        "ncc_mean":           float(np.nanmean(masked_grid)),
        "ncc_min":            float(np.nanmin(masked_grid)),
        "bad_pair_frac":      float(np.mean(pair_means < bad_thresh)),
        "local_ncc_std_mean": float(np.nanmean(pair_stds)),
        "n_tiles":            int(valid.sum()),
    }


def embryo_entropy_summary(
    entropy_grid: np.ndarray,
    bg_entropy_grid: np.ndarray | None,
    mask: np.ndarray,
    tile_size: int,
    stride: int,
    min_tile_coverage: float = 0.10,
) -> dict:
    """
    Entropy (and relative entropy vs background) for a single embryo.

    bg_entropy_grid : if provided, computes rel_entropy = embryo − background.
                      Must have same shape as entropy_grid.
    """
    y_origins, x_origins = tile_origin_coords(mask.shape, tile_size, stride)
    weights = _mask_tile_weights(mask, y_origins, x_origins, tile_size, min_tile_coverage)
    valid = weights > 0

    if not valid.any():
        return {"entropy_mean": np.nan, "entropy_min": np.nan,
                "rel_entropy_mean": np.nan, "rel_entropy_min": np.nan}

    emb_vals = entropy_grid[:, valid]
    result = {
        "entropy_mean": float(np.nanmean(emb_vals)),
        "entropy_min":  float(np.nanmin(emb_vals)),
    }

    if bg_entropy_grid is not None:
        bg_vals = bg_entropy_grid[:, valid]
        diff = emb_vals - bg_vals
        result["rel_entropy_mean"] = float(np.nanmean(diff))
        result["rel_entropy_min"]  = float(np.nanmin(diff))
    else:
        result["rel_entropy_mean"] = np.nan
        result["rel_entropy_min"]  = np.nan

    return result


def embryo_qc_flag(
    ncc_summary: dict,
    entropy_summary: dict,
    ncc_min_thresh: float = 0.85,
    bad_pair_frac_thresh: float = 0.10,
    rel_entropy_mean_thresh: float | None = None,
) -> str:
    """
    Combine embryo summaries into a single QC flag: "PASS" | "FAIL" | "WARN".

    FAIL : ncc_min < ncc_min_thresh  OR  bad_pair_frac > bad_pair_frac_thresh
    WARN : rel_entropy_mean below threshold (within-slice blur, NCC-invisible)
    PASS : everything else
    """
    ncc_min      = ncc_summary.get("ncc_min", np.nan)
    bad_pair_frac = ncc_summary.get("bad_pair_frac", np.nan)
    rel_ent_mean  = entropy_summary.get("rel_entropy_mean", np.nan)

    if (not np.isnan(ncc_min) and ncc_min < ncc_min_thresh) or \
       (not np.isnan(bad_pair_frac) and bad_pair_frac > bad_pair_frac_thresh):
        return "FAIL"

    if rel_entropy_mean_thresh is not None and \
       not np.isnan(rel_ent_mean) and rel_ent_mean < rel_entropy_mean_thresh:
        return "WARN"

    return "PASS"
