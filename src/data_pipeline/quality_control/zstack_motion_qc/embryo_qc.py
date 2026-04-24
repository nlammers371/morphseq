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


def _longest_bad_run(pair_means: np.ndarray, bad_thresh: float) -> int:
    """
    Longest consecutive run of Z-pairs where the pair-level mean masked-tile
    NCC is below bad_thresh. NaN pair-means are treated as bad.

    Returns 0 when no pair is bad, or when pair_means is empty.
    """
    if pair_means.size == 0:
        return 0
    is_bad = np.isnan(pair_means) | (pair_means < bad_thresh)
    longest = current = 0
    for bad in is_bad:
        if bad:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


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
        ncc_grid          : (Z-1, Ny, Nx) from load_grids
        mask              : (H, W) bool array at original image resolution
        tile_size, stride : must match the values used when the grid was built
        bad_thresh        : NCC value below which a tile/pair is considered bad

    Semantics of returned fields:
        ncc_mean           — mean NCC over all masked valid tiles across all Z-pairs
        ncc_min            — minimum NCC over all masked valid tiles across all Z-pairs
        ncc_p05            — 5th-percentile NCC over masked valid tiles across all Z-pairs
        ncc_median         — median NCC over masked valid tiles across all Z-pairs
        bad_pair_frac      — fraction of Z-pairs where the pair-level mean masked-tile
                             NCC is below bad_thresh
        ncc_bad_tile_frac  — fraction of individual masked valid NCC tiles (across all
                             Z-pairs) whose value is below bad_thresh
        local_ncc_std_mean — mean per-Z-pair spatial std of masked tile NCCs;
                             high value flags spatially non-uniform (partial) motion
        longest_bad_run    — longest consecutive Z-pair run where pair-level mean
                             masked-tile NCC is below bad_thresh; NaN pair-means
                             count as bad
        n_tiles            — number of spatially valid (mask-overlapping) tile positions

    All scalar metrics are NaN when no valid tiles overlap the mask (n_tiles == 0).
    """
    y_origins, x_origins = tile_origin_coords(mask.shape, tile_size, stride)
    weights = _mask_tile_weights(mask, y_origins, x_origins, tile_size, min_tile_coverage)
    valid = weights > 0

    _nan = float("nan")
    if not valid.any():
        return {
            "ncc_mean":           _nan,
            "ncc_min":            _nan,
            "ncc_p05":            _nan,
            "ncc_median":         _nan,
            "bad_pair_frac":      _nan,
            "ncc_bad_tile_frac":  _nan,
            "local_ncc_std_mean": _nan,
            "longest_bad_run":    0,
            "n_tiles":            0,
        }

    masked_grid = ncc_grid[:, valid]          # (Z-1, n_valid_tiles)
    pair_means  = np.nanmean(masked_grid, axis=1)   # (Z-1,)
    pair_stds   = np.nanstd(masked_grid,  axis=1)   # (Z-1,)

    # flat view of all masked NCC values across every Z-pair and tile
    flat = masked_grid.ravel()
    flat_valid = flat[~np.isnan(flat)]

    if flat_valid.size == 0:
        # grid exists but every value is NaN (e.g. zero-variance tiles)
        return {
            "ncc_mean":           _nan,
            "ncc_min":            _nan,
            "ncc_p05":            _nan,
            "ncc_median":         _nan,
            "bad_pair_frac":      float(np.mean(np.isnan(pair_means) | (pair_means < bad_thresh))),
            "ncc_bad_tile_frac":  _nan,
            "local_ncc_std_mean": _nan,
            "longest_bad_run":    _longest_bad_run(pair_means, bad_thresh),
            "n_tiles":            int(valid.sum()),
        }

    return {
        "ncc_mean":           float(np.nanmean(masked_grid)),
        "ncc_min":            float(np.nanmin(masked_grid)),
        "ncc_p05":            float(np.percentile(flat_valid, 5)),
        "ncc_median":         float(np.median(flat_valid)),
        "bad_pair_frac":      float(np.mean(pair_means < bad_thresh)),
        "ncc_bad_tile_frac":  float(np.mean(flat_valid < bad_thresh)),
        "local_ncc_std_mean": float(np.nanmean(pair_stds)),
        "longest_bad_run":    _longest_bad_run(pair_means, bad_thresh),
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
