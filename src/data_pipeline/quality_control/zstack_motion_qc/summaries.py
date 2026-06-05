"""
summaries.py
============
Reduce metric grids to scalar stack summaries — no masks, no embryos.

These operate on the raw (Z, Ny, Nx) or (Z-1, Ny, Nx) grids and return
scalars describing the whole stack. Embryo-level summaries (mask-aware)
live in embryo_qc.py.
"""

from __future__ import annotations
import numpy as np


def ncc_stack_summary(ncc_grid: np.ndarray, bad_thresh: float = 0.90) -> dict:
    """
    Scalar summaries derived from an NCC grid of shape (Z-1, Ny, Nx).

    Returns:
        ncc_mean          — mean NCC across all pairs and tiles
        ncc_min           — worst single tile across all pairs
        bad_pair_frac     — fraction of Z-pairs where mean tile NCC < bad_thresh
        local_ncc_std_mean— mean per-pair spatial std (non-uniform motion flag)
    """
    flat_per_pair = ncc_grid.reshape(ncc_grid.shape[0], -1)  # (Z-1, Ny*Nx)
    pair_means = np.nanmean(flat_per_pair, axis=1)
    pair_stds  = np.nanstd(flat_per_pair, axis=1)
    return {
        "ncc_mean":           float(np.nanmean(ncc_grid)),
        "ncc_min":            float(np.nanmin(ncc_grid)),
        "bad_pair_frac":      float(np.mean(pair_means < bad_thresh)),
        "local_ncc_std_mean": float(np.nanmean(pair_stds)),
    }


def entropy_stack_summary(entropy_grid: np.ndarray) -> dict:
    """
    Scalar summaries derived from an entropy grid of shape (Z, Ny, Nx).

    Returns:
        entropy_mean — mean entropy across all slices and tiles
        entropy_min  — minimum (dullest tile/slice)
        entropy_std  — std across slices (focus curve spread)
    """
    return {
        "entropy_mean": float(np.nanmean(entropy_grid)),
        "entropy_min":  float(np.nanmin(entropy_grid)),
        "entropy_std":  float(np.nanstd(entropy_grid)),
    }


def rel_entropy_summary(
    entropy_grid: np.ndarray,
    bg_entropy_grid: np.ndarray,
) -> dict:
    """
    Relative entropy: embryo (or ROI) entropy minus background entropy,
    averaged across Z slices.

    Both grids must have shape (Z, Ny, Nx) and matching tile layouts.

    Returns:
        rel_entropy_mean — mean(embryo_entropy - bg_entropy) over Z
        rel_entropy_min  — worst slice
    """
    diff = entropy_grid - bg_entropy_grid
    return {
        "rel_entropy_mean": float(np.nanmean(diff)),
        "rel_entropy_min":  float(np.nanmin(diff)),
    }
