"""
kernels.py
==========
Pure scalar / patch-level metric functions.

All functions operate on 1-D or 2-D numpy arrays.
No knowledge of stacks, tiles, embryos, files, or experiments.
"""

from __future__ import annotations
import numpy as np


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized cross-correlation between two equal-length pixel arrays.

        NCC = dot(a - mean(a), b - mean(b)) / (||a - mean(a)|| * ||b - mean(b)||)

    Returns NaN if either array has zero variance.
    """
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def shannon_entropy(a: np.ndarray, n_bins: int = 64) -> float:
    """
    Shannon entropy (bits) of pixel intensities in array `a`.

        H = -sum(p * log2(p))   where p is the normalized histogram.
    """
    a = a.ravel().astype(np.float64)
    counts, _ = np.histogram(a, bins=n_bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def laplacian_var(a: np.ndarray) -> float:
    """
    Variance of the Laplacian of a 2-D patch — classical focus measure.
    Higher = sharper.
    """
    import cv2
    lap = cv2.Laplacian(a.astype(np.float64), cv2.CV_64F)
    return float(lap.var())


def phase_corr_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate sub-pixel lateral shift between 2-D images `a` and `b`
    using phase correlation.

    Returns (dy, dx, peak_value).
    Peak near 1.0 = clean translation; low peak = noisy / non-rigid motion.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    denom = np.abs(cross) + 1e-9
    pcorr = np.fft.ifft2(cross / denom).real
    idx = np.unravel_index(np.argmax(pcorr), pcorr.shape)
    peak = float(pcorr[idx])
    H, W = a.shape
    dy = float(idx[0]) if idx[0] <= H // 2 else float(idx[0] - H)
    dx = float(idx[1]) if idx[1] <= W // 2 else float(idx[1] - W)
    return dy, dx, peak
