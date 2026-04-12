"""
kernels.py
----------
Spatial kernel functions for coherence computation.
"""
from __future__ import annotations

import numpy as np


def gaussian_kernel(
    positions: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Pairwise Gaussian spatial kernel at a single time slice.

    Parameters
    ----------
    positions : (N_e, 2)
    sigma : float

    Returns
    -------
    K : (N_e, N_e) — K[i,j] = exp(-||x_i - x_j||^2 / (2 sigma^2))
    """
    diff = positions[:, None, :] - positions[None, :, :]  # (N_e, N_e, 2)
    sq_dist = (diff ** 2).sum(axis=-1)                    # (N_e, N_e)
    return np.exp(-sq_dist / (2.0 * sigma ** 2))
