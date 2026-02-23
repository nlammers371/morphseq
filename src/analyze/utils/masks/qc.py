"""Lightweight mask QC helpers.

These helpers are intentionally conservative: they should not introduce
non-obvious geometry or alignment steps. They exist to normalize masks for
downstream geometry/OT code.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None


def qc_mask(mask: np.ndarray, *, fill_holes: bool = True) -> Tuple[np.ndarray, dict]:
    """Return a QC'd binary mask and a small report dict."""
    m0 = np.asarray(mask) > 0
    report = {"fill_holes": bool(fill_holes)}
    if fill_holes and ndimage is not None:
        m1 = ndimage.binary_fill_holes(m0)
    else:
        m1 = m0
    report["area_before_px"] = int(m0.sum())
    report["area_after_px"] = int(m1.sum())
    return m1.astype(np.uint8), report

