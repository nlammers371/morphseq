"""Back-direction helper for canonical embryo alignment.

This module isolates the yolk-centered back-point calculation used by
`CanonicalAligner`. It keeps the geometry logic separate from the broader
canonical-grid orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import warnings


CenterOfMassFn = Callable[[np.ndarray], tuple[float, float]]
ProjectPointFn = Callable[
    [np.ndarray, tuple[float, float], tuple[float, float], float],
    tuple[float, float],
]


def compute_back_direction(
    mask: np.ndarray,
    yolk_mask: Optional[np.ndarray] = None,
    *,
    center_of_mass: CenterOfMassFn,
    project_point_to_mask_in_disk: ProjectPointFn,
    back_sample_radius_k: float,
) -> tuple[tuple[float, float], dict]:
    """Compute the back point for an embryo mask.

    The back point is derived from the centroid of embryo pixels inside a disk
    centered on the yolk COM. The disk radius is scaled from the yolk area.

    Returns
    -------
    (back_yx, debug)
        back_yx is a ``(y, x)`` tuple. debug contains the derived landmarks and
        fallback path information.
    """
    back_debug: dict = {}

    if yolk_mask is None or np.sum(yolk_mask) == 0:
        warnings.warn(
            "No yolk mask available for back-direction computation. Returning mask COM as fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        fallback = center_of_mass(mask)
        back_debug["selected"] = "no_yolk_fallback"
        back_debug["back_yx"] = (float(fallback[0]), float(fallback[1]))
        return fallback, back_debug

    yolk_com_y, yolk_com_x = center_of_mass(yolk_mask)
    back_debug["yolk_com_yx"] = (float(yolk_com_y), float(yolk_com_x))

    yolk_area = float(yolk_mask.sum())
    r_yolk = np.sqrt(yolk_area / np.pi) if yolk_area > 0 else 0.0
    r_sample = float(back_sample_radius_k) * float(r_yolk)
    back_debug["r_yolk_px"] = float(r_yolk)
    back_debug["r_sample_px"] = float(r_sample)

    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        fallback = (yolk_com_y, yolk_com_x)
        back_debug["selected"] = "empty_mask"
        back_debug["back_yx"] = (float(fallback[0]), float(fallback[1]))
        return fallback, back_debug

    dy = ys.astype(np.float64) - float(yolk_com_y)
    dx = xs.astype(np.float64) - float(yolk_com_x)
    in_disk = (dy**2 + dx**2) <= (r_sample**2)
    n_pixels_in_disk = int(in_disk.sum())
    back_debug["n_pixels_in_disk"] = n_pixels_in_disk

    if n_pixels_in_disk == 0:
        warnings.warn(
            f"No embryo-mask pixels within sampling disk (r_sample={r_sample:.1f}px). Using yolk COM.",
            RuntimeWarning,
            stacklevel=2,
        )
        back_debug["selected"] = "empty_disk"
        back_debug["back_yx"] = (float(yolk_com_y), float(yolk_com_x))
        return (yolk_com_y, yolk_com_x), back_debug

    if n_pixels_in_disk < 50:
        warnings.warn(
            f"Only {n_pixels_in_disk} embryo-mask pixels in sampling disk; result may be noisy.",
            RuntimeWarning,
            stacklevel=2,
        )

    back_centroid_y = float(ys[in_disk].mean())
    back_centroid_x = float(xs[in_disk].mean())
    back_debug["raw_back_centroid_yx"] = (back_centroid_y, back_centroid_x)

    back_y, back_x = project_point_to_mask_in_disk(
        mask,
        (back_centroid_y, back_centroid_x),
        disk_center_yx=(yolk_com_y, yolk_com_x),
        disk_radius=r_sample,
    )
    back_debug["selected"] = "yolk_surrounding_centroid"
    back_debug["back_yx"] = (float(back_y), float(back_x))
    return (back_y, back_x), back_debug
