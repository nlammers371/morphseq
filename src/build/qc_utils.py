from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_fraction_alive(emb_mask: np.ndarray, via_mask: Optional[np.ndarray]) -> float:
    """Compute fraction_alive given embryo mask and viability mask.

    Args:
        emb_mask: Binary numpy array where 1 indicates embryo pixels.
        via_mask: Binary numpy array where 1 indicates non-viable/dead pixels within embryo;
                  or None if not available.

    Returns:
        Fraction of embryo pixels that are alive (in [0, 1]). If via_mask is None or
        embryo has zero pixels, returns np.nan.
    """
    emb_mask = (emb_mask > 0).astype(np.uint8)
    if via_mask is None:
        return np.nan
    via_mask = (via_mask > 0).astype(np.uint8)

    total = int(emb_mask.sum())
    if total == 0:
        return np.nan

    dead = int((emb_mask & via_mask).sum())
    alive = total - dead
    return max(0.0, min(1.0, alive / total))


def compute_qc_flags(
    emb_mask: np.ndarray,
    px_dim_um: float,
    qc_scale_um: int = 150,
    yolk_mask: Optional[np.ndarray] = None,
    focus_mask: Optional[np.ndarray] = None,
    bubble_mask: Optional[np.ndarray] = None,
) -> Dict[str, bool]:
    """Compute QC flags using embryo mask and optional auxiliary masks.

    Flags:
        - frame_flag: True if embryo touches/near frame boundary beyond threshold.
        - no_yolk_flag: True if no overlap between yolk and embryo.
        - focus_flag: True if focus_mask pixels appear within proximity threshold.
        - bubble_flag: True if bubble_mask pixels appear within proximity threshold.

    Args:
        emb_mask: Binary embryo mask (1 indicates embryo).
        px_dim_um: Microns per pixel.
        qc_scale_um: Scale in microns for frame/proximity thresholds (default 150 µm).
        yolk_mask, focus_mask, bubble_mask: Optional binary masks.

    Returns:
        Dict of flags {frame_flag, no_yolk_flag, focus_flag, bubble_flag}
    """
    emb = (emb_mask > 0).astype(np.uint8)
    H, W = emb.shape[:2]
    qc_scale_px = int(math.ceil(qc_scale_um / max(px_dim_um, 1e-9)))

    # frame_flag: if removing a qc-scale border changes embryo area > 2%
    if H > 2 * qc_scale_px and W > 2 * qc_scale_px and emb.sum() > 0:
        inner = emb[qc_scale_px : H - qc_scale_px, qc_scale_px : W - qc_scale_px]
        frame_flag = inner.sum() <= 0.98 * emb.sum()
    else:
        # Image too small or empty embryo; treat as framed safely
        frame_flag = False

    # yolk presence
    if yolk_mask is None:
        no_yolk_flag = True  # unknown -> treat as missing
    else:
        yolk = (yolk_mask > 0).astype(np.uint8)
        no_yolk_flag = ((emb & yolk).sum() == 0)

    # proximity-based flags using distance transform to embryo boundary complement
    focus_flag = False
    bubble_flag = False
    if focus_mask is not None or bubble_mask is not None:
        # Distance from embryo exterior (0 where embryo exists)
        dist = distance_transform_edt(emb == 0)
        thresh = 2 * qc_scale_px
        if focus_mask is not None and (focus_mask > 0).any():
            m = (focus_mask > 0)
            # Guard: if mask has no valid pixels due to size, keep False
            if dist[m].size > 0:
                focus_flag = float(dist[m].min()) <= thresh
        if bubble_mask is not None and (bubble_mask > 0).any():
            m = (bubble_mask > 0)
            if dist[m].size > 0:
                bubble_flag = float(dist[m].min()) <= thresh

    return {
        "frame_flag": bool(frame_flag),
        "no_yolk_flag": bool(no_yolk_flag),
        "focus_flag": bool(focus_flag),
        "bubble_flag": bool(bubble_flag),
    }


def compute_speed(
    prev_xy: Optional[Tuple[float, float]],
    prev_t_s: Optional[float],
    curr_xy: Optional[Tuple[float, float]],
    curr_t_s: Optional[float],
    px_dim_um: float,
) -> float:
    """Compute speed (µm/s) given previous and current positions and times.

    Returns np.nan if any input is missing/invalid or dt <= 0.
    """
    if (
        prev_xy is None
        or curr_xy is None
        or prev_t_s is None
        or curr_t_s is None
        or not np.isfinite(prev_t_s)
        or not np.isfinite(curr_t_s)
    ):
        return float("nan")

    dt = float(curr_t_s) - float(prev_t_s)
    if dt <= 0:
        return float("nan")

    dx_px = float(curr_xy[0]) - float(prev_xy[0])
    dy_px = float(curr_xy[1]) - float(prev_xy[1])
    dr_um = math.hypot(dx_px, dy_px) * float(px_dim_um)
    return dr_um / dt

