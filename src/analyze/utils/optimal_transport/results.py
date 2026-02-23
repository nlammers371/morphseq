"""No-lies UOT result types.

Rule: a result object must not mix coordinate frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Coupling, PairFrameGeometry


@dataclass(frozen=True)
class UOTResultWork:
    """Work-grid-only OT outputs."""

    cost: float
    coupling: Optional[Coupling]

    mass_created_work: np.ndarray  # (Hw, Ww)
    mass_destroyed_work: np.ndarray  # (Hw, Ww)
    velocity_work_px_per_step_yx: np.ndarray  # (Hw, Ww, 2) (dy, dx) in work px/step

    support_src_yx: np.ndarray
    support_tgt_yx: np.ndarray
    weights_src: np.ndarray
    weights_tgt: np.ndarray

    cost_src_support: Optional[np.ndarray] = None
    cost_tgt_support: Optional[np.ndarray] = None
    cost_src_work: Optional[np.ndarray] = None
    cost_tgt_work: Optional[np.ndarray] = None

    diagnostics: Optional[dict] = None

    work_shape_hw: tuple[int, int] = (-1, -1)
    work_um_per_px: float = float("nan")

    pair_frame: Optional[PairFrameGeometry] = None
    meta: Optional[dict] = None


@dataclass(frozen=True)
class UOTResultCanonical:
    """Canonical-grid-only OT outputs (lifted from a work-grid solve)."""

    cost: float

    mass_created_canon: np.ndarray  # (Hc, Wc)
    mass_destroyed_canon: np.ndarray  # (Hc, Wc)
    velocity_canon_px_per_step_yx: np.ndarray  # (Hc, Wc, 2) (dy, dx) in canonical px/step

    cost_src_canon: Optional[np.ndarray] = None
    cost_tgt_canon: Optional[np.ndarray] = None

    diagnostics: Optional[dict] = None

    canonical_shape_hw: tuple[int, int] = (-1, -1)
    canonical_um_per_px: float = float("nan")

    pair_frame: Optional[PairFrameGeometry] = None
    meta: Optional[dict] = None

