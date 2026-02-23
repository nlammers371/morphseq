"""Public datatypes for `analyze.utils.coord`.

This module is public surface only. No algorithms should live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .transforms import TransformChain


CoordFrameId = Literal["work_grid", "canonical_grid", "unknown"]
CoordConvention = Literal["yx"]


@dataclass
class Frame:
    image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    yolk_mask: Optional[np.ndarray] = None
    um_per_px: float = float("nan")
    meta: Optional[dict] = None


@dataclass(frozen=True)
class CanonicalGrid:
    """Descriptor-only canonical reference frame."""

    um_per_px: float
    shape_yx: tuple[int, int]
    anchor_mode: str
    anchor_yx: tuple[float, float]
    coord_convention: CoordConvention = "yx"


@dataclass
class CanonicalMaskResult:
    mask: np.ndarray
    grid: CanonicalGrid
    transform_chain: TransformChain
    meta: dict
    qc: Optional[dict] = None


@dataclass
class CanonicalImageResult:
    image: np.ndarray
    grid: CanonicalGrid
    transform_chain: TransformChain
    meta: dict


@dataclass
class CanonicalFrameResult:
    frame: Frame
    grid: CanonicalGrid
    transform_chain: TransformChain
    meta: dict


@dataclass
class RegisterResult:
    transform: TransformChain
    applied: bool
    meta: dict
    moving_in_fixed: Optional[np.ndarray] = None
