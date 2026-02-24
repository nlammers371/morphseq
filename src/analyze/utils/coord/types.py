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


@dataclass(frozen=True)
class BoxYX:
    """Half-open bounding box [y0, y1) x [x0, x1) in canvas pixel coordinates.

    Contract:
    - Half-open slices: mask[box.to_slices()] selects exactly the content.
    - Canonical pixel coordinates (origin top-left, yx order).
    - Tight by convention; callers apply padding explicitly via .pad().
    - BoxYX is purely index-space. Physical size = h * um_per_px.
    - crop_pad_hw padding is always bottom/right only -- preserves top-left
      anchor alignment.
    """

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def area(self) -> int:
        return self.h * self.w

    def to_slices(self) -> tuple[slice, slice]:
        """(slice(y0,y1), slice(x0,x1)) -- use for direct numpy indexing."""
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))

    def contains(self, other: "BoxYX") -> bool:
        """Check if this box fully contains another box."""
        return (
            self.y0 <= other.y0
            and self.y1 >= other.y1
            and self.x0 <= other.x0
            and self.x1 >= other.x1
        )

    def union(self, other: "BoxYX") -> "BoxYX":
        """Smallest box containing both."""
        return BoxYX(
            y0=min(self.y0, other.y0),
            y1=max(self.y1, other.y1),
            x0=min(self.x0, other.x0),
            x1=max(self.x1, other.x1),
        )

    def pad(self, pad_y: int, pad_x: int) -> "BoxYX":
        """Expand by pad_y/pad_x on all sides. Does NOT clamp to canvas."""
        return BoxYX(
            y0=self.y0 - pad_y,
            y1=self.y1 + pad_y,
            x0=self.x0 - pad_x,
            x1=self.x1 + pad_x,
        )

    def clamp(self, h: int, w: int) -> "BoxYX":
        """Clamp to canvas [0,h) x [0,w)."""
        return BoxYX(
            y0=max(0, self.y0),
            y1=min(h, self.y1),
            x0=max(0, self.x0),
            x1=min(w, self.x1),
        )

    def validate(self, h: int, w: int) -> None:
        """Assert 0 <= y0 <= y1 <= h and 0 <= x0 <= x1 <= w. Raises ValueError."""
        if not (0 <= self.y0 <= self.y1 <= h):
            raise ValueError(
                f"BoxYX y-range invalid: 0 <= {self.y0} <= {self.y1} <= {h} failed"
            )
        if not (0 <= self.x0 <= self.x1 <= w):
            raise ValueError(
                f"BoxYX x-range invalid: 0 <= {self.x0} <= {self.x1} <= {w} failed"
            )

    @staticmethod
    def from_mask(mask: np.ndarray) -> Optional["BoxYX"]:
        """Tight bbox of nonzero pixels. Returns None if mask is empty."""
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
        return BoxYX(
            int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        )


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
    content_bbox_yx: Optional[BoxYX] = None
    # Tight bbox of nonzero pixels in canonical canvas coordinates (half-open,
    # no padding).  Set by to_canonical_grid_mask.  None only for legacy results
    # or empty masks.


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
