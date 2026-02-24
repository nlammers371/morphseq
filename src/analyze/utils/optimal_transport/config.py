"""Core configuration and dataclasses for UOT mask transport."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np

try:
    import scipy.sparse as sp
    Coupling = Union[np.ndarray, sp.coo_matrix]
except Exception:  # pragma: no cover - scipy optional in some envs
    Coupling = np.ndarray

# BoxYX canonical definition lives in coord.types; re-export for backwards compat.
from analyze.utils.coord.types import BoxYX  # noqa: F401


@dataclass(frozen=True)
class PairFrameGeometry:
    """
    Tracks coordinate transformations for a specific OT pair comparison.

    Coordinate flow: canonical → pair_crop → (pad) → work grid

    Note: This is different from uot_grid.GridTransform which handles
    embryo image → canonical grid transformation.

    The crop box is ALWAYS a real region in canonical space. If the crop
    dimensions are not divisible by downsample_factor, we pad the CROPPED
    ARRAYS in memory (not the crop coordinates) with zeros to achieve divisibility.
    """
    # Canonical space
    canon_shape_hw: tuple[int, int]      # Full canonical canvas (e.g., 256x576)
    pair_crop_box_yx: BoxYX              # Real crop region containing both masks

    # Padding applied to cropped arrays (in canonical pixels, bottom/right)
    crop_pad_hw: tuple[int, int]         # (pad_h, pad_w) added to make divisible

    # Work space (after padding + downsampling)
    downsample_factor: int               # s >= 1
    work_shape_hw: tuple[int, int]       # Downsampled shape passed to solver

    # Physical units (canonical is authoritative)
    px_size_um: float                    # Canonical pixel size in μm

    # Bucketing (future, currently unused in MVP)
    work_valid_box_yx: Optional[BoxYX] = None
    work_pad_offsets_yx: tuple[int, int] = (0, 0)

    @classmethod
    def from_shared(cls, geom: "SharedGridGeometry") -> "PairFrameGeometry":
        """Construct from a SharedGridGeometry (lossless mapping)."""
        return cls(
            canon_shape_hw=geom.canon_shape_hw,
            pair_crop_box_yx=geom.crop_box_yx,
            crop_pad_hw=geom.crop_pad_hw,
            downsample_factor=geom.downsample_factor,
            work_shape_hw=geom.work_shape_hw,
            px_size_um=geom.canonical_um_per_px,
        )

    @property
    def px_area_um2(self) -> float:
        return self.px_size_um ** 2

    @property
    def work_px_size_um(self) -> float:
        """Physical size of one work pixel."""
        return self.downsample_factor * self.px_size_um

    @property
    def work_px_area_um2(self) -> float:
        """Physical area of one work pixel."""
        return (self.downsample_factor ** 2) * self.px_area_um2

    @property
    def padded_crop_shape_hw(self) -> tuple[int, int]:
        """Shape of cropped+padded arrays before downsampling."""
        return (self.pair_crop_box_yx.h + self.crop_pad_hw[0],
                self.pair_crop_box_yx.w + self.crop_pad_hw[1])


class SamplingMode(str, Enum):
    AUTO = "auto"
    RAISE = "raise"


class MassMode(str, Enum):
    UNIFORM = "uniform"
    BOUNDARY_BAND = "boundary_band"
    DISTANCE_TRANSFORM = "distance_transform"


@dataclass
class UOTFrame:
    frame: Optional[np.ndarray] = None
    embryo_mask: Optional[np.ndarray] = None
    meta: Optional[dict] = None


@dataclass
class UOTFramePair:
    src: UOTFrame
    tgt: UOTFrame
    pair_meta: Optional[dict] = None


@dataclass
class UOTSupport:
    coords_yx: np.ndarray
    weights: np.ndarray


@dataclass
class UOTProblem:
    src: UOTSupport
    tgt: UOTSupport
    work_shape_hw: tuple[int, int]
    transform_meta: dict
    pair_frame: Optional[PairFrameGeometry] = None  # NEW FIELD


@dataclass
class UOTConfig:
    max_support_points: int = 5000
    sampling_mode: SamplingMode = SamplingMode.AUTO
    sampling_strategy: str = "stratified_boundary_interior"

    epsilon: float = 1e-2
    marginal_relaxation: float = 10.0
    metric: str = "sqeuclidean"
    coord_scale: float = 1.0

    store_coupling: bool = True
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_support_points < 1:
            raise ValueError(f"max_support_points must be >= 1; got {self.max_support_points}")
