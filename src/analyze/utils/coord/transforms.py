"""Transformation primitives for `analyze.utils.coord`.

`GridTransform` uses OpenCV's affine convention on (x, y) coordinates.
Coordinate *values* in meta are in yx convention unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


Interp = Literal["nearest", "linear"]


@dataclass(frozen=True)
class GridTransform:
    name: str
    affine_2x3: np.ndarray
    in_shape_yx: tuple[int, int]
    out_shape_yx: tuple[int, int]
    interp: Interp
    params: dict


@dataclass
class TransformChain:
    transforms: list[GridTransform]

    @staticmethod
    def identity(
        *,
        shape_yx: tuple[int, int],
        interp: Interp,
        name: str = "identity",
    ) -> "TransformChain":
        """Identity convention: affine=[[1,0,0],[0,1,0]], in_shape==out_shape, interp matches downstream apply."""
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return TransformChain(
            transforms=[
                GridTransform(
                    name=name,
                    affine_2x3=affine,
                    in_shape_yx=shape_yx,
                    out_shape_yx=shape_yx,
                    interp=interp,
                    params={"affine_convention": "opencv_xy"},
                )
            ]
        )

    def __len__(self) -> int:
        return len(self.transforms)

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required to apply transforms.")
        out = mask
        for t in self.transforms:
            out = _apply_affine(out, t, is_mask=True)
        return out

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError("cv2 is required to apply transforms.")
        out = image
        for t in self.transforms:
            out = _apply_affine(out, t, is_mask=False)
        return out


def _apply_affine(arr: np.ndarray, t: GridTransform, *, is_mask: bool) -> np.ndarray:
    if cv2 is None:
        raise ImportError("cv2 is required to apply transforms.")
    h_out, w_out = t.out_shape_yx
    if t.name == "flip_x":
        return cv2.flip(arr, 1)
    flags = cv2.INTER_NEAREST if (is_mask or t.interp == "nearest") else cv2.INTER_LINEAR
    return cv2.warpAffine(arr.astype(np.float32), t.affine_2x3.astype(np.float32), (w_out, h_out), flags=flags)
