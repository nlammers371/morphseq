"""Shared LoG focus-stacking utilities copied from the legacy build path."""

from __future__ import annotations

from typing import Union

import cv2
import numpy as np
from skimage import exposure, util
import torch
import torch.nn.functional as F

_KERNELS: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_kernels(filter_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached (gaussian, laplacian) kernels for the given filter size/device."""
    key = (filter_size, device)
    if key in _KERNELS:
        return _KERNELS[key]

    k = filter_size
    ind = k // 2 + 1

    gf = np.zeros((2 * k + 1, 2 * k + 1), np.float32)
    gf[k, k] = 1.0
    gf = cv2.GaussianBlur(gf, (k, k), 0)[ind:-ind, ind:-ind]
    gf_t = torch.from_numpy(gf).unsqueeze(0).unsqueeze(0).to(device)

    full = np.zeros((2 * k + 1, 2 * k + 1), np.float32)
    full[k, k] = 1.0
    lpf = cv2.Laplacian(full, cv2.CV_32F, ksize=k)[ind:-ind, ind:-ind]
    lpf_t = torch.from_numpy(lpf).unsqueeze(0).unsqueeze(0).to(device)

    _KERNELS[key] = (gf_t, lpf_t)
    return gf_t, lpf_t


def LoG_focus_stacker(
    data_zyx: Union[torch.Tensor, np.ndarray],
    filter_size: int,
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (ff, abs_log).

    Input shapes:
    - (Z, Y, X)
    - (N, Z, Y, X)
    """
    device_t = torch.device(device)
    gf, logf = _get_kernels(filter_size, device_t)

    if not torch.is_tensor(data_zyx):
        data = torch.from_numpy(np.asarray(data_zyx))
    else:
        data = data_zyx

    if data.ndim == 3:
        data = data.unsqueeze(0)
        squeeze_first = True
    elif data.ndim == 4:
        squeeze_first = False
    else:
        raise ValueError(f"Expected 3D or 4D input, got {data.ndim}D")

    data = data.to(device_t, dtype=torch.float32, non_blocking=True)

    n, z, y, x = data.shape
    flat = data.reshape(n * z, y, x).unsqueeze(1)

    gb = F.conv2d(flat, gf, padding="same")
    log_rsp = F.conv2d(gb, logf, padding="same").squeeze(1)
    abs_log = log_rsp.abs().reshape(n, z, y, x)

    _, idx = abs_log.max(dim=1)
    ff = data.gather(1, idx.unsqueeze(1)).squeeze(1)

    if squeeze_first:
        ff = ff[0]
        abs_log = abs_log[0]

    return ff, abs_log


def LoG_focus_stacker_batch(
    data_zyx: Union[torch.Tensor, np.ndarray],
    filter_size: int,
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched variant with input shape (B, Z, Y, X)."""
    device_t = torch.device(device)
    gf, logf = _get_kernels(filter_size, device_t)

    if not isinstance(data_zyx, torch.Tensor):
        data = torch.from_numpy(data_zyx)
    else:
        data = data_zyx
    data = data.to(device_t, dtype=torch.float32, non_blocking=True)
    data = data.unsqueeze(1)

    gb = F.conv2d(data, gf, padding="same")
    log_rsp = F.conv2d(gb, logf, padding="same").squeeze(1)
    abs_log = log_rsp.abs()

    _, idx = abs_log.max(dim=0)
    ff = data.squeeze(1).gather(0, idx.unsqueeze(0)).squeeze(0)
    return ff, abs_log


def to_u8_adaptive(img16: np.ndarray, low: float = 0.1, high: float = 99.9) -> np.ndarray:
    """Percentile stretch to uint8."""
    lo, hi = np.percentile(img16, (low, high))
    img_rescaled = exposure.rescale_intensity(img16, in_range=(lo, hi))
    return util.img_as_ubyte(img_rescaled)


def im_rescale(
    image: np.ndarray,
    low: float = 0.01,
    high: float = 99.99,
    lo: float | None = None,
    hi: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Legacy-compatible percentile rescaling used before LoG focus stacking."""
    if (lo is None) or (hi is None):
        flat = image.ravel()
        if flat.size > 1_000_000:
            flat = flat[:: flat.size // 1_000_000]
        lo, hi = np.percentile(flat, (low, high))

    if np.max(image) > 255:
        norm = exposure.rescale_intensity(image, in_range=(lo, hi)) * (2**16 - 1)
    else:
        norm = exposure.rescale_intensity(image, in_range=(lo, hi)) * (2**8 - 1)
    return norm, float(lo), float(hi)
