# ---------- new / consolidated imports ----------
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# ---------- one-time kernel cache ----------
_KERNELS = {}  # {(fs, device) : (gf_tensor, log_tensor)}

def _get_kernels(filter_size: int, device: torch.device):
    """Return (gaussian, laplacian-of-gaussian) kernels on the requested device."""
    key = (filter_size, device)
    if key in _KERNELS:
        return _KERNELS[key]

    k = filter_size                      # alias for brevity
    ind = k // 2 + 1                     # strip border after filtering

    # --- gaussian --------------------------------------------------------
    gf = np.zeros((2 * k + 1, 2 * k + 1), np.float32)
    gf[k, k] = 1.0
    gf = cv2.GaussianBlur(gf, (k, k), 0)[ind:-ind, ind:-ind]
    gf = torch.from_numpy(gf).unsqueeze(0).unsqueeze(0).to(device)

    # --- laplacian -------------------------------------------------------
    lpf = np.zeros_like(gf[0, 0].cpu().numpy())
    pad = (k - lpf.shape[0] // 2)        # difference after cropping
    full = np.pad(lpf, pad, mode="constant")
    full[k, k] = 1
    lpf = cv2.Laplacian(full, cv2.CV_32F, ksize=k)[ind:-ind, ind:-ind]
    lpf = torch.from_numpy(lpf).unsqueeze(0).unsqueeze(0).to(device)

    _KERNELS[key] = (gf, lpf)
    return _KERNELS[key]

# ---------- main function ----------
def LoG_focus_stacker(
    data_zyx: torch.Tensor | np.ndarray,   # Z × Y × X dtype float / uint16
    filter_size: int,
    device: str | torch.device = "cpu",
):
    """
    Return (full-focus tensor, abs(LoG) stack).
    Output dtype is float32 on the requested device.
    """
    device = torch.device(device)
    gf, logf = _get_kernels(filter_size, device)

    # ---- move data once -------------------------------------------------
    if not isinstance(data_zyx, torch.Tensor):
        data = torch.from_numpy(data_zyx)
    else:
        data = data_zyx
    data = data.to(device, dtype=torch.float32, non_blocking=True)
    data = data.unsqueeze(1)             # Z × 1 × Y × X

    # ---- LoG ------------------------------------------------------------
    gb  = F.conv2d(data, gf,  padding="same")
    log = F.conv2d(gb,   logf, padding="same").squeeze(1)  # Z × Y × X
    abs_log = log.abs()

    # ---- choose in-focus plane -----------------------------------------
    best, idx = abs_log.max(dim=0)       # Y × X
    ff = data.squeeze(1).gather(0, idx.unsqueeze(0)).squeeze(0)

    return ff, abs_log