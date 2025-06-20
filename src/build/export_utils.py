import numpy as np
import cv2
from typing import List
from stitch2d.tile import OpenCVTile
from skimage import feature, color
from skimage import exposure, util
from pathlib import Path
import torch
import torch.nn.functional as F


_KERNELS = {}  # {(fs, device) : (gf_tensor, log_tensor)}

# hacky solution for keyence images
def _get_keyence_tile_orientation(experiment_date):
    year_int = int(experiment_date[:4])
    if year_int < 2024:
        orientation = "vertical"
    else:
        orentation = "horizontal"

    return orientation

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

def valid_acq_dirs(root: Path, dir_list: list[str] | None) -> list[Path]:
    if dir_list is not None:
        dirs = [root / d for d in dir_list]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]
    return sorted([d for d in dirs if "ignore" not in d.name])

def to_u8_adaptive(img16, low=.1, high=99.9):
    # percentile stretch → uint8
    lo, hi = np.percentile(img16, (low, high))
    img_rescaled = exposure.rescale_intensity(img16, in_range=(lo, hi))
    return util.img_as_ubyte(img_rescaled)

def im_rescale(im, low=0, high=99.9, n_samp=2000000):
    flat = im.ravel()
    if flat.size > 2_000_000:
        flat = flat[:: flat.size // 2_000_000]
    lo, hi = np.percentile(flat, (low, high))

    # arr = im.astype(np.float32)  # Z × Y × X in host RAM
    norm = exposure.rescale_intensity(im, in_range=(lo, hi))
    # px99 = np.percentile(arr, 99.9)            # very fast C routine
    return norm, lo, hi
    # normalize in PyTorch (or NumPy — either is fine)
    #
    # norm = torch.clamp(tensor / px99, 0, 1) * 65535


# def focus_stack_maxlap(stack: np.ndarray,
#                        lap_ksize: int = 3,
#                        gauss_ksize: int = 3) -> np.ndarray:
#     """
#     Fast Laplacian-max focus stack.
#
#     Parameters
#     ----------
#     stack : (Z, Y, X) uint8/uint16 array of slices
#     lap_ksize, gauss_ksize : odd ints, kernel sizes for Laplacian & blur
#     """
#     if stack.ndim != 3:
#         raise ValueError("stack must be Z×Y×X")
#
#     # gaussian blur (vectorised)
#     blur = np.stack(
#         [cv2.GaussianBlur(z, (gauss_ksize, gauss_ksize), 0) for z in stack],
#         axis=0
#     )
#     lap = np.abs(
#         np.stack(
#             [cv2.Laplacian(z, cv2.CV_32F, ksize=lap_ksize) for z in blur],
#             axis=0
#         )
#     )
#
#     best   = lap.argmax(axis=0)                             # (Y, X) ints
#     y_idx, x_idx = np.indices(best.shape)
#     return stack[best, y_idx, x_idx]

def _findnth(haystack, needle, n):
    parts = haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

def scrape_keyence_metadata(im_path):

    with open(im_path, 'rb') as a:
        fulldata = a.read()
    metadata = fulldata.partition(b'<Data>')[2].partition(b'</Data>')[0].decode()

    meta_dict = dict({})
    keyword_list = ['ShootingDateTime', 'LensName', 'Observation Type', 'Width', 'Height', 'Width', 'Height']
    outname_list = ['Time (s)', 'Objective', 'Channel', 'Width (px)', 'Height (px)', 'Width (um)', 'Height (um)']

    for k in range(len(keyword_list)):
        param_string = keyword_list[k]
        name = outname_list[k]

        if (param_string == 'Width') or (param_string == 'Height'):
            if 'um' in name:
                ind1 = _findnth(metadata, param_string + ' Type', 2)
                ind2 = _findnth(metadata, '/' + param_string, 2)
            else:
                ind1 = _findnth(metadata, param_string + ' Type', 1)
                ind2 = _findnth(metadata, '/' + param_string, 1)
        else:
            ind1 = metadata.find(param_string)
            ind2 = metadata.find('/' + param_string)
        long_string = metadata[ind1:ind2]
        subind1 = long_string.find(">")
        subind2 = long_string.find("<")
        param_val = long_string[subind1+1:subind2]

        sysind = long_string.find("System.")
        dtype = long_string[sysind+7:subind1-1]
        if 'Int' in dtype:
            param_val = int(param_val)

        if param_string == "ShootingDateTime":
            param_val = param_val / 10 / 1000 / 1000  # convert to seconds (native unit is 100 nanoseconds)
        elif "um" in name:
            param_val = param_val / 1000

        # add to dict
        meta_dict[name] = param_val

    return meta_dict

# --- misc image helpers -----------------------------------------------------
def trim_to_shape(img: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """
    Center-crop or zero-pad `img` to `target` (Y, X) shape.
    Returns a **new** array with the original dtype.
    """
    tY, tX = target
    iY, iX = img.shape[:2]

    # pad if needed
    pad_y = max(0, tY - iY)
    pad_x = max(0, tX - iX)
    if pad_y or pad_x:
        img = np.pad(img,
                     ((pad_y // 2, pad_y - pad_y // 2),
                      (pad_x // 2, pad_x - pad_x // 2)),
                     mode="constant")

    # crop if needed
    start_y = (img.shape[0] - tY) // 2
    start_x = (img.shape[1] - tX) // 2
    return img[start_y:start_y + tY, start_x:start_x + tX]
