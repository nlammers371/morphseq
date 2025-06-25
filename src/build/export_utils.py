import numpy as np
import cv2
from typing import List, Sequence
from stitch2d.tile import OpenCVTile
from skimage import feature, color
from skimage import exposure, util
from pathlib import Path
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import skimage.io as skio
import os, psutil, torch


def get_n_cpu_workers(frac: float = 0.5, max_workers: int = 12) -> int:
    """
    Use a fraction of your logical cores for I/O / CPU-bound tasks.
    """
    if os.cpu_count() > 1:
        total = 8
    else: 1

    return total

def estimate_max_blocks(Z: int, Y: int, X: int,
                        dtype: torch.dtype = torch.float16,
                        safety: float = 0.8,
                        device: str = "cuda",
                        rs_thresh: int=2048,
                        batch_max: int = 32) -> int:
    """
    Roughly how many Z×Y×X blocks of type `dtype` you can fit on GPU,
    reserving `safety` fraction of free memory for overhead.
    """
    torch.cuda.empty_cache()
    if device == "cuda":
        # query free GPU memory (in bytes)
        free, _ = torch.cuda.mem_get_info()  
    else:
        free = 2_000_000_000

    # bytes per element
    elem_size = torch.zeros((), dtype=dtype).element_size()
    rs_flag = Y > rs_thresh
    if rs_flag:
        block_bytes = Z * Y * X * elem_size // 4
    else:
        block_bytes = Z * Y * X * elem_size

    # allow only `safety` fraction of free memory
    max_blocks = int((free * safety) // (block_bytes * 16) / 2) * 2 # added fudge factor to account for impact of convolutions etc

    return min(batch_max, max(1, max_blocks)), rs_flag

def estimate_batch_sizes(sample_bytes, gpu_fraction=0.75) -> tuple[int, int]:

    # 2) GPU estimate
    if torch.cuda.is_available():
        dev_idx = torch.cuda.current_device()
        dev     = torch.device(f"cuda:{dev_idx}")
        props    = torch.cuda.get_device_properties(dev)
        reserved = torch.cuda.memory_reserved(dev)
        allocated= torch.cuda.memory_allocated(dev)
        free_gpu = props.total_memory - reserved - allocated
        gpu_bs_raw = int(np.floor(free_gpu * gpu_fraction / 4) * 4)
        gpu_bs   = max(int(gpu_bs_raw // sample_bytes), 1)
    else:
        gpu_bs = 0  # or None, to signal “no GPU”

    # 3) CPU estimate
    cpu_bs   = 1

    return gpu_bs, cpu_bs


def get_n_workers_for_pipeline(max_workers: int=4):
    """
    How many parallel subprocesses (e.g. for loading) should we spawn?
    If you have a GPU, you might want fewer workers to reduce
    overall memory pressure.
    """
    if torch.cuda.is_available():
        return 1
    else:
        return 1
    

def _save_one_image(args):
    img, path = args
    path.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(path, img, check_contrast=False)


def save_images_parallel(images: Sequence[np.ndarray],
                         paths:  Sequence[Path],
                         n_workers: int | None = None):
    """
    images: list of 2D arrays (e.g. your ff tiles)
    paths:  list of same length, where to write each array
    n_workers: threads to use (defaults to cpu_count()-1)
    """
    if n_workers is None:
        import os
        n_workers = max(1, os.cpu_count() - 1)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pool.map(_save_one_image, zip(images, paths))

# hacky solution for keyence images
def _get_keyence_tile_orientation(experiment_date):
    year_int = int(experiment_date[:4])
    if year_int < 2024:
        orientation = "vertical"
    else:
        orentation = "horizontal"

    return orientation

_KERNELS: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_kernels(filter_size: int, device: torch.device):
    """Return cached (gaussian, laplacian) kernels for the given filter_size & device."""
    key = (filter_size, device)
    if key in _KERNELS:
        return _KERNELS[key]

    k   = filter_size
    ind = k // 2 + 1  # for the cropping step

    # --- Gaussian filter kernel ---
    gf = np.zeros((2*k+1, 2*k+1), np.float32)
    gf[k, k] = 1.0
    gf = cv2.GaussianBlur(gf, (k, k), 0)[ind:-ind, ind:-ind]
    gf = torch.from_numpy(gf).unsqueeze(0).unsqueeze(0).to(device)

    # --- Laplacian filter kernel ---
    # build a “full” delta at center then crop to the same shape as gf
    full = np.zeros((2*k+1, 2*k+1), np.float32)
    full[k, k] = 1.0
    lpf  = cv2.Laplacian(full, cv2.CV_32F, ksize=k)[ind:-ind, ind:-ind]
    lpf  = torch.from_numpy(lpf).unsqueeze(0).unsqueeze(0).to(device)

    _KERNELS[key] = (gf, lpf)
    return gf, lpf


def LoG_focus_stacker(
    data_zyx: torch.Tensor | np.ndarray,   # either Z×Y×X or N×Z×Y×X
    filter_size: int,
    device: str | torch.device = "cpu",
):
    """
    Returns (ff, abs_log), where:
      - ff is the full-focus image (float32 on `device`)
      - abs_log is the absolute LoG response.

    Accepts inputs of shape:
      • (Z, Y, X)   → returns ff of shape (Y, X) and abs_log (Z, Y, X)
      • (N, Z, Y, X) → returns ff of shape (N, Y, X) and abs_log (N, Z, Y, X)
    """
    device = torch.device(device)
    gf, logf = _get_kernels(filter_size, device)

    # 1) normalize input to a 4-D tensor of shape (N, Z, Y, X)
    if not torch.is_tensor(data_zyx):
        data = torch.from_numpy(np.asarray(data_zyx))
    else:
        data = data_zyx

    if data.ndim == 3:
        data = data.unsqueeze(0)        # now (1, Z, Y, X)
        squeeze_first = True
    elif data.ndim == 4:
        squeeze_first = False
    else:
        raise ValueError(f"Expected 3D or 4D input, got {data.ndim}D")

    data = data.to(device, dtype=torch.float32, non_blocking=True)

    # 2) flatten N×Z into a single batch dimension so we can conv2d all planes at once
    N, Z, Y, X = data.shape
    flat = data.reshape(N * Z, Y, X).unsqueeze(1)   # (N*Z, 1, Y, X)

    # 3) apply Gaussian then Laplacian
    gb     = F.conv2d(flat, gf, padding="same")
    log_rsp = F.conv2d(gb, logf, padding="same").squeeze(1)  # (N*Z, Y, X)
    abs_flat = log_rsp.abs()

    # 4) un-flatten to (N, Z, Y, X)
    abs_log = abs_flat.reshape(N, Z, Y, X)

    # 5) pick the best focus plane per stack
    #    max over the Z-axis (dimension 1)
    best, idx = abs_log.max(dim=1)       # both are shape (N, Y, X)

    # 6) gather the corresponding pixel from the original data
    #    first reshape data back to (N, Z, Y, X)
    #    then pick along dim=1 using idx
    ff = data.gather(1, idx.unsqueeze(1)).squeeze(1)  # (N, Y, X)

    if squeeze_first:
        ff      = ff[0]      # -> (Y, X)
        abs_log = abs_log[0] # -> (Z, Y, X)

    return ff, abs_log

def LoG_focus_stacker_batch(
    data_zyx: torch.Tensor | np.ndarray,   # B*T x Z × Y × X dtype float / uint16
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


def im_rescale(im, low=0.01, high=99.99, lo=None, hi=None):
    if (lo is None) | (hi is None):
        flat = im.ravel()
        if flat.size > 1_000_000:
            flat = flat[:: flat.size // 1_000_000]
        lo, hi = np.percentile(flat, (low, high))

    # arr = im.astype(np.float32)  # Z × Y × X in host RAM\
    if np.max(im) > 255:
        norm = exposure.rescale_intensity(im, in_range=(lo, hi)) * (2**16-1)
    else:
        norm = exposure.rescale_intensity(im, in_range=(lo, hi)) * (2**8-1)
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
