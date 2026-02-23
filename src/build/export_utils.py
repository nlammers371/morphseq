import numpy as np
import logging
import cv2
from typing import List, Sequence, Optional, Iterable, Union
# Note: stitch2d is an external dep; consider guarding import behind try/except
# and providing a minimal fallback for small fixed mosaics if not installed.
from stitch2d.tile import OpenCVTile
from skimage import feature, color
from skimage import exposure, util
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from concurrent.futures import ThreadPoolExecutor
import skimage.io as skio
# psutil appears unused here; safe to remove to reduce dependencies.
import os, psutil, torch
import itertools
import pandas as pd

log = logging.getLogger(__name__)

def build_experiment_metadata(repo_root: Union[str, Path], exp_name: str, meta_df: pd.DataFrame): #, microscope: str):

    # print("Compiling metadata...")

    # set paths
    # data_root = Path(data_root)
    repo_root = Path(repo_root)
    well_meta_dir = repo_root / "metadata" / "well_metadata"
    plate_meta_dir = repo_root / "metadata" / "plate_metadata"
    expected_name = f"{exp_name}_well_metadata.xlsx"
    
    # Prefer well_metadata directory if present
    search_dirs = []
    if well_meta_dir.exists():
        search_dirs.append(well_meta_dir)
    if plate_meta_dir.exists():
        search_dirs.append(plate_meta_dir)
    
    # Try exact expected name first across preferred dirs
    plate_meta_path = None
    for d in search_dirs:
        p = d / expected_name
        if p.exists():
            plate_meta_path = p
            break
    
    # If not found, try a tolerant glob once
    if plate_meta_path is None:
        candidates = []
        for d in search_dirs:
            candidates += list(d.glob(f"{exp_name}_well_metadata*.xlsx"))
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates) == 1:
            plate_meta_path = candidates[0]
            expected_list = "\n  • ".join(str(d / expected_name) for d in search_dirs)
            log.warning(
                "Expected Excel not found at standard paths:\n  • %s\nUsing non-standard metadata filename: %s",
                expected_list,
                plate_meta_path.name,
            )
        elif len(candidates) > 1:
            raise FileNotFoundError(
                "❌ Multiple candidate well metadata files found. Please keep a single file with the standard name.\n"
                + "\n".join(f"  • {p}" for p in candidates)
            )
        else:
            # Set to expected path in preferred dir for clearer error message below
            plate_meta_path = (search_dirs[0] / expected_name) if search_dirs else (repo_root / "metadata" / "well_metadata" / expected_name)
    # built_meta = meta_root / "built_metadata_files"
    # well_meta = meta_root / "well_metadata"
    # combined_out = meta_root / "combined_metadata_files"
    # exp_meta_csv = meta_root / "experiment_metadata.csv"

    # 1) experiment-level metadata, filter to use_flag
    # exp_df = (
    #     pd.read_csv(exp_meta_csv, parse_dates=["start_date"])
    #       .loc[lambda df: df["use_flag"] == 1, 
    #            ["start_date", "experiment_id", "temperature", "has_sci_data", "microscope"]]
    #       .rename(columns={"start_date": "experiment_date"})
    # )
    # exp_df["experiment_date"] = exp_df["experiment_date"].astype(str)
    # exp_dates = set(exp_df["experiment_date"])

    # 2) well-level built metadata CSVs
    # csv_paths = sorted(built_meta.glob("*_metadata.csv"))
    # dfs = []
    # for p in csv_paths:
    #     date = p.stem.replace("_metadata","")
    #     if date not in exp_dates:
    #         continue
    #     df = pd.read_csv(p)
    #     df = df.drop_duplicates(subset=["well","time_int"], keep="first")
    #     df["experiment_date"] = date
    #     dfs.append(df)
    # if not dfs:
    #     raise RuntimeError("No built metadata files matched experiment dates!")

    # master = pd.concat(dfs, ignore_index=True)
    meta_df["experiment_date"] = exp_name
    # if "microscope" in master_well_table.columns:
    #     master_well_table = master_well_table.drop(labels="microscope", axis=1)

    # master = master.merge(
    #     exp_df, on="experiment_date", how="left", indicator=True
    # )
    # if not (master["_merge"] == "both").all():
    #     missing = master.loc[master["_merge"]!="both","experiment_date"].unique()
    #     raise ValueError(f"Experiment dates missing in experiment_metadata.csv: {missing}")
    # master = master.drop(columns=["_merge"])


    # 3) load per-well Excel sheets and stack
    # if well_sheets is None:
    well_sheets = [
        "medium",
        "genotype",
        "chem_perturbation",
        "start_age_hpf",
        "embryos_per_well",
        "temperature",
        "pair",
    ]
    # well_meta_dir = meta_root / "well_metadata"
    # well_xl_paths = sorted(well_meta.glob("*_well_metadata.xlsx"))

    # sheet_dfs = []
    
    # xlf = pd.ExcelFile(xl)
    try:
        with pd.ExcelFile(plate_meta_path) as xlf:
            # build base well names A01…H12
            wells = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1,13)]
            plate_df = pd.DataFrame({"well": wells})
            plate_df["experiment_date"] = exp_name

            # parse each sheet into one vector
            for sheet in well_sheets:
                if sheet in xlf.sheet_names:
                    df = xlf.parse(sheet, header=0)           # keep blanks as NaN
                    block = df.iloc[:8, 1:13]                    # whatever made it through

                    # ensure 8 rows, 12 cols
                    if sheet in ["temperature", "embryos_per_well"]:
                        # Numeric sheets only
                        block = block.reindex(index=range(8), columns=range(1,13), fill_value=np.nan)
                        arr = block.to_numpy(dtype=float).ravel()
                    elif sheet == "start_age_hpf":
                        # Can be string identifiers OR numeric values
                        block = block.reindex(index=range(8), columns=range(1,13), fill_value='')
                        arr = block.to_numpy(dtype=str).ravel()
                    else:
                        block = block.reindex(index=range(8), columns=range(1,13), fill_value='')
                        arr = block.to_numpy(dtype=str).ravel()
                    plate_df[sheet] = arr
                else:                   
                    plate_df[sheet] = np.nan
                

            # Drop rows where start_age_hpf is empty (now stored as string)
            plate_df = plate_df[plate_df["start_age_hpf"] != ''].reset_index(drop=True)
    
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"❌ CRITICAL: Well metadata Excel file not found: {plate_meta_path}\n\n"
            f"This file is REQUIRED and contains experiment design information:\n"
            f"  • Medium conditions\n"
            f"  • Genotype information\n"
            f"  • Chemical perturbations\n"
            f"  • Start age (hpf)\n"
            f"  • Embryos per well\n"
            f"  • Temperature settings\n\n"
            f"Expected file: {plate_meta_path}\n\n"
            f"The Excel file should contain these sheets:\n"
            f"  • medium\n"
            f"  • genotype\n"
            f"  • chem_perturbation\n"
            f"  • start_age_hpf\n"
            f"  • embryos_per_well\n"
            f"  • temperature\n"
            f"  • pair\n\n"
            f"Each sheet should have 8 rows × 12 columns (A01-H12 plate layout)"
        ) from e
    
    except Exception as e:
        raise RuntimeError(
            f"❌ Error reading well metadata Excel file: {plate_meta_path}\n\n"
            f"File exists but cannot be processed. Common issues:\n"
            f"  • File is corrupted or not a valid Excel file\n"
            f"  • File is currently open in Excel (close it and try again)\n"
            f"  • Missing required sheets (medium, genotype, chem_perturbation, start_age_hpf, embryos_per_well, temperature, pair)\n"
            f"  • Incorrect sheet format (should be 8 rows × 12 columns)\n\n"
            f"Original error: {e}"
        ) from e
    
    # # optional temperature sheet
    # if "temperature" in xlf.sheet_names:
    #     temp = xlf.parse("temperature").iloc[:8,1:13].to_numpy().ravel()
    #     plate_df["temperature"] = np.nan_to_num(temp, nan=0).astype(float)
    # else:
    #     plate_df["temperature"] = np.nan

    # # optional QC sheet
    # if "qc" in xlf.sheet_names:
    #     qc = xlf.parse("qc").iloc[:8,1:13].to_numpy().ravel()
    #     plate_df["well_qc_flag"] = np.nan_to_num(qc, nan=0).astype(int)
    # else:
    #     plate_df["well_qc_flag"] = 0

    # well_df_long = pd.concat(sheet_dfs, axis=0, ignore_index=True)
    # plate_df["experiment_date"] = well_df_long["experiment_date"].astype(str)

    # add to main dataset
    # 4) final merge
    meta_df = meta_df.merge(
        plate_df, on=["well","experiment_date"], how="left", indicator=True
    )
    if not (meta_df["_merge"]=="both").all():
        missing = meta_df.loc[meta_df["_merge"]!="both", ["well","experiment_date"]]
        raise ValueError(f"Missing well metadata for:\n{missing}")
    meta_df = meta_df.drop(columns=["_merge"])

    meta_df["well_id"] = meta_df["experiment_date"] + "_" + meta_df["well"]

    # 6) reorder columns so well_id is first
    cols = ["well_id"] + [c for c in meta_df.columns if c!="well_id"]
    meta_df = meta_df[cols]

    # 7) write out
    # built_meta.mkdir(parents=True, exist_ok=True)
    # out_csv = built_meta / f"{exp_name}_metadata.csv"
    # meta_df.to_csv(out_csv, index=False)

    
    return meta_df



# patterns to use for file checking
PATTERNS = {
    "raw"     : ("W*", "XY*", "*.nd2"),
    "ff"      : ("*.jpg", "*.png", "ff_*"),
    "stitch"  : ("*_stitch.jpg",),
    "stitch_z"  : ("*_stack.tif",), 
    "segment" : ("*.jpg"),
    "snips" : ("*.jpg"),
}

# def _match_files(p: Path, patterns: Sequence[str], max_files:int=1) -> list[Path]:
#     """Return every file in *p* that matches ANY of the glob patterns."""
#     if not p or not p.exists():
#         return []
#     matches = itertools.chain.from_iterable(p.glob(pat) for pat in patterns)[:max_files]
#     return [m for m in matches if m.is_file()]

def _match_files(folder: Union[Path, str], patterns: Iterable[str]) -> List[Path]:
    """
    Return a list with *at most one* matching file in `folder`
    (empty list ↔ no match).  Fast because we stop after the first hit.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    folder = Path(folder)
    if not folder.is_dir():
        return []

    for pat in patterns:
        hit = next(folder.glob(pat), None)     # grab first match, if any
        if hit is not None:
            return [hit]                       # early exit

    return []                                  


def has_output(path: Optional[Path], patterns: Sequence[str]) -> bool:
    """Does *path* contain at least one matching file?"""
    if not path:
        return False
    return bool(_match_files(Path(path), patterns))

def newest_mtime(path: Optional[Path], patterns: Sequence[str]) -> float:
    """
    mtime of the newest file that matches *patterns*.
    Returns 0 if nothing matches.
    """
    if not path:
        return 0.0
    
    files = _match_files(Path(path), patterns)
    if not files:
        return 0.0
    
    return max(f.stat().st_mtime for f in files)

    


def _mod_time(path: Optional[Path]) -> float:
    return path.stat().st_mtime if path and path.exists() else 0.0

def _write_ff(out_root: Path,
        well: str,
        t_idx: int,
        ch_idx: int,
        ff: np.ndarray,
        overwrite: bool = False,
    ):
    name = f"{well}_t{t_idx:04}_ch{ch_idx:02}_stitch.jpg"
    out_root.mkdir(parents=True, exist_ok=True)
    f = out_root / name
    if f.exists() and not overwrite:
        return
    skio.imsave(f, ff, check_contrast=False)


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
                         n_workers: Optional[int] = None):
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
        orientation = "horizontal"

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
    data_zyx: Union[torch.Tensor, np.ndarray],   # either Z×Y×X or N×Z×Y×X
    filter_size: int,
    device: Union[str, torch.device] = "cpu",
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
    data_zyx: Union[torch.Tensor, np.ndarray],   # B*T x Z × Y × X dtype float / uint16
    filter_size: int,
    device: Union[str, torch.device] = "cpu",
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


# ============================================================================
# Z-Stack Filtering Functions (Motion Blur & Out-of-Focus Rejection)
# ============================================================================

def compute_slice_quality_metrics(
    data_zyx: torch.Tensor,      # shape (Z, Y, X) or (N, Z, Y, X)
    log_scores: torch.Tensor,    # shape (Z, Y, X) or (N, Z, Y, X)
    enable_correlation: bool = True
) -> dict:
    """
    Compute per-slice quality metrics from LoG focus stacker outputs.

    Parameters:
    -----------
    data_zyx : torch.Tensor
        Input z-stack, shape (Z, Y, X) or (N, Z, Y, X)
    log_scores : torch.Tensor
        LoG response from LoG_focus_stacker, same shape as data_zyx
    enable_correlation : bool
        If True, compute frame-to-frame correlation metrics

    Returns:
    --------
    metrics : dict
        - mean_log_per_slice: np.ndarray, mean |LoG| response per slice
        - peak_log: float, maximum mean |LoG| across slices
        - mean_log: float, average mean |LoG| across slices
        - min_corr: float, minimum frame-to-frame correlation (if enabled)
        - median_corr: float, median correlation (if enabled)
    """
    # Handle batched input (process first sample only)
    if data_zyx.ndim == 4:
        data = data_zyx[0]  # shape (Z, Y, X)
        log_s = log_scores[0]
    else:
        data = data_zyx
        log_s = log_scores

    # Per-slice mean LoG
    mean_log_per_slice = log_s.mean(dim=(1, 2)).cpu().numpy()  # shape (Z,)

    metrics = {
        "mean_log_per_slice": mean_log_per_slice,
        "peak_log": float(mean_log_per_slice.max()),
        "mean_log": float(mean_log_per_slice.mean()),
    }

    if enable_correlation:
        Z = data.shape[0]
        correlations = []
        for i in range(Z - 1):
            a = data[i].cpu().numpy().ravel()
            b = data[i+1].cpu().numpy().ravel()
            # Normalize
            a = (a - a.mean()) / (a.std() + 1e-8)
            b = (b - b.mean()) / (b.std() + 1e-8)
            # Pearson correlation
            corr = np.dot(a, b) / len(a)
            correlations.append(corr)

        if correlations:
            metrics["min_corr"] = float(np.min(correlations))
            metrics["median_corr"] = float(np.median(correlations))
        else:
            metrics["min_corr"] = 1.0
            metrics["median_corr"] = 1.0

    return metrics


def filter_bad_slices(
    data_zyx: torch.Tensor,      # shape (Z, Y, X) or (N, Z, Y, X)
    log_scores: torch.Tensor,    # same shape as data_zyx
    alpha: float = 0.7,
    corr_threshold: float = 0.9,
    method: str = "peak_relative"
) -> tuple:
    """
    Filter out bad slices before focus stacking.

    Parameters:
    -----------
    data_zyx : torch.Tensor
        Input z-stack
    log_scores : torch.Tensor
        LoG response from LoG_focus_stacker
    alpha : float
        Peak-relative threshold (0.5-0.8 typical)
        Slices with mean_log < peak * alpha are rejected
    corr_threshold : float
        Correlation threshold (0.85-0.95 typical)
        Frame pairs with corr < threshold are rejected (both frames)
    method : str
        "peak_relative" - filter by sharpness only
        "correlation" - filter by motion only
        "hybrid" - combine both (recommended)

    Returns:
    --------
    filtered_data : torch.Tensor
        Stack with bad slices removed
    kept_indices : np.ndarray
        Boolean mask of kept slices
    metrics : dict
        Quality and filtering diagnostics
    """
    # Handle batched input (process first sample only)
    if data_zyx.ndim == 4:
        data = data_zyx[0]
        log_s = log_scores[0]
        was_batched = True
    else:
        data = data_zyx
        log_s = log_scores
        was_batched = False

    Z = data.shape[0]

    # Compute mean LoG per slice
    mean_log_per_slice = log_s.mean(dim=(1, 2)).cpu().numpy()

    # Filter by sharpness (peak-relative)
    if method in ["peak_relative", "hybrid"]:
        peak_score = mean_log_per_slice.max()
        threshold = peak_score * alpha
        is_sharp = mean_log_per_slice >= threshold
    else:
        is_sharp = np.ones(Z, dtype=bool)

    # Filter by correlation (motion detection)
    if method in ["correlation", "hybrid"]:
        is_stable = np.ones(Z, dtype=bool)
        for i in range(Z - 1):
            a = data[i].cpu().numpy().ravel()
            b = data[i+1].cpu().numpy().ravel()
            # Normalize
            a = (a - a.mean()) / (a.std() + 1e-8)
            b = (b - b.mean()) / (b.std() + 1e-8)
            # Pearson correlation
            corr = np.dot(a, b) / len(a)

            # Bilateral flagging: if correlation is bad, flag BOTH frames
            if corr < corr_threshold:
                is_stable[i] = False
                is_stable[i+1] = False
    else:
        is_stable = np.ones(Z, dtype=bool)

    # Combine filters
    keep_mask = is_sharp & is_stable

    # Safety: never drop all frames
    if not keep_mask.any():
        keep_mask = is_sharp if is_sharp.any() else np.ones(Z, dtype=bool)

    # Filter the data
    kept_indices_torch = torch.from_numpy(keep_mask).to(data.device)
    filtered_data = data[kept_indices_torch]

    if was_batched:
        filtered_data = filtered_data.unsqueeze(0)  # restore batch dimension

    metrics = {
        "n_total": Z,
        "n_sharp": int(is_sharp.sum()),
        "n_stable": int(is_stable.sum()),
        "n_kept": int(keep_mask.sum()),
        "rejection_rate": 1.0 - (keep_mask.sum() / Z),
    }

    return filtered_data, keep_mask, metrics


def LoG_focus_stacker_with_filtering(
    data_zyx: Union[torch.Tensor, np.ndarray],
    filter_size: int,
    device: Union[str, torch.device] = "cpu",
    enable_filtering: bool = False,
    filter_alpha: float = 0.7,
    filter_corr_threshold: float = 0.9,
    filter_method: str = "peak_relative",
    return_metrics: bool = False
):
    """
    LoG focus stacker with optional pre-filtering of bad slices.

    This wrapper extends LoG_focus_stacker to optionally filter out:
    - Out-of-focus frames (via peak-relative sharpness threshold)
    - Motion artifacts (via frame-to-frame correlation)

    Parameters:
    -----------
    data_zyx : torch.Tensor or np.ndarray
        Input z-stack, shape (Z, Y, X) or (N, Z, Y, X)
    filter_size : int
        Kernel size for LoG filter
    device : str or torch.device
        Device to run computation on
    enable_filtering : bool
        If True, filter bad slices before stacking
    filter_alpha : float
        Peak-relative threshold (0.5-0.8 typical)
    filter_corr_threshold : float
        Correlation threshold (0.85-0.95 typical)
    filter_method : str
        "peak_relative" (sharpness only),
        "correlation" (motion only),
        "hybrid" (both - recommended)
    return_metrics : bool
        If True, return quality metrics dict

    Returns:
    --------
    ff : torch.Tensor
        Full-focus image
    abs_log : torch.Tensor
        LoG scores (from filtered stack if filtering enabled)
    metrics : dict (only if return_metrics=True)
        Quality and filtering diagnostics
    """
    device = torch.device(device)

    # Convert to tensor if needed
    if not torch.is_tensor(data_zyx):
        data = torch.from_numpy(np.asarray(data_zyx))
    else:
        data = data_zyx
    data = data.to(device, dtype=torch.float32)

    # First pass: compute LoG scores on full stack
    ff_initial, log_scores_initial = LoG_focus_stacker(data, filter_size, device)

    metrics = {}

    if enable_filtering:
        # Filter bad slices
        filtered_data, keep_mask, filter_metrics = filter_bad_slices(
            data, log_scores_initial,
            alpha=filter_alpha,
            corr_threshold=filter_corr_threshold,
            method=filter_method
        )
        metrics.update(filter_metrics)

        # Second pass: focus stack on filtered data
        ff, log_scores = LoG_focus_stacker(filtered_data, filter_size, device)
    else:
        ff = ff_initial
        log_scores = log_scores_initial
        keep_mask = np.ones(data.shape[0] if data.ndim == 3 else data.shape[1], dtype=bool)

    # Compute quality metrics
    if return_metrics:
        quality_metrics = compute_slice_quality_metrics(
            data if not enable_filtering else filtered_data,
            log_scores,
            enable_correlation=True
        )
        metrics.update(quality_metrics)
        return ff, log_scores, metrics

    return ff, log_scores


def valid_acq_dirs(root: Path, dir_list: Optional[list[str]]) -> list[Path]:
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
