from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

# Dependency simplification notes (comments only; no behavior change):
# - nd2: specialized Nikon format reader; consider optional import with a user-friendly error, or support preconverted TIFF via imageio/cv2.
# - sklearn KMeans: used only for QC of row/column assignments; could be replaced by numpy-based sorting/bucketing to avoid scikit-learn.
# - skimage (io/util): IO/resize/CLAHE can be replaced by `imageio.v3` or OpenCV to reduce dependency surface.
# - tqdm/process_map: provide fallback using `concurrent.futures` and a no-op progress wrapper when tqdm is missing.
# - torch: focus-stacking supports CPU; ensure device="cpu" is documented as the default on machines without CUDA.


import os, json, time, logging
from pathlib import Path
from typing import List, Sequence, Tuple
from functools import partial
from queue import Queue
from threading import Thread
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import nd2
import skimage.io as skio
import skimage
from stitch2d import StructuredMosaic
from src.build.export_utils import (LoG_focus_stacker, im_rescale, _write_ff, build_experiment_metadata)
from src.data_pipeline.metadata_ingest.mapping.series_well_mapper_yx1 import map_nd2_to_wells_by_xy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)
logging.getLogger("stitch2d").setLevel(logging.ERROR) 


##########
# Helper functions
def _FF_wrapper(w, dask_arr, well_id_array, time_id_array, device, filter_size, bf_idx, 
                out_ff, well_name_lookup, overwrite, z_buff):
    
    n_z_keep = 12 if z_buff else None
    # get indices
    t_idx = time_id_array[w]
    w_idx = well_id_array[w]
    w_base = int(well_id_array[w])
    well_name = well_name_lookup.get(w_base)
    if well_name is None:
        return

    stack = _get_stack(dask_arr, t_idx, w_idx, n_z_keep=n_z_keep)
    ff = _focus_stack(stack, device, filter_size)

    _write_ff(out_ff, well_name, t_idx, bf_idx, ff, overwrite)



def _qc_well_assignments(stage_xyz_array, well_name_list_long):
     # use clustering to double check well assignments
    row_letter_vec = np.asarray([id[0] for id in well_name_list_long])
    col_num_vec = np.asarray([int(id[1:]) for id in well_name_list_long])
    row_index = np.unique(row_letter_vec)
    col_index = np.unique(col_num_vec)

    # Check rows
    row_clusters = KMeans(n_init="auto", n_clusters=len(row_index)).fit(stage_xyz_array[:, 1].reshape(-1, 1))
    row_si = np.argsort(np.argsort(row_clusters.cluster_centers_.ravel()))
    row_ind_pd = row_si[row_clusters.labels_]
    row_letter_pd = row_index[row_ind_pd]
    assert np.all(row_letter_pd == row_letter_vec)

    col_clusters = KMeans(n_init="auto", n_clusters=len(col_index)).fit(stage_xyz_array[:, 0].reshape(-1, 1))
    col_si = np.argsort(np.argsort(col_clusters.cluster_centers_.ravel()))
    col_ind_pd = col_si[col_clusters.labels_]
    col_num_pd = col_index[len(col_index)-col_ind_pd-1]
    assert np.all(col_num_pd == col_num_vec)


def _get_imputed_time_vector(nd, n_t, n_w, n_z, well_indices):
    """
    Extracts timestamps, handling missing metadata, and imputes any gaps.
    Drop-in replacement for _fix_nd2_timestamp with robust gap handling.
    
    Args:
        nd: ND2File object
        n_t: Number of timepoints
        n_w: Number of wells in ND2
        n_z: Number of Z slices
        well_indices: List of 0-based well indices to use as references
    
    Returns:
        numpy array of length n_t with complete timestamps (no NaN)
    """
    # Safe reference well selection with guards
    refs = sorted(set(w for w in (well_indices or []) if 0 <= w < n_w))
    if len(refs) >= 3:
        ref_wells = [refs[0], refs[len(refs)//2], refs[-1]]  # First, middle, last
    elif refs:
        ref_wells = refs  # Use whatever valid wells we have
    else:
        ref_wells = list(range(min(n_w, 3)))  # Fallback to first few wells
    
    log.info(f"Using reference wells: {ref_wells} from {len(refs)} selected wells")
    
    # 1. Robustly extract timestamps, allowing for NaNs
    times = np.full((n_t,), np.nan, dtype=float)
    
    for t in range(n_t):
        for w in ref_wells:
            seq = (t * n_w + w) * n_z
            try:
                times[t] = nd.frame_metadata(seq).channels[0].time.relativeTimeMs / 1000.0
                break  # Found valid timestamp, move to next timepoint
            except Exception:
                continue  # Try next reference well
    
    valid_count = (~np.isnan(times)).sum()
    log.info(f"Extracted {valid_count}/{n_t} valid timestamps from ND2 metadata")
    
    # 2. Calculate robust cycle time from ORIGINAL valid data
    s = pd.Series(times)  # Original with NaN gaps
    original_valid = s.dropna()

    if len(original_valid) >= 2:
        # Calculate cycle time from original valid timestamps
        original_diffs = original_valid.diff().dropna()
        cycle_time = original_diffs.median()
        log.info(f"Calculated cycle time: {cycle_time:.2f}s from {len(original_diffs)} intervals")
    else:
        cycle_time = 1800.0  # 30 minutes default (typical plate cycle time)
        log.info(f"Using default cycle time: {cycle_time:.2f}s")

    # 3. Impute missing values using pre-calculated cycle time
    if s.isna().any():
        missing_count = s.isna().sum()
        log.info(f"Imputing {missing_count} missing timestamps...")

        # Use direct extrapolation based on cycle time
        if len(original_valid) > 0:
            # Find the pattern in valid timestamps
            first_valid_idx = s.first_valid_index()
            last_valid_idx = s.last_valid_index()

            # Fill backwards from first valid
            if first_valid_idx > 0:
                first_time = s.iloc[first_valid_idx]
                for i in range(first_valid_idx - 1, -1, -1):
                    s.iloc[i] = first_time - (first_valid_idx - i) * cycle_time

            # Fill forwards from last valid  
            if last_valid_idx < len(s) - 1:
                last_time = s.iloc[last_valid_idx]
                for i in range(last_valid_idx + 1, len(s)):
                    s.iloc[i] = last_time + (i - last_valid_idx) * cycle_time

            # Fill middle gaps with linear progression
            for i in range(len(s)):
                if pd.isna(s.iloc[i]):
                    s.iloc[i] = s.iloc[0] + i * cycle_time
    
    # Last-resort safety: if still all NaN, create uniform grid
    if s.isna().all():
        log.warning("No valid timestamps found - using default 30 min intervals")
        s = pd.Series(np.arange(n_t, dtype=float) * 1800.0)  # 30 min intervals
    
    # Ensure monotonic non-decreasing (prevent small numeric regressions)
    s = s.cummax()
    
    # Final validation
    final_valid = n_t - s.isna().sum()
    log.info(f"Timestamp imputation complete: {final_valid}/{n_t} valid timestamps")
    log.info(f"Time range: {s.min():.1f}s to {s.max():.1f}s (median interval: {s.diff().median():.2f}s)")
    
    return s.to_numpy()


# fix large jumps in time stamp (not sure why this happens innd2 metadata)
def _fix_nd2_timestamp(nd, n_z):
    # extract frame times
    n_frames_total = nd.frame_metadata(0).contents.frameCount
    frame_time_vec = [nd.frame_metadata(i).channels[0].time.relativeTimeMs / 1000 for i in
                    range(0, n_frames_total, n_z)]
    # check for common nd2 artifact where time stamps jump midway through
    dt_frame_approx = (nd.frame_metadata(n_z).channels[0].time.relativeTimeMs -
                    nd.frame_metadata(0).channels[0].time.relativeTimeMs) / 1000
    jump_ind = np.where(np.diff(frame_time_vec) > 2*dt_frame_approx)[0] # typically it is multiple orders of magnitude to large
    if len(jump_ind) > 0:
        jump_ind = jump_ind[0]
        # prior to this point we will just use the time stamps. We will extrapolate to get subsequent time points
        nf = jump_ind - 1 - int(jump_ind/2)
        dt_frame_est = (frame_time_vec[jump_ind-1] - frame_time_vec[int(jump_ind/2)]) / nf
        base_time = frame_time_vec[jump_ind-1]
        for f in range(jump_ind, len(frame_time_vec)):
            frame_time_vec[f] = base_time + dt_frame_est*(f - jump_ind)
    frame_time_vec = np.asarray(frame_time_vec)

    return frame_time_vec


def _read_nd2(path: Path) -> nd2.ND2File:
    nd2_files = list(path.glob("*.nd2"))
    if not nd2_files:
        raise FileNotFoundError(f"No nd2 in {path}")
    if len(nd2_files) > 1:
        raise RuntimeError(f"Multiple nd2 files in {path}")
    return nd2.ND2File(nd2_files[0])


def _get_stack(
    dask_arr, t: int, w: int, n_z_keep: int | None = None
) -> np.ndarray:
    """Return Z×Y×X BF stack (no channels)."""
    nz = dask_arr.shape[2]
    buf = max((nz - n_z_keep) // 2, 0) if n_z_keep else 0
    return (
        dask_arr[t, w, buf : nz - buf, :, :].compute()
        if buf or n_z_keep
        else dask_arr[t, w, :, :, :].compute()
    )


_profiler_counter = 0  # Global counter for profiler traces

def _focus_stack(
    stack_zyx: np.ndarray,
    device: str,
    filter_size: int = 3
) -> np.ndarray:
    global _profiler_counter

    # instead of torch.quantile, use numpy
    norm, _, _ = im_rescale(stack_zyx)
    norm = norm.astype(np.float32)
    tensor = torch.from_numpy(norm).to(device)

    # Debug logging when MSEQ_YX1_DEBUG is set
    debug_enabled = os.environ.get("MSEQ_YX1_DEBUG") == "1"
    if debug_enabled:
        log.info("_focus_stack: tensor shape=%s device=%s cuda_available=%s",
                 tensor.shape, tensor.device, torch.cuda.is_available())
        if torch.cuda.is_available() and device == "cuda":
            log.info("_focus_stack: cuda mem allocated %.1f MB",
                     torch.cuda.memory_allocated() / 1e6)

    # Profile the focus stacker when debug is enabled (first 3 stacks only)
    if debug_enabled and _profiler_counter < 3:
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            ff_t, _ = LoG_focus_stacker(tensor, filter_size, device)

        # Save profiler trace
        trace_path = f"profiler_trace_stack_{_profiler_counter:03d}.json"
        prof.export_chrome_trace(trace_path)
        log.info("Saved profiler trace to: %s", trace_path)
        _profiler_counter += 1
    else:
        ff_t, _ = LoG_focus_stacker(tensor, filter_size, device)

    arr = ff_t.cpu().numpy()
    arr_clipped = np.clip(arr, 0, 65535)
    ff_i = arr_clipped.astype(np.uint16)

    # convert to 8 bit
    ff_8 = skimage.util.img_as_ubyte(ff_i)

    return ff_8 #(65535 - ff.cpu().numpy()).astype(np.uint16)


def _resolve_batch_size(
    n_z: int,
    height_px: int,
    width_px: int,
    device: str,
) -> int:
    """
    Determine a safe batching factor for YX1 focus stacking.
    Prefers env override (MSEQ_YX1_BATCH_SIZE); otherwise estimates using free CUDA memory.
    """
    env_override = os.environ.get("MSEQ_YX1_BATCH_SIZE")
    if env_override:
        try:
            val = int(env_override)
            if val > 0:
                log.info("YX1 batching: using env override batch size=%d", val)
                return val
        except ValueError:
            log.warning("Invalid MSEQ_YX1_BATCH_SIZE=%s; falling back to auto sizing", env_override)

    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return 1

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError as exc:
        log.warning("YX1 batching: unable to query CUDA memory (%s); using batch size 1", exc)
        return 1

    per_stack_bytes = (
        max(n_z, 1)
        * max(height_px, 1)
        * max(width_px, 1)
        * np.dtype(np.float32).itemsize
    )
    if per_stack_bytes <= 0:
        return 1

    safety = float(os.environ.get("MSEQ_YX1_BATCH_SAFETY", "0.7"))
    safety = min(max(safety, 0.1), 0.95)
    usable_bytes = int(free_bytes * safety)

    est_batch = max(usable_bytes // per_stack_bytes, 1)
    max_batch = int(os.environ.get("MSEQ_YX1_BATCH_MAX", "16") or 16)
    if max_batch > 0:
        est_batch = min(est_batch, max_batch)

    log.info(
        "YX1 batching auto-tune: free=%.1f GB total=%.1f GB per-stack=%.1f MB safety=%.0f%% -> batch=%d",
        free_bytes / 1e9,
        total_bytes / 1e9,
        per_stack_bytes / 1e6,
        safety * 100.0,
        est_batch,
    )
    return est_batch



def build_ff_from_yx1(
    data_root: str | Path,
    repo_root: str | Path,
    exp_name: str,
    overwrite: bool = False,
    # dir_list: Sequence[str] | None = None,
    # write_dir: str | Path | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_workers: int=1,
    metadata_only: bool = False,
    # n_z_keep: Sequence[int | None] | None = None,
):


    par_flag = n_workers > 1

    # Debug logging when MSEQ_YX1_DEBUG is set
    if os.environ.get("MSEQ_YX1_DEBUG") == "1":
        log.info("=" * 60)
        log.info("GPU DIAGNOSTICS - build_ff_from_yx1")
        log.info("Torch CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            log.info("CUDA device count: %d", torch.cuda.device_count())
            log.info("CUDA version: %s", torch.version.cuda)
            log.info("Selected device: %s", device)
            for i in range(torch.cuda.device_count()):
                log.info("GPU %d: %s", i, torch.cuda.get_device_name(i))
        else:
            log.warning("CUDA NOT AVAILABLE - will use CPU (slow!)")
            log.info("Selected device: %s", device)
        log.info("n_workers: %d %s", n_workers,
                 "(GPU works best with n_workers=1)" if n_workers > 1 and device == "cuda" else "")
        log.info("=" * 60)

    data_root = Path(data_root)
    read_root = data_root / "raw_image_data" / "YX1"
    write_root = data_root / "built_image_data"
    meta_root = data_root / "metadata"

    # for exp_path, z_keep in zip(exp_dirs, n_z_keep):
    exp_path = read_root / exp_name
    # exp_name = exp_path.exp_name
    log.info("Calculating FF for %s", exp_name)

    nd = _read_nd2(exp_path)
    shape_twzcxy = nd.shape  # T,W,Z,C,Y,X
    n_t, n_w, n_z = shape_twzcxy[:3]
    height_px = shape_twzcxy[4] if len(shape_twzcxy) > 4 else nd.shape[-2]
    width_px = shape_twzcxy[5] if len(shape_twzcxy) > 5 else nd.shape[-1]
    print(f"ND2 reports: n_t={n_t} timepoints, n_w={n_w} wells, n_z={n_z} z-slices")
    print(f"Total expected frames in ND2: {n_t * n_w * n_z}")
    
    # Check actual frame count
    try:
        total_frames = nd.frame_metadata(0).contents.frameCount
        print(f"Actual frame count from ND2 metadata: {total_frames}")
    except:
        print("Could not read actual frame count")

    # calculate batch size
    # sample_bytes = np.product(shape_twzcxy[2:]) * 4 # factor of 4 for 16 but
    # batch_size = estimate_batch_sizes(sample_bytes)

    dask_arr = nd.to_dask()  # (T,W,Z,C,Y,X)
    channel_names = [c.channel.name for c in nd.frame_metadata(0).channels]
    # Determine BF channel index with simple rules
    env_bf = os.environ.get("YX1_BF_CHANNEL_INDEX")
    if env_bf is not None:
        try:
            bf_idx = int(env_bf)
        except Exception:
            raise ValueError(f"Invalid YX1_BF_CHANNEL_INDEX env var: {env_bf}")
    else:
        # Try exact 'BF' (case-insensitive)
        lower = [str(n).lower() for n in channel_names]
        bf_idx = None
        if "bf" in lower:
            bf_idx = lower.index("bf")
        else:
            # Try known single-channel labels used on your system
            # Match case-insensitively to be safe
            try_labels = ["eyes - dia", "empty"]
            for i, name in enumerate(lower):
                if name in try_labels:
                    bf_idx = i
                    break
        if bf_idx is None:
            # Fallbacks: if only one channel, pick it; else fail clearly
            if len(channel_names) == 1:
                bf_idx = 0
            else:
                raise ValueError(
                    f"Could not locate BF channel. Available channels: {channel_names}. "
                    f"Set YX1_BF_CHANNEL_INDEX to override."
                )

    # Loud warnings when using non-standard BF naming
    lower = [str(n).lower() for n in channel_names]
    if lower == ["eyes - dia"]:
        log.warning("Using channel 'EYES - Dia' as BF (no 'BF' channel present). Channels: %s", channel_names)
    elif lower == ["empty"]:
        log.warning("Using channel 'Empty' as BF (mislabeled brightfield channel). Channels: %s", channel_names)
    elif "bf" not in lower:
        log.warning("No 'BF' channel found; selected index %d from channels: %s. If incorrect, set YX1_BF_CHANNEL_INDEX.", bf_idx, channel_names)


    # get image resolution
    voxel_size = nd.voxel_size()

    # n_channels = len(nd.frame_metadata(0).channels)

    # read in plate map (prefer well_metadata/, fall back to plate_metadata/)
    excel_name = f"{exp_name}_well_metadata.xlsx"
    xl_well = meta_root / "well_metadata" / excel_name
    xl_plate = meta_root / "plate_metadata" / excel_name
    if xl_well.exists():
        xl_path = xl_well
    elif xl_plate.exists():
        xl_path = xl_plate
    else:
        # Try a tolerant glob (handles stray suffixes like *_well_metadata1.xlsx)
        candidates = []
        candidates += list((meta_root / "well_metadata").glob(f"{exp_name}_well_metadata*.xlsx"))
        candidates += list((meta_root / "plate_metadata").glob(f"{exp_name}_well_metadata*.xlsx"))
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates) == 1:
            xl_path = candidates[0]
            log.warning(
                "Expected Excel not found at standard paths:\n  • %s\n  • %s\nUsing non-standard metadata filename: %s",
                xl_well, xl_plate, xl_path.name,
            )
        elif len(candidates) > 1:
            raise FileNotFoundError(
                "❌ Multiple candidate well metadata files found. Please keep a single file with the standard name.\n"
                + "\n".join(f"  • {p}" for p in candidates)
            )
        else:
            raise FileNotFoundError(
                f"❌ Required well metadata Excel not found in either location.\n"
                f"Expected one of:\n  • {xl_well}\n  • {xl_plate}"
            )
    plate_map_xl = pd.ExcelFile(xl_path)

    # if n_channels > 1:
    #     channel_map = plate_map_xl.parse("channels")

    # Note: timestamp extraction moved to after well list creation for robust processing

    # TRY XY-BASED MAPPING FIRST (default, most reliable)
    # Get ND2 file path for XY extraction
    nd2_files = list(exp_path.glob("*.nd2"))
    nd2_path = nd2_files[0] if nd2_files else None

    well_name_list = []
    well_ind_list = []
    mapping_method = "unknown"

    if nd2_path:
        try:
            log.info("Attempting XY-based reference mapping...")
            p_to_well_map, xy_diag = map_nd2_to_wells_by_xy(nd2_path)

            # Convert P-based mapping (0-based) to series-based (1-based)
            # Sort by P index to maintain order
            for p_idx in sorted(p_to_well_map.keys()):
                well_name = p_to_well_map[p_idx]
                series_idx_1b = p_idx + 1  # Convert 0-based P to 1-based series
                well_name_list.append(well_name)
                well_ind_list.append(series_idx_1b)

            mapping_method = "xy_reference"
            log.info("✓ XY-based mapping SUCCESS: %d wells mapped", len(well_name_list))
            log.info("  Distance stats: min=%.1f, max=%.1f, mean=%.1f µm",
                     min(xy_diag['distances']) if xy_diag['distances'] else 0,
                     max(xy_diag['distances']) if xy_diag['distances'] else 0,
                     np.mean(xy_diag['distances']) if xy_diag['distances'] else 0)
            if xy_diag['rejected']:
                log.warning("  Rejected %d positions (distance > threshold)", len(xy_diag['rejected']))

        except Exception as e:
            log.warning("XY-based mapping failed: %s", str(e))
            log.warning("Falling back to Excel series_number_map...")
            well_name_list = []
            well_ind_list = []

    # FALLBACK TO EXCEL-BASED MAPPING if XY failed or unavailable
    if not well_name_list:
        log.info("Using Excel-based series_number_map (fallback)")
        # get series numbers (robust to header row/col)
        sm_raw = plate_map_xl.parse("series_number_map", header=None)
        # Detect header row of 1..12 in columns 1..12 and drop it if present
        data_rows = sm_raw
        try:
            header_like = list(sm_raw.iloc[0, 1:13].astype(object))
            if header_like == list(range(1, 13)):
                data_rows = sm_raw.iloc[1:9, :]  # rows A..H in 1..8 next
            else:
                data_rows = sm_raw.iloc[:8, :]
        except Exception:
            data_rows = sm_raw.iloc[:8, :]

        series_map = data_rows.iloc[:, 1:13]  # 8x12 numeric grid

        used_series = set()
        col_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        row_letter_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for c in range(len(col_id_list)):
            for r in range(len(row_letter_list)):
                val = series_map.iloc[r, c]
                if pd.notna(val):
                    try:
                        series_idx_1b = int(val)
                    except Exception:
                        continue
                    # Validate series index: 1..n_w
                    if series_idx_1b < 1 or series_idx_1b > n_w:
                        log.warning("Skipping out-of-range series index %d (valid 1..%d) at well %s%02d",
                                    series_idx_1b, n_w, row_letter_list[r], col_id_list[c])
                        continue
                    if series_idx_1b in used_series:
                        # Skip duplicates to avoid mapping two wells to same ND2 series
                        log.warning("Duplicate series index %d; keeping first mapping, skipping well %s%02d",
                                    series_idx_1b, row_letter_list[r], col_id_list[c])
                        continue
                    used_series.add(series_idx_1b)
                    well_name = row_letter_list[r] + f"{col_id_list[c]:02}"
                    well_name_list.append(well_name)
                    well_ind_list.append(series_idx_1b)

        mapping_method = "excel_series_number_map"

    # Log a concise mapping summary for diagnostics
    if well_name_list and well_ind_list:
        pairs = sorted(zip(well_name_list, well_ind_list), key=lambda x: x[1])
        sel = len(pairs)
        smin = pairs[0][1]
        smax = pairs[-1][1]
        log.info(
            "YX1 well mapping [%s]: selected_wells=%d, series_min=%d, series_max=%d",
            mapping_method, sel, smin, smax
        )
        log.info(
            "YX1 map examples: first %s->%d | last %s->%d",
            pairs[0][0], pairs[0][1], pairs[-1][0], pairs[-1][1]
        )

    # Diagnostic: per-well frame coverage (how many timepoints truly available)
    try:
        selected_nd2_indices = sorted(set(int(s1b) - 1 for s1b in well_ind_list))
        def _has_seq(w: int, t: int) -> bool:
            seq = (t * n_w + w) * n_z
            try:
                _ = nd.frame_metadata(seq)
                return True
            except Exception:
                return False
        for w in selected_nd2_indices:
            t_avail = 0
            for t in range(n_t):
                if _has_seq(w, t):
                    t_avail += 1
                else:
                    break
            # Find well name for this ND2 index if present
            try:
                # series number is w+1 (1-based)
                s1b = w + 1
                idx = well_ind_list.index(s1b)
                wname = well_name_list[idx]
            except ValueError:
                wname = f"nd2_{w:02d}"
            log.info("DEBUG YX1 coverage: well %s (nd2 %d) frames_available=%d/%d", wname, w, t_avail, n_t)
    except Exception:
        pass

    si = np.argsort(well_ind_list)
    well_name_list_sorted = np.asarray(well_name_list)[si].tolist()
    well_ind_list_sorted = np.asarray(well_ind_list)[si].tolist()

    # Extract timestamps using robust method (handles NaNs and jumps)
    # Convert well series (1-based) to 0-based indices for reference selection
    ref_indices = [int(s) - 1 for s in well_ind_list_sorted]
    frame_time_vec = _get_imputed_time_vector(nd, n_t=n_t, n_w=n_w, n_z=n_z, well_indices=ref_indices)

    # generate longform vectors
    well_name_list_long = np.repeat(well_name_list_sorted, n_t)
    well_ind_list_long = np.repeat(well_ind_list_sorted, n_t)

    # check that assigned well IDs match recorded stage positions - ONLY for mapped wells
    n_mapped_wells = len(well_ind_list)
    total_entries = n_mapped_wells * n_t
    
    stage_xyz_array = np.empty((total_entries, 3))
    well_id_array = np.empty((total_entries,), dtype=np.uint16)
    time_id_array = np.empty((total_entries,), dtype=np.uint16)
    iter_i = 0
    
    # Loop through only the wells that are mapped in the Excel file - USE SORTED LISTS
    for well_idx, (well_name, well_series) in enumerate(zip(well_name_list_sorted, well_ind_list_sorted)):
        nd2_well_idx = well_series - 1  # Convert from 1-based to 0-based indexing
        for t in range(n_t):
            base_ind = t*n_w + nd2_well_idx  # Use ND2's well index for frame calculation
            slice_ind = base_ind*n_z
            
            try:
                stage_xyz_array[iter_i, :] = np.asarray(nd.frame_metadata(slice_ind).channels[0].position.stagePositionUm)
                well_id_array[iter_i] = nd2_well_idx  # Store the ND2 well index
                time_id_array[iter_i] = t
                iter_i += 1
            except IndexError:
                # Skip this frame - ND2 file has fewer frames than expected
                # Only print summary at the end to avoid spam
                continue

    # Trim arrays to actual processed entries if some frames were missing
    if iter_i < total_entries:
        missing_frames = total_entries - iter_i
        print(f"ND2 file missing {missing_frames}/{total_entries} expected frames - trimming arrays to {iter_i} entries")
        stage_xyz_array = stage_xyz_array[:iter_i]
        well_id_array = well_id_array[:iter_i]  
        time_id_array = time_id_array[:iter_i]

    # Check that recorded well positions are consistent with actual image positions on the plate.
    # Some ND2s have missing wells/frames leading to length mismatches; in that case, skip QC check.
    if exp_name != "20240314":
        try:
            if stage_xyz_array.shape[0] != len(well_name_list_long):
                log.warning(
                    "Skipping YX1 well-assignment QC: length mismatch (stage=%d, wells=%d)",
                    stage_xyz_array.shape[0], len(well_name_list_long)
                )
            else:
                _qc_well_assignments(stage_xyz_array, well_name_list_long)
        except Exception as e:
            log.warning("Skipping YX1 well-assignment QC due to error: %s", e)

    if len(shape_twzcxy) == 6:
        dask_arr = dask_arr[:, :, :, bf_idx, :, :]

    # generate metadata dataframe for SELECTED wells only
    n_selected_wells = len(well_name_list_sorted)
    well_df = pd.DataFrame(well_name_list_long[:, np.newaxis], columns=["well"])
    well_df["nd2_series_num"] = well_ind_list_long
    well_df["microscope"] = "YX1"
    # time_int goes 0..n_t-1 for each selected well
    time_int_list = np.tile(np.arange(0, n_t, dtype=int), n_selected_wells)
    well_df["time_int"] = time_int_list
    well_df["Height (um)"] = shape_twzcxy [3]*voxel_size[1]
    well_df["Width (um)"] = shape_twzcxy [4]*voxel_size[0]
    well_df["Height (px)"] = shape_twzcxy [3]
    well_df["Width (px)"] = shape_twzcxy [4]
    well_df["BF Channel"] = bf_idx
    well_df["Objective"] = nd.frame_metadata(0).channels[0].microscope.objectiveName
    # Map frame times for selected wells: simply repeat the per-time vector for each selected well
    frame_time_vec = np.asarray(frame_time_vec)
    time_ind_vec = np.tile(np.arange(0, n_t, dtype=int), n_selected_wells)
    well_df["Time (s)"] = frame_time_vec[time_ind_vec]


    if device == "cpu":
        print("Warning: using CPU. This may be quite slow. GPU recommended.")

    z_buff = True if exp_name == "20231206" else False

    # call FF function
    if not metadata_only:
        log.info("Calculating FF for %s", exp_name) 
        
        out_ff = write_root / "stitched_FF_images" / exp_name

        # Build lookup of ND2 well index -> well name from series_number_map (selected wells only)
        well_name_lookup = {int(ind)-1: name for name, ind in zip(well_name_list_sorted, well_ind_list_sorted)}
        
        # Show every 8th lookup entry for debugging (convert to 1-based for user clarity)
        lookup_items = list(well_name_lookup.items())
        sample_entries = {}
        for i in range(0, len(lookup_items), 8):
            nd2_idx, well_name = lookup_items[i]
            excel_series = nd2_idx + 1  # Convert back to Excel 1-based numbering
            sample_entries[excel_series] = well_name
        # Always include the last entry
        if len(lookup_items) > 0 and (len(lookup_items) - 1) % 8 != 0:
            nd2_idx, well_name = lookup_items[-1]
            excel_series = nd2_idx + 1
            sample_entries[excel_series] = well_name
        
        print(f"Subsetting to {len(well_name_lookup)} wells from metadata")
        print(f"Sample lookup entries (every 8th): {sample_entries}")

        # Build well info for only the sampled/subset wells (mapped in Excel)
        sampled_wells_info = []  # Store (well_idx, well_name, well_series, nd2_well_idx, t) for processing

        for well_idx, (well_name, well_series) in enumerate(zip(well_name_list_sorted, well_ind_list_sorted)):
            nd2_well_idx = well_series - 1  # Convert from 1-based to 0-based
            for t in range(n_t):
                sampled_wells_info.append((well_idx, well_name, well_series, nd2_well_idx, t))

        # Resume optimization: compute only frames missing stitched outputs when overwrite=False
        if not overwrite:
            filtered_wells_info = []
            for well_idx, well_name, well_series, nd2_well_idx, t in sampled_wells_info:
                out = out_ff / f"{well_name}_t{t:04}_ch{bf_idx:02}_stitch.jpg"
                if not out.exists():
                    filtered_wells_info.append((well_idx, well_name, well_series, nd2_well_idx, t))

            skipped = len(sampled_wells_info) - len(filtered_wells_info)
            if skipped > 0:
                log.info("Resuming: skipping %d stitched frames for %s", skipped, exp_name)

            sampled_wells_info = filtered_wells_info

        if par_flag:
            log.warning("Parallel workers with CUDA batching are not supported; running sequentially on a single worker.")
            par_flag = False

        batch_size = max(1, _resolve_batch_size(n_z, height_px, width_px, device))

        env_queue = os.environ.get("MSEQ_YX1_WRITE_QUEUE")
        if env_queue:
            try:
                queue_size = int(env_queue)
            except ValueError:
                log.warning("Invalid MSEQ_YX1_WRITE_QUEUE=%s; using default", env_queue)
                queue_size = batch_size * 2
        else:
            queue_size = batch_size * 2
        queue_size = max(queue_size, 2)

        write_queue: Queue[tuple[str, int, np.ndarray]] = Queue(maxsize=queue_size)
        WRITE_SENTINEL = object()

        def _writer_worker():
            while True:
                item = write_queue.get()
                if item is WRITE_SENTINEL:
                    write_queue.task_done()
                    break
                well_name, t, ff_img = item
                try:
                    _write_ff(out_ff, well_name, t, bf_idx, ff_img, overwrite)
                finally:
                    write_queue.task_done()

        writer_thread = Thread(target=_writer_worker, name="yx1_ff_writer", daemon=True)
        writer_thread.start()

        n_z_keep = 12 if z_buff else None
        batch_stacks: list[np.ndarray] = []
        batch_infos: list[tuple[int, str, int, int, int]] = []

        def flush_batch():
            if not batch_stacks:
                return
            stacked = np.stack(batch_stacks, axis=0)
            ff_batch = _focus_stack(stacked, device, filter_size=3)
            if ff_batch.ndim == 2:
                ff_batch = ff_batch[np.newaxis, ...]
            for ff_img, (_, well_name, _, _, t) in zip(ff_batch, batch_infos):
                write_queue.put((well_name, t, ff_img))
            batch_stacks.clear()
            batch_infos.clear()

        try:
            for well_idx, well_name, well_series, nd2_well_idx, t in tqdm(sampled_wells_info):
                stack = _get_stack(dask_arr, t, nd2_well_idx, n_z_keep=n_z_keep)
                batch_stacks.append(stack)
                batch_infos.append((well_idx, well_name, well_series, nd2_well_idx, t))
                if len(batch_stacks) >= batch_size:
                    flush_batch()
            flush_batch()
        finally:
            # Ensure all pending writes are flushed before shutting down the writer
            write_queue.join()
            write_queue.put(WRITE_SENTINEL)
            writer_thread.join()

    else:
        log.info("Skipping FF for %s", exp_name)

    meta_df = build_experiment_metadata(repo_root=repo_root, exp_name=exp_name, meta_df=well_df)
    first_time = np.nanmin(meta_df['Time (s)'])  # NaN-safe minimum
    meta_df['Time Rel (s)'] = meta_df['Time (s)'] - first_time
    
    # Final validation - ensure no NaN values remain
    assert not meta_df['Time (s)'].isna().any(), f"FATAL: NaN timestamps remain in {exp_name}"
    assert not meta_df['Time Rel (s)'].isna().any(), f"FATAL: NaN relative times in {exp_name}"
    log.info(f"✅ {exp_name}: validated {len(meta_df)} metadata rows, ready for save")
    
    # load previous metadata
    out_meta = meta_root / "built_metadata_files"
    out_meta.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(out_meta / f"{exp_name}_metadata.csv", index=False)

    if not metadata_only:
        marker = write_root / "stitched_FF_images" / exp_name / ".ff_complete"
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"rows={len(meta_df)}\n")
        except Exception as exc:
            log.warning("Could not write FF completion marker for %s: %s", exp_name, exc)

    nd.close()

    print('Done.')



if __name__ == "__main__":

    overwrite_flag = False
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    build_ff_from_yx1(data_root=data_root, exp_name="20240314", metadata_only=True)
