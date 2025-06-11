# script to define functions_folder for loading and standardizing fish movies
import os
import numpy as np
import skimage.io as skio
from tqdm.contrib.concurrent import process_map 
from functools import partial
from typing import List, Tuple, Union, Optional
import glob2 as glob
import logging
import cv2
from stitch2d import StructuredMosaic
import json
from tqdm import tqdm
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, Any, Union
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

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

# def _findnth(haystack, needle, n):
#     parts = haystack.split(needle, n+1)
#     if len(parts)<=n+1:
#         return -1
#     return len(haystack)-len(parts[-1])-len(needle)

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


def focus_stack_maxlap(stack: np.ndarray,
                       lap_ksize: int = 3,
                       gauss_ksize: int = 3) -> np.ndarray:
    """
    Fast Laplacian-max focus stack.

    Parameters
    ----------
    stack : (Z, Y, X) uint8/uint16 array of slices
    lap_ksize, gauss_ksize : odd ints, kernel sizes for Laplacian & blur
    """
    if stack.ndim != 3:
        raise ValueError("stack must be Z×Y×X")

    # gaussian blur (vectorised)
    blur = np.stack(
        [cv2.GaussianBlur(z, (gauss_ksize, gauss_ksize), 0) for z in stack],
        axis=0
    )
    lap = np.abs(
        np.stack(
            [cv2.Laplacian(z, cv2.CV_32F, ksize=lap_ksize) for z in blur],
            axis=0
        )
    )

    best   = lap.argmax(axis=0)                             # (Y, X) ints
    y_idx, x_idx = np.indices(best.shape)
    return stack[best, y_idx, x_idx]      

### Helper functions
def _load_images(indices: List[int], file_list: List[str]) -> List[np.ndarray]:
    return [skio.imread(file_list[i]) for i in indices]

def _write_ff(ff: np.ndarray, out_dir: Path, pos_string: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    skio.imsave(out_dir / f"im_{pos_string}.png", ff, check_contrast=False)


def process_well(
    w: int,
    well_list: list[str],
    cytometer_flag: bool,
    ff_root: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Drop-in replacement: returns the same metadata DF,
    writes full-focus JPGs to `ff_root`.
    """
    well_dir  = Path(well_list[w])
    well_name = well_dir.name[-4:]                # e.g. 'A01a'
    well_conv = sorted(well_dir.glob("_*"))[0].name[-3:]

    # handle optional 'P*' and 'T*' layers
    pos_dirs  = sorted(well_dir.glob("P*")) or [well_dir]
    meta_rows = []

    for p, pos_dir in enumerate(pos_dirs):
        time_dirs = sorted(pos_dir.glob("T*")) or [pos_dir]

        for time_dir in time_dirs:
            t_idx = int(time_dir.name[1:]) if time_dir.stem.startswith("T") else 0
            im_files = sorted(time_dir.glob("*CH*"))

            # deduce sub-positions inside one folder
            sub_pos = (
                np.ones(len(im_files)) if cytometer_flag
                else [int(f.name.split(well_name)[1][1:6]) for f in im_files]
            )
            for pi in np.unique(sub_pos).astype(int):
                pos_idx   = np.where(sub_pos == pi)[0]
                pos_str   = f"p{(p if cytometer_flag else pi):04}"
                out_dir   = Path(ff_root) / f"ff_{well_conv}_t{t_idx:04}"
                out_file  = out_dir / f"im_{pos_str}.png"

                if out_file.exists() and not overwrite:
                    if p == t_idx == w == 0:
                        print("Skipping existing PNGs (set overwrite=True to rebuild).")
                    continue

                stack = np.stack(_load_images(pos_idx, im_files), axis=0)
                well_res = float(scrape_keyence_metadata(im_files[0])["Width (um)"]) / \
                           scrape_keyence_metadata(im_files[0])["Width (px)"]
                ksize = int(np.floor(26 / well_res / 2) * 2 + 1)
                ff_img = focus_stack_maxlap(stack, lap_ksize=ksize, gauss_ksize=ksize)

                _write_ff(ff_img, out_dir, pos_str)

                # metadata only once per (well, time point)
                meta = scrape_keyence_metadata(im_files[0])
                meta.update({"well": well_conv,
                             "time_string": f"T{t_idx:04}",
                             "time_int": t_idx})
                meta_rows.append(meta)

    return pd.DataFrame(meta_rows)


def _valid_acq_dirs(root: Path, dir_list: list[str] | None) -> list[Path]:
    if dir_list is not None:
        dirs = [root / d for d in dir_list]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]
    return [d for d in dirs if "ignore" not in d.name]

#############################################################################
#  A.  MASTER-PARAM SAMPLER
#############################################################################
def _coords_to_array(coords: dict[int, List[float]], n_tiles: int) -> np.ndarray:
    arr = np.full((n_tiles, 2), np.nan)
    for k, v in coords.items():
        arr[k, :] = v
    return arr

def build_master_params(
    sample_dirs : List[Path],
    *,
    orientation : str,
    n_tiles     : int,
    outfile     : Path,
) -> None:
    """
    Randomly samples folders, runs stitch2d alignment, and stores the
    *median* tile coordinates as a JSON prior.  Called ONCE per acquisition.
    """
    all_coords = []
    for fld in sample_dirs:
        try:
            m = StructuredMosaic(str(fld), dim=n_tiles,
                                 origin="upper left", direction=orientation,
                                 pattern="raster")
            m.align()
            if len(m.params["coords"]) == n_tiles:          # good alignment
                all_coords.append(_coords_to_array(m.params["coords"], n_tiles))
        except Exception:
            log.debug("Sampling failed in %s", fld)

    if not all_coords:
        log.warning("No good samples – master params not written")
        return

    med = np.nanmedian(np.stack(all_coords), axis=0)        # (n_tiles, 2)
    prior = {"coords": {i: med[i].tolist() for i in range(n_tiles)}}

    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(prior))
    log.info("Wrote master params to %s", outfile)


#############################################################################
#  B.  ONE-FOLDER STITCHER
#############################################################################
_SHIFT_TOL = 2.0  # px tolerance per axis for “good” translation

def _coords_good(coords: dict[int, List[float]],
                 n_tiles: int,
                 orientation: str) -> bool:
    if len(coords) != n_tiles:
        return False
    arr = _coords_to_array(coords, n_tiles)     # (n,2)

    axis = 1 if orientation == "vertical" else 0
    # ensure successive tiles don’t jump more than +-2 px laterally
    return np.nanmax(np.abs(np.diff(arr[:, axis]))) <= _SHIFT_TOL


def stitch_experiment(
    idx            : int,
    ff_folders     : List[str],
    ff_tile_dir    : str,
    out_dir        : str,
    overwrite      : bool,
    size_factor    : float,
    orientation    : str = "vertical",
) -> dict:
    """
    Replaces your old function 1-for-1.
    • Still uses StructuredMosaic.
    • Will *only* fall back to master_params.json when alignment is missing
      or clearly exceeds the tolerance.
    """
    ff_path  = Path(ff_folders[idx])
    out_png  = Path(out_dir) / (ff_path.name[3:] + "_stitch.png")
    if out_png.exists() and not overwrite:
        return {}

    n_tiles = len(list(ff_path.glob("*.png")))
    target  = {2: np.array([800,  630]),
               3: np.array([1140, 630]) if orientation == "vertical"
                  else np.array([1140, 480])}[n_tiles] * size_factor
    target  = tuple(target.astype(int))

    mosaic = StructuredMosaic(str(ff_path), dim=n_tiles,
                              origin="upper left", direction=orientation,
                              pattern="raster")
    try:
        mosaic.align()
    except Exception:
        log.debug("Primary alignment failed in %s", ff_path)

    if not _coords_good(mosaic.params["coords"], n_tiles, orientation):
        prior_file = Path(ff_tile_dir) / "master_params.json"
        if prior_file.exists():
            mosaic.load_params(str(prior_file))
            mosaic.reset_tiles()
            log.info("Fallback to master params for %s", ff_path)
        else:
            raise RuntimeError(f"No valid alignment and no prior for {ff_path}")

    mosaic.smooth_seams()
    stitched = mosaic.stitch()
    if orientation == "horizontal":
        stitched = stitched.T

    # trim & invert
    stitched = trim_to_shape(stitched, target)
    maxv = np.iinfo(stitched.dtype).max
    stitched = maxv - stitched

    out_png.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(out_png, stitched, check_contrast=False)
    return {}

def build_ff_from_keyence(data_root, *, n_workers=4,
                          overwrite=False, dir_list=None, write_dir=None):
    
    par_flag = n_workers > 1

    RAW   = Path(data_root) / "raw_image_data" / "keyence"
    BUILT = Path(write_dir or data_root) / "built_image_data" / "keyence"
    META  = Path(data_root) / "metadata" / "built_metadata_files"
    acq_dirs = _valid_acq_dirs(RAW, dir_list)

    for acq in tqdm(acq_dirs, desc="Building FF"):

        ff_dir = BUILT / "FF_images" / acq.name
        ff_dir.mkdir(parents=True, exist_ok=True)

        well_list = sorted((acq.glob("XY*") or acq.glob("W0*")))
        cytometer_flag = not any(acq.glob("XY*"))

        # print(f'Building full-focus images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        meta_frames = []
        if not par_flag:
            for w in tqdm(range(len(well_list))):
                temp_df = process_well(w, well_list, cytometer_flag, ff_dir, overwrite)
                meta_frames.append(temp_df)
        else:
            metadata_df_temp = process_map(partial(process_well, well_list=well_list, cytometer_flag=cytometer_flag, 
                                                                        ff_dir=ff_dir, overwrite_flag=overwrite), 
                                        range(len(well_list)), max_workers=n_workers)
            meta_frames += metadata_df_temp
            
        # process_well(w, well_list, cytometer_flag, ff_dir, overwrite_flag=False)
        # meta_frames = pmap(process_well, range(len(well_list)), (well_list, cytometer_flag, ff_dir, depth_dir, ch_to_use, overwrite_flag), rP=0.5)
        # if len(meta_frames) > 0:
        #     meta_frames = pd.concat(meta_frames)
        #     first_time = np.min(meta_frames['Time (s)'].copy())
        #     meta_frames['Time Rel (s)'] = meta_frames['Time (s)'] - first_time
        # else:
        #     meta_frames = []

        # load previous metadata
        df_path = META / f"{acq.name}_metadata.csv"
        if not overwrite and os.path.isfile(df_path):
            df_prev = pd.read_csv(df_path)
            df_prev = [df_prev]
        else:
            df_prev = []

        if meta_frames:
            df = pd.concat(meta_frames + df_prev)
            df.drop_duplicates(subset=["well", "time_string"])
            df["Time Rel (s)"] = df["Time (s)"] - df["Time (s)"].min()
            df.to_csv(META / f"{acq.name}_metadata.csv", index=False)

    print('Done.')


def stitch_ff_from_keyence(data_root, n_workers=4, overwrite_flag=False, n_stitch_samples=15, dir_list=None, write_dir=None, orientation_list=None):
    
    par_flag = n_workers > 1

    RAW_ROOT   = Path(data_root) / "raw_image_data" / "keyence"
    WRITE_ROOT = Path(write_dir or data_root)
    META_ROOT  = Path(data_root) / "metadata" / "built_metadata_files"

    # --- discover acquisition folders --------------------------------------
    acq_dirs = (
        [RAW_ROOT / d for d in dir_list]
        if dir_list
        else [p for p in RAW_ROOT.iterdir() if p.is_dir() and "ignore" not in p.name]
    )
    acq_dirs = sorted(acq_dirs)

    if orientation_list is None:
        orientation_list = ["vertical"] * len(acq_dirs)
    if len(orientation_list) != len(acq_dirs):
        raise ValueError("orientation_list length must match dir_list length")

    # -----------------------------------------------------------------------
    for acq, orientation in zip(acq_dirs, orientation_list):
        # inside stitch_ff_from_keyence() – one acquisition folder “acq”
        ff_tile_root = WRITE_ROOT / "built_image_data" / "keyence" / "FF_images" / acq.name
        stitch_root  = WRITE_ROOT / "built_image_data" / "stitched_FF_images" / acq.name
        stitch_root.mkdir(parents=True, exist_ok=True)

        metadata_path = os.path.join(META_ROOT, acq.name + '_metadata.csv')
        metadata_df = pd.read_csv(metadata_path,)
        size_factor = metadata_df["Width (px)"].iloc[0] / 640

        ff_folders = sorted(ff_tile_root.glob("ff_*"))
        n_tiles    = len(list(ff_folders[0].glob("*.png")))  # assume constant

        # ----- build (or reuse) master params once -------------------------------
        prior_file = ff_tile_root / "master_params.json"
        if overwrite_flag or not prior_file.exists():
            sample_dirs = np.random.choice(ff_folders,
                                        min(n_stitch_samples, len(ff_folders)),
                                        replace=False)
            build_master_params(list(sample_dirs), orientation=orientation,
                                n_tiles=n_tiles, outfile=prior_file)

        # print(f'Stitching images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        # Call parallel function to stitch images
        if not par_flag:
            for f in tqdm(range(len(ff_folders))):
                stitch_experiment(f, ff_folders, str(ff_tile_root), str(stitch_root), overwrite_flag, size_factor, orientation=orientation)

        else:
            process_map(partial(stitch_experiment, ff_folder_list=ff_folders, ff_tile_dir=str(ff_tile_root), 
                                stitch_ff_dir=str(stitch_root), overwrite_flag=overwrite_flag, size_factor=size_factor, orientation=orientation),
                                        range(len(ff_folders)), max_workers=n_workers, chunksize=1)



if __name__ == "__main__":

    overwrite_flag = True

    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = ["20230525", "20231207"]
    # build FF images
    build_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list)
    # stitch FF images
    stitch_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list)