from pathlib import Path
import sys

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

# script to define functions_folder for loading and standardizing fish movies
import os
import numpy as np
import json
from tqdm.contrib.concurrent import process_map 
from functools import partial
from typing import List
import logging
from stitch2d import StructuredMosaic
import json
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any, Union
from pathlib import Path
from glob2 import glob
import skimage
import skimage.io as skio
from skimage import exposure
from src.build.export_utils import scrape_keyence_metadata, trim_to_shape, to_u8_adaptive, valid_acq_dirs, im_rescale, LoG_focus_stacker
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)
logging.getLogger("stitch2d").setLevel(logging.ERROR) 

SHIFT_TOL = {               # px tolerances before we fall back
    2: {"vertical": 1, "horizontal": 1},
    3: {"vertical": 2, "horizontal": 2},
}


def align_with_qc(mosaic, n_tiles, orientation):
    """
    • try mosaic.align(); ignore exceptions
    • test #coords and intra-tile shift against thresholds
    • load master params if any test fails
    """

    fallback_flag = False
    try:
        mosaic.align()
    except Exception as e:
        log.debug("align() raised %s – will fall back", e)
        fallback_flag = True
        return fallback_flag

    coords = mosaic.params.get("coords", {})
    if len(coords) != n_tiles:           # lost a tile
        log.debug("Only %d/%d tiles aligned – falling back", len(coords), n_tiles)
        fallback_flag = True
        return fallback_flag

    # shift QC: Δx for vertical stacks, Δy for horizontal
    arr = np.array([coords[i] for i in range(n_tiles)])
    axis = 1 if orientation == "vertical" else 0
    if np.abs(arr[:, axis]).max() > SHIFT_TOL[n_tiles][orientation]:
        log.debug("Shifts exceed tolerance – falling back")
        fallback_flag = True
        return fallback_flag


def flatten_master_params(master_json):
    with open(master_json) as fh:
        d = json.load(fh)
    # Move 'shape' to the root if nested under 'metadata'
    if "metadata" in d and "shape" in d["metadata"]:
        d["shape"] = d["metadata"]["shape"]
    if "coords" in d and isinstance(next(iter(d["coords"])), str):
        # Convert keys to int if needed
        d["coords"] = {int(k): v for k, v in d["coords"].items()}
    # Save to a temp file or overwrite
    with open(master_json, "w") as fh:
        json.dump(d, fh)
    return master_json


### Helper functions
def _load_images(indices: List[int], file_list: List[str]) -> List[np.ndarray]:
    return [skio.imread(file_list[i]) for i in indices]

def _write_ff(ff: np.ndarray, out_dir: Path, pos_string: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if ff.dtype == np.uint16:
        ff = skimage.util.img_as_ubyte(ff)
    # skio.use_plugin("imageio")
    skio.imsave(out_dir / f"im_{pos_string}.jpg", ff, check_contrast=False)


def process_well(
    w: int,
    well_list: list[str],
    cytometer_flag: bool,
    device: str,
    ff_root: str | Path,
    ff_filter_size: int=3,
    overwrite: bool = False) -> pd.DataFrame:
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
            # First loop: load all positions into z-stacks
            stack_list = []
            for pi in np.unique(sub_pos).astype(int):
                pos_idx   = np.where(sub_pos == pi)[0]
                pos_str   = f"p{(p if cytometer_flag else pi):04}"

                stack = np.stack(_load_images(pos_idx, im_files), axis=0)

                stack_list.append(stack)

            # Normalize intensity consistently across stack
            im_cb = np.stack(stack_list)
            _, lo, hi = im_rescale(im_cb)

            # Second loop: FF-project and save
            for pi in np.unique(sub_pos).astype(int):
                pos_idx   = np.where(sub_pos == pi)[0]
                pos_str   = f"p{(p if cytometer_flag else pi):04}"
                out_dir   = Path(ff_root) / f"ff_{well_conv}_t{t_idx:04}"
                out_file  = out_dir / f"im_{pos_str}.jpg"

                if out_file.exists() and not overwrite:
                    if p == t_idx == w == 0:
                        print("Skipping existing JPGs (set overwrite=True to rebuild).")
                    continue
                
                # normalize
                stack = np.stack(_load_images(pos_idx, im_files), axis=0)
                norm = exposure.rescale_intensity(stack, in_range=(lo, hi))
                norm = norm.astype(np.float32)
                tensor = torch.from_numpy(norm).to(device)          

                # ksize = int(np.floor(26 / well_res / 2) * 2 + 1)
                ff_t, _ = LoG_focus_stacker(tensor, filter_size=ff_filter_size, device=device)
                arr = ff_t.numpy()
                if stack.dtype == np.uint16:
                    arr_clipped = np.clip(arr, 0, 65535)
                    ff_i = arr_clipped.astype(np.uint16)
                elif stack.dtype == np.uint8:
                    arr_clipped = np.clip(arr, 0, 255)
                    ff_i = arr_clipped.astype(np.uint8)

                _write_ff(ff_i, out_dir, pos_str)

                # metadata only once per (well, time point)
                meta = scrape_keyence_metadata(im_files[0])
                meta.update({"well": well_conv,
                             "time_string": f"T{t_idx:04}",
                             "time_int": t_idx})
                meta_rows.append(meta)

    return pd.DataFrame(meta_rows)

#############################################################################
#  A.  MASTER-PARAM SAMPLER
#############################################################################

def build_master_params(
    sample_dirs,           # list of Path or str: folders to sample
    orientation,           # str: "vertical" or "horizontal"
    n_tiles,               # int: how many tiles
    outfile,               # Path: destination for master_params.json
):
    align_array = []
    last_good_mosaic = None

    for fld in sample_dirs:
        try: 
            mosaic = StructuredMosaic(
                str(fld), dim=n_tiles,
                origin="upper left", direction=orientation,
                pattern="raster"
            )
            mosaic.align()
            if len(mosaic.params["coords"]) == n_tiles:
                coords = mosaic.params["coords"]
                arr = np.array([coords[i] for i in range(n_tiles)])
                align_array.append(arr)
                last_good_mosaic = mosaic
        except Exception as e:
            print(f"Stitch2d alignment failed for {fld}: {e}")

    if not align_array or last_good_mosaic is None:
        print("No good samples – master params not written")
        return

    # median across all sampled alignments
    med_coords = np.nanmedian(np.stack(align_array), axis=0)
    coords_dict = {i: med_coords[i].tolist() for i in range(n_tiles)}

    # Use a real, full params dict as template
    master_params = last_good_mosaic.params.copy()
    master_params["coords"] = coords_dict

    # (optional) If you want to remove volatile keys, do it here:
    # for key in ["filenames", ...]: master_params.pop(key, None)

    # Write valid JSON for stitch2d
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(master_params))
    print(f"Wrote master params to {outfile}")


#############################################################################
#  B.  ONE-FOLDER STITCHER
#############################################################################

def stitch_experiment(
    idx            : int,
    ff_folders     : List[str],
    ff_tile_dir    : str,
    out_dir        : str,
    overwrite      : bool,
    size_factor    : float,
    orientation    : str = "vertical",
) -> dict:

    ff_path  = Path(ff_folders[idx])
    out_png  = Path(out_dir) / (ff_path.name[3:] + "_stitch.jpg")
    out_params  = Path(ff_path) / "params.json"
    if out_png.exists() and not overwrite:
        return {}

    n_tiles = len(list(glob(str(ff_path) + "/*.jpg")))
    target  = {2: np.array([800,  630]),
               3: np.array([1140, 630]) if orientation == "vertical"
                  else np.array([1140, 480])}[n_tiles] * size_factor
    target  = tuple(target.astype(int))

    mosaic = StructuredMosaic(str(ff_path), dim=n_tiles,
                              origin="upper left", direction=orientation,
                              pattern="raster")
    
    fallback_flag = align_with_qc(mosaic,
              n_tiles=n_tiles,
              orientation=orientation)
    
    if fallback_flag:
        master_json=str(Path(ff_tile_dir) / "master_params.json")
        # flatten_master_params(master_json)
        # Now:
        mosaic.load_params(master_json)

    mosaic.reset_tiles()

   

    mosaic.smooth_seams()
    mosaic.save_params(out_params)
    stitched = mosaic.stitch()
    if orientation == "horizontal":
        stitched = stitched.T

    # trim & invert
    stitched = trim_to_shape(stitched, target)
    # maxv = np.iinfo(stitched.dtype).max
    # stitched = maxv - stitched

    out_png.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(out_png, stitched, check_contrast=False)
    return {}


def build_ff_from_keyence(data_root, *, n_workers=4,
                          overwrite=False, dir_list=None, write_dir=None):
    
    par_flag = n_workers > 1

    RAW   = Path(data_root) / "raw_image_data" / "keyence"
    BUILT = Path(write_dir or data_root) / "built_image_data" / "keyence"
    META  = Path(data_root) / "metadata" / "built_metadata_files"
    acq_dirs = valid_acq_dirs(RAW, dir_list)

    # get compute device to use
    device = (
                "cuda"
                if torch.cuda.is_available() #and (not par_flag)
                else "cpu"
            )

    if device == "cpu":
        print("Warning: using CPU. This may be quite slow. GPU recommended.")

    # iterate through directories
    for acq in tqdm(acq_dirs, desc="Building FF"):

        ff_dir = BUILT / "FF_images" / acq.name
        ff_dir.mkdir(parents=True, exist_ok=True)

        well_list = sorted((acq.glob("XY*") or acq.glob("W0*")))
        cytometer_flag = not any(acq.glob("XY*"))

        # print(f'Building full-focus images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        meta_frames = []
        run_process_well = partial(process_well, well_list=well_list, cytometer_flag=cytometer_flag, device=device,
                                                                        ff_root=ff_dir, overwrite=overwrite)

        if not par_flag:
            for w in tqdm(range(len(well_list))):
                temp_df = run_process_well(w)
                meta_frames.append(temp_df)
        else:
            metadata_df_temp = process_map(run_process_well, range(len(well_list)), max_workers=n_workers)
            meta_frames += metadata_df_temp
            
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


def stitch_ff_from_keyence(data_root, n_workers=4, overwrite=False, n_stitch_samples=50, dir_list=None, write_dir=None, orientation_list=None):
    
    par_flag = n_workers > 1

    RAW   = Path(data_root) / "raw_image_data" / "keyence"
    WRITE_ROOT = Path(write_dir or data_root)
    META_ROOT  = Path(data_root) / "metadata" / "built_metadata_files"

    # --- discover acquisition folders --------------------------------------
    acq_dirs = valid_acq_dirs(RAW, dir_list)

    if orientation_list is None:
        orientation_list = ["horizontal"] * len(acq_dirs)
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
        n_tiles    = len(list(ff_folders[0].glob("*.jpg")))  # assume constant

        # ----- build (or reuse) master params once -------------------------------
        prior_file = ff_tile_root / "master_params.json"
        if overwrite or not prior_file.exists():
            sample_dirs = np.random.choice(ff_folders,
                                        min(n_stitch_samples, len(ff_folders)),
                                        replace=False)
            build_master_params(list(sample_dirs), orientation=orientation,
                                n_tiles=n_tiles, outfile=prior_file)

        # print(f'Stitching images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        # Call parallel function to stitch images
        if not par_flag:
            for f in tqdm(range(len(ff_folders))):
                stitch_experiment(f, ff_folders, str(ff_tile_root), str(stitch_root), overwrite, size_factor, orientation=orientation)

        else:
            process_map(partial(stitch_experiment, ff_folder_list=ff_folders, ff_tile_dir=str(ff_tile_root), 
                                stitch_ff_dir=str(stitch_root), overwrite=overwrite, size_factor=size_factor, orientation=orientation),
                                        range(len(ff_folders)), max_workers=n_workers, chunksize=1)



if __name__ == "__main__":

    overwrite = True

    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = ["20230525", "20231207"]
    # build FF images
    build_ff_from_keyence(data_root, overwrite=overwrite, dir_list=dir_list)
    # stitch FF images
    stitch_ff_from_keyence(data_root, overwrite=overwrite, dir_list=dir_list)