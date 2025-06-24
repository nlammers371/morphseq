from pathlib import Path
import sys

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

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
from typing import Dict, Any, Union, Literal
from pathlib import Path
from glob2 import glob
import skimage
import skimage.io as skio
from skimage import exposure, util
from src.build.export_utils import (trim_to_shape, _get_keyence_tile_orientation, save_images_parallel, LoG_focus_stacker, 
                                    get_n_cpu_workers, get_n_workers_for_pipeline, estimate_batch_sizes, scrape_keyence_metadata)
from src.build.data_classes import MultiTileZStackDataset
from torch.utils.data import DataLoader

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
# def _load_images(indices: List[int], file_list: List[str]) -> List[np.ndarray]:
#     return [skio.imread(file_list[i]) for i in indices]

# def _write_ff(ff: np.ndarray, out_dir: Path, pos_string: str):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     if ff.dtype == np.uint16:
#         ff = skimage.util.img_as_ubyte(ff)
#     # skio.use_plugin("imageio")
#     skio.imsave(out_dir / f"im_{pos_string}.jpg", ff, check_contrast=False)

# build path dictionary
def get_image_paths(
    well_list: list[str],
    cytometer_flag: bool):

    sample_dict = {}
    exp_name = os.path.basename(os.path.dirname(well_list[0]))

    meta_rows = []
    pixel_size_um = None
    for w, well_dir in enumerate(tqdm(well_list, "Building metadata...")):

        well_name = Path(well_dir).name[-4:]                # e.g. 'A01a'
        well_conv = sorted(well_dir.glob("_*"))[0].name[-3:]

        # handle optional 'P*' and 'T*' layers
        pos_dirs  = sorted(well_dir.glob("P*")) or [well_dir]

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
                for pi in np.unique(sub_pos).astype(int):
                    # update metadata once per well-timepoint
                    p_iter = p if cytometer_flag else pi-1
                    if p_iter == 0:
                        meta = scrape_keyence_metadata(im_files[0])
                        meta.update({"well": well_conv,
                                    "time_string": f"T{t_idx:04}",
                                    "time_int": t_idx})
                        meta_rows.append(meta)

                        pixel_size_um = meta['Width (um)'] / meta['Width (px)']

                    pos_idx   = np.where(sub_pos == pi)[0]
                    key = f"{well_conv}_t{t_idx:04}"

                    if key in sample_dict:
                        tile_zpaths = sample_dict[key]["tile_zpaths"]
                        tile_zpaths.append([str(im_files[idx]) for idx in pos_idx])
                        sample_dict[key]["tile_zpaths"] = tile_zpaths
                    else:
                        sample_dict[key] = {
                                            "date": exp_name,
                                            "well": well_conv,
                                            "time_id": t_idx,
                                            "tile_zpaths": [[str(im_files[idx]) for idx in pos_idx]],
                                            "pixel_size_um": pixel_size_um
                                        }
                        
                    

                

    # convert dict to a list                
    return list(sample_dict.values()), pd.DataFrame(meta_rows)
    

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
    orientation    : str,
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


def build_ff_from_keyence(data_root: Path | str, 
                          exp_name: str,
                          ff_filter_res_um: float=3.0,
                          overwrite: bool=False):
    
    # figure out how to utilize compute resources
    # 1) decide how many threads to write images
    n_write_threads = get_n_cpu_workers(frac=0.5)

    # 2) decide how many workers to load/process stacks in parallel
    n_load_workers  = get_n_workers_for_pipeline()


    # get path info
    RAW   = Path(data_root) / "raw_image_data" / "Keyence"
    BUILT = Path(data_root) / "built_image_data" / "Keyence"
    META  = Path(data_root) / "metadata" / "built_metadata_files"
    os.makedirs(BUILT, exist_ok=True)
    os.makedirs(META, exist_ok=True)

    # iterate through directories
    # for acq in tqdm(acq_dirs, desc="Building FF"):
    ff_dir = BUILT / "FF_images" / exp_name
    raw_dir = RAW / exp_name 
    ff_dir.mkdir(parents=True, exist_ok=True)

    well_list = sorted((raw_dir.glob("XY*") or raw_dir.glob("W0*")))
    cytometer_flag = not any(raw_dir.glob("XY*"))
    if cytometer_flag:
        well_list = sorted(raw_dir.glob("W0*"))
    else:
        well_list = sorted(raw_dir.glob("XY*"))
    
    # call function to walk through directories and compile list of image paths
    sample_list, meta_df = get_image_paths(well_list=well_list, cytometer_flag=cytometer_flag)


    ds = MultiTileZStackDataset(sample_list)

    # check image size to gauge memory footprint
    im0 = ds[0]["data"]
    sample_bytes = int(im0.nbytes)
    gpu_bs, cpu_bs = estimate_batch_sizes(sample_bytes)
    # batch_size = gpu_bs if gpu_bs > 0 else cpu_bs
    if gpu_bs > 0:
        device = "cuda"
    else:
        device = "cpu"


    loader = DataLoader(ds,
                        batch_size=4,            # only ever load 1 sample at a time
                        num_workers=2,           # only one worker process
                        pin_memory=True,
                        prefetch_factor=1,       # only prefetch 1 batch
                        persistent_workers=False)
    
    
    for batch in tqdm(loader, "Doing FF projections..."):
        # batch.shape == (batch_size, n_tiles, Z, H, W)
       
        # now you can FF-project each tile—for instance by reshaping:
        data = batch["data"]
        meta_dict = batch["path"]
        data = data.to(device, non_blocking=True)
        b, t, z, h, w = data.shape
        flat = data.view(b*t, z, h, w) # treat each tile as its own volume

        # calculate size of filter to use
        pixel_size_um = meta_dict["pixel_size_um"][0].numpy()
        filter_rad = max(1, int(round(ff_filter_res_um/ pixel_size_um)))
        filter_size = 2 * filter_rad + 1
        ff, _   = LoG_focus_stacker(flat, filter_size=filter_size) 
        
        # prepare to save              # returns shape (b*t, H, W)
        ff   = ff.view(b, t, h, w).cpu().numpy()
        ff_list = [util.img_as_ubyte(ff[b][p]) for b in range(ff.shape[0]) for p in range(ff.shape[1])]

        # build lists file and folder names
        
        n_tiles = len(meta_dict["tile_zpaths"])
        well_names = meta_dict["well"]
        time_indices = meta_dict["time_id"]
        all_paths = [
            Path(ff_dir) / f"ff_{w}_t{t:04}" / f"im_p{p:04}.jpg"
            for w, t in zip(well_names, time_indices)
            for p in range(n_tiles)
            ]
        
        
        # us parallel processing to save images
        save_images_parallel(images=ff_list,
                             paths=all_paths,
                             n_workers=n_write_threads)

        
    # load previous metadata
    df_path = META / f"{exp_name}_metadata.csv"
    if not overwrite and os.path.isfile(df_path):
        df_prev = pd.read_csv(df_path)
        df_prev = [df_prev]
    else:
        df_prev = []

    if meta_df.shape[0] > 0:
        meta_df.drop_duplicates(subset=["well", "time_string"])
        meta_df["Time Rel (s)"] = meta_df["Time (s)"] - meta_df["Time (s)"].min()
        meta_df.to_csv(META / f"{exp_name}_metadata.csv", index=False)

    print('Done.')


def stitch_ff_from_keyence(data_root: str | Path, 
                           exp_name: str, 
                           n_workers: int=4, 
                           overwrite: bool=False, 
                           n_stitch_samples: int=50):
    
    par_flag = n_workers > 1
    orientation = _get_keyence_tile_orientation(exp_name)
    # RAW   = Path(data_root) / "raw_image_data" / "keyence"
    WRITE_ROOT = Path(data_root)
    META_ROOT  = Path(data_root) / "metadata" / "built_metadata_files"

    # # --- discover acquisition folders --------------------------------------
    # acq_dirs = valid_acq_dirs(RAW, dir_list)

    # if orientation_list is None:
    #     orientation_list = ["horizontal"] * len(acq_dirs)
    # if len(orientation_list) != len(acq_dirs):
    #     raise ValueError("orientation_list length must match dir_list length")

    # -----------------------------------------------------------------------
    # for acq, orientation in zip(acq_dirs, orientation_list):
        # inside stitch_ff_from_keyence() – one acquisition folder “acq”
    ff_tile_root = WRITE_ROOT / "built_image_data" / "keyence" / "FF_images" / exp_name
    stitch_root  = WRITE_ROOT / "built_image_data" / "stitched_FF_images" / exp_name
    stitch_root.mkdir(parents=True, exist_ok=True)

    metadata_path = os.path.join(META_ROOT, exp_name + '_metadata.csv')
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
    
    build_ff_from_keyence(data_root=data_root, exp_name="20250529_36hpf_ctrl_atf6")