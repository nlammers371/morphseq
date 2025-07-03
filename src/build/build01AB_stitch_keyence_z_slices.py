from pathlib import Path
import sys

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

# script to define functions_folder for loading and standardizing fish movies
import os
import numpy as np
from PIL import Image
from tqdm.contrib.concurrent import process_map 
from functools import partial
from src.functions.utilities import path_leaf
from tqdm import tqdm
import glob2 as glob
from stitch2d import StructuredMosaic
from tqdm import tqdm
import skimage.io as io
import pandas as pd
from stitch2d.tile import Tile # OpenCVTile as
from pathlib import Path
import logging 
import skimage
from src.build.export_utils import trim_to_shape, to_u8_adaptive, valid_acq_dirs, _get_keyence_tile_orientation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)
logging.getLogger("stitch2d").setLevel(logging.ERROR) 
#####################
# Helpers

# -------------------------------------------------------------
def list_time_dirs(pos_dir: Path) -> list[Path]:
    """Return [T0001/, T0002/, …] or [pos_dir] if no sub-folders."""
    pos_dir = Path(pos_dir)
    t_dirs = sorted(p for p in pos_dir.glob("T*") if p.is_dir())
    return t_dirs or [pos_dir]

def list_z_images(time_dir: Path, ch_tag="CH") -> list[Path]:
    """All Z-slice images for a given time directory."""
    return sorted(time_dir.glob(f"*{ch_tag}*"))

def split_subpos(im_list: list[Path], well_tag: str, cytometer=False) -> dict[int, list[Path]]:
    """
    Group filenames by sub-position id.
    Returns {pos_id: [img0, img1, …]}.
    """
    if cytometer:
        return {0: im_list}

    by_pos = {}
    for p in im_list:
        stem = p.name
        pos_id = int(stem[stem.find(well_tag) + 5 : stem.find(well_tag) + 10])
        by_pos.setdefault(pos_id, []).append(p)
    return by_pos

def build_well_path_table(position_dirs, well_tag, cytometer) -> list[list[list[Path]]]:
    """
    3-D ragged list: [time][pos][z] == Path
    """
    table = []
    for pos_dir in position_dirs:
        for t, time_dir in enumerate(list_time_dirs(pos_dir)):
            im_list = list_z_images(time_dir)
            subpos_dict = split_subpos(im_list, well_tag, cytometer)
            # ensure inner dims are aligned
            if len(table) <= t:
                table.append([])
            table[t].extend(subpos_dict[k] for k in sorted(subpos_dict))
    return table  # shape: T × P × Z

# -------------------------------------------------------------
def choose_canvas(n_tiles, orientation, size_factor) -> tuple[int,int]:
    lookup = {
        1: np.array([480, 640]),
        2: np.array([800, 630]),
        3: np.array([1140, 630]) if orientation=="vertical" else np.array([1140, 480]),
    }
    if n_tiles not in lookup:
        raise ValueError("unexpected #tiles")
    return tuple((lookup[n_tiles] * size_factor).astype(int))

def get_alignment_coords(n_pos_tiles, orientation, ff_tile_dir):

    folder_name = os.path.basename(ff_tile_dir)
    
    prior_path = Path(ff_tile_dir) / "params.json"
    if prior_path.exists():  
        template = StructuredMosaic(
            os.path.dirname(str(prior_path)),
            dim=n_pos_tiles,
            origin="upper left",
            direction=orientation,        # "vertical" or "horizontal"
            pattern="raster",
            )
                          # (a) load existing params
        template.load_params(str(prior_path))
    else:                                      # (b) compute once on first slice
        # -- load slice z=0 just for alignment --
        raise Exception(f"No alignment info found for {folder_name}")

    coords_prior = template.params["coords"] 

    return coords_prior


def stitch_well_z(
    w: int,
    well_list: list[Path],
    cytometer: bool,
    ff_tile_dir: Path,
    size_factor: float,
    orientation: str,
    out_dir: Path,
    overwrite=False,
):
    
    well_dir = Path(well_list[w])
    well_tag = well_dir.name[-4:]

    well_name_conv = sorted(glob.glob(os.path.join(well_dir, "_*")))
    well_name_conv = well_name_conv[0][-3:]

    position_dirs = sorted(glob.glob(str(well_dir) + "/P*"))
    if len(position_dirs) == 0:
        position_dirs = [well_dir]

    table = build_well_path_table(position_dirs, well_tag, cytometer)  # T×P×Z
    _, n_pos, n_z = len(table), len(table[0]), len(table[0][0])
    canvas_shape = choose_canvas(n_pos, orientation, size_factor)

    # build placeholder Tile list once
    # tiles = [Tile(np.zeros((1,1), np.uint8)) for _ in range(n_pos)]
    
    master_json = str(Path(ff_tile_dir) / "master_params.json")

    for t_idx, stack in enumerate(table):                         # loop time
        out_png = out_dir / f"{well_name_conv}_t{t_idx:04}_stack.tif"
        if out_png.exists() and not overwrite:
            continue
        z_cube = np.empty((n_z, *canvas_shape), dtype=np.uint8)
        
        # iterate through Z
        for z in range(n_z): 
            tiles = []                                     # loop z
            for p in range(n_pos):
                img = io.imread(stack[p][z])
                if img.dtype == np.uint16:
                    img = skimage.util.img_as_ubyte(img)          # or adaptive
                tiles.append(Tile(img))

            if len(tiles) != n_pos:
                raise Exception(f"Mismatched numbers of stitch coords and tiles in {well_name_conv}")
            
            z_mosaic = StructuredMosaic(
                    tiles,
                    dim=n_pos,  # number of tiles in primary axis
                    origin="upper left",  # position of first tile
                    direction=orientation,
                    pattern="raster"
                )

            # z_mosaic.params["coords"] = coords_prior   # 1. inject pre-computed shifts
            z_mosaic.load_params(master_json)

            z_mosaic.reset_tiles()                     # 2. tell stitch2d to trust them
            z_mosaic.smooth_seams()                    # 3. OPTIONAL but recommended
            stitched = z_mosaic.stitch() 
            if orientation == "horizontal":
                stitched = stitched.T
            z_cube[z] = trim_to_shape(stitched, canvas_shape)

        # save 
        io.imsave(out_png, z_cube,  check_contrast=False)
        
                
    # well_dict_out = dict({well_tag: well_dict})

    return {}


def stitch_z_from_keyence(data_root: Path | str, 
                          exp_name: str,
                          n_workers:int=4, 
                          overwrite: bool=False):

    par_flag = n_workers > 1
    orientation = _get_keyence_tile_orientation(exp_name)

    RAW = Path(data_root) / "raw_image_data" / "Keyence"
    META  = Path(data_root) / "metadata" / "built_metadata_files"
    BUILT = Path(data_root) / "built_image_data" / "Keyence"
    BUILTZ = Path(data_root) / "built_image_data" / "Keyence_stitched_z"
    
    # handle paths
    # acq_dirs = valid_acq_dirs(RAW, dir_list)

    # for d, acq in enumerate(tqdm(acq_dirs, "Stitching z stacks...")):
    # initialize dictionary to metadata)
    acq = Path(RAW / exp_name)
    dir_path = str(acq) #os.path.join(read_dir, sub_name, '')

    # depth_dir = os.path.join(write_dir, "D_images", sub_name)
    out_dir = BUILTZ /  acq.name
    os.makedirs(out_dir, exist_ok=True)

    # Each folder at this level pertains to a single well
    well_list = sorted(glob.glob(os.path.join(dir_path, "XY*")))
    cytometer_flag = False
    if len(well_list) == 0:
        cytometer_flag = True
        well_list = sorted(glob.glob(dir_path + "/W0*"))

    # get list of FF tile folders
    ff_tile_dir = BUILT /  "FF_images" / acq.name
    metadata_path = META / f"{acq.name}_metadata.csv"
    metadata_df = pd.read_csv(metadata_path)
    size_factor = metadata_df["Width (px)"].iloc[0] / 640

    # print(f'Stitching z slices in directory {d+1:01} of ' + f'{len(dir_indices)}')
    run_stitch_well_z = partial(stitch_well_z, well_list=well_list, orientation=orientation, cytometer=cytometer_flag, 
                                out_dir=out_dir, overwrite=overwrite, size_factor=size_factor, ff_tile_dir=ff_tile_dir)
    if not par_flag:
        for w in tqdm(range(len(well_list))):
            run_stitch_well_z(w)  
    else:
        process_map(run_stitch_well_z, range(len(well_list)), chunksize=1)

    print('Done.')


if __name__ == "__main__":

    overwrite = True

    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    stitch_z_from_keyence(data_root=data_root, 
                          exp_name='20250612_24hpf_wfs1_ctcf',
                          n_workers=1, 
                          overwrite=True)