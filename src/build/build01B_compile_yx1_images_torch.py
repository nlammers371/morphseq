from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[3]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))


import os, json, time, logging
from pathlib import Path
from typing import List, Sequence, Tuple
from functools import partial
from tqdm.contrib.concurrent import process_map
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import nd2
import skimage.io as skio
import skimage
from stitch2d import StructuredMosaic      
from src.build.export_utils import LoG_focus_stacker, im_rescale
# 
log = logging.getLogger(__name__)
logging.basicConfig(format="%(level_name)s | %(message)s", level=logging.INFO)


##########
# Helper functions
def _FF_wrapper(w, dask_arr, well_id_array, time_id_array, device, filter_size, bf_idx, 
                out_ff, well_name_list_long, overwrite):
    
    # get indices
    t_idx = time_id_array[w]
    w_idx = well_id_array[w]
    well_name = well_name_list_long[w]

    stack = _get_stack(dask_arr, t_idx, w_idx)
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


def _find_dirs(root: Path) -> List[Path]:
    return sorted([d for d in root.iterdir() if d.is_dir() and "ignore" not in d.name])

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

def _focus_stack(
    stack_zyx: np.ndarray,
    device: str,
    filter_size: int = 3
) -> np.ndarray:

    # instead of torch.quantile, use numpy
    norm, _, _ = im_rescale(stack_zyx)
    norm = norm.astype(np.float32)
    tensor = torch.from_numpy(norm).to(device)

    ff_t, _ = LoG_focus_stacker(tensor, filter_size, device)
    arr = ff_t.cpu().numpy()
    arr_clipped = np.clip(arr, 0, 65535)
    ff_i = arr_clipped.astype(np.uint16)
    # convert to 8 bit 
    ff_8 = skimage.util.img_as_ubyte(ff_i)

    return ff_8 #(65535 - ff.cpu().numpy()).astype(np.uint16)

def _write_ff(
    out_root: Path,
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


def build_ff_from_yx1(
    data_root: str | Path,
    exp_name: str,
    overwrite: bool = False,
    # dir_list: Sequence[str] | None = None,
    # write_dir: str | Path | None = None,
    device: str="cpu",
    n_workers: int=1,
    metadata_only: bool = False,
    # n_z_keep: Sequence[int | None] | None = None,
):

    par_flag = n_workers > 1

    data_root = Path(data_root)
    read_root = data_root / "raw_image_data" / "YX1"
    write_root = data_root / "built_image_data"
    meta_root = data_root / "metadata"

    # exp_dirs = (
    #     [read_root / d for d in dir_list] if dir_list else _find_dirs(read_root)
    # )
    # if n_z_keep is None:
    #     n_z_keep = [None] * len(exp_dirs)


    # for exp_path, z_keep in zip(exp_dirs, n_z_keep):
    exp_path = read_root / exp_name
    # exp_name = exp_path.exp_name
    log.info("⏳  %s", exp_name)

    nd = _read_nd2(exp_path)
    shape_twzcxy = nd.shape  # T,W,Z,C,Y,X
    n_t, n_w, n_z = shape_twzcxy[:3]

    dask_arr = nd.to_dask()  # (T,W,Z,C,Y,X)
    channel_names = [c.channel.name for c in nd.frame_metadata(0).channels]
    bf_idx = channel_names.index("BF")

    # non_bf_indices = [i for i in range(len(channel_names)) if channel_names[i] != "BF"]

    # if n_channels > 1:
    #     fluo_dir = os.path.join(write_dir, "stitched_fluo_images", sub_name)
    #     if not os.path.isdir(fluo_dir):
    #         os.makedirs(fluo_dir)


    # get image resolution
    voxel_size = nd.voxel_size()

    # n_channels = len(nd.frame_metadata(0).channels)

    # read in plate map
    plate_map_xl = pd.ExcelFile(meta_root / "well_metadata" / f"{exp_name}_well_metadata.xlsx")

    # if n_channels > 1:
    #     channel_map = plate_map_xl.parse("channels")

    # fix jumps in nd2 time stamp
    frame_time_vec = _fix_nd2_timestamp(nd, n_z)

    # get series numbers
    series_map = plate_map_xl.parse("series_number_map").iloc[:8, 1:13]

    well_name_list = []
    well_ind_list = []
    col_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    row_letter_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for c in range(len(col_id_list)):
        for r in range(len(row_letter_list)):
            ind_float = series_map.iloc[r, c]
            if ~np.isnan(ind_float):
                well_name = row_letter_list[r] + f"{col_id_list[c]:02}"
                well_name_list.append(well_name)
                well_ind_list.append(int(ind_float))

    si = np.argsort(well_ind_list)
    well_name_list_sorted = np.asarray(well_name_list)[si].tolist()
    well_ind_list_sorted = np.asarray(well_ind_list)[si].tolist()

    # generate longform vectors
    well_name_list_long = np.repeat(well_name_list_sorted, n_t)
    well_ind_list_long = np.repeat(np.asarray(well_ind_list)[si], n_t)

    # check that assigned well IDs match recorded stage positions
    stage_xyz_array = np.empty((n_w*n_t, 3))
    well_id_array = np.empty((n_w*n_t,))
    time_id_array = np.empty((n_w*n_t,))
    iter_i = 0
    for w in range(n_w):
        for t in range(n_t):
            base_ind = t*n_w + w
            slice_ind = base_ind*n_z
            
            stage_xyz_array[iter_i, :] = np.asarray(nd.frame_metadata(slice_ind).channels[0].position.stagePositionUm)
            well_id_array[iter_i] = w
            time_id_array[iter_i] = t
            iter_i += 1


    # chec that recorded well positions are consistent with actual image positions on the plate
    _qc_well_assignments(stage_xyz_array, well_name_list_long)

    # generate metadata dataframe
    well_df = pd.DataFrame(well_name_list_long[:, np.newaxis], columns=["well"])
    well_df["nd2_series_num"] = well_ind_list_long
    well_df["microscope"] = "YX1"
    time_int_list = np.tile(np.arange(0, n_t), n_w)
    well_df["time_int"] = time_int_list
    well_df["Height (um)"] = shape_twzcxy [3]*voxel_size[1]
    well_df["Width (um)"] = shape_twzcxy [4]*voxel_size[0]
    well_df["Height (px)"] = shape_twzcxy [3]
    well_df["Width (px)"] = shape_twzcxy [4]
    well_df["BF Channel"] = bf_idx
    well_df["Objective"] = nd.frame_metadata(0).channels[0].microscope.objectiveName
    time_ind_vec = []
    for n in range(n_w):
        time_ind_vec += np.arange(n, n_w*n_t, n_w).tolist()
    well_df["Time (s)"] = frame_time_vec[time_ind_vec]

    # get device
    # device = (
    #         "cuda"
    #         if torch.cuda.is_available() #and (not par_flag)
    #         else "cpu"
    #     )

    if device == "cpu":
        print("Warning: using CPU. This may be quite slow. GPU recommended.")

    # call FF function
    if not metadata_only:

        out_ff = write_root / "stitched_FF_images" / exp_name

        call_ff = partial(_FF_wrapper, dask_arr=dask_arr, well_id_array=well_id_array, 
                          time_id_array=time_id_array, device=device, filter_size=3, 
                          bf_idx=bf_idx, out_ff=out_ff, well_name_list_long=well_ind_list_long, overwrite=overwrite)
    
        if not par_flag:
            for w in tqdm(range(len(well_id_array))):
                call_ff(w)
        else:
            process_map(call_ff, range(len(well_id_array)), max_workers=n_workers, chunksize=1)

                


    first_time = np.min(well_df['Time (s)'].copy())
    well_df['Time Rel (s)'] = well_df['Time (s)'] - first_time
    
    # load previous metadata
    out_meta = meta_root / "built_metadata_files"
    out_meta.mkdir(parents=True, exist_ok=True)
    well_df.to_csv(out_meta / f"{exp_name}_metadata.csv", index=False)

    nd.close()


    print('Done.')



if __name__ == "__main__":

    overwrite_flag = False
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = ["2020306"]
    # build FF images
    # build_ff_from_keyence(data_root, write_dir=write_dir, overwrite_flag=True, dir_list=dir_list, ch_to_use=4)
    # stitch FF images
    build_ff_from_yx1(data_root=data_root, dir_list=dir_list, overwrite_flag=overwrite_flag, n_z_keep_in=8)