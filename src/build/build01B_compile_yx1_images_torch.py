from __future__ import annotations
import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]   # adjust “2” if levels differ

# Put that directory at the *front* of sys.path so Python looks there first
sys.path.insert(0, str(REPO_ROOT))

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import nd2
from skimage import util
from src.build.export_utils import (LoG_focus_stacker, im_rescale, save_images_parallel, estimate_max_blocks)
from skimage.measure import block_reduce
import dask.array as da

# pick a name for your logger (usually __name__)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)            # or DEBUG, WARNING, etc.

# create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# note: use %(levelname)s, not %(level_name)s
fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt=datefmt)

ch.setFormatter(formatter)
log.addHandler(ch)



# def batch_blocks_fast(dask_arr, batch_size, dtype,
#                       z_pad=False, downsample=False):
#     """
#     Iterate through (t,w) in row-major order; build batches lazily.

#     Returns  (coords, tensor)  where
#        coords : list[tuple[int,int]]  – (t,w) for each item in batch
#        tensor : torch.Tensor (B, Z, Y, X) on *CPU*
#     """
#     T, W, Z, Y, X = dask_arr.shape
#     buf = 5 if z_pad else 0
#     ds  = 2 if downsample else 1
#     # new_Y, new_X = Y//ds, X//ds            # integer division is safe here

#     coords, blocks = [], []
#     for t in range(T):
#         for w in range(W):
#             # --- 1. pull one stack (identical to your old code) ----------
#             stk = dask_arr[t, w, buf:Z-buf].compute()       # → (Z, Y, X)
#             # if downsample:
#             #     stk = block_reduce(stk, block_size=(1, ds, ds), func=np.mean)

#             norm, _, _ = im_rescale(stk)      # your existing rescale
#             blocks.append( torch.from_numpy(norm).type(dtype) )
#             coords.append((t, w))

#             # --- 2. yield when batch is full ----------------------------
#             if len(blocks) == batch_size:
#                 yield coords, torch.stack(blocks, 0)
#                 coords, blocks = [], []

#     # tail
#     if blocks:
#         yield coords, torch.stack(blocks, 0)


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



def build_ff_from_yx1(
    data_root: str | Path,
    exp_name: str,
    overwrite: bool = False,
    metadata_only: bool = False,
    ff_proc_dtype: torch.dtype = torch.float16,
    ff_filter_res_um: float=3.0
):
    
    # set directories
    BUILT = Path(data_root) / "built_image_data" / "Keyence"
    ff_dir = BUILT / "stitched_FF_images" / exp_name

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # par_flag = n_workers > 1

    data_root = Path(data_root)
    read_root = data_root / "raw_image_data" / "YX1"
    meta_root = data_root / "metadata"

    # for exp_path, z_keep in zip(exp_dirs, n_z_keep):
    exp_path = read_root / exp_name
    # exp_name = exp_path.exp_name
    log.info("Calculating FF for %s", exp_name)

    nd = _read_nd2(exp_path)
    shape_twzcxy = nd.shape  # T,W,Z,C,Y,X
    n_t, n_w, n_z = shape_twzcxy[:3]

    dask_arr = nd.to_dask()  # (T,W,Z,C,Y,X)
    channel_names = [c.channel.name for c in nd.frame_metadata(0).channels]
    bf_idx = channel_names.index("BF")
    if len(shape_twzcxy) == 6: # force to be BF only
        print("Detected multiple channles. Keeping only BF.")
        dask_arr = dask_arr[:, :, :, bf_idx, :, :]

    # get image resolution
    voxel_size = nd.voxel_size()

    # read in plate map
    plate_map_xl = pd.ExcelFile(meta_root / "well_metadata" / f"{exp_name}_well_metadata.xlsx")


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

    # suppose you already have dask_arr = nd.to_dask()[..., bf_idx, ...] with shape (T,W,Z,Y,X)
    Z, Y, X = dask_arr.shape[2:]
    bs, rs_flag = estimate_max_blocks(Z, Y, X, dtype=ff_proc_dtype, safety=0.75, device=device)
    mb = 3
    if bs > mb:
        bs = mb
    print(f"Batch size: {bs}")


    if rs_flag:
        shape_twzcxy = list(shape_twzcxy)
        shape_twzcxy[-1] = shape_twzcxy[-1] // 2
        shape_twzcxy[-2] = shape_twzcxy[-2] // 2
        shape_twzcxy = tuple(shape_twzcxy)

    # generate metadata dataframe
    well_df = pd.DataFrame(well_name_list_long[:, np.newaxis], columns=["well"])
    well_df["nd2_series_num"] = well_ind_list_long
    well_df["microscope"] = "YX1"
    time_int_list = np.tile(np.arange(0, n_t), n_w)
    well_df["time_int"] = time_int_list
    if not rs_flag:
        well_df["Height (um)"] = shape_twzcxy [3]*voxel_size[1]
        well_df["Width (um)"] = shape_twzcxy [4]*voxel_size[0]
    else:
        well_df["Height (um)"] = shape_twzcxy [3]*voxel_size[1] * 2
        well_df["Width (um)"] = shape_twzcxy [4]*voxel_size[0] * 2
    well_df["Height (px)"] = shape_twzcxy [3]
    well_df["Width (px)"] = shape_twzcxy [4]
    well_df["BF Channel"] = bf_idx
    well_df["Objective"] = nd.frame_metadata(0).channels[0].microscope.objectiveName
    time_ind_vec = []
    for n in range(n_w):
        time_ind_vec += np.arange(n, n_w*n_t, n_w).tolist()
    well_df["Time (s)"] = frame_time_vec[time_ind_vec]

    n_coords  = dask_arr.shape[0] * dask_arr.shape[1]
    n_batches = (n_coords + bs - 1) // bs 
    z_pad_flag = exp_name == "20231206"

    if not metadata_only:

        for coords, batch_t in  tqdm(
                                    batch_blocks_fast(dask_arr, bs, dtype=ff_proc_dtype, downsample=rs_flag, z_pad=z_pad_flag),
                                    desc="Generating FF projections…",
                                    total=n_batches,
                                    unit="batch"
                                ):
            
            # Pull batch np_batch.shape == (B, Z, Y, X)
            batch_t = batch_t.to(device, non_blocking=True)

            # instead of torch.quantile, use numpy
            if rs_flag:
                pixel_size_um = voxel_size[0] * 2
            else:
                pixel_size_um = voxel_size[0]
            filter_rad = max(1, int(round(ff_filter_res_um/ pixel_size_um)))
            filter_size = 2 * filter_rad + 1

            # pass to batcher
            # if device == "cuda":
            #     ff_t = batched_focus(batch_t, filter_size, device)
            # else:
            # ff_t, _ = LoG_focus_stacker(batch_t, filter_size, device)

            # arr = ff_t.cpu().numpy()
            # arr_clipped = np.clip(arr, 0, 65535)
            # ff = arr_clipped.astype(np.uint16)
            # # prepare to save              # returns shape (b*t, H, W)
            
            # ff_list = [util.img_as_ubyte(ff[b]) for b in range(ff.shape[0])]
            
            # time_indices = [coord[0] for coord in coords]
            # well_names = [well_name_list_sorted[coords[b][1]] for b in range(bs)]
            # ff_paths = [
            #     Path(ff_dir) / f"{w}_t{t:04}_stitch.jpg"
            #     for w, t in zip(well_names, time_indices)]

            # # save
            # # us parallel processing to save images
            # save_images_parallel(images=ff_list,
            #                     paths=ff_paths,
            #                     n_workers=min(bs, 2))
            
            # cleanup
            del batch_t#, ff_t, arr, ff, ff_list
            torch.cuda.empty_cache()       # clears un-referenced CUDA buffers
            # import gc; gc.collect() 

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
    # dir_list = ["2020306"]
    # build FF images
    # build_ff_from_keyence(data_root, write_dir=write_dir, overwrite_flag=True, dir_list=dir_list, ch_to_use=4)
    # stitch FF images
    build_ff_from_yx1(data_root = data_root,
                      exp_name = "20231110" #"20240314" 
                     )