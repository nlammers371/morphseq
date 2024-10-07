import os
import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import skimage
import cv2
import pandas as pd
from src.functions.utilities import path_leaf
from src.functions.image_utils import crop_embryo_image, get_embryo_angle, process_masks, rotate_image
from src.functions.image_utils import LoG_focus_stacker
from src.build.build03A_process_embryos_main_par import estimate_image_background
from functools import partial
import scipy
import warnings
# from parfor import pmap
import skimage
from scipy.stats import truncnorm
import numpy as np
import skimage.io as io
from tqdm.contrib.concurrent import process_map 
import nd2
import torch

# Set the logging level to ERROR
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'


def export_embryo_snips_z(r, root, embryo_metadata_df, experiment_log, dl_rad_um, outscale, outshape, device, px_mean, px_std, 
                          n_z_max, z_range_um, filter_size=3):


    ############################
    # set generic path variables

    # set path to segmentation data
    segmentation_path = os.path.join(root, 'built_image_data', 'segmentation', '')

    # make directory for embryo snips
    im_snip_dir = os.path.join(root, 'training_data', f'bf_embryo_snips_z{n_z_max:02}', '')
    im_snip_dir_uc = os.path.join(im_snip_dir[:-1] + "_uncropped")
    if not os.path.isdir(im_snip_dir[:-1] + "_uncropped"):
        os.makedirs(im_snip_dir[:-1] + "_uncropped")

    # initialize folder to save temporary metadata files
    metadata_temp_dir = os.path.join(root, 'metadata', f'metadata_files_temp_z{n_z_max:02}', '')
    if not os.path.isdir(metadata_temp_dir):
        os.makedirs(metadata_temp_dir)

    # generate path and image name
    seg_dir_list_raw = glob.glob(segmentation_path + "*")
    seg_dir_list = [s for s in seg_dir_list_raw if os.path.isdir(s)]

    emb_path = [m for m in seg_dir_list if "mask" in m][0]
    yolk_path = [m for m in seg_dir_list if "yolk" in m][0]

    ##########################
    # get metadata
    row = embryo_metadata_df.iloc[r].copy()

    # z resolution
    scope = row["microscope"]
    if not type(scope)==str and not np.isnan(row["nd2_series_num"]):
        scope = "YX1"

    # get surface area
    px_dim_raw = row["Height (um)"] / row["Height (px)"]  # to adjust for size reduction (need to automate this)

    # generate file names
    im_name = row["snip_id"]
    
    # load well info 
    well = row["well"]
    time_int = row["time_int"]
    date = str(row["experiment_date"])

    ############
    # Load masks from segmentation
    ############
    im_stub = well + f"_t{time_int:04}*"

    # load main embryo mask
    im_emb_path = glob.glob(os.path.join(emb_path, date, im_stub))[0]
    im_mask = io.imread(im_emb_path)

    # load yolk mask
    im_yolk_path = glob.glob(os.path.join(yolk_path, date, im_stub))[0]
    im_yolk = io.imread(im_yolk_path)

    im_mask_ft, im_yolk = process_masks(im_mask, im_yolk, row)

    ############
    # Load Image Stack
    ############
    # check which microscope we're working with
    
    # set path to image stack
    if scope == "Keyence":
        try:
            image_path = os.path.join(root, 'built_image_data', 'keyence_stitched_z', '')
            im_stack_path = glob.glob(os.path.join(image_path, date, im_stub))[0]        
            im_stack = io.imread(im_stack_path)
        except:
            print("Could not open image stack for " + date + "_" + im_stub) 
            return
        
        zres = experiment_log.loc[experiment_log["start_date"]==date, "z_res_um"].to_numpy()[0]

    elif scope == "YX1":
        image_path = os.path.join(root, 'raw_image_data', 'YX1', date, '')
        image_list = sorted(glob.glob(image_path + "*.nd2")) 
        if len(image_list) > 1:
            raise Exception("Multiple nd2 files found in " + date + ". Please move extra nd2 files to a subfolder." )
        elif len(image_list) == 0:
            raise Exception("No nd2 files found in " + date)

        # get array
        with nd2.ND2File(image_list[0]) as imObject:
            # imObject= nd2.ND2File(image_list[0])
            im_array_dask = imObject.to_dask()

            n_channels = len(imObject.frame_metadata(0).channels)
            channel_names = [c.channel.name for c in imObject.frame_metadata(0).channels]
            bf_channel_ind = channel_names.index("BF")

            if n_channels > 1:
                im_array_bf = np.squeeze(im_array_dask[:, :, :, bf_channel_ind, :, :])#.compute()
            else:
                im_array_bf = im_array_dask

            series_num = row["nd2_series_num"].astype(int)
            im_stack = im_array_bf[time_int, series_num-1, :, :, :].compute()

            # get z resolution
            voxel_size = imObject.voxel_size()
            zres = voxel_size[2]
    else:
        raise Exception("Unrecognized microscope type: " + scope)
    
    # except: # if we can't load the image, just return
    #     print("Skipping " + date + "_" + im_stub + "...")
    #     return
    
    if len(im_stack.shape) < 3:
        raise Exception("Incorrect image dimensions for " + date + "_" + im_stub) 
    
    if im_stack.shape[1] < im_stack.shape[2]:
        im_stack = np.transpose(im_stack, (0, 2, 1))

    # get z window size    
    window_size = np.round(z_range_um / zres / 2).astype(int)

    if np.max(im_stack) <= 255:
        im_stack = im_stack.astype(np.uint8)
    else:
        im_stack = im_stack.astype(np.uint16)
    dtype = im_stack.dtype
    # im_stack = skimage.util.img_as_ubyte(im_stack)

    # rescale masks and image
    im_rs0 = cv2.resize(im_stack[0], None, fx=px_dim_raw / outscale, fy=px_dim_raw / outscale)
    im_stack_rs = np.zeros((im_stack.shape[0], im_rs0.shape[0], im_rs0.shape[1]), dtype=dtype)
    im_stack_rs[0, :, :] = im_rs0
    for z in range(1, im_stack.shape[0]):
        im_stack_rs[z] = cv2.resize(im_stack[z], None, fx=px_dim_raw / outscale, fy=px_dim_raw / outscale)

    mask_emb_rs = cv2.resize(im_mask_ft, im_rs0.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    mask_yolk_rs = cv2.resize(im_yolk, im_rs0.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    ###################
    # Rotate image
    ###################

    # get embryo mask orientation
    angle_to_use = get_embryo_angle(mask_emb_rs, mask_yolk_rs)

    emb_mask_rotated = rotate_image(mask_emb_rs.astype(np.uint8), np.rad2deg(angle_to_use))
    im_yolk_rotated = rotate_image(mask_yolk_rs.astype(np.uint8), np.rad2deg(angle_to_use))
    im_stack_rotated = np.zeros((im_stack.shape[0], emb_mask_rotated.shape[0], emb_mask_rotated.shape[1]), dtype=dtype)
    for z in range(im_stack.shape[0]):
        im_stack_rotated[z, :, :] = rotate_image(im_stack_rs[z, :, :], np.rad2deg(angle_to_use))

    #######################
    # Crop
    #######################
    im_cropped, emb_mask_cropped, yolk_mask_cropped = crop_embryo_image(im_stack_rotated, emb_mask_rotated, im_yolk_rotated, outshape=outshape)
    
    # fill holes in embryo and yolk masks
    emb_mask_cropped2 = scipy.ndimage.binary_fill_holes(emb_mask_cropped).astype(np.uint8)
    yolk_mask_cropped = scipy.ndimage.binary_fill_holes(yolk_mask_cropped).astype(np.uint8)

    # calculate the distance transform
    im_dist_cropped = scipy.ndimage.distance_transform_edt(1 * (emb_mask_cropped2 == 0))

    # crop out background
    dl_rad_px = int(np.ceil(dl_rad_um / outscale))

    # noise_array = np.random.normal(px_mean, px_std, outshape)
    noise_array_raw = np.reshape(truncnorm.rvs(-px_mean/px_std, 4, size=outshape[0]*outshape[1]), outshape)
    noise_array = noise_array_raw*px_std + px_mean
    noise_array[np.where(noise_array < 0)] = 0 # This is redundant, but just in case someone fiddles with the above distributioon

    #############
    # Calculate LAPS to find most in-focus slices
    image_tensor = torch.tensor(im_cropped.astype(np.float64))
    _, lap_scores = LoG_focus_stacker(image_tensor, filter_size, device)

    # calculate focus scores just in non-yolk embryo regions of image
    body_mask = emb_mask_cropped2.copy()
    body_mask[yolk_mask_cropped==1] = 0

    lap_score_vec = np.empty((image_tensor.shape[0],))
    for z in range(image_tensor.shape[0]):
        lap_score_vec[z] = np.mean(np.asarray(lap_scores[z])[body_mask==1])

    # get moving average
    avg_ft = np.ones((window_size*2+1,)) / (window_size*2+1)
    lap_avg = np.convolve(lap_score_vec, avg_ft, mode="same")
    max_z_i = np.argmax(lap_avg)
    
    if n_z_max == 1:
        frames_to_use = [max_z_i]
    elif n_z_max == 3:
        frames_to_use = [max_z_i-window_size, max_z_i, max_z_i+window_size]
    elif n_z_max == 5:
        frames_to_use = [max_z_i-window_size, max_z_i, max_z_i+window_size]
        f4 = [np.round((max_z_i-window_size + max_z_i)/2).astype(int)]
        f5 = [np.round((max_z_i+window_size + max_z_i)/2).astype(int)]
        frames_to_use = frames_to_use + f4 + f5
    else:
        raise Exception("Invalid z slice number specified (must be 1, 3 or 5)")
    
    # basic QC checks
    frames_to_use = np.asarray(frames_to_use)
    frames_to_use[frames_to_use < 0] = 0
    frames_to_use[frames_to_use > im_stack.shape[0]] = im_stack.shape[0]
    frames_to_use = np.unique(frames_to_use)
    

    # iterate through slices and export
    if im_cropped.dtype != np.uint8:
        im_cropped_out = skimage.util.img_as_ubyte(im_cropped)
    else:
        im_cropped_out = im_cropped.copy()

    metadata_list = []
    for ind, z in enumerate(frames_to_use):

        im_slice = 255 - im_cropped_out[z] # invert
        im_slice_masked = im_slice.copy()
        
        # im_masked_cropped += noise_array_scaled # [np.where(emb_mask_cropped == 0)] = np.random.choice(other_pixel_array, np.sum(emb_mask_cropped == 0)).astype(np.uint8)
        im_slice_masked[np.where(im_dist_cropped > dl_rad_px)] = np.round(noise_array[np.where(im_dist_cropped > dl_rad_px)]).astype(np.uint8)
        # im_masked_cropped[np.where(im_masked_cropped < 0)] = 0
        im_slice_masked[np.where(im_slice_masked > 255)] = 255    # NL: I think this is redundant but will leave it
        im_slice_masked = np.round(im_slice_masked).astype(np.uint8)

        # write to file
        im_save_name = os.path.join(im_snip_dir, im_name + f"_z{ind:02}.jpg")
        im_save_name_uc = os.path.join(im_snip_dir_uc, im_name + f"_z{ind:02}.jpg")
        
        io.imsave(im_save_name, im_slice_masked, check_contrast=False)
        io.imsave(im_save_name_uc, im_slice, check_contrast=False)

        # store metadat
        df_temp = row.copy()
        df_temp["z_res_um"] = zres
        df_temp["z_ind"] = ind
        df_temp["z_pos_rel"] = (z - max_z_i) * zres
        metadata_list.append(df_temp)

    z_meta_df = pd.concat(metadata_list, axis=1, ignore_index=True).T

    # save stripped-down data
    z_meta_simp = z_meta_df.loc[:, ["snip_id", "z_res_um", "z_ind", "z_pos_rel"]]
    snip_name = z_meta_simp.loc[0, "snip_id"]
    z_meta_simp.to_csv(os.path.join(metadata_temp_dir, snip_name + ".csv"), index=False)

    # return z_meta_df



def extract_embryo_z_snips(root, outscale=5.66, overwrite_flag=False, par_flag=False, outshape=None, dl_rad_um=75, z_range_um=350, n_z_max=5):

    if outshape == None:
        outshape = [576, 256]

    device = (
                "cuda"
                if torch.cuda.is_available() and (not par_flag)
                else "cpu"
            )

    # read in metadata
    metadata_path = os.path.join(root, 'metadata', "combined_metadata_files", '')
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df.csv"), index_col=0)
    experiment_log_df = pd.read_csv(os.path.join(root, "metadata", "experiment_metadata.csv"))

    # make directory for embryo snips
    im_snip_dir = os.path.join(root, 'training_data', f'bf_embryo_snips_z{n_z_max:02}', '')
    if not os.path.isdir(im_snip_dir):
        os.makedirs(im_snip_dir)
        export_indices = range(embryo_metadata_df.shape[0])
    elif overwrite_flag:
        export_indices = range(embryo_metadata_df.shape[0])
    else: # if directory exists, check for previously-exported images
        extant_images = sorted(glob.glob(im_snip_dir + "*.jpg"))
        extant_df = pd.DataFrame(np.asarray([path_leaf(im)[:-8] for im in extant_images]), columns=["snip_id"]).drop_duplicates()
        embryo_metadata_df = embryo_metadata_df.merge(extant_df, on="snip_id", how="left", indicator=True)
        export_indices = np.where(embryo_metadata_df["_merge"]=="left_only")[0]
        embryo_metadata_df = embryo_metadata_df.drop(labels=["_merge"], axis=1)

        # # load previous metadata
        # embryo_metadata_df_z_prev = pd.read_csv(os.path.join(metadata_path, f"embryo_metadata_z{n_z_max:02}_df.csv"))
        # cat_flag = True

    # draw random sample to estimate background
    px_mean, px_std = estimate_image_background(root, embryo_metadata_df, bkg_seed=309, n_bkg_samples=100)

    if par_flag:
        n_workers = 8 #np.ceil(os.cpu_count() / 2).astype(int)
        process_map(partial(export_embryo_snips_z, root=root, embryo_metadata_df=embryo_metadata_df, experiment_log=experiment_log_df,
                                      dl_rad_um=dl_rad_um, outscale=outscale, outshape=outshape, device=device, n_z_max=n_z_max, z_range_um=z_range_um,
                                      px_mean=0.1*px_mean, px_std=0.1*px_std),
                                      export_indices, max_workers=n_workers, chunksize=1)

        
    else:
        # frame_df_list = []
        for r in tqdm(export_indices, "Exporting snips..."):
            export_embryo_snips_z(r, root=root, embryo_metadata_df=embryo_metadata_df, experiment_log=experiment_log_df,
                                      dl_rad_um=dl_rad_um, outscale=outscale, outshape=outshape, device=device, n_z_max=n_z_max, z_range_um=z_range_um,
                                      px_mean=0.1*px_mean, px_std=0.1*px_std)

            # frame_df_list.append(frame_df)
    
    # combine metadata
    # embryo_metadata_df_z = pd.concat(frame_df_list, axis=0, ignore_index=True)
    # if cat_flag:
    #     embryo_metadata_df_z = pd.concat([embryo_metadata_df_z, embryo_metadata_df_z_prev], axis=0, ignore_index=True)
    
    # # save
    # embryo_metadata_df_z.to_csv(os.path.join(metadata_path, f"embryo_metadata_z{n_z_max:02}_df.csv"), index=False)


if __name__ == "__main__":

    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
    
    # print('Compiling well metadata...')
    extract_embryo_z_snips(root, par_flag=False, overwrite_flag=False, dl_rad_um=10)
