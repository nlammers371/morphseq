# script to define functions_folder for loading and standardizing fish movies
import os
import numpy as np
from skimage import io
import glob2 as glob
import torchvision
import torch
import torch.nn.functional as F
from src.functions.utilities import path_leaf
from tqdm import tqdm
import pandas as pd
import time
import nd2
import cv2
from sklearn.cluster import KMeans

def set_inputs_to_device(input_tensor, device):

    inputs_on_device = input_tensor

    if device == "cuda":
        cuda_inputs = input_tensor

        # for key in inputs.keys():
        #     if torch.is_tensor(inputs[key]):
        #         cuda_inputs[key] = inputs[key].cuda()

        #     else:
        #         cuda_inputs[key] = inputs[key]
        cuda_inputs = input_tensor.cuda()
        inputs_on_device = cuda_inputs

    return inputs_on_device


def findnth(haystack, needle, n):
    parts = haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

def trim_image(im, out_shape):
    im_shape = im.shape
    im_diffs = im_shape - out_shape

    pad_width = -im_diffs
    pad_width[np.where(pad_width < 0)] = 0
    im_out = np.pad(im.copy(), ((0, pad_width[0]), (0, pad_width[1])), mode='constant').astype('uint8')

    im_diffs[np.where(im_diffs < 0)] = 0
    sv = np.floor(im_diffs / 2).astype(int)
    if np.all(sv>0):
        im_out = im_out[sv[0]:-(im_diffs[0] - sv[0]), sv[1]:-(im_diffs[1] - sv[1])]
    elif sv[0]==0:
        im_out = im_out[:, sv[1]:-(im_diffs[1] - sv[1])]
    elif sv[1]==0:
        im_out = im_out[sv[0]:-(im_diffs[0] - sv[0]), :]

    return im_out



def process_frame(w, im_data_dask, well_name_list, well_time_list, well_ind_list, ff_dir, device, rs_dims_yx=None, rs_res_yx=None, overwrite_flag=False, n_z_keep=10, ch_to_use=0):

    # set scene
    well_name_conv = well_name_list[w]
    time_int = well_time_list[w]
    well_int = well_ind_list[w]

    # get data
    # start = time.time()
    n_z_slices = im_data_dask.shape[2]
    # buffer = np.max([int((n_z_slices - n_z_keep)/2), 0])
    data_zyx = im_data_dask[time_int, well_int, :, :, :].compute()
    # print(time.time() - start)

    # generate save names
    im_out_name = 'im_stack_' + well_name_conv + f'_t{time_int:04}_' + f'ch{ch_to_use:02}_stitch'

    io.imsave(os.path.join(ff_dir, im_out_name + ".tif"), data_zyx.astype(np.uint16))

    return {}


def extract_sample_yx1_stacks(data_root, dir_list=None, write_dir=None):

    read_dir_root = os.path.join(data_root, 'raw_image_data', 'YX1') 
    if write_dir is None:
        write_dir = os.path.join(data_root, 'built_image_data', 'YX1', 'sample_stacks') 
        
    # handle paths
    if dir_list is None:
        # Get a list of directories
        dir_list_raw = sorted(glob.glob(read_dir_root + "*"))
        dir_list = []
        for dd in dir_list_raw:
            if os.path.isdir(dd):
                dir_list.append(path_leaf(dd))


    # filter for desired directories
    dir_indices = [d for d in range(len(dir_list)) if "ignore" not in dir_list[d]]

    for d in dir_indices:

        # initialize dictionary to metadata
        sub_name = dir_list[d]
        dir_path = os.path.join(read_dir_root, sub_name, "")

        # depth_dir = os.path.join(write_dir, "stitched_depth_images", sub_name)
        ff_dir = os.path.join(write_dir, "stitched_FF_images",  sub_name + "_test")

        # if not os.path.isdir(depth_dir):
        #     os.makedirs(depth_dir)
        if not os.path.isdir(ff_dir):
            os.makedirs(ff_dir)

        # Read in  metadata object
        image_list = sorted(glob.glob(dir_path + "*.nd2")) 
        if len(image_list) > 1:
            raise Exception("Multiple nd2 files found in " + sub_name + ". Please move extra nd2 files to a subfolder." )
        elif len(image_list) == 0:
            raise Exception("No nd2 files found in " + sub_name)

        # Read in experiment metadata 
        print(f"Processing {sub_name}...")

        imObject= nd2.ND2File(image_list[0])
        im_shape = imObject.shape
        n_time_points = im_shape[0]
        n_wells = im_shape[1]
        n_z_slices = im_shape[2]

        # pull dask array
        im_array_dask = imObject.to_dask()
        # use first 10 frames to infer time resolution

        # extract frame times
        n_frames_total = imObject.frame_metadata(0).contents.frameCount
        frame_time_vec = [imObject.frame_metadata(i).channels[0].time.relativeTimeMs / 1000 for i in
                        range(0, n_frames_total, im_shape[2])]
        # check for common nd2 artifact where time stamps jump midway through
        dt_frame_approx = (imObject.frame_metadata(n_z_slices).channels[0].time.relativeTimeMs -
                        imObject.frame_metadata(0).channels[0].time.relativeTimeMs) / 1000
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

        # get image resolution
        voxel_size = imObject.voxel_size()
        voxel_yx = np.asarray([voxel_size[1], voxel_size[0]])
        # rs_factor = np.divide(voxel_yx, rs_res)

        # rs_dims_yx = np.round(np.multiply(np.asarray(im_shape[3:]), rs_factor)).astype(int)
        # resample images to a standardized resolution


        # # initialize metadata data frame
        # well_df = pd.DataFrame([], columns=['well', 'nd2_series_num', 'microscope', 'time_int', 'Height (um)', 'Width (um)', 'Height (px)', 'Width (px)', 'Objective', 'Time (s)'])

        # read in plate map
        plate_map_xl = pd.ExcelFile(dir_path + sub_name + "_plate_map.xlsx")
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
        well_name_list_long = np.repeat(well_name_list_sorted, n_time_points)
        well_ind_list_long = np.repeat(np.asarray(well_ind_list)[si], n_time_points)

        
        time_int_list = np.tile(np.arange(0, n_time_points), n_wells)
        


        # get device
        device = (
                "cuda"
                if torch.cuda.is_available() 
                else "cpu"
            )

        # for indexing dask array
        well_int_list = np.repeat(np.arange(0, n_wells), n_time_points)

        # call FF function
    
        for w in tqdm(range(0, n_wells*n_time_points, 100)): #n_wells*n_time_points)):
            process_frame(w, im_array_dask, well_name_list_long, time_int_list, well_int_list, ff_dir, device=device)#, rs_dims_yx=rs_dims_yx, rs_res_yx=rs_res)
        
   

        imObject.close()
        # with open(os.path.join(ff_dir, 'metadata.pickle'), 'wb') as handle:
        #     pickle.dump(metadata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('Done.')



if __name__ == "__main__":

    overwrite_flag = True
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = ["20231206", "20231110", "20231218"]
    # build FF images
    # build_ff_from_keyence(data_root, write_dir=write_dir, overwrite_flag=True, dir_list=dir_list, ch_to_use=4)
    # stitch FF images
    extract_sample_yx1_stacks(data_root=data_root, dir_list=dir_list)