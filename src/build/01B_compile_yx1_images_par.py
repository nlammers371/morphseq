# script to define functions_folder for loading and standardizing fish movies
import os
import numpy as np
from PIL import Image
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import matplotlib
from tqdm import tqdm
from PIL import Image
import glob2 as glob
from src.functions.image_utils import doLap
from src.functions.utilities import path_leaf
from aicsimageio import AICSImage
import json
from tqdm import tqdm
import pickle
from parfor import pmap
import pandas as pd
import time
import nd2

def scrape_yx1_metadata(im_path):

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
                ind1 = findnth(metadata, param_string + ' Type', 2)
                ind2 = findnth(metadata, '/' + param_string, 2)
            else:
                ind1 = findnth(metadata, param_string + ' Type', 1)
                ind2 = findnth(metadata, '/' + param_string, 1)
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



def process_well(w, im_data_dask, well_name_list, well_time_list, well_ind_list, ff_dir, depth_dir, rs_scale_yx, overwrite_flag=False):

    # set scene
    well_name_conv = well_name_list[w]
    time_int = well_time_list[w]
    well_int = well_ind_list[w]

    # get data
    data_zyx = im_data_dask[time_int, well_int, :, :, :].compute()

    # generate save names
    ff_out_name = 'ff_' + well_name_conv + f'_t{time_int:04}' + '.tif'

    if os.path.isfile(ff_out_name) and not overwrite_flag:
        print(f"Skipping time point {t} for well {well_name_conv}.")
        continue

    # resize image

    # calculate FF image
    laps.append(doLap(images[i]))
    laps_d.append(doLap(images[i], lap_size=7, blur_size=7))  # I've found that depth stacking works better with larger filters

        laps = np.asarray(laps)
        abs_laps = np.absolute(laps)

        laps_d = np.asarray(laps_d)
        abs_laps_d = np.absolute(laps_d)

        # calculate full-focus and depth images
        ff_image = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
        depth_image = np.argmax(abs_laps_d, axis=0)
        maxima = abs_laps.max(axis=0)
        bool_mask = abs_laps == maxima
        mask = bool_mask.astype(np.uint8)
        for i in range(len(images)):
            ff_image[np.where(mask[i] == 1)] = images[i][np.where(mask[i] == 1)]

        # ff_image = 255 - ff_image  # take the negative

        tt = int(time_dir[-4:])

        if cytometer_flag:
            # pos_id_list.append(p)
            pos_string = f'p{p:04}'
        else:
            # pos_id_list.append(pi)
            pos_string = f'p{pi:04}'

        # save images
        ff_out_name = 'ff_' + well_name_conv + f'_t{tt:04}_' + f'ch{ch_to_use:02}/'
        depth_out_name = 'depth_' + well_name_conv + f'_t{tt:04}_' + f'ch{ch_to_use:02}/'
        op_ff = os.path.join(ff_dir, ff_out_name)
        op_depth = os.path.join(depth_dir, depth_out_name)

        if not os.path.isdir(op_ff):
            os.makedirs(op_ff)
        if not os.path.isdir(op_depth):
            os.makedirs(op_depth)

        # convet depth image to 8 bit
        max_z = abs_laps.shape[0]
        depth_image_int8 = np.round(depth_image / max_z * 255).astype('uint8')

        cv2.imwrite(os.path.join(ff_dir, ff_out_name, 'im_' + pos_string + '.tif'), ff_image)

        cv2.imwrite(os.path.join(depth_dir, depth_out_name, 'im_' + pos_string + '.tif'), depth_image_int8)

    # well_dict_out = dict({well_name_conv: well_dict})

    return well_df


def build_ff_from_yx1(data_root, overwrite_flag=False, ch_to_use=0, dir_list=None, write_dir=None, rs_res=None):

    read_dir_root = os.path.join(data_root, 'raw_image_data', 'YX1') 
    if write_dir is None:
        write_dir = os.path.join(data_root, 'built_image_data', 'YX1') 
        
    # handle paths
    if dir_list is None:
        # Get a list of directories
        dir_list_raw = sorted(glob.glob(read_dir_root + "*"))
        dir_list = []
        for dd in dir_list_raw:
            if os.path.isdir(dd):
                dir_list.append(path_leaf(dd))

    if rs_res is None:
        rs_res = np.asarray([5.66, 5.66])

    # filter for desired directories
    dir_indices = [d for d in range(len(dir_list)) if "ignore" not in dir_list[d]]

    for d in dir_indices:

        # initialize dictionary to metadata
        sub_name = dir_list[d]
        dir_path = os.path.join(read_dir_root, sub_name, "")

        depth_dir = os.path.join(write_dir, "D_images", sub_name)
        ff_dir = os.path.join(write_dir, "FF_images",  sub_name)

        if not os.path.isdir(depth_dir):
            os.makedirs(depth_dir)
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
        # imObject = AICSImage(image_list[0])       
        # n_wells = len(imObject.scenes)
        # well_list = imObject.scenes
        # n_time_points = imObject.dims["T"][0]

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
        rs_factor = np.divide(voxel_yx, rs_res)

        rs_dims_yx = np.round(np.multiply(np.asarray(im_shape[3:]), rs_factor)).astype(int)
        # resample images to a standardized resolution


        # # initialize metadata data frame
        # well_df = pd.DataFrame([], columns=['well', 'nd2_series_num', 'microscope', 'time_int', 'Height (um)', 'Width (um)', 'Height (px)', 'Width (px)', 'Objective', 'Time (s)'])

        # read in plate map
        plate_map_xl = pd.ExcelFile(dir_path + "plate_map.xlsx")
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
        well_ind_list_long = np.repeat(well_ind_list, n_time_points)

        # generate metadata dataframe
        well_df = pd.DataFrame(well_name_list_long[:, np.newaxis], columns=["well"])
        well_df["nd2_series_num"] = well_ind_list_long
        well_df["microscope"] = "YX1"
        time_int_list = np.tile(np.arange(1, n_time_points+1), n_wells)
        well_int_list = np.repeat(np.arange(1, n_wells+1), n_time_points)
        well_df["time_int"] = time_ind_list
        well_df["Height (um)"] = im_shape[3]*voxel_size[1]
        well_df["Width (um)"] = im_shape[4]*voxel_size[0]
        well_df["Height (px)"] = rs_dims_yx[0]
        well_df["Width (px)"] = rs_dims_yx[1]
        well_df["Channel"] = imObject.frame_metadata(0).channels[0].channel.name
        well_df["Objective"] = imObject.frame_metadata(0).channels[0].microscope.objectiveName
        time_ind_vec = []
        for n in range(n_wells):
            time_ind_vec += np.arange(n, n_wells*n_time_points, n_wells).tolist()
        well_df["Time (s)"] = frame_time_vec[time_ind_vec]

        print(f'Building full-focus images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        # metadata_df_list = pmap(process_well, range(len(well_list)), (well_list, cytometer_flag, ff_dir, depth_dir, ch_to_use, overwrite_flag), rP=0.5)
        for i in range(n_wells*n_time_points):
            process_well(w, im_array_dask, well_name_list_long, time_int_list, well_int_list, ff_dir, depth_dir, ch_to_use=1, overwrite_flag=False)
    


        if len(metadata_df_list) > 0:
            metadata_df = pd.concat(metadata_df_list)
            first_time = np.min(metadata_df['Time (s)'].copy())
            metadata_df['Time Rel (s)'] = metadata_df['Time (s)'] - first_time
        else:
            metadata_df = []

        # load previous metadata
        metadata_path = os.path.join(ff_dir, 'metadata.csv')

        if len(metadata_df) > 0:
            metadata_df.reset_index()
            metadata_df.to_csv(metadata_path)
        # with open(os.path.join(ff_dir, 'metadata.pickle'), 'wb') as handle:
        #     pickle.dump(metadata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('Done.')




if __name__ == "__main__":

    overwrite_flag = False
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = ["20231206"]
    # build FF images
    # build_ff_from_keyence(data_root, write_dir=write_dir, overwrite_flag=True, dir_list=dir_list, ch_to_use=4)
    # stitch FF images
    build_ff_from_yx1(data_root=data_root, dir_list=dir_list)