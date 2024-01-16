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



def process_well(w, imObject, well_list, well_name_list, ff_dir, depth_dir, overwrite_flag=False):

    # set scene
    well_id_string = well_list[w]
    well_num = int(well_id_string.replace("XYPos:", ""))
    imObject.set_scene(well_id_string)

    # initialize
    well_df = pd.DataFrame([], columns=['well', 'nd2_series_num', 'time_int', 'time_string', 'Height (um)', 'Width (um)', 'Height (px)', 'Width (px)', 'Channel', 'Objective', 'Time (s)'])

    # get conventional well name
    well_name_conv = well_name_list[w]

    # get number of time points
    n_time_points = 10 # NL: replace with dynamic lookup
    
    # each pos dir contains one or more time points
    for t in range(n_time_points):

        ff_out_name = 'ff_' + well_name_conv + f'_t{t:04}' + '.tif'

        if os.path.isfile(ff_out_name) and not overwrite_flag:
            print(f"Skipping time point {t} for well {well_name_conv}.")
            continue
        
        # get zyx data stack
        print("Starting to load zyx...")
        start = time.time()
        data_zyx_raw = np.squeeze(imObject.get_image_data("CZYX", T=t))
        print(time.time() - start)

        # each time directoy contains a list of Z slices for each channel
        ch_string = "CH" + str(ch_to_use)
        im_list = sorted(glob.glob(time_dir + "/*" + ch_string + "*"))

        # get pixel resolution 
        pixel_scale_vec = np.asarray(imObject.physical_pixel_sizes)

        # check to see if images have already been generated
        do_flags = [1] * len(sub_pos_index)
        # print(sub_pos_index)
        for pi in sub_pos_index:
            tt = int(time_dir[-4:])
            
            if cytometer_flag:
                # pos_id_list.append(p)
                pos_string = f'p{p:04}'
            else:
                # pos_id_list.append(pi)
                pos_string = f'p{pi:04}'

            
            if os.path.isfile(
                    os.path.join(ff_dir, ff_out_name, 'im_' + pos_string + '.tif')) and not overwrite_flag:
                do_flags[pi - 1] = 0
                if t == 0 and p == 0 and w == 0:
                    print("Skipping pre-existing files. Set 'overwrite_flag=True' to overwrite existing images")

        for pi in sub_pos_index:
            pos_indices = np.where(np.asarray(sub_pos_list) == pi)[0]
            # load
            images = []
            for iter_i, i in enumerate(pos_indices):
                if do_flags[pi - 1]:
                    im = cv2.imread(im_list[i])
                    if im is not None:
                        images.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

                # scrape metadata from first image
                if (iter_i == 0) and (p == 0):
                    temp_dict = scrape_keyence_metadata(im_list[i])
                    k_list = list(temp_dict.keys())
                    temp_df = pd.DataFrame(np.empty((1, len(k_list))), columns=k_list)
                    for k in k_list:
                        temp_df[k] = temp_dict[k]
                    
                    tstring = 'T' + time_dir[-4:]
                    temp_df["time_string"] = tstring
                    temp_df["time_int"] = int(time_dir[-4:])
    

                    temp_df["well"] = well_name_conv
                    temp_df = temp_df[temp_df.columns[::-1]]
                    # add to main dataframe
                    well_df.loc[master_iter_i] = temp_df.loc[0]
                    master_iter_i += 1

            if do_flags[pi - 1]:
                laps = []
                laps_d = []
                for i in range(len(images)):
                    # print
                    # "Lap {}".format(i)
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


def build_ff_from_yx1(data_root, overwrite_flag=False, ch_to_use=0, dir_list=None, write_dir=None):

    read_dir_root = os.path.join(data_root, 'raw_image_data', 'YX1') 
    if write_dir is None:
        write_dir = os.path.join(data_root, 'built_image_data', 'YX1') 
        
    # handle paths
    if dir_list == None:
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
        imObject = AICSImage(image_list[0])       
        n_wells = len(imObject.scenes)
        well_list = imObject.scenes
        # n_time_points = imObject.dims["T"][0]

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
        print(f'Building full-focus images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        # metadata_df_list = pmap(process_well, range(len(well_list)), (well_list, cytometer_flag, ff_dir, depth_dir, ch_to_use, overwrite_flag), rP=0.5)
        metadata_df_list = []
        for w in range(n_wells):
            metadata_well = process_well(w, imObject, well_list, well_name_list_sorted, ff_dir, depth_dir, ch_to_use=1, overwrite_flag=False)
            metadata_df_list.append(metadata_well)


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