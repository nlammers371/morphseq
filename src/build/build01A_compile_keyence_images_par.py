# script to define functions_folder for loading and standardizing fish movies
import os
import numpy as np
from PIL import Image
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import matplotlib
from src.functions.utilities import path_leaf
from tqdm import tqdm
from PIL import Image
import glob2 as glob
import cv2
from stitch2d import StructuredMosaic
import json
from tqdm import tqdm
import pickle
from parfor import pmap
import pandas as pd

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

def doLap(image, lap_size=3, blur_size=3):

    # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
#     kernel_size = 5  # Size of the laplacian window
#     blur_size = 5  # How big of a kernal to use for the gaussian blur
    # Generally, keeping these two values the same or very close works well
    # Also, odd numbers, please...

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_size)

def process_well(w, well_list, cytometer_flag, ff_dir, depth_dir, ch_to_use=1, overwrite_flag=False, no_timelapse_flag=False):

    well_dir = well_list[w]
    # extract basic well info
    well_name = well_dir[-4:]
    # well_num = well_name[-2:]
    # well_dict = dict({})
    well_df = pd.DataFrame([], columns=['well', 'time_int', 'time_string', 'Height (um)', 'Width (um)', 'Height (px)', 'Width (px)', 'Channel', 'Objective', 'Time (s)'])
    master_iter_i = 0

    # get conventional well name
    well_name_conv = sorted(glob.glob(os.path.join(well_dir, "_*")))
    well_name_conv = well_name_conv[0][-3:]

    # if multiple positions were taken per well, then there will be a layer of position folders
    position_dir_list = sorted(glob.glob(well_dir + "/P*"))
    if no_timelapse_flag:
        position_dir_list = [""]
    for p, pos_dir in enumerate(position_dir_list):

        # each pos dir contains one or more time points
        time_dir_list = sorted(glob.glob(pos_dir + "/T*"))
        if no_timelapse_flag:
            time_dir_list = [well_dir]
        for t, time_dir in enumerate(time_dir_list):

            # each time directoy contains a list of Z slices for each channel
            ch_string = "CH" + str(ch_to_use)
            im_list = sorted(glob.glob(time_dir + "/*" + ch_string + "*"))

            # it is possible that multiple positionas are present within the same time folder
            if not cytometer_flag:
                sub_pos_list = []
                for i, im in enumerate(im_list):
                    im_name = im.replace(time_dir, "")
                    well_ind = im_name.find(well_name)
                    pos_id = int(im_name[well_ind + 5:well_ind + 10])
                    sub_pos_list.append(pos_id)
            else:
                sub_pos_list = np.ones((len(im_list),))

            sub_pos_index = np.unique(sub_pos_list).astype(int)

            # check to see if images have already been generated
            do_flags = [1] * len(sub_pos_index)
            # print(sub_pos_index)
            for pi in sub_pos_index:
                if not no_timelapse_flag:
                    tt = int(time_dir[-4:])
                else:
                    tt = 0
                if cytometer_flag:
                    # pos_id_list.append(p)
                    pos_string = f'p{p:04}'
                else:
                    # pos_id_list.append(pi)
                    pos_string = f'p{pi:04}'

                ff_out_name = 'ff_' + well_name_conv + f'_t{tt:04}_' + f'ch{ch_to_use:02}/'
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
                        # if (t == 0) and (w == 0):
                        #     base_time = temp_dict["Time (s)"]
                        # temp_dict["Time (s)"] = temp_dict["Time (s)"]# - base_time
                        # add to main dictionary
                        if no_timelapse_flag:
                            tstring = 'T0'
                            temp_df["time_string"] = tstring
                            temp_df["time_int"] = 0
                        else:
                            tstring = 'T' + time_dir[-4:]
                            temp_df["time_string"] = tstring
                            temp_df["time_int"] = int(time_dir[-4:])

                        temp_df["well"] = well_name_conv
                        temp_df = temp_df[temp_df.columns[::-1]]
                        # add to main dataframe
                        well_df.loc[master_iter_i] = temp_df.loc[0]
                        master_iter_i += 1
                            # temp_df = temp_df[temp_df.columns[::-1]]
                            # well_dict[tstring] = temp_dict
                             # metadata_dict[well_name_conv] = temp_dict2

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

                    if not no_timelapse_flag:
                        tt = int(time_dir[-4:])
                    else:
                        tt = 0
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


def stitch_experiment(t, ff_folder_list, ff_tile_dir, depth_folder_list, depth_tile_dir, stitch_ff_dir,
                      stitch_depth_dir, overwrite_flag, out_shape):


    # time_indices = np.where(np.asarray(time_id_list) == tt)[0]
    ff_path = os.path.join(ff_folder_list[t], '')
    ff_name = path_leaf(ff_path)
    n_images = len(glob.glob(ff_path + '*.tif'))
    depth_path = os.path.join(depth_folder_list[t], '')
    depth_name = depth_path.replace(depth_tile_dir, "")

    ff_out_name = ff_name[3:-1] + '_stitch.tif'
    depth_out_name = depth_name[6:-1] + '_stitch.tif'

    if not os.path.isfile(os.path.join(stitch_ff_dir, ff_out_name)) or overwrite_flag:

        # perform stitching
        ff_mosaic = StructuredMosaic(
            ff_path,
            dim=n_images,  # number of tiles in primary axis
            origin="upper left",  # position of first tile
            direction="vertical",
            pattern="raster"
        )

        try:
            # mosaic.downsample(0.6)
            ff_mosaic.align()
        except:
            pass

        try:  # NL: skipping one problematic well for now
            default_flag = False
            if len(ff_mosaic.params["coords"]) != n_images:
                default_flag = True
            else:
                c_params = ff_mosaic.params["coords"]
                if n_images == 3:
                    lr_shifts = np.asarray([c_params[0][1], c_params[1][1], c_params[2][1]])
                    if np.max(lr_shifts) > 2:
                        default_flag = True
                # elif n_images == 2:
                #     lr_shifts = np.asarray([c_params[0][1], c_params[1][1]])
                #     ud_shifts = np.asarray([c_params[0][0], c_params[1][0]])
                    # if np.max(lr_shifts) > 1:
                    #     default_flag = True

            if default_flag:
                ff_mosaic.load_params(ff_tile_dir + "/master_params.json")

            ff_mosaic.reset_tiles()
            ff_mosaic.save_params(ff_path + 'params.json')
            # ff_mosaic.save_params(path=depth_path)
            ff_mosaic.smooth_seams()

            # perform stitching
            depth_mosaic = StructuredMosaic(
                depth_path,
                dim=n_images,  # number of tiles in primary axis
                origin="upper left",  # position of first tile
                direction="vertical",
                pattern="raster"
            )

            # mosaic.downsample(0.6)
            depth_mosaic.load_params(ff_path + 'params.json')
            depth_mosaic.smooth_seams()

            # name_start_ind = ff_path.find("/ff_")
            # well_name = ff_path[name_start_ind+4:name_start_ind+7]

            # trim to standardize the size
            ff_arr = ff_mosaic.stitch()
            ff_out = trim_image(ff_arr, out_shape)

            # invert
            ff_out = 255 - ff_out

            depth_arr = depth_mosaic.stitch()
            depth_out = trim_image(depth_arr, out_shape)

            cv2.imwrite(os.path.join(stitch_ff_dir, ff_out_name), ff_out)
            cv2.imwrite(os.path.join(stitch_depth_dir, depth_out_name), depth_out)
        except:
            pass

    return{}

def build_ff_from_keyence(data_root, overwrite_flag=False, ch_to_use=1, dir_list=None, write_dir=None, no_timelapse_flag=False):

    read_dir = os.path.join(data_root, 'raw_image_data', 'keyence', '') 
    if write_dir is None:
        write_dir = os.path.join(data_root, 'built_image_data', 'keyence', '')
        
    # # handle paths
    # if dir_list == None:
    #     # Get a list of directories
    #     dir_list_raw = sorted(glob.glob(read_dir + "*"))
    #     dir_list = []
    #     for dd in dir_list_raw:
    #         if os.path.isdir(dd):
    #             dir_list.append(dd)

    # filter for desired directories
    dir_indices = [d for d in range(len(dir_list)) if "ignore" not in dir_list[d]]

    for d in dir_indices:
        # initialize dictionary to metadata
        # metadata_dict = dict({})
        sub_name = dir_list[d]
        dir_path = os.path.join(read_dir, sub_name, '')

        depth_dir = os.path.join(write_dir, "D_images", sub_name)
        ff_dir = os.path.join(write_dir, "FF_images",  sub_name)

        if not os.path.isdir(depth_dir):
            os.makedirs(depth_dir)
        if not os.path.isdir(ff_dir):
            os.makedirs(ff_dir)

        # Each folder at this level pertains to a single well
        well_list = sorted(glob.glob(dir_path + "XY*"))
        cytometer_flag = False
        if len(well_list) == 0:
            cytometer_flag = True
            well_list = sorted(glob.glob(dir_path + "/W0*"))

        print(f'Building full-focus images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        # for w in tqdm(range(len(well_list))):
        # (w, well_list, cytometer_flag, ff_dir, depth_dir, ch_to_use=1)
        # metadata_dict_list = []
        metadata_df_list = []
        for w in tqdm(range(len(well_list))):
            temp_df = process_well(w, well_list, cytometer_flag, ff_dir, depth_dir, ch_to_use, overwrite_flag, no_timelapse_flag)
            metadata_df_list.append(temp_df)

        # metadata_df_list = pmap(process_well, range(len(well_list)), (well_list, cytometer_flag, ff_dir, depth_dir, ch_to_use, overwrite_flag), rP=0.5)
        if len(metadata_df_list) > 0:
            metadata_df = pd.concat(metadata_df_list)
            first_time = np.min(metadata_df['Time (s)'].copy())
            metadata_df['Time Rel (s)'] = metadata_df['Time (s)'] - first_time
        else:
            metadata_df = []

        # load previous metadata
        metadata_path = os.path.join(ff_dir, 'metadata.csv')
        # if os.path.isfile(metadata_path) and len(metadata_df) > 0:
        #     prev_metadata = pd.read_csv(metadata_path, index_col=0)
        #     updated_metadata_df = pd.concat([metadata_df, prev_metadata])
        #     updated_metadata_df.drop_duplicates(subset=["well", "Time (s)"])

        if len(metadata_df) > 0:
            metadata_df.reset_index()
            metadata_df.to_csv(metadata_path)
        # with open(os.path.join(ff_dir, 'metadata.pickle'), 'wb') as handle:
        #     pickle.dump(metadata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('Done.')

def stitch_ff_from_keyence(data_root, overwrite_flag=False, n_stitch_samples=15, dir_list=None, write_dir=None, no_timelapse_flag=True):
    
    read_dir = os.path.join(data_root, 'raw_image_data', 'keyence')
    if write_dir is None:
        write_dir = data_root

    # handle paths
    if dir_list == None:
        # Get a list of directories
        dir_list_raw = sorted(glob.glob(read_dir + "*"))
        dir_list = []
        for dd in dir_list_raw:
            if os.path.isdir(dd):
                dir_list.append(dd)

    # print('Estimating stitching prior...')
    dir_indices = [d for d in range(len(dir_list)) if "ignore" not in dir_list[d]]
    for d in dir_indices:
        sub_name = dir_list[d]
        # sub_name = dir_path.replace(read_dir, "")

        # directories containing image tiles
        depth_tile_dir = os.path.join(write_dir, "built_image_data", "keyence", "D_images", sub_name, '')
        ff_tile_dir = os.path.join(write_dir, "built_image_data", "keyence", "FF_images", sub_name, '')

        
        metadata_path = os.path.join(ff_tile_dir, 'metadata.csv')
        metadata_df = pd.read_csv(metadata_path, index_col=0)
        size_factor = metadata_df["Width (px)"].iloc[0] / 640
        if no_timelapse_flag:
            out_shape = np.asarray([800, 630])*size_factor
        else:
            out_shape = np.asarray([1140, 630])*size_factor
        out_shape = out_shape.astype(int)

        # get list of subfolders
        # depth_folder_list = sorted(glob.glob(depth_tile_dir + "depth*"))
        ff_folder_list = sorted(glob.glob(ff_tile_dir + "ff*"))

        # # if not no_timelapse_flag:
        if not os.path.isfile(ff_tile_dir + "/master_params.json") or overwrite_flag:

            # select a random set of directories to iterate through to estimate stitching prior
            folder_options = range(len(ff_folder_list))
            stitch_samples = np.random.choice(folder_options, np.min([n_stitch_samples, len(folder_options)]), replace=False)


            print(f'Estimating stitch priors for images in directory {d+1:01} of ' + f'{len(dir_indices)}')
            for n in tqdm(range(len(stitch_samples))):
                im_ind = stitch_samples[n]
                ff_path = ff_folder_list[im_ind]
                n_images = len(glob.glob(ff_path + '/*.tif'))
                if n == 0:
                    align_array = np.empty((n_stitch_samples, 2, n_images))
                    align_array[:] = np.nan
                n_pass = 0
                # perform stitching
                ff_mosaic = StructuredMosaic(
                    ff_path,
                    dim=n_images,  # number of tiles in primary axis
                    origin="upper left",  # position of first tile
                    direction="vertical",
                    pattern="raster"
                )
                try:
                    ff_mosaic.align()
                    n_pass += 1
                    # ff_mosaic.reset_tiles()
                except:
                    pass

                if (len(ff_mosaic.params["coords"]) == n_images) and n_images > 1:       # NL: need to make this more general eventually
                    c_params = ff_mosaic.params["coords"]
                    for c in range(len(c_params)):
                        align_array[n, :, c] = c_params[c]

            # now make a master mosaic to use when alignment fails
            master_mosaic = ff_mosaic.copy()
            master_params = master_mosaic.params
            # calculate median parameters for each tile
            med_coords = np.nanmedian(align_array, axis=0)
            c_dict = dict({})
            for c in range(align_array.shape[2]):
                c_dict[c] =  med_coords[:, c].tolist()
            master_params["coords"] = c_dict

            # save params for subsequent use
            jason_params = json.dumps(master_params)
            # Writing to json
            with open(ff_tile_dir + "/master_params.json", "w") as outfile:
                outfile.write(jason_params)

            # Writing to json
            with open(depth_tile_dir + "/master_params.json", "w") as outfile:
                outfile.write(jason_params)

    # # Now perform the stitching
    # print('Done.')

    # for d in dir_indices:
    #     dir_path = dir_list[d]
    #     sub_name = dir_path.replace(read_dir, "")

    #     # directories containing image tiles
    #     depth_tile_dir = os.path.join(write_dir, "built_keyence_data", "D_images", sub_name, '')
    #     ff_tile_dir = os.path.join(write_dir, "built_keyence_data", "FF_images", sub_name, '')


        # directories to write stitched files to
        stitch_depth_dir = os.path.join(write_dir, "built_image_data", "stitched_depth_images", sub_name)
        stitch_ff_dir = os.path.join(write_dir, "built_image_data", "stitched_FF_images", sub_name)

        if not os.path.isdir(stitch_depth_dir):
            os.makedirs(stitch_depth_dir)
        if not os.path.isdir(stitch_ff_dir):
            os.makedirs(stitch_ff_dir)

        # get list of subfolders
        depth_folder_list = sorted(glob.glob(depth_tile_dir + "depth*"))
        ff_folder_list = sorted(glob.glob(ff_tile_dir + "ff*"))

        print(f'Stitching images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        # Call parallel function to stitch images
        if no_timelapse_flag:
            # out_shape[0] = 1230
            for f in tqdm(range(len(ff_folder_list))):
                stitch_experiment(f, ff_folder_list, ff_tile_dir, depth_folder_list, depth_tile_dir, stitch_ff_dir,
                                  stitch_depth_dir, overwrite_flag, out_shape)
        else:
            raise Warning("Some compute environments may not be compatible with parfor pmap function")
            pmap(stitch_experiment, range(len(ff_folder_list)), (ff_folder_list, ff_tile_dir, depth_folder_list,
                              depth_tile_dir, stitch_ff_dir,
                              stitch_depth_dir, overwrite_flag, out_shape), rP=0.5)
            
            # pmap(stitch_experiment, range(len(ff_folder_list)),
            #                             (ff_folder_list, ff_tile_dir, depth_folder_list, depth_tile_dir, stitch_ff_dir,
            #                              stitch_depth_dir, overwrite_flag, out_shape), rP=0.5)

        # stitch_experiment(1, ff_folder_list, ff_tile_dir, depth_folder_list, depth_tile_dir, stitch_ff_dir,
        #  stitch_depth_dir, overwrite_flag, out_shape)



if __name__ == "__main__":

    overwrite_flag = True

    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = ["lmx1b_20231207", "lmx1b_20231208"]
    # build FF images
    build_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list, ch_to_use=3, no_timelapse_flag=True)
    # stitch FF images
    stitch_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list)