# script to define functions for loading and standardizing fish movies
import os
import numpy as np
from PIL import Image
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import matplotlib
from tqdm import tqdm
import glob2 as glob
import cv2
from stitch2d import StructuredMosaic
import json
from tqdm import tqdm
import pickle

def scrape_keyence_metadata(im_path):
    # im_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/20230531/bf_timeseries_stack0850_pitch040/W001/P00001/T0004/wt_W001_P00001_T0004_Z005_CH1.tif"
    with open(im_path, 'rb') as a:
        fulldata = a.read()
    metadata = fulldata.partition(b'<Data>')[2].partition(b'</Data>')[0].decode()

    meta_dict = dict({})
    keyword_list = ['ShootingDateTime', 'LensName', 'Observation Type', 'Width', 'Height']
    outname_list = ['Time (s)', 'Objective', 'Channel', 'Width (um)', 'Height (um)']

    for k in range(len(keyword_list)):
        param_string = keyword_list[k]
        name = outname_list[k]

        if (param_string == 'Width') or (param_string == 'Height'):
            ind1 = findnth(metadata, param_string + ' Type', 2)
            ind2 = findnth(metadata, '/' + param_string, 2)
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
        elif (param_string=='Height') or (param_string=='Width'):
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
    im_out = im.copy()
    sv = np.floor(im_diffs / 2).astype(int)
    im_out = im_out[sv[0]:-(im_diffs[0] - sv[0]), sv[1]:-(im_diffs[1] - sv[1])]

    return im_out

def doLap(image, lap_size=5, blur_size=5):

    # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
#     kernel_size = 5  # Size of the laplacian window
#     blur_size = 5  # How big of a kernal to use for the gaussian blur
    # Generally, keeping these two values the same or very close works well
    # Also, odd numbers, please...

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_size)

def build_ff_from_keyence(read_dir, db_path, overwrite_flag=False, ch_to_use=1, n_stitch_samples=500,
                          out_shape=None, dir_list=None):

    # handle paths
    if dir_list == None:
        # Get a list of directories
        dir_list_raw = sorted(glob.glob(read_dir + "*"))
        dir_list = []
        for dd in dir_list_raw:
            if os.path.isdir(dd):
                dir_list.append(dd)

    # filter for desired directories
    dir_indices = [d for d in range(len(dir_list)) if "ignore" not in dir_list[d]]

    for d in dir_indices:
        # initialize dictionary to metadata
        metadata_dict = dict({})

        dir_path = dir_list[d]
        sub_name = dir_path.replace(read_dir, "")

        depth_dir = os.path.join(db_path[:-1], "built_keyence_data", "D_images", sub_name)
        ff_dir = os.path.join(db_path[:-1], "built_keyence_data", "FF_images",  sub_name)

        if not os.path.isdir(depth_dir):
            os.makedirs(depth_dir)
        if not os.path.isdir(ff_dir):
            os.makedirs(ff_dir)


        # Each folder at this level pertains to a single well
        well_list = sorted(glob.glob(dir_path + "/XY*"))
        cytometer_flag = False
        if len(well_list) == 0:
            cytometer_flag = True
            well_list = sorted(glob.glob(dir_path + "/W0*"))

        print(f'Building full-focus images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        for w in tqdm(range(len(well_list))):

            well_dir = well_list[w]
            # extract basic well info
            well_name = well_dir[-4:]
            # well_num = well_name[-2:]

            # get conventional well name
            well_name_conv = sorted(glob.glob(well_dir + "/_*"))
            well_name_conv = well_name_conv[0][-3:]

            # if multiple positions were taken per well, then there will be a layer of position folders
            position_dir_list = sorted(glob.glob(well_dir + "/P*"))

            for p, pos_dir in enumerate(position_dir_list):

                # each pos dir contains one or more time points
                time_dir_list = sorted(glob.glob(pos_dir + "/T*"))

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
                    do_flags = [1]*len(sub_pos_index)
                    # print(sub_pos_index)
                    for pi in sub_pos_index:
                        tt = int(time_dir[-4:])
                        if cytometer_flag:
                            # pos_id_list.append(p)
                            pos_string = f'p{p:04}'
                        else:
                            # pos_id_list.append(pi)
                            pos_string = f'p{pi:04}'

                        ff_out_name = 'ff_' + well_name_conv + f'_t{tt:04}_' + f'ch{ch_to_use:02}/'
                        if os.path.isfile(os.path.join(ff_dir, ff_out_name, 'im_' + pos_string + '.tif')) and not overwrite_flag:
                            do_flags[pi-1] = 0
                            if t==0 and p==0 and w==0:
                                print("Skipping pre-existing files. Set 'overwrite_flag=True' to overwrite existing images")


                    for pi in sub_pos_index:
                            pos_indices = np.where(np.asarray(sub_pos_list) == pi)[0]
                            # load
                            images = []
                            for iter_i, i in enumerate(pos_indices):
                                if do_flags[pi - 1]:
                                    im = cv2.imread(im_list[i])
                                    images.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

                                # scrape metadata from first image
                                if (iter_i == 0) and (p == 0):
                                    temp_dict = scrape_keyence_metadata(im_list[i])
                                    if (t == 0) and (w == 0):
                                        base_time = temp_dict["Time (s)"]
                                    temp_dict["Time (s)"] = temp_dict["Time (s)"] - base_time
                                    # add to main dictionary
                                    tstring = 'T' + time_dir[-4:]
                                    if t == 0:
                                        metadata_dict[well_name_conv] = dict({tstring: temp_dict})
                                    else:
                                        temp_dict2 = metadata_dict[well_name_conv]
                                        temp_dict2[tstring] = temp_dict
                                        metadata_dict[well_name_conv] = temp_dict2

                            if do_flags[pi - 1]:
                                laps = []
                                for i in range(len(images)):
                                    # print
                                    # "Lap {}".format(i)
                                    laps.append(doLap(images[i]))

                                laps = np.asarray(laps)
                                abs_laps = np.absolute(laps)

                                # calculat full-focus and depth images
                                ff_image = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
                                depth_image = np.argmax(abs_laps, axis=0)
                                maxima = abs_laps.max(axis=0)
                                bool_mask = abs_laps == maxima
                                mask = bool_mask.astype(np.uint8)
                                for i in range(len(images)):
                                    ff_image[np.where(mask[i] == 1)] = images[i][np.where(mask[i] == 1)]

                                ff_image = 255-ff_image

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
                                depth_image_int8 = np.round(depth_image/max_z*255).astype('uint8')

                                cv2.imwrite(os.path.join(ff_dir, ff_out_name, 'im_' + pos_string + '.tif'), ff_image)
                                cv2.imwrite(os.path.join(depth_dir, depth_out_name, 'im_' + pos_string + '.tif'), depth_image_int8)

        with open(os.path.join(ff_dir, 'metadata.pickle'), 'wb') as handle:
            pickle.dump(metadata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(depth_dir, 'metadata.pickle'), 'wb') as handle:
            pickle.dump(metadata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')

def stitch_ff_from_keyence(read_dir, db_path, overwrite_flag=False, ch_to_use=1, n_stitch_samples=500,
                           out_shape=None, dir_list=None):

    if out_shape == None:
        out_shape = np.asarray([1140, 630])

    # handle paths
    if dir_list == None:
        # Get a list of directories
        dir_list_raw = sorted(glob.glob(read_dir + "*"))
        dir_list = []
        for dd in dir_list_raw:
            if os.path.isdir(dd):
                dir_list.append(dd)

    print('Estimating stitching prior...')
    dir_indices = [d for d in range(len(dir_list)) if "ignore" not in dir_list[d]]
    for d in dir_indices:
        dir_path = dir_list[d]
        sub_name = dir_path.replace(read_dir, "")

        # directories containing image tiles
        depth_tile_dir = os.path.join(db_path[:-1], "built_keyence_data", "D_images", sub_name, '')
        ff_tile_dir = os.path.join(db_path[:-1], "built_keyence_data", "FF_images", sub_name, '')

        # get list of subfolders
        # depth_folder_list = sorted(glob.glob(depth_tile_dir + "depth*"))
        ff_folder_list = sorted(glob.glob(ff_tile_dir + "ff*"))

        # select a random set of directories to iterate through to estimate stitching prior
        folder_options = range(len(ff_folder_list))
        stitch_samples = np.random.choice(folder_options, np.min([n_stitch_samples, len(folder_options)]), replace=False)

        align_array = np.empty((n_stitch_samples, 2, 3))
        align_array[:] = np.nan

        print(f'Estimating stitch priors for images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        for n in tqdm(range(len(stitch_samples))):
            im_ind = stitch_samples[n]
            ff_path = ff_folder_list[im_ind]
            n_images = len(glob.glob(ff_path + '/*.tif'))
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

            if len(ff_mosaic.params["coords"]) == 3:       # NL: need to make this more general eventually
                c_params = ff_mosaic.params["coords"]
                for c in range(len(c_params)):
                    align_array[n, :, c] = c_params[c]

        # now make a master mosaic to use when alignment fails
        master_mosaic = ff_mosaic.copy()
        master_params = master_mosaic.params
        # calculate median parameters for each tile
        med_coords = np.nanmedian(align_array, axis=0)
        c_dict = {0: med_coords[:, 0].tolist(),
                  1: med_coords[:, 1].tolist(),
                  2: med_coords[:, 2].tolist()}
        master_params["coords"] = c_dict

        # save params for subsequent use
        jason_params = json.dumps(master_params)
        # Writing to json
        with open(ff_tile_dir + "/master_params.json", "w") as outfile:
            outfile.write(jason_params)

        # Writing to json
        with open(depth_tile_dir + "/master_params.json", "w") as outfile:
            outfile.write(jason_params)

    # Now perform the stitching
    print('Done.')

    for d in dir_indices:
        dir_path = dir_list[d]
        sub_name = dir_path.replace(read_dir, "")

        # directories containing image tiles
        depth_tile_dir = os.path.join(db_path[:-1], "built_keyence_data", "D_images", sub_name, '')
        ff_tile_dir = os.path.join(db_path[:-1], "built_keyence_data", "FF_images", sub_name, '')

        # directories to write stitched files to
        stitch_depth_dir = os.path.join(db_path[:-1], "built_keyence_data", "stitched_depth_images", sub_name)
        stitch_ff_dir = os.path.join(db_path[:-1], "built_keyence_data", "stitched_FF_images", sub_name)

        if not os.path.isdir(stitch_depth_dir):
            os.makedirs(stitch_depth_dir)
        if not os.path.isdir(stitch_ff_dir):
            os.makedirs(stitch_ff_dir)

        # get list of subfolders
        depth_folder_list = sorted(glob.glob(depth_tile_dir + "depth*"))
        ff_folder_list = sorted(glob.glob(ff_tile_dir + "ff*"))

        print(f'Stitching images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        for t in tqdm(range(len(ff_folder_list))):

            # time_indices = np.where(np.asarray(time_id_list) == tt)[0]
            ff_path = os.path.join(ff_folder_list[t], '')
            ff_name = ff_path.replace(ff_tile_dir, "")
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

                default_flag = False
                if len(ff_mosaic.params["coords"]) != 3:
                    default_flag = True
                else:
                    c_params = ff_mosaic.params["coords"]
                    lr_shifts = np.asarray([c_params[0][1], c_params[1][1], c_params[2][1]])
                    if np.max(lr_shifts) > 2:
                        default_flag = True

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

                name_start_ind = ff_path.find("/ff_")
                well_name = ff_path[name_start_ind+4:name_start_ind+7]

                # trim to standardie the size
                ff_arr = ff_mosaic.stitch()
                ff_out = trim_image(ff_arr, out_shape)

                depth_arr = depth_mosaic.stitch()
                depth_out = trim_image(depth_arr, out_shape)

                cv2.imwrite(os.path.join(stitch_ff_dir, ff_out_name), ff_out)
                cv2.imwrite(os.path.join(stitch_depth_dir, depth_out_name), depth_out)


if __name__ == "__main__":

    overwrite_flag = False

    # set path to excel doc with metadata
    db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/"
    # read_path = "/Volumes/LaCie/Keyence/"
    read_dir = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/raw_keyence_data/"
    # excel_path = db_path + "Nick/morphSeq/data/embryo_metadata.xlsx"
    genotype = 'wt_pseudo'
    # set path to save images
    # data_folder = "20230518"
    # data_folder = "20230525"

    # ch_to_use = [1]  # ,2,3]

    # build FF images
    build_ff_from_keyence(read_dir, db_path, overwrite_flag=False)
    # stitch FF images
    stitch_ff_from_keyence(read_dir, db_path, overwrite_flag=False)