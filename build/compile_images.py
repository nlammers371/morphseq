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

def doLap(image, lap_size=5, blur_size=5):

    # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
#     kernel_size = 5  # Size of the laplacian window
#     blur_size = 5  # How big of a kernal to use for the gaussian blur
    # Generally, keeping these two values the same or very close works well
    # Also, odd numbers, please...

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=lap_size)

def build_ff_from_keyence(read_path, db_path, data_folder, overwrite_flag=False, ch_to_use=1):

    # handle paths
    read_dir = read_path + data_folder + "/"

    # set number of sample folders to read
    n_stitch_samples = 500

    # Get a list of directories
    dir_list = sorted(glob.glob(read_dir + "*"))

    # filter for desired directories
    dir_indices = [d for d in range(len(dir_list)) if "ignore" not in dir_list[d]]

    for d in dir_indices:
        dir_path = dir_list[d]
        sub_name = dir_path.replace(read_dir, "")

        depth_dir = os.path.join(db_path[:-1], "built_data", "depth_images", data_folder + '_' + sub_name)
        ff_dir = os.path.join(db_path[:-1], "built_data", "FF_images", data_folder + '_' + sub_name)

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
            # stitch_flag = True
            # if len(position_dir_list) == 0:
            #     stitch_flag = False
            #     position_dir_list = [well_dir] # add dummy directory
            # elif len(position_dir_list) == 1:
            #     stitch_flag = False

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

                    sub_pos_index = np.unique(sub_pos_list)
                    # if len(sub_pos_index) > 1:      # switch stitching back on if there are multiple positions detected
                    #     stitch_flag = True

                    # load
                    images = []
                    for i in range(len(im_list)):
                        im = cv2.imread(im_list[i])
                        images.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

                    laps = []
                    for i in range(len(images)):
                        # print
                        # "Lap {}".format(i)
                        laps.append(doLap(images[i]))

                    laps = np.asarray(laps)
                    abs_laps = np.absolute(laps)
                    for pi in sub_pos_index:

                        # calculat full-focus and depth images
                        ff_image = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
                        pos_indices = np.where(np.asarray(sub_pos_list) == pi)[0]
                        abs_laps_pos = abs_laps[pos_indices]
                        depth_image = np.argmax(abs_laps_pos, axis=0)
                        maxima = abs_laps_pos.max(axis=0)
                        bool_mask = abs_laps_pos == maxima
                        mask = bool_mask.astype(np.uint8)
                        for i in range(len(pos_indices)):
                            ff_image[np.where(mask[i] == 1)] = images[pos_indices[i]][np.where(mask[i] == 1)]

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
                        # if op_ff not in fn_ff_list:
                        #     fn_ff_list.append(op_ff)
                        #     time_id_list.append(tt)
                        #     fn_depth_list.append(op_depth)

                        # convet depth image to 8 bit
                        max_z = abs_laps_pos.shape[0]
                        depth_image_int8 = np.round(depth_image/max_z*255).astype('uint8')

                        cv2.imwrite(os.path.join(ff_dir, ff_out_name, 'im_' + pos_string + '.tif'), ff_image)
                        cv2.imwrite(os.path.join(depth_dir, depth_out_name, 'im_' + pos_string + '.tif'), depth_image_int8)

    print('Done.')
    print('Estimating stitching prior...')
    for d in dir_indices:
        dir_path = dir_list[d]
        sub_name = dir_path.replace(read_dir, "")

        # directories containing image tiles
        depth_tile_dir = os.path.join(db_path[:-1], "built_data", "depth_images", data_folder + '_' + sub_name, '')
        ff_tile_dir = os.path.join(db_path[:-1], "built_data", "FF_images", data_folder + '_' + sub_name, '')

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
        depth_tile_dir = os.path.join(db_path[:-1], "built_data", "depth_images", data_folder + '_' + sub_name, '')
        ff_tile_dir = os.path.join(db_path[:-1], "built_data", "FF_images", data_folder + '_' + sub_name, '')

        # directories to write stitched files to
        stitch_depth_dir = os.path.join(db_path[:-1], "built_data", "stitched_depth_images",
                                        data_folder + '_' + sub_name)
        stitch_ff_dir = os.path.join(db_path[:-1], "built_data", "stitched_FF_images", data_folder + '_' + sub_name)

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
            if len(ff_mosaic.params["coords"]) != 3:
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

            ff_out_name = ff_name[3:-1] + '_stitch.tif'
            depth_out_name = depth_name[6:-1] + '_stitch.tif'
            ff_mosaic.save(stitch_ff_dir + '/' + ff_out_name)
            depth_mosaic.save(stitch_depth_dir + '/' + depth_out_name)
            # cv2.imwrite(os.path.join(stitch_ff_dir, ff_out_name), im_ff)
            # cv2.imwrite(os.path.join(stitch_depth_dir, depth_out_name), im_depth)


if __name__ == "__main__":
    overwrite_flag = True

    # set path to excel doc with metadata
    db_path = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphSeq/data/"
    # read_path = "/Volumes/LaCie/Keyence/"
    read_path = db_path
    # excel_path = db_path + "Nick/morphSeq/data/embryo_metadata.xlsx"

    # set path to save images
    # data_folder = "20230518"
    data_folder = "20230525"

    ch_to_include = [1]  # ,2,3]

    build_ff_from_keyence(read_path, db_path, data_folder, overwrite_flag)