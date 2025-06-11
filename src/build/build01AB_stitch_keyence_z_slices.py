# script to define functions_folder for loading and standardizing fish movies
import os
import numpy as np
from PIL import Image
from tqdm.contrib.concurrent import process_map 
from functools import partial
from src.functions.utilities import path_leaf
from src.functions.image_utils import gaussian_focus_stacker, LoG_focus_stacker
from tqdm import tqdm
from PIL import Image
import glob2 as glob
import cv2
from stitch2d import StructuredMosaic
import json
from tqdm import tqdm
import skimage.io as io
import pandas as pd
from stitch2d.tile import Tile

from src.build.keyence_export_utils import trim_to_shape



def trim_to_shape(im, out_shape):
    im_shape = im.shape
    im_diffs = im_shape - out_shape
    if np.any(np.abs(im_diffs) > 0):
        pad_width = -im_diffs
        pad_width[np.where(pad_width < 0)] = 0
        im_out = np.pad(im.copy(), ((0, pad_width[0]), (0, pad_width[1])), mode='constant').astype(im.dtype)

        im_diffs[np.where(im_diffs < 0)] = 0
        sv = np.floor(im_diffs / 2).astype(int)
        if np.all(sv>0):
            im_out = im_out[sv[0]:-(im_diffs[0] - sv[0]), sv[1]:-(im_diffs[1] - sv[1])]
        elif sv[0]==0:
            im_out = im_out[:, sv[1]:-(im_diffs[1] - sv[1])]
        elif sv[1]==0:
            im_out = im_out[sv[0]:-(im_diffs[0] - sv[0]), :]
    else:
        im_out = im
        
    return im_out[:out_shape[0], :out_shape[1]]

def stitch_well(w, well_list, cytometer_flag, out_dir, size_factor, ff_tile_dir, orientation, overwrite_flag=False):

    well_dir = well_list[w]
    # extract basic well info
    well_name = well_dir[-4:]

    # master_iter_i = 0

    # get conventional well name
    well_name_conv = sorted(glob.glob(os.path.join(well_dir, "_*")))
    well_name_conv = well_name_conv[0][-3:]

    # if multiple positions were taken per well, then there will be a layer of position folders
    position_dir_list = sorted(glob.glob(well_dir + "/P*"))
    if len(position_dir_list) == 0:
        position_dir_list = [well_dir]

    #####
    # load all paths into a parsable list object
    first_flag = True

    for p, pos_dir in enumerate(position_dir_list):
        # each pos dir contains one or more time points
        time_dir_list = sorted(glob.glob(pos_dir + "/T*"))
        time_dir_list = [t for t in time_dir_list if os.path.isdir(t)]
        no_timelapse_flag = len(time_dir_list) == 0

        if no_timelapse_flag:
            time_dir_list = [well_dir]

        for t, time_dir in enumerate(time_dir_list):

            # each time directoy contains a list of Z slices for each channel
            ch_string = "CH" #+ str(ch_to_use)
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

            if first_flag:
                if cytometer_flag:
                    n_pos = len(position_dir_list)
                else:
                    n_pos = len(sub_pos_index)
                well_path_list =  [[ [""] for i in range(n_pos) ] for i in range(len(time_dir_list)) ] #[[""]*n_pos]*len(time_dir_list)
                first_flag = False

            for sp, pi in enumerate(sub_pos_index):
                pos_indices = np.where(np.asarray(sub_pos_list) == pi)[0]
                # save paths
                image_paths = [im_list[pos_i] for pos_i in pos_indices]

                if cytometer_flag:
                    well_path_list[t][p] = image_paths
                else:
                    well_path_list[t][sp] = image_paths

    # get dim stats
    n_time_points = len(well_path_list)
    n_pos_tiles = len(well_path_list[0])
    n_z_slices = len(well_path_list[0][0])

    # set target stitched image size
    if n_pos_tiles == 1:
        out_shape = np.asarray([480, 640]) * size_factor
    elif n_pos_tiles == 2:
        out_shape = np.asarray([800, 630]) * size_factor
    elif n_pos_tiles == 3:
        if orientation == "vertical":
            out_shape = np.asarray([1140, 630]) * size_factor
        else: 
            out_shape = np.asarray([1140, 480]) * size_factor
    else:
        raise Exception("Unrecognized number of images to stitch")
    out_shape = out_shape.astype(int)
    
    for t in range(n_time_points):
        if n_time_points > 1:
            ff_out_name = 'ff_' + well_name_conv + f'_t{t+1:04}'
        else:
            ff_out_name = 'ff_' + well_name_conv + f'_t{t:04}'
        ff_tile_path = os.path.join(ff_tile_dir, ff_out_name, "")

        out_name = ff_out_name
        out_name = out_name.replace("ff_", "")
        out_name = out_name +  "_stack.tif"

        save_path = os.path.join(out_dir, out_name)

        if (not os.path.isfile(save_path)) or overwrite_flag: 
            z_slice_array = np.zeros((n_z_slices, out_shape[0], out_shape[1]))
            for z in range(n_z_slices):
                im_z_list = []
                for p in range(n_pos_tiles):
                    load_string = well_path_list[t][p][z]
                    if load_string == '/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/keyence/20230608/W045/P00003/T0040/wt_11ss_W045_P00003_T0040_Z001_CH1.tif':
                        im = np.zeros(out_shape, dtype=np.uint8) # handle a one-time issue with a corrupt tile
                    else:
                        im = io.imread(load_string)
                        out_dtype = im.dtype
                    im_z_list.append(Tile(im))

                n_images = len(im_z_list)

                z_mosaic = StructuredMosaic(
                        im_z_list,
                        dim=n_images,  # number of tiles in primary axis
                        origin="upper left",  # position of first tile
                        direction=orientation,
                        pattern="raster"
                    )

                if n_images > 1:

                    # load saved parameters
                    if os.path.isfile(os.path.join(ff_tile_path, "params.json")):
                        z_mosaic.load_params(os.path.join(ff_tile_path, "params.json"))
                    else:
                        z_mosaic.load_params(os.path.join(ff_tile_dir, "master_params.json"))
                    z_mosaic.smooth_seams()
                    
                    z_arr = z_mosaic.stitch()

                else:
                    z_arr = z_mosaic.stitch()

                if orientation == "horizontal":
                    z_arr = z_arr.T
                z_out = trim_to_shape(z_arr.astype(out_dtype), out_shape)
                z_slice_array[z, :, :] = z_out

            # save 
            io.imsave(save_path, z_slice_array, check_contrast=False)
        
                
    # well_dict_out = dict({well_name_conv: well_dict})

    return {}


def stitch_z_from_keyence(data_root, orientation_list, par_flag=False, n_workers=4, overwrite_flag=False, dir_list=None, write_dir=None):

    read_dir = os.path.join(data_root, 'raw_image_data', 'keyence', '') 
    # built_dir = os.path.join(data_root, 'built_image_data', 'keyence', '') 
    write_dir = data_root #os.path.join(data_root, 'built_image_data', 'keyence_stitched_z', '')
    
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
        sub_name = path_leaf(dir_list[d])
        dir_path = os.path.join(read_dir, sub_name, '')

        orientation = orientation_list[d]

        # depth_dir = os.path.join(write_dir, "D_images", sub_name)
        out_dir = os.path.join(write_dir, 'built_image_data', 'keyence_stitched_z', sub_name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # Each folder at this level pertains to a single well
        well_list = sorted(glob.glob(dir_path + "XY*"))
        cytometer_flag = False
        if len(well_list) == 0:
            cytometer_flag = True
            well_list = sorted(glob.glob(dir_path + "/W0*"))

        # get list of FF tile folders
        ff_tile_dir = os.path.join(write_dir, "built_image_data", "keyence", "FF_images", sub_name, '')
        metadata_path = os.path.join(data_root, 'metadata', 'built_metadata_files', sub_name + '_metadata.csv')
        metadata_df = pd.read_csv(metadata_path)
        size_factor = metadata_df["Width (px)"].iloc[0] / 640
        time_ind_index = np.unique(metadata_df["time_int"])
        # no_timelapse_flag = len(time_ind_index) == 1
        # if no_timelapse_flag:
        #     out_shape = np.asarray([800, 630])*size_factor
        # else:
        #     out_shape = np.asarray([1140, 630])*size_factor
        # out_shape = out_shape.astype(int)

        # get list of subfolders
        # ff_folder_list = sorted(glob.glob(ff_tile_dir + "ff*"))

        print(f'Stitching z slices in directory {d+1:01} of ' + f'{len(dir_indices)}')
        if not par_flag:
            for w in tqdm(range(len(well_list))):
                stitch_well(w, well_list=well_list, orientation=orientation, cytometer_flag=cytometer_flag, out_dir=out_dir, overwrite_flag=overwrite_flag, size_factor=size_factor, ff_tile_dir=ff_tile_dir)
                
        else:
            process_map(partial(stitch_well, well_list=well_list, orientation=orientation, cytometer_flag=cytometer_flag, 
                                                                        out_dir=out_dir, overwrite_flag=overwrite_flag, size_factor=size_factor, ff_tile_dir=ff_tile_dir), 
                                        range(len(well_list)), chunksize=1)

    print('Done.')


if __name__ == "__main__":

    overwrite_flag = True

    # data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    # dir_list = ["20230525", "20231207"]
    # # build FF images
    # build_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list)
    # # stitch FF images
    # stitch_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list)