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
import pickle
from parfor import pmap
import skimage.io as io
import pandas as pd
from stitch2d.tile import Tile



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

def stitch_well(w, well_list, cytometer_flag, out_dir, out_shape, ff_tile_dir, overwrite_flag=False):

    well_dir = well_list[w]
    # extract basic well info
    well_name = well_dir[-4:]

    master_iter_i = 0

    # get conventional well name
    well_name_conv = sorted(glob.glob(os.path.join(well_dir, "_*")))
    well_name_conv = well_name_conv[0][-3:]

    # if multiple positions were taken per well, then there will be a layer of position folders
    position_dir_list = sorted(glob.glob(well_dir + "/P*"))
    if len(position_dir_list) == 0:
        position_dir_list = [""]

    #####
    # load all paths into a parsable list object
    first_flag = True

    for p, pos_dir in enumerate(position_dir_list):
        # each pos dir contains one or more time points
        time_dir_list = sorted(glob.glob(pos_dir + "/T*"))
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

    for t in range(n_time_points):
        
        ff_out_name = 'ff_' + well_name_conv + f'_t{t+1:04}'
        ff_tile_path = os.path.join(ff_tile_dir, ff_out_name, "")

        z_slice_array = np.zeros((n_z_slices, out_shape[0], out_shape[1]), dtype=np.uint8)
        for z in range(n_z_slices):
            im_z_list = []
            for p in range(n_pos_tiles):
                im = io.imread(well_path_list[t][p][z])
                im_z_list.append(Tile(im))

            n_images = len(im_z_list)
            # initialize mosaic
            z_mosaic = StructuredMosaic(
                im_z_list,
                dim=n_images,  # number of tiles in primary axis
                origin="upper left",  # position of first tile
                direction="vertical",
                pattern="raster"
            )

            # load saved parameters
            z_mosaic.load_params(os.path.join(ff_tile_path, "params.json"))
            z_mosaic.reset_tiles()
            z_mosaic.smooth_seams()
            z_arr = z_mosaic.stitch()
            z_out = trim_image(z_arr, out_shape)
            z_slice_array[z, :, :] = z_out

        # save 
        out_name = ff_out_name
        out_name = out_name.replace("ff_", "")
        out_name = out_name +  "_stack.tif"
        io.imsave(os.path.join(out_dir, out_name), z_slice_array, check_contrast=False)
                
    # well_dict_out = dict({well_name_conv: well_dict})

    return {}


def stitch_z_from_keyence(data_root, par_flag=False, n_workers=4, overwrite_flag=False, dir_list=None, write_dir=None,):

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
        # metadata_dict = dict({})
        sub_name = path_leaf(dir_list[d])
        dir_path = os.path.join(read_dir, sub_name, '')

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
        metadata_path = os.path.join(ff_tile_dir, 'metadata.csv')
        metadata_df = pd.read_csv(metadata_path, index_col=0)
        size_factor = metadata_df["Width (px)"].iloc[0] / 640
        time_ind_index = np.unique(metadata_df["time_int"])
        no_timelapse_flag = len(time_ind_index) == 0
        if no_timelapse_flag:
            out_shape = np.asarray([800, 630])*size_factor
        else:
            out_shape = np.asarray([1140, 630])*size_factor
        out_shape = out_shape.astype(int)

        # get list of subfolders
        # ff_folder_list = sorted(glob.glob(ff_tile_dir + "ff*"))

        print(f'Stitching z slices in directory {d+1:01} of ' + f'{len(dir_indices)}')
        if not par_flag:
            for w in tqdm(range(len(well_list))):
                stitch_well(w, well_list=well_list, cytometer_flag=cytometer_flag, out_dir=out_dir, overwrite_flag=overwrite_flag, out_shape=out_shape, ff_tile_dir=ff_tile_dir)
                
        else:
            process_map(partial(stitch_well, well_list=well_list, cytometer_flag=cytometer_flag, 
                                                                        out_dir=out_dir, overwrite_flag=overwrite_flag, out_shape=out_shape, ff_tile_dir=ff_tile_dir), 
                                        range(len(well_list)), max_workers=n_workers, stacksize=1)


    print('Done.')

def stitch_ff_from_keyence(data_root, n_workers=4, par_flag=False, overwrite_flag=False, n_stitch_samples=15, dir_list=None, write_dir=None):
    
    read_dir = os.path.join(data_root, 'raw_image_data', 'keyence', '')
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
        sub_name = path_leaf(dir_list[d])
        # sub_name = dir_path.replace(read_dir, "")

        # directories containing image tiles
        # depth_tile_dir = os.path.join(write_dir, "built_image_data", "keyence", "D_images", sub_name, '')
        ff_tile_dir = os.path.join(write_dir, "built_image_data", "keyence", "FF_images", sub_name, '')

        metadata_path = os.path.join(ff_tile_dir, 'metadata.csv')
        metadata_df = pd.read_csv(metadata_path, index_col=0)
        size_factor = metadata_df["Width (px)"].iloc[0] / 640
        time_ind_index = np.unique(metadata_df["time_int"])
        no_timelapse_flag = len(time_ind_index) == 0
        if no_timelapse_flag:
            out_shape = np.asarray([800, 630])*size_factor
        else:
            out_shape = np.asarray([1140, 630])*size_factor
        out_shape = out_shape.astype(int)

        # get list of subfolders
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
                n_images = len(glob.glob(ff_path + '/*.jpg'))
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
            # with open(depth_tile_dir + "/master_params.json", "w") as outfile:
            #     outfile.write(jason_params)

        # directories to write stitched files to
        # stitch_depth_dir = os.path.join(write_dir, "built_image_data", "stitched_depth_images", sub_name)
        stitch_ff_dir = os.path.join(write_dir, "built_image_data", "stitched_FF_images", sub_name)

        # if not os.path.isdir(stitch_depth_dir):
        #     os.makedirs(stitch_depth_dir)
        if not os.path.isdir(stitch_ff_dir):
            os.makedirs(stitch_ff_dir)

        # get list of subfolders
        # depth_folder_list = sorted(glob.glob(depth_tile_dir + "depth*"))
        ff_folder_list = sorted(glob.glob(ff_tile_dir + "ff*"))

        print(f'Stitching images in directory {d+1:01} of ' + f'{len(dir_indices)}')
        # Call parallel function to stitch images
        if not par_flag: #no_timelapse_flag:
            # out_shape[0] = 1230
            for f in tqdm(range(len(ff_folder_list))):
                stitch_experiment(f, ff_folder_list, ff_tile_dir, stitch_ff_dir, overwrite_flag, out_shape)

        else:
            process_map(partial(stitch_experiment, ff_folder_list=ff_folder_list, ff_tile_dir=ff_tile_dir, 
                                stitch_ff_dir=stitch_ff_dir, overwrite_flag=overwrite_flag, out_shape=out_shape), 
                                        range(len(ff_folder_list)), max_workers=n_workers, chunksize=1)

    



if __name__ == "__main__":

    overwrite_flag = True

    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    dir_list = ["20230525", "20231207"]
    # build FF images
    build_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list)
    # stitch FF images
    stitch_ff_from_keyence(data_root, overwrite_flag=overwrite_flag, dir_list=dir_list)