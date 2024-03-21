# script to define functions_folder for loading and standardizing fish movies
import os
import numpy as np
from skimage import io
import glob as glob
import torchvision
import torch
import torch.nn.functional as F
from src.functions.utilities import path_leaf
from src.functions.image_utils import gaussian_focus_stacker, LoG_focus_stacker
from src.functions.dataset_utils import set_inputs_to_device
from tqdm.contrib.concurrent import process_map
from functools import partial
from tqdm import tqdm
import pandas as pd
import time
import nd2
import cv2
from sklearn.cluster import KMeans


def create_composite_im(w, max_dir, ff_dir, out_dir, ff_image_list, overwrite_flag):

    # set patjs
    ff_path = ff_image_list[w]
    ff_name = path_leaf(ff_path)
    ff_dir = ff_path.replace(ff_name, "")
    max_dir = ff_dir.replace("FF", "fluo")
    max_name = ff_name.replace("ff", "max")
    max_path = os.path.join(max_dir, max_name)

    out_name = ff_name.replace("ff", "overlay")
    out_name = out_name.replace("png", "tif")

    if (not os.path.isfile(os.path.join(out_dir, out_name))) or overwrite_flag:
        # load
        im_fluo = io.imread(max_path)
        im_bf = io.imread(ff_path)

        # stack to composit RGB
        im_rgb = np.zeros((im_fluo.shape[0], im_fluo.shape[1], 3), dtype=np.float32)
        for i in range(3):
            im_rgb[:, :, i] = im_bf*0.25

        im_rgb[:, :, 1] += im_fluo
        im_rgb = im_rgb / (2**16-1)
        im_rgb[im_rgb > 1] = 1
        # generate save names
        io.imsave(os.path.join(out_dir, out_name), im_rgb, check_contrast=False)

    return {}



def build_composite_images(data_root, overwrite_flag=False, dir_list=None, write_dir=None, par_flag=False, n_workers=12):

    read_dir_root = os.path.join(data_root, 'raw_image_data', 'YX1') 
    if write_dir is None:
        write_dir = os.path.join(data_root, 'built_image_data') 
    #
    # metadata_path = os.path.join(data_root, "metadata", "well_metadata", "")
    # handle paths
    if dir_list is None:
        # Get a list of directories
        dir_list_raw = sorted(glob.glob(read_dir_root + "*"))
        dir_list = []
        for dd in dir_list_raw:
            if os.path.isdir(dd):
                dir_list.append(path_leaf(dd))

    # if rs_res is None:
    #     rs_res = np.asarray([3.2, 3.2])

    # filter for desired directories
    dir_indices = [d for d in range(len(dir_list)) if "ignore" not in dir_list[d]]

    for d in dir_indices:

        # initialize dictionary to metadata
        sub_name = dir_list[d]

        # depth_dir = os.path.join(write_dir, "stitched_depth_images", sub_name)
        ff_dir = os.path.join(write_dir, "stitched_FF_images", sub_name, "")
        max_dir = os.path.join(write_dir, "stitched_fluo_images", sub_name, "")
        out_dir = os.path.join(write_dir, "composite_images", sub_name, "")

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # get list of images
        ff_image_list = sorted(glob.glob(ff_dir + "*"))

        print("Generating composite images...")

        if par_flag:
            process_map(partial(create_composite_im, max_dir=max_dir, ff_dir=ff_dir, out_dir=out_dir, ff_image_list=ff_image_list),
                        range(len(ff_image_list)), max_workers=n_workers, chunksize=1)

        else:
            for w in tqdm(range(len(ff_image_list))):
                create_composite_im(w, max_dir=max_dir, ff_dir=ff_dir, out_dir=out_dir, ff_image_list=ff_image_list, overwrite_flag=overwrite_flag)



    print('Done.')



if __name__ == "__main__":

    overwrite_flag = True
    data_root = "E:/Nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    dir_list = ["20240314"]
    # build FF images
    # build_ff_from_keyence(data_root, write_dir=write_dir, overwrite_flag=True, dir_list=dir_list, ch_to_use=4)
    # stitch FF images
    build_composite_images(data_root=data_root, dir_list=dir_list, overwrite_flag=overwrite_flag)
