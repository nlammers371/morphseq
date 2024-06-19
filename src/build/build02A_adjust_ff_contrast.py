import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import glob
import ntpath
from tqdm import tqdm
from src.functions.utilities import path_leaf
import skimage.io as io
from skimage import exposure
from functools import partial
from tqdm.contrib.concurrent import process_map

def adjust_contrast(index, image_path_list):

    # load image
    im_path = image_path_list[index]
    im = io.imread(im_path)

    dtype = im.dtype

    # adjust
    im = exposure.equalize_hist(im)

    # convert
    if dtype == np.uint8:
        im = (im * 255).astype(np.uint8)
    elif dtype == np.uint16:
        im = (im * 65535).astype(np.uint16)
    else:
        raise ValueError('Unsupported image type')

    # save
    temp_path = im_path.replace("stitched_FF_images_raw", "stitched_FF_images")
    im_stub = path_leaf(temp_path)
    date_string = path_leaf(os.path.dirname(temp_path))
    root_dir = os.path.dirname(os.path.dirname(temp_path))
    out_path = os.path.join(root_dir, date_string + "_" + im_stub)
    io.imsave(out_path, im, check_contrast=False)

def adjust_contrast_wrapper(root, overwrite_flag=False, par_flag=True, n_workers=None):

    """
    Adjust the contrast of an image
    Parameters
    """

    if n_workers is None:
        n_workers = np.floor(os.cpu_count() / 4).astype(int)

    # make write directory
    out_dir = os.path.join(root, "built_image_data", "stitched_FF_images", "")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get folders with images to adjust
    path_to_images = os.path.join(root, "built_image_data", 'stitched_FF_images_raw', '*')
    project_list = sorted(glob.glob(path_to_images))
    project_list = [p for p in project_list if "ignore" not in p]
    project_list = [p for p in project_list if os.path.isdir(p)]

    # select subset of images to label
    image_path_list = []
    label_path_list = []
    exist_flags = []
    for ind, p in enumerate(tqdm(project_list, "Checking for pre-existing images")):
        im_list_temp = glob.glob(os.path.join(p, '*.png')) + glob.glob(os.path.join(p, '*.tif')) + glob.glob(
            os.path.join(p, '*.jpg'))
        image_path_list += im_list_temp

        _, project_name = ntpath.split(p)
        project_path_root = os.path.join(out_dir, project_name)
        if not os.path.isdir(project_path_root):
            os.makedirs(project_path_root)

        if not overwrite_flag:
            for imp in im_list_temp:
                _, tail = ntpath.split(imp)
                label_path = os.path.join(project_path_root, tail)
                # label_path = label_path.replace(".png", ".jpg")
                label_path_list.append(label_path)
                exist_flags.append(os.path.isfile(label_path))

    # remove images with previously existing labels if overwrite_flag=False
    if not overwrite_flag:
        image_path_list = [image_path_list[e] for e in range(len(image_path_list)) if not exist_flags[e]]
        # label_path_list = [label_path_list[e] for e in range(len(label_path_list)) if not exist_flags[e]]
        n_ex = np.sum(np.asarray(exist_flags) == 1)
        if n_ex > 0:
            print('Skipping ' + str(
                n_ex) + " previously segmented images. Set 'overwrite_flag=True' to overwrite")

    image_path_list = sorted(image_path_list)

    # iterate through images and adjust contrast
    if par_flag:
        process_map(partial(adjust_contrast, image_path_list=image_path_list),
                    range(len(image_path_list)), max_workers=n_workers, chunksize=10)
    else:
        for i in tqdm(range(len(image_path_list)), "Adjusting image contrast..."):
            adjust_contrast(i, image_path_list)

